"""Measure retrieval only. Test questions are never searched during a dev run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import statistics
import time

from sqlalchemy import func, select, text

from app.config import settings
from app.db import SessionLocal
from app.models import QueryLog
from eval.retrievers import (RERANKER_MODEL, TfidfRetriever, cross_encoder, e5_search,
                             e5_rerank, hybrid_rrf, hybrid_rerank, load_corpus)

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def test_integrity(questions):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"Контрольная сумма замороженного test:\s*`([0-9a-f]{64})`", readme)
    expected = match.group(1) if match else None
    actual = digest([row for row in questions if row["split"] == "test"])
    matches = actual == expected
    print(f"TEST_SHA256 expected={expected} actual={actual} match={matches}", flush=True)
    if not matches:
        print("WARNING: frozen test SHA-256 differs from README (or README hash is missing); stored hash was NOT changed", flush=True)
    return {"expected": expected, "actual": actual, "matches": matches}


def relevance(question, hits):
    gold_pages = {(source["filename"], page) for source in question["gold"] for page in source["pages"]}
    filenames = {source["filename"] for source in question["gold"]}
    rank = next((i for i, hit in enumerate(hits, 1) if (hit.filename, hit.page) in gold_pages), None)
    return rank, any(hit.filename in filenames for hit in hits)


def summarize(rows, k):
    count = len(rows)
    return {"n": count, "Hit@1": sum(row["rank"] == 1 for row in rows) / count,
            f"Hit@{k}": sum(row["rank"] is not None for row in rows) / count,
            f"MRR@{k}": sum(1 / row["rank"] if row["rank"] else 0 for row in rows) / count,
            f"Doc-Hit@{k}": sum(row["doc_hit"] for row in rows) / count,
            "Hit@20": sum(row["candidate_rank"] is not None for row in rows) / count,
            "mean_search_ms": statistics.mean(row["latency_ms"] for row in rows),
            "by_type": {kind: {"n": len(group), "hits": sum(r["rank"] is not None for r in group),
                               f"Hit@{k}": sum(r["rank"] is not None for r in group) / len(group)}
                        for kind in sorted({r["type"] for r in rows})
                        if (group := [r for r in rows if r["type"] == kind])}}


def history_count():
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(QueryLog))


def candidate_pipeline(search, query, k):
    candidates = search(query, 20)
    return candidates[:k], candidates


def write_reports(report):
    k, split = report["k"], report["split"]
    output = ROOT / "eval/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"retrieval_{split}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Retrieval: {split}", "",
             "Assistant-prepared labels; not human-reviewed ground truth. Baseline measurement, no tuning.", "",
             f"Answerable questions: {report['answerable_count']}; ignored unanswerable: {report['ignored_unanswerable_count']}.",
             f"Corpus: {report['corpus']['documents']} documents, {report['corpus']['chunks']} chunks; ready only, same repeatable-read snapshot.",
             f"Frozen test hash matches README: {report['test_integrity']['matches']}. Test evaluated: {split == 'test'}.", "",
             "n=24, один вопрос ≈ 4 п.п.; различия в Hit@5 менее 10 п.п. следует интерпретировать осторожно: статистическая значимость не проверялась." if report["answerable_count"] == 24 else "Small sample; statistical significance was not evaluated.", "",
             f"| Method | n | Hit@1 | Hit@{k} | MRR@{k} | Doc-Hit@{k} | Hit@20* | Mean search ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in report["methods"].items():
        lines.append(f"| {name} | {result['n']} | {result['Hit@1']:.4f} | {result[f'Hit@{k}']:.4f} | {result[f'MRR@{k}']:.4f} | {result[f'Doc-Hit@{k}']:.4f} | {result['Hit@20']:.4f} | {result['mean_search_ms']:.3f} |")
    lines += ["", "*Hit@20 for rerank methods is PRE-rerank candidate recall, not recall of five returned outputs. Other methods use their genuine first 20 hits. RRF sums 1/(60+rank), rank starts at 1, chunk_id is the fusion key; up to 40 union candidates are truncated to 20 before hybrid reranking.", "",
              f"Reranker: `{report['settings']['reranker_model']}`, CPU, raw (query, chunk_text) pairs without E5 prefixes. Rerank latency includes fresh candidate retrieval plus CrossEncoder scoring/ranking; no cached-candidate shortcut."]
    lines += ["", f"## Hit@{k} by type", "", "| Method | Type | Hits / n | Hit rate |", "| --- | --- | ---: | ---: |"]
    for name, result in report["methods"].items():
        for kind, values in result["by_type"].items():
            lines.append(f"| {name} | {kind} | {values['hits']} / {values['n']} | {values[f'Hit@{k}']:.4f} |")
    lines += ["", "## Excluded setup timings", "", "| Method | Index build ms | Warmup ms |", "| --- | ---: | ---: |"]
    for name, values in report["setup_timings"].items():
        lines.append(f"| {name} | {values['index_build_ms']:.3f} | {values['warmup_ms']:.3f} |")
    lines += ["", "Search timing is wall time around the retriever call. TF-IDF includes query vectorization, cosine similarity, sorting and hit construction. E5 includes query encoding, DB cosine search and hit construction. Imports, index building and one generic warmup per method are excluded. E5 warmup includes model initialization; no separate model-load-only claim is made.", "",
              f"QueryLog before/after: {report['query_log_before']} / {report['query_log_after']}; unchanged: {report['query_log_unchanged']}.",
              f"Duplicate ready filenames: {report['corpus']['duplicate_filenames']}. Missing gold pages: {report['missing_gold']}.", "",
              f"Model setting: `{report['settings']['embedding_model']}`; chunk size/overlap: {report['settings']['chunk_size']} / {report['settings']['chunk_overlap']}.",
              f"Corpus text+metadata SHA-256: `{report['corpus']['sha256']}`.",
              f"Evaluated questions SHA-256: `{report['evaluated_questions_sha256']}`.",
              f"Frozen test SHA-256: `{report['test_integrity']['actual']}`.", "",
              "No QueryLog writes, API requests, /ask, external LLM calls or automated tests. No chunk-to-page deduplication. Filename+page relevance and first relevant rank are used; document-only hit ignores page. Timings are one run on this shared desktop, not a hardware-independent benchmark. DB schema does not persist embedding model revisions, so the embedding-model setting is reported, not verified per stored vector. Dev/test share source documents; semantic labels still require human review.", ""]
    (output / f"retrieval_{split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]), flush=True)
    print(f"SAVED {output / f'retrieval_{split}.json'} and .md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.k <= 100:
        parser.error("k must be between 1 and 100; report metric names use the supplied k")
    if args.k > 20:
        parser.error("rerank methods have only 20 candidates; k must not exceed 20")
    questions = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    integrity = test_integrity(questions)
    selected = [row for row in questions if row["split"] == args.split and row["answerable"]]
    if not selected:
        raise ValueError("No answerable questions in requested split")
    before = history_count()
    report = {"split": args.split, "k": args.k, "created_at": datetime.now(timezone.utc).isoformat(),
              "answerable_count": len(selected), "ignored_unanswerable_count": sum(row["split"] == args.split and not row["answerable"] for row in questions),
              "type_denominators": dict(Counter(row["type"] for row in selected)),
              "test_integrity": integrity, "evaluated_questions_sha256": digest(selected),
              "settings": {"embedding_model": settings.EMBEDDING_MODEL, "embedding_dim": settings.EMBEDDING_DIM,
                           "chunk_size": settings.CHUNK_SIZE, "chunk_overlap": settings.CHUNK_OVERLAP,
                           "reranker_model": RERANKER_MODEL, "reranker_device": "cpu", "rrf_constant": 60},
              "environment": {"platform": platform.platform(), "python": platform.python_version(),
                              "packages": {name: version(name) for name in ("scikit-learn", "sentence-transformers", "torch", "SQLAlchemy", "pgvector")}},
              "setup_timings": {}, "methods": {}, "records": {}, "query_log_before": before}
        # All methods see the same transaction snapshot even if the user uploads later.
    with SessionLocal() as db:
        db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        corpus = load_corpus(db)
        documents = {row["document_id"]: row["filename"] for row in corpus}
        names = Counter(documents.values())
        report["corpus"] = {"documents": len(documents), "chunks": len(corpus), "sha256": digest(corpus),
                            "duplicate_filenames": {name: count for name, count in names.items() if count > 1},
                            "document_ids": sorted(documents), "filenames": sorted(names)}
        coverage = {(row["filename"], row["page"]) for row in corpus}
        missing = [{"id": q["id"], "filename": g["filename"], "page": page}
                   for q in selected for g in q["gold"] for page in g["pages"]
                   if (g["filename"], page) not in coverage]
        report["missing_gold"] = missing
        if missing:
            print("MISSING_GOLD " + json.dumps(missing, ensure_ascii=False), flush=True)
            raise ValueError("Gold coverage incomplete; no questions silently dropped")
        print(f"EVALUATING split={args.split} answerable={len(selected)} docs={len(documents)} chunks={len(corpus)}", flush=True)
        retrievers = {}
        for name in ("tfidf_word", "tfidf_char"):
            start = time.perf_counter()
            retrievers[name] = TfidfRetriever(corpus, name).search
            build_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            retrievers[name]("прогрев поиска", 20)
            report["setup_timings"][name] = {"index_build_ms": build_ms, "warmup_ms": (time.perf_counter() - start) * 1000}
        retrievers["e5"] = lambda query, k: e5_search(db, query, k)
        start = time.perf_counter()
        retrievers["e5"]("прогрев поиска", 20)
        report["setup_timings"]["e5"] = {"index_build_ms": 0, "warmup_ms": (time.perf_counter() - start) * 1000}
        char_search = retrievers["tfidf_char"]
        pipelines = {name: (lambda query, k, search=retrieve: candidate_pipeline(search, query, k))
                     for name, retrieve in retrievers.items()}
        pipelines["hybrid_rrf"] = lambda query, k: candidate_pipeline(lambda q, count: hybrid_rrf(db, char_search, q, count), query, k)
        start = time.perf_counter()
        cross_encoder().predict([("прогрев поиска", corpus[0]["text"])], show_progress_bar=False)
        report["setup_timings"]["cross_encoder_model"] = {"index_build_ms": 0, "warmup_ms": (time.perf_counter() - start) * 1000}
        pipelines["e5_rerank"] = lambda query, k: e5_rerank(db, query, k)
        pipelines["hybrid_rerank"] = lambda query, k: hybrid_rerank(db, char_search, query, k)
        for name in ("hybrid_rrf", "e5_rerank", "hybrid_rerank"):
            start = time.perf_counter()
            pipelines[name]("прогрев поиска", args.k)
            report["setup_timings"][name] = {"index_build_ms": 0, "warmup_ms": (time.perf_counter() - start) * 1000}
        for name, retrieve in pipelines.items():
            rows = []
            for question in selected:
                start = time.perf_counter()
                hits, candidates = retrieve(question["question"], args.k)
                latency = (time.perf_counter() - start) * 1000
                rank, doc_hit = relevance(question, hits)
                candidate_rank, _ = relevance(question, candidates[:20])
                rows.append({"id": question["id"], "type": question["type"], "rank": rank,
                             "candidate_rank": candidate_rank,
                             "doc_hit": doc_hit, "latency_ms": latency,
                             "candidate_ids": [hit.chunk_id for hit in candidates[:20]],
                             "hits": [{key: hit.model_dump()[key] for key in ("chunk_id", "document_id", "filename", "page", "score")} for hit in hits]})
            report["methods"][name] = summarize(rows, args.k)
            report["records"][name] = rows
            print(f"DONE {name}: {report['methods'][name]}", flush=True)
    report["query_log_after"] = history_count()
    report["query_log_unchanged"] = report["query_log_after"] == before
    if not report["query_log_unchanged"]:
        print("WARNING QueryLog count changed; possible concurrent writer, no records deleted", flush=True)
    write_reports(report)


if __name__ == "__main__":
    main()
