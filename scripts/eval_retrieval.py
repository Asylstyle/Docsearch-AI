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
from eval.retrievers import TfidfRetriever, e5_search, load_corpus

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
            "mean_search_ms": statistics.mean(row["latency_ms"] for row in rows),
            "by_type": {kind: {"n": len(group), "hits": sum(r["rank"] is not None for r in group),
                               f"Hit@{k}": sum(r["rank"] is not None for r in group) / len(group)}
                        for kind in sorted({r["type"] for r in rows})
                        if (group := [r for r in rows if r["type"] == kind])}}


def history_count():
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(QueryLog))
