# Retrieval: test

Assistant-prepared labels; not human-reviewed ground truth. Baseline measurement, no tuning.

Answerable questions: 16; ignored unanswerable: 4.
Corpus: 10 documents, 644 chunks; ready only, same repeatable-read snapshot.
Frozen test hash matches README: True. Test evaluated: True.

Small sample; statistical significance was not evaluated.

| Method | n | Hit@1 | Hit@5 | MRR@5 | Doc-Hit@5 | Hit@20* | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf_word | 16 | 0.2500 | 0.3750 | 0.3021 | 1.0000 | 0.8125 | 2.416 | 1.887 | 4.641 |
| tfidf_char | 16 | 0.3750 | 0.6250 | 0.4740 | 1.0000 | 0.9375 | 9.330 | 8.780 | 12.566 |
| e5 | 16 | 0.4375 | 0.6250 | 0.5052 | 0.9375 | 1.0000 | 34.720 | 33.476 | 50.806 |
| hybrid_rrf | 16 | 0.4375 | 0.7500 | 0.5094 | 1.0000 | 0.9375 | 42.185 | 41.879 | 51.679 |
| e5_rerank | 16 | 0.5000 | 0.7500 | 0.5885 | 1.0000 | 1.0000 | 1346.634 | 1241.143 | 2029.721 |
| hybrid_rerank | 16 | 0.5625 | 0.8125 | 0.6427 | 1.0000 | 0.9375 | 1361.644 | 1242.834 | 1866.427 |

*Hit@20 for rerank methods is PRE-rerank candidate recall, not recall of five returned outputs. Other methods use their genuine first 20 hits. RRF sums 1/(60+rank), rank starts at 1, chunk_id is the fusion key; up to 40 union candidates are truncated to 20 before hybrid reranking.

Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, CPU, raw (query, chunk_text) pairs without E5 prefixes. Rerank latency includes fresh candidate retrieval plus CrossEncoder scoring/ranking; no cached-candidate shortcut.

## Hit@5 by type

| Method | Type | Hits / n | Hit rate |
| --- | --- | ---: | ---: |
| tfidf_word | cross_university | 1 / 4 | 0.2500 |
| tfidf_word | paraphrase | 2 / 6 | 0.3333 |
| tfidf_word | simple | 3 / 6 | 0.5000 |
| tfidf_char | cross_university | 2 / 4 | 0.5000 |
| tfidf_char | paraphrase | 4 / 6 | 0.6667 |
| tfidf_char | simple | 4 / 6 | 0.6667 |
| e5 | cross_university | 2 / 4 | 0.5000 |
| e5 | paraphrase | 5 / 6 | 0.8333 |
| e5 | simple | 3 / 6 | 0.5000 |
| hybrid_rrf | cross_university | 3 / 4 | 0.7500 |
| hybrid_rrf | paraphrase | 5 / 6 | 0.8333 |
| hybrid_rrf | simple | 4 / 6 | 0.6667 |
| e5_rerank | cross_university | 2 / 4 | 0.5000 |
| e5_rerank | paraphrase | 6 / 6 | 1.0000 |
| e5_rerank | simple | 4 / 6 | 0.6667 |
| hybrid_rerank | cross_university | 4 / 4 | 1.0000 |
| hybrid_rerank | paraphrase | 6 / 6 | 1.0000 |
| hybrid_rerank | simple | 3 / 6 | 0.5000 |

## Excluded setup timings

| Method | Index build ms | Warmup ms |
| --- | ---: | ---: |
| tfidf_word | 79.955 | 18.711 |
| tfidf_char | 385.488 | 78.250 |
| e5 | 0.000 | 7913.506 |
| cross_encoder_model | 0.000 | 12940.367 |
| hybrid_rrf | 0.000 | 228.612 |
| e5_rerank | 0.000 | 8360.442 |
| hybrid_rerank | 0.000 | 3998.788 |

Search timing is wall time around the complete retriever call, including all fresh candidate retrieval. TF-IDF includes query vectorization, cosine similarity, sorting and hit construction. E5 includes query encoding, DB cosine search and hit construction. Three fixed non-evaluation strings of varied length warm every pipeline; models/index setup are excluded. Methods are shuffled deterministically per question (seed 42) to avoid fixed method-block order. No candidate or query-embedding cache is added.

## Mean measured components (ms)

| Method | Components |
| --- | --- |
| tfidf_word | {"retrieval_ms": 2.407810687429901} |
| tfidf_char | {"retrieval_ms": 9.31803043738455} |
| e5 | {"retrieval_ms": 34.70170224983349} |
| hybrid_rrf | {"char_ms": 9.504019125074592, "e5_ms": 31.09404750006206, "fusion_ms": 0.08533543751809702} |
| e5_rerank | {"e5_ms": 34.00346981260327, "rerank_ms": 1311.9905055000345} |
| hybrid_rerank | {"char_ms": 9.175747000028878, "e5_ms": 30.91233318770037, "fusion_ms": 0.08431200001268735, "rerank_ms": 1210.0037542498967} |

Torch thread state: {'before_warmup': [6, 6], 'after_warmup': [6, 6], 'after_measurement': [6, 6]}.
Each question/method is searched three times freshly; latency_ms is the median. p50/p95 summarize per-question medians (linear interpolation). Component medians need not sum exactly to total median. No query/result caches shared across methods.
The prior fixed-block run had E5 min/median/max 36.4/124.6/1215.4 ms, versus RRF 29.9/36.8/43.0 ms. This showed order/startup sensitivity, not evidence that fusion inherently speeds up E5. Installed CPU encode had no general embedding cache (only an unrelated HPU tensor-cache branch). No application memoization exists; each E5 path calls search_chunks -> embed_query -> encode + SQL. Exact low-level runtime cause was not isolated. The corrected protocol and component boundaries are reported rather than claiming statistical significance.

QueryLog before/after: 3 / 3; unchanged: True.
Duplicate ready filenames: {}. Missing gold pages: [].

Model setting: `intfloat/multilingual-e5-small`; chunk size/overlap: 800 / 100.
Corpus text+metadata SHA-256: `237478d700f815e724f4e23aa46204f992a039a6f1c0bd6d6ee60dc48e02a26d`.
Evaluated questions SHA-256: `bb29a71fb9d534438e6c85a3ce082f08417c0d799f15ab92cbababdeef0c45fa`.
Frozen test SHA-256: `a89172651d233cb9ddde66c68939cc51377ab20a40066b337644917c3773b0f2`.

No QueryLog writes, API requests, /ask, external LLM calls or automated tests. No chunk-to-page deduplication. Filename+page relevance and first relevant rank are used; document-only hit ignores page. Timings are one run on this shared desktop, not a hardware-independent benchmark. DB schema does not persist embedding model revisions, so the embedding-model setting is reported, not verified per stored vector. Dev/test share source documents; semantic labels still require human review.
