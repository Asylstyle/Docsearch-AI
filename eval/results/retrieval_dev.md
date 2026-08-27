# Retrieval: dev

Assistant-prepared labels; not human-reviewed ground truth. Baseline measurement, no tuning.

Answerable questions: 24; ignored unanswerable: 6.
Corpus: 10 documents, 644 chunks; ready only, same repeatable-read snapshot.
Frozen test hash matches README: True. Test evaluated: False.

n=24, один вопрос ≈ 4 п.п.; различия в Hit@5 менее 10 п.п. следует интерпретировать осторожно: статистическая значимость не проверялась.

| Method | n | Hit@1 | Hit@5 | MRR@5 | Doc-Hit@5 | Hit@20* | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf_word | 24 | 0.2917 | 0.6250 | 0.3896 | 0.9583 | 0.8750 | 2.066 | 1.964 | 2.658 |
| tfidf_char | 24 | 0.3333 | 0.7083 | 0.4875 | 1.0000 | 1.0000 | 9.001 | 8.723 | 10.835 |
| e5 | 24 | 0.5417 | 0.7500 | 0.6111 | 1.0000 | 0.9167 | 31.885 | 29.895 | 39.008 |
| hybrid_rrf | 24 | 0.4583 | 0.7917 | 0.6042 | 1.0000 | 1.0000 | 43.308 | 41.320 | 53.822 |
| e5_rerank | 24 | 0.4583 | 0.8333 | 0.6125 | 1.0000 | 0.9167 | 1188.411 | 1153.978 | 1419.956 |
| hybrid_rerank | 24 | 0.4167 | 0.7917 | 0.5764 | 1.0000 | 1.0000 | 1268.308 | 1235.246 | 1554.045 |

*Hit@20 for rerank methods is PRE-rerank candidate recall, not recall of five returned outputs. Other methods use their genuine first 20 hits. RRF sums 1/(60+rank), rank starts at 1, chunk_id is the fusion key; up to 40 union candidates are truncated to 20 before hybrid reranking.

Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, CPU, raw (query, chunk_text) pairs without E5 prefixes. Rerank latency includes fresh candidate retrieval plus CrossEncoder scoring/ranking; no cached-candidate shortcut.

## Hit@5 by type

| Method | Type | Hits / n | Hit rate |
| --- | --- | ---: | ---: |
| tfidf_word | cross_university | 4 / 6 | 0.6667 |
| tfidf_word | paraphrase | 6 / 9 | 0.6667 |
| tfidf_word | simple | 5 / 9 | 0.5556 |
| tfidf_char | cross_university | 3 / 6 | 0.5000 |
| tfidf_char | paraphrase | 7 / 9 | 0.7778 |
| tfidf_char | simple | 7 / 9 | 0.7778 |
| e5 | cross_university | 4 / 6 | 0.6667 |
| e5 | paraphrase | 7 / 9 | 0.7778 |
| e5 | simple | 7 / 9 | 0.7778 |
| hybrid_rrf | cross_university | 4 / 6 | 0.6667 |
| hybrid_rrf | paraphrase | 7 / 9 | 0.7778 |
| hybrid_rrf | simple | 8 / 9 | 0.8889 |
| e5_rerank | cross_university | 6 / 6 | 1.0000 |
| e5_rerank | paraphrase | 6 / 9 | 0.6667 |
| e5_rerank | simple | 8 / 9 | 0.8889 |
| hybrid_rerank | cross_university | 5 / 6 | 0.8333 |
| hybrid_rerank | paraphrase | 6 / 9 | 0.6667 |
| hybrid_rerank | simple | 8 / 9 | 0.8889 |

## Excluded setup timings

| Method | Index build ms | Warmup ms |
| --- | ---: | ---: |
| tfidf_word | 75.717 | 38.778 |
| tfidf_char | 540.422 | 116.591 |
| e5 | 0.000 | 13260.446 |
| cross_encoder_model | 0.000 | 6640.465 |
| hybrid_rrf | 0.000 | 110.979 |
| e5_rerank | 0.000 | 6214.832 |
| hybrid_rerank | 0.000 | 3728.808 |

Search timing is wall time around the complete retriever call, including all fresh candidate retrieval. TF-IDF includes query vectorization, cosine similarity, sorting and hit construction. E5 includes query encoding, DB cosine search and hit construction. Three fixed non-evaluation strings of varied length warm every pipeline; models/index setup are excluded. Methods are shuffled deterministically per question (seed 42) to avoid fixed method-block order. No candidate or query-embedding cache is added.

## Mean measured components (ms)

| Method | Components |
| --- | --- |
| tfidf_word | {"retrieval_ms": 2.0581932500742064} |
| tfidf_char | {"retrieval_ms": 8.988809499934783} |
| e5 | {"retrieval_ms": 31.86837637520057} |
| hybrid_rrf | {"char_ms": 10.286716541637967, "e5_ms": 32.49538600005053, "fusion_ms": 0.09327804169364147} |
| e5_rerank | {"e5_ms": 32.0848492500924, "rerank_ms": 1157.0856667082883} |
| hybrid_rerank | {"char_ms": 9.917962791784399, "e5_ms": 32.416155666624036, "fusion_ms": 0.088823083274292, "rerank_ms": 1220.6568097915351} |

Torch thread state: {'before_warmup': [6, 6], 'after_warmup': [6, 6], 'after_measurement': [6, 6]}.
The prior fixed-block run had E5 min/median/max 36.4/124.6/1215.4 ms, versus RRF 29.9/36.8/43.0 ms. This showed order/startup sensitivity, not evidence that fusion inherently speeds up E5. Installed CPU encode had no general embedding cache (only an unrelated HPU tensor-cache branch). No application memoization exists; each E5 path calls search_chunks -> embed_query -> encode + SQL. Exact low-level runtime cause was not isolated. The corrected protocol and component boundaries are reported rather than claiming statistical significance.

QueryLog before/after: 2 / 2; unchanged: True.
Duplicate ready filenames: {}. Missing gold pages: [].

Model setting: `intfloat/multilingual-e5-small`; chunk size/overlap: 800 / 100.
Corpus text+metadata SHA-256: `237478d700f815e724f4e23aa46204f992a039a6f1c0bd6d6ee60dc48e02a26d`.
Evaluated questions SHA-256: `5e175890b865be39a8074381ae32050534da92db0596995271ba9f887c58af20`.
Frozen test SHA-256: `a89172651d233cb9ddde66c68939cc51377ab20a40066b337644917c3773b0f2`.

No QueryLog writes, API requests, /ask, external LLM calls or automated tests. No chunk-to-page deduplication. Filename+page relevance and first relevant rank are used; document-only hit ignores page. Timings are one run on this shared desktop, not a hardware-independent benchmark. DB schema does not persist embedding model revisions, so the embedding-model setting is reported, not verified per stored vector. Dev/test share source documents; semantic labels still require human review.
