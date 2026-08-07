# Retrieval: dev

Assistant-prepared labels; not human-reviewed ground truth. Baseline measurement, no tuning.

Answerable questions: 24; ignored unanswerable: 6.
Corpus: 10 documents, 644 chunks; ready only, same repeatable-read snapshot.
Frozen test hash matches README: True. Test evaluated: False.

| Method | n | Hit@1 | Hit@5 | MRR@5 | Doc-Hit@5 | Mean search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf_word | 24 | 0.2917 | 0.6250 | 0.3896 | 0.9583 | 1.733 |
| tfidf_char | 24 | 0.3333 | 0.7083 | 0.4701 | 1.0000 | 9.619 |
| e5 | 24 | 0.5417 | 0.7083 | 0.5951 | 1.0000 | 230.155 |

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
| e5 | paraphrase | 6 / 9 | 0.6667 |
| e5 | simple | 7 / 9 | 0.7778 |

## Excluded setup timings

| Method | Index build ms | Warmup ms |
| --- | ---: | ---: |
| tfidf_word | 69.932 | 2.001 |
| tfidf_char | 461.358 | 31.172 |
| e5 | 0.000 | 15814.094 |

Search timing is wall time around the retriever call. TF-IDF includes query vectorization, cosine similarity, sorting and hit construction. E5 includes query encoding, DB cosine search and hit construction. Imports, index building and one generic warmup per method are excluded. E5 warmup includes model initialization; no separate model-load-only claim is made.

QueryLog before/after: 2 / 2; unchanged: True.
Duplicate ready filenames: {}. Missing gold pages: [].

Model setting: `intfloat/multilingual-e5-small`; chunk size/overlap: 800 / 100.
Corpus text+metadata SHA-256: `237478d700f815e724f4e23aa46204f992a039a6f1c0bd6d6ee60dc48e02a26d`.
Evaluated questions SHA-256: `97fbf7f2da1ddc9ff8325c6e83fc7674521bb15ca0396d5944f2f4f75c98c012`.
Frozen test SHA-256: `369910a237ad347605eafe18b76192db51681e357fd37e17d5a8caf60c76836f`.

No QueryLog writes, API requests, /ask, external LLM calls or automated tests. No chunk-to-page deduplication. Filename+page relevance and first relevant rank are used; document-only hit ignores page. Timings are one run on this shared desktop, not a hardware-independent benchmark. DB schema does not persist embedding model revisions, so the embedding-model setting is reported, not verified per stored vector. Dev/test share source documents; semantic labels still require human review.
