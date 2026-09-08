# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit

| Tokenizer  | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
| ---------- | -----------: | -----------: | ---------: | ---------: | ----------: |
| mBERT      |        2.153 |        1.510 |       27.0 |       25.0 |     0.00453 |
| XLM-R      |        1.672 |        1.434 |       21.0 |       23.0 |     0.00000 |
| CAMeLBERT  |        1.405 |        2.705 |       20.0 |       38.0 |     0.00804 |
| DistilBERT |        4.527 |        1.298 |       47.0 |       21.0 |     0.00215 |

* Golden preprocessing: 25 / 25 passed
* PII masking recall: 60 / 60 = 100%

## Lab 2 — Attention diagnostics

| Metric                                       |           Result |
| -------------------------------------------- | ---------------: |
| Maximum absolute difference vs PyTorch       |     0.0000002384 |
| Numerical equivalence tolerance              |    < 1e-6 — PASS |
| Future-token attention mass with causal mask |              0.0 |
| PAD attention mass with correct mask         |       0.00000000 |
| PAD attention mass without mask              |       0.07816089 |
| Best adjacency-looking head                  | Layer 5, Head 10 |
| Adjacency mass                               |         0.941012 |
| Strongest `[SEP]` sink head                  |  Layer 1, Head 5 |
| `[SEP]` attention mass                       |         0.273421 |

* Causal attention matrix: lower triangular — PASS
* Pad-attention leakage: detected when the attention mask is omitted


## Lab 3 — Models

| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | CPU baseline |
| XLM-R topic classifier | macro-F1 | 1.0000 | 1.0000 | not captured in initial run |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

- Topic-classifier delta over TF-IDF baseline: +0.0000
- Grouped split integrity: PASS — zero `citizen_group_id` overlap between train, validation, and test.
- Transformer target of baseline +0.08 was mathematically unattainable in this run because the leakage-safe TF-IDF baseline already achieved macro-F1 = 1.0000.

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
