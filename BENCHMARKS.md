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
|---|---|---:|---:|---|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | not captured (CPU) |
| XLM-R topic classifier | macro-F1 | 1.0000 | 0.9992 | not captured (Colab T4) |
| NER | entity-F1 | 1.0000 | 1.0000 | not captured (Colab T4) |
| QA | span/null smoke | 9/9 | 3/3 | N/A (pretrained SQuAD2) |

- QA smoke target: PASS — overall 12/12.
- Grouped split integrity: PASS — zero `citizen_group_id` overlap between train, validation, and test.
- Topic-classifier validation delta over TF-IDF baseline: +0.0000.
- Topic-classifier frozen-test delta over TF-IDF baseline: -0.0008.
- The requested transformer target of baseline +0.08 was mathematically unattainable in this run because the leakage-safe TF-IDF baseline already achieved macro-F1 = 1.0000, while macro-F1 cannot exceed 1.0000.
- Results above reflect the latest documented Google Colab run in `notebooks/00_colab_setup.ipynb`.

## Lab 4 — Arabic pipeline and model bake-off

### NER clitic segmentation

| NER setting | Frozen test LOCATION recall |
|---|---:|
| Baseline (without segmentation) | 1.0000 |
| CAMeL Tools d3tok segmentation | 1.0000 |
| LOCATION recall delta | +0.0000 |

- The requested improvement of about +4 LOCATION recall points was not attainable on this supplied dataset because the baseline recall was already 1.0000, which is the maximum possible recall.
- The d3tok segmentation path preserved LOCATION recall without degradation.

### Arabic model bake-off

| Checkpoint | macro-F1 all Arabic | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| XLM-R incumbent | 1.0000 | 1.0000 | 1.0000 | 1.685 |
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 1.412 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | 1.412 |

- All three checkpoints tied at 1.0000 macro-F1 on the Gulf slice.
- The requested +4-point Gulf improvement over the Day-2 incumbent was not attainable because the incumbent already achieved the maximum possible macro-F1 of 1.0000.
- CAMeLBERT-mix and CAMeLBERT-DA achieved lower Arabic token fertility (1.412) than XLM-R (1.685).
- No Arabic checkpoint demonstrated a measurable Gulf macro-F1 advantage over the incumbent in this run.

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | 0.0256 | 0.0175 | 18.35 ms |
| + cross-encoder rerank | 0.0026 | 0.0015 | 331.28 ms |
| cross-lingual slice (cross-language gold, reranked) | 0.0000 | 0.0000 | 331.28 ms |

- no-answer empty-correct: 20 / 20 at min_score=-3.0
- threshold rationale: observed top reranker scores ranged from -5.0148 for no-answer queries to -1.3641 or higher for answerable queries; -3.0 lies between these observed ranges. The supplied no-answer queries are identical, so this threshold result should be interpreted with that limitation.
- cross-lingual gap (same-language - cross-language, reranked): recall@10 = 0.0038; MRR@10 = 0.0015
- Hit@10 diagnostic: bi-encoder = 0.0692; reranked = 0.0077
- target status: recall@10 >= 0.80 and MRR@10 >= 0.70 were not met; no-answer correctness target was met.
- L2-normalisation diagnosis: normalized recall@10/MRR@10 = 0.0256/0.0175; unnormalised = 0.0154/0.0080.
- FAISS tie diagnostic: direct k=10 versus k=50 then taking the first 10 changed the top-10 ordering for 104/130 queries.

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
