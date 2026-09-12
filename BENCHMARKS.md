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
| topic classifier (XLM-R) | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 93.75% |
| dialect-aware (CAMeLBERT-DA) | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 31.25% |

- paired comparison verdict: XLM-R and CAMeLBERT-DA are tied on the frozen Arabic and Gulf macro-F1 evaluations; paired bootstrap delta is 0.0000 [0.0000, 0.0000]. Retain XLM-R as the incumbent because the quality comparison does not justify replacement.
- error taxonomy top categories: Systematic class confusion — 67/120 (55.8%); Label ambiguity — 40/120 (33.3%); Arabic orthographic variation — 13/120 (10.8%).
- top-3 prioritised fixes: (1) target `parks -> roads` confusion, estimated upper-bound validation accuracy delta +7.0 pp; (2) improve contextual disambiguation of park walkways, +4.2 pp; (3) strengthen Arabic orthographic normalisation and augmentation, +1.4 pp.

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


---

## Lab 7 — CPU Inference Optimisation and Serving

### Benchmark methodology

Lab 7 latency measurements were collected on a CPU runtime with the
CPU thread count pinned to **4 threads**.

The supplied production serving mix was used:

`data/serving/bench_mix.npy`

- Samples: **2,000**
- Warm-up requests were excluded.
- Model loading was excluded.
- Tokenization was performed before the timer and therefore excluded.
- Reported latency is **model inference latency only**.
- GPU latency was not used as Lab 7 evidence.

### Optimisation ladder

| Configuration | Samples | Mean (ms) | p50 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|
| PyTorch FP32 — padded 512 | 2000 | 1941.57 | 1693.32 | 3038.94 |
| PyTorch FP32 — dynamic <=128 | 2000 | 140.98 | 120.92 | 397.69 |
| ONNX FP32 — dynamic <=128 | 2000 | 176.85 | 151.67 | 382.99 |
| **ONNX INT8 — dynamic <=128** | **2000** | **114.02** | **96.76** | **297.19** |

### Optimisation observations

Dynamic sequence length was the largest single latency improvement.
Replacing fixed padding to length 512 with dynamic sequences capped at
128 reduced unnecessary computation substantially.

ONNX FP32 did not improve mean or p50 latency over the PyTorch dynamic
configuration on this CPU environment. It did, however, slightly improve
p99 latency.

The final ONNX INT8 configuration achieved the best overall latency:

- Mean: **114.02 ms**
- p50: **96.76 ms**
- p99: **297.19 ms**

Compared with the original PyTorch FP32 padded-512 baseline:

- Mean speed-up: approximately **17.03x**
- p50 speed-up: approximately **17.50x**
- p99 speed-up: approximately **10.23x**

### Model size

| Artefact | Approximate size |
|---|---:|
| ONNX FP32 | 1.1 GB |
| ONNX INT8 | 266 MB |

The INT8 artefact is approximately **4.1x smaller** than ONNX FP32.

### Correctness / quantisation parity

ONNX FP32 and ONNX INT8 predictions were compared over the complete
2,000-sample production benchmark mix.

| Check | Result |
|---|---:|
| Samples checked | 2000 |
| Prediction matches | 2000 |
| Prediction mismatches | 0 |
| Prediction agreement | **100.00%** |
| Maximum absolute logit difference | 2.12650514 |

This is prediction agreement evidence and is not a replacement for the
required labelled macro-F1 quality-tax measurement.

### FastAPI serving smoke tests

The Lab 7 classifier is exposed through:

- `GET /health`
- `POST /v1/classify`

| Test | HTTP status | Result |
|---|---:|---|
| Health endpoint | 200 | ready |
| Arabic classification | 200 | lighting |
| English classification | 200 | lighting |
| Empty text validation | 422 | correctly rejected |

### Preliminary API load test

A preliminary local load test was performed with:

- Requests: **100**
- Concurrency: **4**
- Warm-up: **10 requests**

| Metric | Result |
|---|---:|
| Successful requests | **100** |
| Failed requests | **0** |
| Total time | 4.74 s |
| Throughput | **21.08 requests/s** |
| Mean response latency | 186.30 ms |
| p50 response latency | 183.86 ms |
| p99 response latency | 244.78 ms |

This preliminary test does **not** replace the official Lab 7 `hey`
load test at 16 concurrent clients for 60 seconds.

### Current classifier production choice

The current selected classifier is:

**ONNX INT8 + dynamic sequence length <=128 + CPUExecutionProvider**

This choice will be finalised after the required labelled quality-tax,
NER comparison, startup-canary, serving-contract and official HTTP-load
evidence are completed.


### Lab 7 — Paired FP32 vs INT8 quality check

The quantised models were evaluated against their FP32 ONNX rollback
artefacts on the labelled frozen test sets.

#### Classifier

| Metric | FP32 | INT8 |
|---|---:|---:|
| Frozen-test samples | 1208 | 1208 |
| Macro-F1 | 1.000000 | 1.000000 |
| Accuracy | 1.000000 | 1.000000 |

- Macro-F1 quality tax: **0.0000 points**
- 95% bootstrap CI for quality tax: **[0.0000, 0.0000] points**
- FP32/INT8 prediction agreement: **100.00%**
- Lab target: classifier quality tax <= **1 macro-F1 point**
- Result: **PASS**

#### NER

| Metric | FP32 | INT8 |
|---|---:|---:|
| Frozen-test sentences | 400 | 400 |
| Evaluated labelled tokens | 4343 | 4343 |
| Entity-F1 | 1.000000 | 1.000000 |
| Token accuracy | 1.000000 | 1.000000 |

- Entity-F1 quality tax: **0.0000 points**
- 95% bootstrap CI for quality tax: **[0.0000, 0.0000] points**
- FP32/INT8 labelled-token agreement: **100.00%**

The NER INT8 artefact therefore shows no measured quality degradation on
the frozen test set. The final NER serving decision will also consider
its measured CPU latency, as required by Lab 7.


### Lab 7 — Official HTTP load test

The final FastAPI classifier service was tested with `hey` using the
required Lab 7 load profile:

- Duration: **60 seconds**
- Concurrent clients: **16**
- Endpoint: `POST /v1/classify`
- Serving model: **ONNX dynamic INT8 classifier**
- CPU thread setting: **OMP_NUM_THREADS=4**
- Colab host available logical CPUs: **2**
- Startup canaries: **GREEN**
- Serving contract: **2 passed**

| Metric | Measured |
|---|---:|
| Total requests | 1759 |
| Successful HTTP 200 | 1759 |
| Errors | 0 |
| Requests/sec | 29.1732 |
| Average latency | 547.3 ms |
| p50 | 494.0 ms |
| p90 | 799.2 ms |
| p95 | 843.7 ms |
| p99 | 930.9 ms |
| Slowest | 1090.7 ms |

Lab HTTP target:

- p99 <= **40 ms**: **NOT MET on this Colab CPU**
- 0 errors: **PASS**
- startup canaries green: **PASS**

The absolute HTTP latency target was not met on the measured Colab
environment. The host exposed only two logical CPUs while the lab
thread configuration was pinned to four threads. The measured result
is reported without modification.
