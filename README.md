# Bayan | بيان

## SDA-AIE-211 — Natural Language Processing with Transformers

**Bayan** is a bilingual Arabic-English Natural Language Processing project developed as part of the **SDA-AIE-211 Natural Language Processing with Transformers** course.

The project builds an end-to-end NLP pipeline for analysing citizen feedback. It covers bilingual preprocessing, privacy protection, transformer-based topic classification, Arabic-aware modelling, semantic search, confidence-aware evaluation, behavioural testing, and model documentation.

---

## Repository Links

### Student Repository

[ns3271585-arch/Bayan](https://github.com/ns3271585-arch/Bayan)

### SDAIA Academy

[SDAIAAcademy](https://github.com/SDAIAAcademy)

### Course Repository

[AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course](https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course)

---

## Project Overview

Bayan processes Arabic and English citizen feedback and provides a complete NLP workflow that includes:

- Arabic and English text preprocessing
- Personally Identifiable Information (PII) protection
- Transformer tokenisation and diagnostics
- Topic classification
- Named Entity Recognition
- Extractive Question Answering
- Arabic normalisation
- Dialect-aware model comparison
- Semantic search using FAISS
- Cross-encoder re-ranking
- Confidence intervals
- Sliced evaluation
- Behavioural testing
- Manual error analysis
- Model cards
- Reproducible evaluation reports

---

## Completed Labs

### Lab 1 — Bilingual Preprocessing

Implemented a bilingual preprocessing pipeline for Arabic and English text.

Key work:

- Arabic and English text normalisation
- PII masking and protection
- Reproducible preprocessing versioning
- Transformer tokenisation
- Preprocessing tests
- PII recall evaluation

**Preprocessing version:** `1.2.0`

Lab 1 validation:

```text
Preprocessing tests: passed
PII recall cases: 60 / 60
```

---

### Lab 2 — Attention and Transformer Diagnostics

Implemented and evaluated transformer attention utilities and diagnostic checks.

Key work:

- Attention implementation
- Transformer diagnostics
- Shape and masking validation
- Automated attention tests

Lab 2 validation:

```text
Attention tests: passed
```

---

### Lab 3 — Topic Classification, NER, and QA

Built the main supervised NLP components using leakage-safe grouped data splits.

Tasks implemented:

- Topic classification
- Named Entity Recognition
- Extractive Question Answering
- Group-aware train/validation/test splitting

Topic classification models included:

- TF-IDF baseline
- XLM-R transformer classifier

Additional evaluation covered NER alignment and QA behaviour.

---

### Lab 4 — Arabic Model Bake-Off

Evaluated Arabic-specific transformer models against the bilingual XLM-R incumbent.

Models:

- `xlm-roberta-base`
- `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- `CAMeL-Lab/bert-base-arabic-camelbert-da`

Arabic frozen-test results:

| Model | All Arabic Macro-F1 | Gulf Macro-F1 | MSA Macro-F1 | Tokenizer Fertility |
|---|---:|---:|---:|---:|
| XLM-R incumbent | 1.0000 | 1.0000 | 1.0000 | 1.685 |
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 | 1.412 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 | 1.412 |

Although the models achieved the same frozen-test quality, the Arabic CAMeLBERT tokenizers produced lower fertility.

XLM-R remained the incumbent because the quality comparison did not demonstrate a measurable improvement that justified replacement.

---

### Lab 5 — Bilingual Semantic Search

Implemented semantic search for bilingual citizen feedback.

The retrieval pipeline includes:

1. Text preprocessing
2. Dense embedding generation
3. L2 normalisation
4. FAISS indexing
5. Bi-encoder retrieval
6. Cross-encoder re-ranking
7. Minimum-score rejection
8. Index manifest validation

The persisted search index records model and preprocessing metadata to prevent serving/index version mismatch.

Evaluation included:

- Recall@10
- MRR
- Hit@10
- Arabic evaluation
- English evaluation
- Cross-lingual evaluation
- No-answer threshold evaluation
- Retrieval latency
- L2-normalisation bug diagnosis

A tuned no-answer threshold correctly rejected all supplied no-answer examples:

```text
No-answer evaluation: 20 / 20
```

The search evaluation also exposed important retrieval-quality limitations, which are retained in the benchmark evidence rather than hidden.

---

### Lab 6 — Honest Evaluation and Model Cards

Lab 6 adds statistical, sliced, behavioural, and manual evaluation on top of the trained models.

Implemented:

- Bootstrap confidence intervals
- Paired bootstrap comparison
- Sliced evaluation
- Behavioural evaluation
- Minimum-functionality testing
- Manual review of 120 validation errors
- Extended error taxonomy
- Prioritised fixes
- Evaluation report generation
- Three model cards
- Hand-written known limitations

Automated evaluation tests:

```text
6 passed
```

---

### Lab 7 — CPU Optimisation and Model Serving

Lab 7 adds CPU-oriented inference optimisation and a production-style FastAPI serving path for the Bayan topic classifier.

Implemented work includes:

- CPU inference benchmarking with four pinned inference threads
- Dynamic padding and sequence-length optimisation
- Classifier export from PyTorch to ONNX FP32
- Dynamic INT8 quantisation
- FP32 rollback artefact retention
- Paired FP32/INT8 quality evaluation
- Bootstrap confidence intervals for quantisation quality tax
- FastAPI `POST /v1/classify`
- `GET /health`
- Startup compatibility and behaviour canaries
- HTTP load testing with 16 concurrent clients for 60 seconds

#### Classifier CPU Benchmark

The production benchmark uses the supplied `data/serving/bench_mix.npy` mixture with 2,000 examples.

| Runtime | Input strategy | p50 | p99 |
|---|---|---:|---:|
| PyTorch FP32 | padded to 512 | 1693.32 ms | 3038.94 ms |
| PyTorch FP32 | dynamic, max 128 | 120.92 ms | 397.69 ms |
| ONNX FP32 | dynamic, max 128 | 151.67 ms | 382.99 ms |
| ONNX INT8 | dynamic, max 128 | 96.76 ms | 297.19 ms |

The ONNX INT8 classifier reduced p99 latency by approximately **10.23x** relative to the original padded-512 PyTorch baseline.

The absolute Lab 7 bare-inference target of **p99 <= 25 ms** was not reached on the measured Colab CPU environment.

#### Classifier Artefact Size

| Artefact | Size |
|---|---:|
| ONNX FP32 | 1060.91 MB |
| ONNX INT8 | 265.93 MB |

Dynamic INT8 quantisation reduced the classifier artefact size by approximately **74.93%**.

#### Classifier Quantisation Quality

The labelled frozen test set contains **1,208 examples**.

| Metric | ONNX FP32 | ONNX INT8 |
|---|---:|---:|
| Macro-F1 | 1.000000 | 1.000000 |
| Accuracy | 1.000000 | 1.000000 |

Measured classifier quantisation evidence:

```text
Macro-F1 quality tax: 0.0000 points
95% bootstrap CI:     [0.0000, 0.0000] points
FP32/INT8 agreement:  100.00%
```

The Lab 7 classifier quality-tax target of at most one macro-F1 point is satisfied.

#### NER Quantisation Quality

The segmented XLM-R NER model was evaluated on the deterministic frozen test split used during training.

```text
Frozen-test sentences:              400
Evaluated labelled tokens:          4343
FP32 entity-F1:                     1.000000
INT8 entity-F1:                     1.000000
Entity-F1 quality tax:              0.0000 points
95% bootstrap CI:                   [0.0000, 0.0000] points
FP32/INT8 labelled-token agreement: 100.00%
```

The quantised NER model showed no measured quality degradation on the frozen test set.

#### FastAPI Serving

The classifier is exposed through:

```text
GET  /health
POST /v1/classify
```

The production classifier path uses:

- Shared Bayan preprocessing
- XLM-R tokenizer
- ONNX Runtime CPU execution
- Dynamic INT8 classifier
- Maximum sequence length of 128
- Four inference threads

Startup canaries validate required serving artefacts, model metadata, label mappings, tokenizer and configuration hashes, deterministic preprocessing, CAMeL Tools `d3tok` segmentation behaviour, and a pinned Arabic classifier behaviour check.

Observed startup result:

```text
STARTUP CANARIES: GREEN
```

Serving-contract tests:

```text
2 passed
```

Health check:

```text
HTTP 200
{"status":"ready","model":"ONNX INT8 topic classifier","threads":4}
```

#### Official HTTP Load Test

The FastAPI endpoint was tested with `hey` using the required Lab 7 profile.

```text
Duration:             60 seconds
Concurrent clients:   16
Successful responses: 1759
HTTP errors:          0
Requests/second:      29.1732
Average latency:      547.3 ms
p50:                  494.0 ms
p90:                  799.2 ms
p95:                  843.7 ms
p99:                  930.9 ms
```

The zero-error requirement and startup-canary requirement passed.

The HTTP target of **p99 <= 40 ms** was not reached on the measured Colab environment. The benchmark host exposed only **2 logical CPUs** while the Lab 7 inference configuration was pinned to four threads. The measured latency is reported without modification.

Detailed optimisation and serving evidence is retained in `BENCHMARKS.md`.

---


## Lab 6 Validation Results

The validation prediction set contains:

```text
Validation examples: 2400
Correct predictions: 2100
Errors: 300
Validation accuracy: 87.5%
```

Bootstrap evaluation reports the aggregate estimate together with a 95% confidence interval.

---

## Sliced Evaluation

Important validation slices include:

| Slice | Examples | Accuracy |
|---|---:|---:|
| Overall | 2400 | 87.50% |
| Arabic | 1200 | 75.00% |
| English | 1200 | 100.00% |
| MSA | 1200 | 75.00% |
| Billing | 300 | 100.00% |
| Digital services | 300 | 100.00% |
| Licensing | 300 | 100.00% |
| Lighting | 300 | 100.00% |
| Parks | 300 | 0.00% |
| Roads | 300 | 100.00% |
| Waste | 300 | 100.00% |
| Water | 300 | 100.00% |
| Medium length | 1646 | 88.94% |
| Short length | 754 | 84.35% |

The largest observed class-level weakness is the `parks` class, which is systematically confused with `roads`.

---

## Model Comparison

| Model | Aggregate Macro-F1 [95% CI] | Gulf [95% CI] | Invariance Pass | MFT Pass |
|---|---|---|---:|---:|
| XLM-R incumbent | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 93.75% |
| CAMeLBERT-DA | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 31.25% |

Paired bootstrap comparison:

```text
XLM-R - CAMeLBERT-DA
All Arabic delta: 0.0000 [0.0000, 0.0000]
Gulf delta:       0.0000 [0.0000, 0.0000]
```

The models are tied on the frozen Arabic and Gulf evaluation sets.

---

## Behavioural Evaluation

### XLM-R Incumbent

```text
Invariance: 140 / 200 = 70.00%
MFT:         15 / 16 = 93.75%
Directional: not evaluated
```

Arabic-only MFT:

```text
7 / 8 = 87.50%
```

### CAMeLBERT-DA

```text
Invariance: 140 / 200 = 70.00%
MFT:          5 / 16 = 31.25%
Directional: not evaluated
```

Arabic-only MFT:

```text
4 / 8 = 50.00%
```

The Lab 6 invariance target is 95%, so neither evaluated model reaches the invariance target.

The XLM-R model exceeds the 90% full-suite MFT target, while CAMeLBERT-DA does not.

The minimum-functionality cases were authored during Lab 6 using the service names available in the project service registry because the supplied behavioural template data contains invariance and directional cases but no MFT rows.

Directional behavioural templates require sentiment scoring. A trained sentiment scorer is not available in the current model artefacts, so directional cases are documented as not evaluated instead of assigning an artificial score.

---

## Manual Error Analysis

A reproducible random sample of **120 validation errors** was manually reviewed and tagged.

Sampling configuration:

```text
Total validation errors: 300
Manually reviewed errors: 120
Random state: 42
```

All sampled errors were read and classified manually.

### Error Taxonomy Results

| Error Category | Count | Share |
|---|---:|---:|
| Systematic class confusion | 67 | 55.8% |
| Label ambiguity | 40 | 33.3% |
| Arabic orthographic variation | 13 | 10.8% |
| **Total** | **120** | **100.0%** |

Observed histogram:

```text
Systematic class confusion        67 | #################################
Label ambiguity                   40 | ####################
Arabic orthographic variation     13 | #######
```

The dominant failure pattern is systematic:

```text
parks -> roads
```

Examples involving park irrigation and playground maintenance are repeatedly predicted as road-related feedback.

---

## Prioritised Improvements

### 1. Reduce `parks -> roads` confusion

Add hard-negative and contrastive examples that explicitly distinguish park maintenance from road maintenance.

Estimated upper-bound validation accuracy improvement:

```text
+7.0 percentage points
```

### 2. Improve park-walkway contextual disambiguation

Increase training coverage for terms such as `الممر` when the surrounding context explicitly refers to a park such as `حديقة`.

Estimated upper-bound validation accuracy improvement:

```text
+4.2 percentage points
```

### 3. Strengthen Arabic orthographic robustness

Improve Arabic augmentation and normalisation for spelling and elongation variants observed during manual error review.

Examples include:

```text
لووووسمحت
ألعأب
ألري
ألممر
```

Estimated upper-bound validation accuracy improvement:

```text
+1.4 percentage points
```

These values are scenario estimates derived from the manually reviewed error distribution. They are not measured improvements and require retraining and reevaluation for confirmation.

---

## Model Cards

Three model cards are included:

```text
docs/model_cards/xlm_r_incumbent.md
docs/model_cards/camelbert_mix.md
docs/model_cards/camelbert_da.md
```

Each model card documents:

- Intended use
- Model/checkpoint
- Preprocessing version
- Dataset snapshot
- Evaluation metrics
- Slice evidence
- Behavioural evidence
- Hand-written known limitations

---

## Evaluation Evidence

Detailed project evidence is available in:

- `BENCHMARKS.md`
- `DECISIONS.md`
- `EVALUATION_REPORT.md`
- `docs/ERROR_TAXONOMY.md`
- `docs/model_cards/`
- `manual_error_sample.csv`

---

## Project Structure

```text
Bayan/
├── data/
│   └── eval/
│       ├── behavioural_templates.csv
│       ├── mft_cases.csv
│       └── validation_predictions.csv
│
├── docs/
│   ├── ERROR_TAXONOMY.md
│   └── model_cards/
│       ├── xlm_r_incumbent.md
│       ├── camelbert_mix.md
│       └── camelbert_da.md
│
├── scripts/
│   ├── arabic_bakeoff.py
│   ├── evaluation_report.py
│   └── ...
│
├── src/
│   └── bayan/
│       ├── evaluation/
│       │   ├── behavioural.py
│       │   ├── bootstrap.py
│       │   └── slices.py
│       ├── search/
│       └── ...
│
├── tests/
├── BENCHMARKS.md
├── DECISIONS.md
├── EVALUATION_REPORT.md
├── manual_error_sample.csv
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ns3271585-arch/Bayan.git
cd Bayan
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Running the Lab 6 Evaluation

Run the evaluation utility tests:

```bash
pytest tests/test_evaluation.py -q
```

Generate the evaluation report and model cards:

```bash
python scripts/evaluation_report.py
```

The expected automated Lab 6 test result is:

```text
6 passed
```

---

## Lab Commands

The repository Makefile includes lab-specific checks.

```bash
make lab1
make lab2
make lab3
make lab4
make lab6
```

To run the complete available test suite:

```bash
pytest -q
```

---

## Known Evaluation Limitations

The project deliberately reports observed weaknesses instead of presenting aggregate metrics alone.

Important limitations include:

- Arabic validation accuracy is lower than English validation accuracy.
- The `parks` class has a severe systematic confusion with `roads`.
- Both evaluated classifiers achieve only 70% invariance.
- CAMeLBERT-DA performs poorly on the full MFT suite despite perfect frozen-test macro-F1.
- Directional sentiment behaviour cannot currently be measured because a trained sentiment scoring artefact is not available.
- The Lab 6 MFT cases are student-authored because no minimum-functionality rows were supplied in the behavioural template CSV.
- Perfect scores on the frozen course test data do not guarantee robustness on real-world data.
- Estimated improvements from the error taxonomy are hypotheses and require retraining for validation.

---

## Reproducibility

The project uses fixed random seeds and reproducible evaluation samples where applicable.

The Lab 6 manual error sample uses:

```text
random_state = 42
```

Bootstrap evaluation also uses fixed seeds to make confidence intervals reproducible.

Model and preprocessing versions are documented in the generated model cards and evaluation evidence.

---

## Course and Academy

This repository was developed for the **SDA-AIE-211 Natural Language Processing with Transformers** learning project.

### SDAIA Academy GitHub

https://github.com/SDAIAAcademy

### Course Repository

https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course

### Student Repository

https://github.com/ns3271585-arch/Bayan

---

## Current Status

```text
Lab 1: Completed
Lab 2: Completed
Lab 3: Completed
Lab 4: Completed
Lab 5: Completed
Lab 6: Completed
Lab 7: Serving and classifier optimisation implemented
```

Lab 6 final checkpoint:

```text
Evaluation tests: passed
Manual errors tagged: 120 / 120
Model cards generated: 3
Evaluation report generated: yes
```
