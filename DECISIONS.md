# Decision Records

## tokenizer
## tokenizer

* Chosen checkpoint(s): XLM-R (`xlm-roberta-base`)

* Arabic fertility evidence: 1.672

* English fertility evidence: 1.434

* p95 length evidence: Arabic = 21.0, English = 23.0

* Operational trade-off / rationale: XLM-R was selected because it provides the best overall balance for Bayan's bilingual Arabic/English data. CAMeLBERT achieved better Arabic fertility at 1.405, but its English fertility was much worse at 2.705 and its English p95 sequence length increased to 38.0. DistilBERT performed well on English but poorly on Arabic, with Arabic fertility of 4.527 and an Arabic p95 sequence length of 47.0. mBERT was more balanced, but XLM-R achieved better fertility and shorter sequence lengths in both Arabic and English. Therefore, XLM-R provides the best bilingual trade-off for Bayan.


## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
