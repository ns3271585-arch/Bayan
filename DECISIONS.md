# Decision Records

## tokenizer
## tokenizer

* Chosen checkpoint(s): XLM-R (`xlm-roberta-base`)

* Arabic fertility evidence: 1.672

* English fertility evidence: 1.434

* p95 length evidence: Arabic = 21.0, English = 23.0

* Operational trade-off / rationale: XLM-R was selected because it provides the best overall balance for Bayan's bilingual Arabic/English data. CAMeLBERT achieved better Arabic fertility at 1.405, but its English fertility was much worse at 2.705 and its English p95 sequence length increased to 38.0. DistilBERT performed well on English but poorly on Arabic, with Arabic fertility of 4.527 and an Arabic p95 sequence length of 47.0. mBERT was more balanced, but XLM-R achieved better fertility and shorter sequence lengths in both Arabic and English. Therefore, XLM-R provides the best bilingual trade-off for Bayan.


## arabic-model
- Incumbent: XLM-R (`xlm-roberta-base`).
- Candidate: CAMeLBERT-DA (`CAMeL-Lab/bert-base-arabic-camelbert-da`). CAMeLBERT-mix tied with it on all measured Lab 4 metrics.
- All/Gulf/MSA evidence: XLM-R = 1.0000 / 1.0000 / 1.0000 with Arabic fertility 1.685; CAMeLBERT-mix = 1.0000 / 1.0000 / 1.0000 with fertility 1.412; CAMeLBERT-DA = 1.0000 / 1.0000 / 1.0000 with fertility 1.412.
- CI-backed verdict: No confidence intervals were computed in Lab 4, so there is no statistically supported quality winner yet. Retain XLM-R as the incumbent for now; CAMeLBERT-DA remains an Arabic-specific candidate because it preserved the same Gulf macro-F1 while using fewer tokens per Arabic word. Confirm the decision with slice confidence intervals in Lab 6.
- Segmentation contract: CAMeL Tools `d3tok` is used for Arabic clitic segmentation. The frozen-test LOCATION recall was 1.0000 without segmentation and 1.0000 with d3tok, giving a measured delta of +0.0000. The requested +4-point gain was unattainable because the baseline recall was already at the maximum value of 1.0000.

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
