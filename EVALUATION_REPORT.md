# Bayan Lab 6 Evaluation Report

## Executive Summary

Validation accuracy is 87.5% with a 95% bootstrap confidence interval
of [86.2%, 88.9%].

The sliced evaluation exposes a substantial language and class-level gap:
Arabic accuracy is lower than English accuracy, and the `parks` class is the
dominant observed topic-classification failure.

Behavioural evaluation also shows that perfect frozen-test macro-F1 does not
imply robust behaviour. Both evaluated models achieve only 70.0% invariance.

## Bootstrap Result

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Validation accuracy | 0.8750 | [0.8617, 0.8888] |

## Sliced Evaluation

| slice_type   | slice_value      |    n |   accuracy |   ci_low |   ci_high | small_slice   |
|:-------------|:-----------------|-----:|-----------:|---------:|----------:|:--------------|
| overall      | all              | 2400 |     0.875  |   0.8617 |    0.8892 | False         |
| language     | ar               | 1200 |     0.75   |   0.725  |    0.7733 | False         |
| language     | en               | 1200 |     1      |   1      |    1      | False         |
| dialect      | MSA              | 1200 |     0.75   |   0.725  |    0.7742 | False         |
| dialect      | NA               | 1200 |     1      |   1      |    1      | False         |
| class        | billing          |  300 |     1      |   1      |    1      | False         |
| class        | digital_services |  300 |     1      |   1      |    1      | False         |
| class        | licensing        |  300 |     1      |   1      |    1      | False         |
| class        | lighting         |  300 |     1      |   1      |    1      | False         |
| class        | parks            |  300 |     0      |   0      |    0      | False         |
| class        | roads            |  300 |     1      |   1      |    1      | False         |
| class        | waste            |  300 |     1      |   1      |    1      | False         |
| class        | water            |  300 |     1      |   1      |    1      | False         |
| length       | medium           | 1646 |     0.8894 |   0.8736 |    0.9046 | False         |
| length       | short            |  754 |     0.8435 |   0.8143 |    0.8687 | False         |

## Model Comparison

| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---:|---:|---:|---:|
| XLM-R incumbent | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 93.75% |
| CAMeLBERT-DA | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 31.25% |

The paired XLM-R minus CAMeLBERT-DA macro-F1 difference is 0.0000 with
bootstrap interval [0.0000, 0.0000] on both the full Arabic and Gulf frozen
test slices. The quality tie therefore does not justify replacing the XLM-R
incumbent on aggregate quality alone.

## Behavioural Evaluation

### XLM-R incumbent

| Test        | Result           |
|:------------|:-----------------|
| invariance  | 140/200 (70.00%) |
| mft         | 15/16 (93.75%)   |
| directional | N/A              |

### CAMeLBERT-DA

| Test        | Result           |
|:------------|:-----------------|
| invariance  | 140/200 (70.00%) |
| mft         | 5/16 (31.25%)    |
| directional | N/A              |

The Lab 6 invariance target is 95%, so both evaluated models miss the target.

The MFT target is 90%. XLM-R passes the full 16-case student-authored MFT
suite at 93.75%, while CAMeLBERT-DA passes 31.25%. The Arabic-only MFT rates
are 87.5% for XLM-R and 50.0% for CAMeLBERT-DA.

The supplied directional templates describe a sentiment relation, but the
current project artefacts do not contain a trained sentiment scoring model.
Directional cases are therefore reported as not evaluated rather than being
assigned an artificial pass rate.

The 16 MFT cases were authored for Lab 6 from the service names in the
project service registry because the supplied behavioural template file
contains invariance and directional cases but no minimum-functionality rows.

## Manual Error Taxonomy

A reproducible random sample of 120 errors was drawn from the 300 validation
errors with `random_state=42`. All 120 sampled errors were read and tagged.

| Error category                |   Count | Share   |
|:------------------------------|--------:|:--------|
| Systematic class confusion    |      67 | 55.8%   |
| Label ambiguity               |      40 | 33.3%   |
| Arabic orthographic variation |      13 | 10.8%   |

The dominant pattern is systematic `parks -> roads` confusion. The next
largest category is ambiguity around park walkways, followed by Arabic
orthographic variation.

## Top 3 Prioritised Fixes

1. **Target `parks -> roads` confusion.** Add hard-negative and contrastive
   park-versus-road training examples. Estimated upper-bound validation
   accuracy delta: **+7.0 percentage points**.

2. **Improve contextual disambiguation of park walkways.** Strengthen examples
   in which `الممر` occurs with clear park context such as `حديقة`.
   Estimated upper-bound validation accuracy delta:
   **+4.2 percentage points**.

3. **Strengthen Arabic orthographic normalisation and augmentation.** Cover
   elongation and observed alef/hamza spelling variation.
   Estimated upper-bound validation accuracy delta:
   **+1.4 percentage points**.

These deltas are scenario estimates derived from the manually reviewed error
shares. They are not measured improvements and assume that the corresponding
error category can be corrected without introducing regressions.

## Model Cards

Three model cards were generated:

- `docs/model_cards/xlm_r_incumbent.md`
- `docs/model_cards/camelbert_mix.md`
- `docs/model_cards/camelbert_da.md`

## Known Evaluation Limitations

- The validation prediction file contains a concentrated `parks -> roads`
  failure, so aggregate accuracy hides a severe class-level weakness.
- Behavioural invariance is below target for both evaluated models.
- Directional sentiment behaviour cannot be measured from the currently
  trained model artefacts.
- The MFT cases are student-authored because minimum-functionality rows were
  not supplied in the behavioural template CSV.
- Frozen-test scores of 1.0000 should not be treated as evidence of
  production robustness.
- Predicted metric deltas for proposed fixes are estimates and require
  retraining to verify.

## Evidence Summary

- Bootstrap evaluation: complete.
- Sliced evaluation: complete.
- Behavioural evaluation: invariance and MFT measured; directional documented
  as unavailable with current model artefacts.
- Manual error review: 120/120 tagged.
- Error taxonomy and top fixes: complete.
- Model cards: 3 generated.
