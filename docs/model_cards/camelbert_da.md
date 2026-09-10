# Model Card — CAMeLBERT-DA

## Intended use
Arabic and dialect-aware feedback topic classification, with emphasis on Gulf and Arabic evaluation slices.

## Artefact / data versions
- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-da
- Preprocessing version: 1.2.0
- Data version/snapshot: Bayan course dataset / frozen Lab 3 split

## Metrics
| Metric                     | Result                  |
|:---------------------------|:------------------------|
| All Arabic macro-F1        | 1.0000 [1.0000, 1.0000] |
| Gulf macro-F1              | 1.0000 [1.0000, 1.0000] |
| MSA macro-F1               | 1.0000                  |
| Arabic tokenizer fertility | 1.412                   |

## Slice metrics
_No model-specific Lab 6 validation slice predictions were recorded for this checkpoint. Arabic/Gulf/MSA frozen-test metrics are reported in the Metrics section above._

## Behavioural tests
| Test        | Result           |
|:------------|:-----------------|
| invariance  | 140/200 (70.00%) |
| mft         | 5/16 (31.25%)    |
| directional | N/A              |

## Known limitations
- Behavioural invariance passed 140/200 cases (70.0%), below the Lab 6 target of 95%.
- The full student-authored MFT suite passed only 5/16 cases (31.25%); the Arabic-only subset passed 4/8 cases.
- English MFT performance is weak because this checkpoint was trained on the Arabic slice.
- Directional sentiment behaviour was not evaluated because the current project artefacts do not include a trained sentiment scorer.
- Perfect frozen Arabic-test macro-F1 does not remove the behavioural robustness weaknesses observed in Lab 6.

## Contact / owner
Bayan course team