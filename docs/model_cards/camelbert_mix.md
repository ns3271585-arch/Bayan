# Model Card — CAMeLBERT-mix

## Intended use
Arabic feedback topic classification, used as an Arabic-centric comparison model in the Lab 4 bake-off.

## Artefact / data versions
- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-mix
- Preprocessing version: 1.2.0
- Data version/snapshot: Bayan course dataset / frozen Lab 3 split

## Metrics
| Metric                     |   Result |
|:---------------------------|---------:|
| All Arabic macro-F1        |    1     |
| Gulf macro-F1              |    1     |
| MSA macro-F1               |    1     |
| Arabic tokenizer fertility |    1.412 |

## Slice metrics
_No model-specific Lab 6 validation slice predictions were recorded for this checkpoint. Arabic/Gulf/MSA frozen-test metrics are reported in the Metrics section above._

## Behavioural tests
| Test        | Result   |
|:------------|:---------|
| invariance  | N/A      |
| mft         | N/A      |
| directional | N/A      |

## Known limitations
- CAMeLBERT-mix is Arabic-centric and is not intended to replace the bilingual XLM-R model for English feedback.
- A separate Lab 6 behavioural evaluation was not recorded for this checkpoint, so invariance, MFT, and directional rates are not reported.
- Perfect frozen Arabic-test scores come from the constrained course dataset and should not be interpreted as production-level robustness.
- Additional behavioural and distribution-shift evaluation is required before deployment.

## Contact / owner
Bayan course team