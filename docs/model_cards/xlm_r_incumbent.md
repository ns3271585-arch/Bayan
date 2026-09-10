# Model Card — XLM-R incumbent

## Intended use
Bilingual Arabic/English feedback topic classification for the Bayan service-feedback domain.

## Artefact / data versions
- Model/checkpoint: xlm-roberta-base
- Preprocessing version: 1.2.0
- Data version/snapshot: Bayan course dataset / frozen Lab 3 split

## Metrics
| Metric                     | Result                  |
|:---------------------------|:------------------------|
| All Arabic macro-F1        | 1.0000 [1.0000, 1.0000] |
| Gulf macro-F1              | 1.0000 [1.0000, 1.0000] |
| MSA macro-F1               | 1.0000                  |
| Arabic tokenizer fertility | 1.685                   |

## Slice metrics
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

## Behavioural tests
| Test        | Result           |
|:------------|:-----------------|
| invariance  | 140/200 (70.00%) |
| mft         | 15/16 (93.75%)   |
| directional | N/A              |

## Known limitations
- Behavioural invariance passed 140/200 cases (70.0%), below the Lab 6 target of 95%.
- The full student-authored MFT suite passed 15/16 cases (93.75%), while the Arabic-only subset passed 7/8 cases.
- Directional sentiment behaviour was not evaluated because the current project artefacts do not include a trained sentiment scorer.
- Perfect frozen Arabic-test macro-F1 does not demonstrate robustness to unseen real-world distributions or behavioural perturbations.

## Contact / owner
Bayan course team