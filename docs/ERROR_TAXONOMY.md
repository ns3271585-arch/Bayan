# Error Taxonomy

The taxonomy started from the supplied Lab 6 categories and was extended
after manually reviewing a reproducible sample of 120 validation errors.

## Taxonomy

1. Label ambiguity
2. Arabic orthographic variation
3. Dialect or code-switching
4. Entity boundary or clitic alignment
5. Long-context truncation
6. Retrieval relevance mismatch
7. Preprocessing or serving skew
8. Annotation defect
9. Systematic class confusion

## Manual Error Review

A fixed random sample of 120 errors was drawn from the 300 errors in
`data/eval/validation_predictions.csv` using `random_state=42`.

All 120 examples were read and tagged manually.

| Error category | Count | Share |
|---|---:|---:|
| Systematic class confusion | 67 | 55.8% |
| Label ambiguity | 40 | 33.3% |
| Arabic orthographic variation | 13 | 10.8% |
| **Total** | **120** | **100.0%** |

## Error-Category Histogram

```text
Systematic class confusion        67 | #################################
Label ambiguity                   40 | ####################
Arabic orthographic variation     13 | #######