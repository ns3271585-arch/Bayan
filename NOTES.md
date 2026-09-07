# Lab Notes

## Lab 1 — Defect Safari
## Lab 1 — Defect Safari

Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.

For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1

* Class: Extra whitespace
* Example: `  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   `
* Why it matters: Extra spaces create inconsistent text and may affect tokenization and model input.
* Decision: Clean — remove leading/trailing whitespace and collapse repeated whitespace into a single space.

### Defect 2

* Class: Phone number PII
* Example: `تطبيق خدمات المياه يتوقف عند تسجيل الدخول 0551234567`
* Why it matters: Phone numbers are personally identifiable information and should not be exposed to downstream models.
* Decision: Clean — replace supported Saudi phone numbers with `<PHONE>`.

### Defect 3

* Class: National ID PII
* Example: `إنارة الممر لا تعمل عند طريق الملك فهد 1023456789`
* Why it matters: National ID numbers are sensitive personal information and must be protected.
* Decision: Clean — replace Saudi national-ID-shaped values with `<NATIONAL_ID>`.

### Defect 4

* Class: HTML remnants
* Example: `الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة <br>`
* Why it matters: HTML tags such as `<br>` are formatting artifacts rather than useful linguistic content and may create unnecessary tokens.
* Decision: Clean — remove HTML remnants before model processing.

### Defect 5

* Class: Repeated / elongated characters
* Example: `لووووسمحت الري متوقف في حديقة طريق الملك فهد`
* Why it matters: Excessive character repetition creates inconsistent word forms and can increase subword fragmentation during tokenization.
* Decision: Clean — reduce runs of three or more repeated characters to two characters.

### Defect 6

* Class: Emoji / sentiment signal
* Example: `الحاوية ممتلئة في حي الياسمين ولم تُفرغ منذ 4 أيام 😡`
* Why it matters: Emoji can carry useful sentiment information, especially for complaint and feedback classification.
* Decision: Preserve — keep emoji because they may provide useful task-related information.


### Defect 1
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 2
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 3
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 4
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 5
- Class:
- Example:
- Why it matters:
- Decision:

### Defect 6
- Class:
- Example:
- Why it matters:
- Decision:
### Sentence Segmentation Spot Check

I tested the sentence segmentation pipeline on five long bilingual/Arabic-style examples, including a numbered-list complaint.

Observations:

- Arabic full stop punctuation was segmented correctly.
- Arabic question marks `؟` were handled correctly.
- English punctuation such as `.`, `!`, and `?` was also handled correctly.
-- Numbered-list complaints were segmented, but the sentencizer treated list markers such as `1.`, `2.`, and `3.` as separate sentences. This is a known limitation of the current lightweight segmentation approach.
- PII masking occurred before segmentation, so phone numbers and national IDs were replaced before the text was split.
- Emoji were preserved after preprocessing and sentence segmentation.
- Empty sentence strings were not returned.

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
