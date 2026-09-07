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
**Why is the embedding share different?**  
mBERT has a much larger multilingual vocabulary, so a larger share of its parameters is spent on the embedding matrix; this is the multilingual vocabulary tax.
### Attention Diagnostics

* Scaled dot-product attention matched the PyTorch reference with a maximum absolute difference of `0.0000002384`, which is below the required `1e-6` tolerance.
* The causal attention matrix was lower triangular, and the future-token attention mass was `0.0`.
* This masking pattern corresponds to decoder-style causal attention.
* The diagnostic batch contained 5 Bayan examples and 27 PAD tokens.
* With the correct attention mask, average attention mass assigned to `[PAD]` tokens was `0.00000000`.
* Without the attention mask, average `[PAD]` attention mass increased to `0.07816089`.
* This confirms pad-attention leakage when the attention mask is omitted.
* The strongest adjacency-looking head was Layer 5, Head 10, with adjacency mass `0.941012`.
* The strongest `[SEP]` sink behaviour was observed in Layer 1, Head 5, with `[SEP]` attention mass `0.273421`.

**Conclusion:** Attention masks are essential when batching padded sequences because omitting the mask allows the model to allocate non-zero attention to meaningless `[PAD]` positions.


| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | 177,853,440 | 51.84% | More than half of the model parameters are in embeddings. Attention = 15.94%, FFN = 31.86%. |
| CAMeLBERT | 109,081,344 | 21.48% | Smaller embedding share than mBERT. Attention = 25.99%, FFN = 51.95%. |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
