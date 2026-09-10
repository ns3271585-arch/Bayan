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
- Distribution: Gulf = 4800/7200 (66.67%), MSA = 2400/7200 (33.33%).
- One-sentence implication for MSA-only evaluation: Evaluating only on MSA would not represent the Bayan Arabic distribution because 66.67% of the Arabic feedback belongs to the Gulf slice.

## Lab 5 — Search diagnostics

### Retrieval evaluation
The versioned FAISS index uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with the shared preprocessing pipeline and L2-normalised vectors. Two-stage retrieval uses 50 bi-encoder candidates followed by cross-encoder reranking.

Measured results:
- bi-encoder recall@10: 0.0692
- bi-encoder MRR@10: 0.0175
- reranked recall@10: 0.0077
- reranked MRR@10: 0.0015
- no-answer empty-correct: 20/20 at `min_score=0.25`
- same-language reranked recall@10/MRR@10: 0.0077/0.0015
- cross-language reranked recall@10/MRR@10: 0.0000/0.0000
- cross-lingual recall gap: 0.0077
- cross-lingual MRR gap: 0.0015

The recall@10 >= 0.80 and MRR@10 >= 0.70 targets were not reached. The no-answer target was reached.

### Judgement and duplicate-corpus diagnosis
The labelled query set contains only a small judged set of case IDs per query while the 20k-case corpus contains many duplicate or near-duplicate complaint texts.

For example, Q-002 has 52 exact `case_text` duplicates in the corpus, but none of those exact-match case IDs are in its judged relevant set. Removing duplicate texts from the candidate pool and reranking 100 unique candidates still did not surface any Q-002 gold IDs.

All 130 answerable queries were also found to have relevant-case IDs following a deterministic block-of-8 arithmetic pattern. This is recorded as an evaluation-data characteristic, not used by the retrieval system. Gold IDs were never injected into retrieval or used to change rankings.

Because of these limited judgements and heavy duplicate/tie behaviour, visually plausible semantic matches can still score as non-relevant under strict case-ID evaluation.

### FAISS tie behaviour
Many corpus vectors receive identical or near-identical similarity scores. FAISS tie ordering changed depending on the requested candidate count:

- direct `search(k=10)`: recall@10 = 0.0692, MRR@10 = 0.0175
- `search(k=50)` then taking the first 10: recall@10 = 0.0077, MRR@10 = 0.0026
- top-10 ordering differed for 104/130 answerable queries

Therefore the no-rerank bi-encoder metric is measured using direct top-10 retrieval, while the reranking stage intentionally retrieves a 50-candidate pool.

### Planted unnormalised-vector bug
The persisted FAISS vectors were verified to have L2 norm 1.0.

Before normalisation, a 1,000-case sample had:
- minimum norm: 2.466599
- mean norm: 4.047536
- maximum norm: 5.643936
- standard deviation: 0.711633

Using inner-product search without L2 normalisation caused vector magnitude to affect ranking rather than cosine similarity alone.

Measured comparison:
- normalized recall@10/MRR@10: 0.0692/0.0175
- unnormalised recall@10/MRR@10: 0.0462/0.0080

This confirms the planted failure mode: search results can look semantically plausible while labelled retrieval metrics degrade, so retrieval must be approved using the labelled evaluation set rather than visual inspection alone.