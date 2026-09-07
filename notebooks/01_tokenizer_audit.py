"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Calculate total subword pieces divided by whitespace words."""

    total_subwords = 0
    total_words = 0

    for text in texts:
        text = str(text)

        words = text.split()
        tokens = tokenizer.tokenize(text)

        total_words += len(words)
        total_subwords += len(tokens)

    if total_words == 0:
        return 0.0

    return total_subwords / total_words


def sequence_lengths(tokenizer, texts) -> list[int]:
    """Return tokenized sequence lengths including special tokens."""

    lengths = []

    for text in texts:
        encoded = tokenizer(
            str(text),
            add_special_tokens=True,
            truncation=False,
        )

        lengths.append(len(encoded["input_ids"]))

    return lengths


def unknown_rate(tokenizer, texts) -> float:
    """Calculate the proportion of tokens mapped to the unknown token."""

    if tokenizer.unk_token is None:
        return 0.0

    total_tokens = 0
    unknown_tokens = 0

    for text in texts:
        tokens = tokenizer.tokenize(str(text))

        total_tokens += len(tokens)
        unknown_tokens += sum(
            token == tokenizer.unk_token
            for token in tokens
        )

    if total_tokens == 0:
        return 0.0

    return unknown_tokens / total_tokens


def main():
    """Run the tokenizer audit for Arabic and English Bayan text."""

    df = pd.read_csv(DATA)

    arabic_texts = (
        df.loc[df["lang"] == "ar", "text"]
        .dropna()
        .astype(str)
        .tolist()
    )

    english_texts = (
        df.loc[df["lang"] == "en", "text"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(f"Arabic examples: {len(arabic_texts)}")
    print(f"English examples: {len(english_texts)}")
    print()

    results = []

    for checkpoint, name in CANDIDATES.items():

        print(f"Loading {name}...")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_fertility = fertility(
            tokenizer,
            arabic_texts,
        )

        en_fertility = fertility(
            tokenizer,
            english_texts,
        )

        ar_lengths = sequence_lengths(
            tokenizer,
            arabic_texts,
        )

        en_lengths = sequence_lengths(
            tokenizer,
            english_texts,
        )

        ar_p95 = float(
            np.percentile(ar_lengths, 95)
        )

        en_p95 = float(
            np.percentile(en_lengths, 95)
        )

        ar_unk_rate = unknown_rate(
            tokenizer,
            arabic_texts,
        )

        results.append(
            {
                "Tokenizer": name,
                "AR fertility": round(ar_fertility, 3),
                "EN fertility": round(en_fertility, 3),
                "AR p95 len": round(ar_p95, 1),
                "EN p95 len": round(en_p95, 1),
                "AR UNK rate": round(ar_unk_rate, 5),
            }
        )

    results_df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("BAYAN TOKENIZER AUDIT")
    print("=" * 80)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()