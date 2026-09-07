"""Lab 1 starter: sentence segmentation."""

import spacy

from bayan.preprocessing.core import preprocess


def build_pipeline():
    """Build the spaCy sentence segmentation pipeline."""

    nlp = spacy.blank("xx")

    nlp.add_pipe(
        "sentencizer",
        config={
            "punct_chars": [".", "!", "?", "؟"]
        },
    )

    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess text then return non-empty sentence strings."""

    cleaned = preprocess(raw)

    doc = nlp(cleaned)

    return [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]