"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

import re
import unicodedata

PREPROC_VERSION = "1.2.0"


PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?966|0)5\d{8}(?!\d)"
)

NATIONAL_ID_RE = re.compile(
    r"(?<!\d)[12]\d{9}(?!\d)"
)


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    # Normalize Unicode forms
    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic tatweel / kashida
    text = text.replace("ـ", "")

    # Reduce 3 or more repeated characters to 2
    # Examples:
    # هلووو -> هلوو
    # GOOD!!! -> GOOD!!
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Collapse spaces, tabs, and newlines into one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""

    # Mask Saudi phone numbers
    text = PHONE_RE.sub("<PHONE>", text)

    # Mask Saudi national IDs
    text = NATIONAL_ID_RE.sub("<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""

    # First mask personal information
    text = mask_pii(text)

    # Then normalize the text
    text = normalize(text)

    return text