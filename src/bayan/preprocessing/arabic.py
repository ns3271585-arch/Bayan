"""Lab 4: per-model Arabic normalisation profiles."""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


# Arabic diacritics / harakat.
_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Normalisation rules required by the supplied bayan_ar_v1 golden pairs.
_BAYAN_AR_V1_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ؤ": "و",
        "ئ": "ي",
        "ى": "ي",
        "ة": "ه",
    }
)


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Return model-normalised Arabic text for the selected profile.

    The caller should keep the original input separately whenever the
    original display form is required.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if profile.name != "bayan_ar_v1":
        raise ValueError(f"Unknown Arabic normalisation profile: {profile.name}")

    # Unicode canonicalisation while leaving the caller's original text intact.
    normalized = unicodedata.normalize("NFKC", text)

    # Remove tatweel/kashida.
    normalized = normalized.replace("ـ", "")

    # Remove Arabic diacritics when requested by the profile.
    if profile.dediacritize:
        normalized = _ARABIC_DIACRITICS.sub("", normalized)

    # Model-specific character normalisation.
    normalized = normalized.translate(_BAYAN_AR_V1_TRANSLATION)

    return normalized


@lru_cache(maxsize=1)
def _build_d3_tokenizer():
    """Build and cache the CAMeL Tools d3tok tokenizer."""
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer

    disambiguator = MLEDisambiguator.pretrained("calima-msa-r13")

    return MorphologicalTokenizer(
        disambiguator,
        scheme="d3tok",
        split=True,
        diac=False,
    )


def segment(text: str) -> list[str]:
    """Segment Arabic text using CAMeL Tools d3tok."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return []

    tokenizer = _build_d3_tokenizer()

    return tokenizer.tokenize(text.split())