
"""Lab 7 startup skew and behaviour canaries."""

import hashlib
import json
from pathlib import Path

from bayan.preprocessing.arabic import segment
from bayan.preprocessing.core import preprocess


EXPECTED_LABELS = [
    "billing",
    "digital_services",
    "licensing",
    "lighting",
    "parks",
    "roads",
    "waste",
    "water",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def run_startup_canaries(classifier=None) -> None:
    """Fail startup when serving artefacts or preprocessing are incompatible."""

    if classifier is None:
        from bayan.serving.classifier import TopicClassifier

        classifier = TopicClassifier(
            threads=4
        )

    artifact_dir = Path(
        classifier.artifact_dir
    )

    required = [
        "model.onnx",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "manifest.json",
    ]

    for filename in required:
        path = artifact_dir / filename

        if not path.exists():
            raise RuntimeError(
                f"Serving canary failed: missing {path}"
            )

    manifest = json.loads(
        (artifact_dir / "manifest.json")
        .read_text(encoding="utf-8")
    )

    if manifest.get("task") != "text-classification":
        raise RuntimeError(
            "Serving canary failed: wrong task"
        )

    if manifest.get("format") != "onnx":
        raise RuntimeError(
            "Serving canary failed: wrong model format"
        )

    if manifest.get("quantization") != "dynamic-int8":
        raise RuntimeError(
            "Serving canary failed: expected INT8 artefact"
        )

    if manifest.get("max_length") != 128:
        raise RuntimeError(
            "Serving canary failed: max_length skew"
        )

    if manifest.get("labels") != EXPECTED_LABELS:
        raise RuntimeError(
            "Serving canary failed: label mapping skew"
        )

    config_hash = _sha256(
        artifact_dir / "config.json"
    )

    tokenizer_hash = _sha256(
        artifact_dir / "tokenizer.json"
    )

    if config_hash != manifest.get("config_sha256"):
        raise RuntimeError(
            "Serving canary failed: config hash mismatch"
        )

    if tokenizer_hash != manifest.get("tokenizer_sha256"):
        raise RuntimeError(
            "Serving canary failed: tokenizer hash mismatch"
        )

    # Shared preprocessing must remain deterministic.
    raw = "  الخدمة ممتازة  "
    first = preprocess(raw)
    second = preprocess(raw)

    if first != second:
        raise RuntimeError(
            "Serving canary failed: preprocessing is not deterministic"
        )

    # CAMeL d3tok behaviour used by Bayan NER.
    segmented = segment(
        "وبالرياض"
    )

    if segmented != [
        "و+",
        "ب+",
        "ال+",
        "رياض",
    ]:
        raise RuntimeError(
            "Serving canary failed: segmentation skew "
            f"{segmented}"
        )

    # Pinned behaviour canary for the quantised classifier.
    result = classifier.classify(
        "إنارة الشارع لا تعمل"
    )

    if result["topic"] != "lighting":
        raise RuntimeError(
            "Serving canary failed: expected lighting, "
            f"got {result['topic']}"
        )
