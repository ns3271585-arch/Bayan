
"""Lab 7: export Bayan classifier and NER to ONNX and dynamic INT8."""

import argparse
import shutil
import subprocess
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


DEFAULT_CLASSIFIER = (
    "/content/drive/MyDrive/Bayan/artifacts/"
    "arabic_bakeoff/xlm_r_incumbent/"
    "checkpoints/checkpoint-524"
)

DEFAULT_NER = (
    "/content/drive/MyDrive/Bayan/artifacts/"
    "ner_segmented"
)

DEFAULT_OUTPUT = (
    "/content/drive/MyDrive/Bayan/artifacts/"
    "lab7_onnx"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export Bayan classifier and NER to "
            "FP32 ONNX and dynamic INT8."
        )
    )

    parser.add_argument(
        "--classifier-model",
        default=DEFAULT_CLASSIFIER,
        help="Trained Hugging Face classifier directory.",
    )

    parser.add_argument(
        "--ner-model",
        default=DEFAULT_NER,
        help="Trained Hugging Face NER directory.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Directory for Lab 7 ONNX artefacts.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export existing FP32 artefacts.",
    )

    return parser.parse_args()


def require_model_dir(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} model directory not found: {path}"
        )

    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"{name} config.json not found: {path}"
        )

    return path


def run_optimum_export(
    model_dir,
    output_dir,
    task,
    *,
    force=False,
):
    """Export a Hugging Face model to FP32 ONNX."""

    model_path = output_dir / "model.onnx"

    if model_path.exists() and not force:
        print(
            f"FP32 rollback already exists: {model_path}"
        )
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "/content/venv312/bin/optimum-cli",
        "export",
        "onnx",
        "--model",
        str(model_dir),
        "--task",
        task,
        str(output_dir),
    ]

    print()
    print("=" * 70)
    print(f"EXPORTING FP32: {task}")
    print("=" * 70)
    print("Source:", model_dir)
    print("Output:", output_dir)

    subprocess.run(
        command,
        check=True,
    )

    if not model_path.exists():
        raise RuntimeError(
            f"ONNX export did not create: {model_path}"
        )


def copy_sidecar_files(
    source_dir,
    destination_dir,
):
    """Copy tokenizer/config files beside the quantised model."""

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "label_mapping.json",
    ]

    for name in names:
        source = source_dir / name

        if source.exists():
            shutil.copy2(
                source,
                destination_dir / name,
            )


def quantize_onnx(
    fp32_dir,
    int8_dir,
):
    """Create a dynamic QInt8 ONNX model while retaining FP32 rollback."""

    input_model = fp32_dir / "model.onnx"
    output_model = int8_dir / "model.onnx"

    if not input_model.exists():
        raise FileNotFoundError(
            f"FP32 ONNX model missing: {input_model}"
        )

    int8_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_model.exists():
        output_model.unlink()

    print()
    print("=" * 70)
    print("DYNAMIC INT8 QUANTISATION")
    print("=" * 70)
    print("Input :", input_model)
    print("Output:", output_model)

    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8,
    )

    copy_sidecar_files(
        fp32_dir,
        int8_dir,
    )

    if not output_model.exists():
        raise RuntimeError(
            f"INT8 quantisation did not create: {output_model}"
        )


def size_mb(path):
    return (
        Path(path).stat().st_size
        / (1024 ** 2)
    )


def report_pair(
    name,
    fp32_dir,
    int8_dir,
):
    fp32_model = fp32_dir / "model.onnx"
    int8_model = int8_dir / "model.onnx"

    fp32_size = size_mb(fp32_model)
    int8_size = size_mb(int8_model)

    reduction = (
        1.0 - (int8_size / fp32_size)
    ) * 100.0

    ratio = (
        fp32_size / int8_size
    )

    print()
    print("=" * 70)
    print(f"{name.upper()} ARTEFACT SUMMARY")
    print("=" * 70)

    print(
        f"FP32 rollback: {fp32_model}"
    )

    print(
        f"INT8 model   : {int8_model}"
    )

    print(
        f"FP32 size    : {fp32_size:.2f} MB"
    )

    print(
        f"INT8 size    : {int8_size:.2f} MB"
    )

    print(
        f"Reduction    : {reduction:.2f}%"
    )

    print(
        f"Size ratio   : {ratio:.2f}x smaller"
    )


def main():
    args = parse_args()

    classifier_model = require_model_dir(
        args.classifier_model,
        "Classifier",
    )

    ner_model = require_model_dir(
        args.ner_model,
        "NER",
    )

    root = Path(
        args.output_dir
    )

    classifier_fp32 = (
        root / "classifier_fp32"
    )

    classifier_int8 = (
        root / "classifier_int8"
    )

    ner_fp32 = (
        root / "ner_fp32"
    )

    ner_int8 = (
        root / "ner_int8"
    )

    print("=" * 70)
    print("BAYAN LAB 7 — ONNX EXPORT")
    print("=" * 70)

    print(
        "Classifier:",
        classifier_model,
    )

    print(
        "NER:",
        ner_model,
    )

    print(
        "Output root:",
        root,
    )

    # ---------------------------------------------------------
    # Classifier
    # ---------------------------------------------------------

    run_optimum_export(
        classifier_model,
        classifier_fp32,
        "text-classification",
        force=args.force,
    )

    quantize_onnx(
        classifier_fp32,
        classifier_int8,
    )

    report_pair(
        "Classifier",
        classifier_fp32,
        classifier_int8,
    )

    # ---------------------------------------------------------
    # NER
    # ---------------------------------------------------------

    run_optimum_export(
        ner_model,
        ner_fp32,
        "token-classification",
        force=args.force,
    )

    quantize_onnx(
        ner_fp32,
        ner_int8,
    )

    report_pair(
        "NER",
        ner_fp32,
        ner_int8,
    )

    print()
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)

    print(
        "FP32 rollback artefacts were retained."
    )

    print(
        "Next: paired FP32/INT8 quality "
        "and latency evaluation."
    )


if __name__ == "__main__":
    main()
