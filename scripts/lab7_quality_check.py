
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score as cls_accuracy
from sklearn.metrics import f1_score as cls_f1
from seqeval.metrics import (
    accuracy_score as ner_accuracy,
    f1_score as ner_f1,
)
from transformers import AutoTokenizer

from bayan.models.data import build_topic_dataset
from bayan.models.ner import align_labels
from bayan.preprocessing.core import preprocess


ROOT = Path(
    "/content/drive/MyDrive/Bayan/artifacts/lab7_onnx"
)

CLASSIFIER_FP32 = ROOT / "classifier_fp32/model.onnx"
CLASSIFIER_INT8 = ROOT / "classifier_int8/model.onnx"

NER_FP32 = ROOT / "ner_fp32/model.onnx"
NER_INT8 = ROOT / "ner_int8/model.onnx"

CLASSIFIER_TOKENIZER = ROOT / "classifier_fp32"
NER_TOKENIZER = ROOT / "ner_fp32"

NER_SOURCE_MODEL = Path(
    "/content/drive/MyDrive/Bayan/artifacts/ner_segmented"
)

RESULT_FILE = Path(
    "/content/drive/MyDrive/Bayan/lab7_results/"
    "quality_results.json"
)

BOOTSTRAPS = 1000
SEED = 42


def make_session(model_path):
    options = ort.SessionOptions()

    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def session_predict(session, encoded):
    input_names = {
        item.name
        for item in session.get_inputs()
    }

    inputs = {
        key: value
        for key, value in encoded.items()
        if key in input_names
    }

    return session.run(
        None,
        inputs,
    )[0]


def bootstrap_classifier_tax(
    y_true,
    pred_fp32,
    pred_int8,
    labels,
):
    rng = np.random.default_rng(SEED)

    y_true = np.asarray(y_true)
    pred_fp32 = np.asarray(pred_fp32)
    pred_int8 = np.asarray(pred_int8)

    n = len(y_true)
    taxes = []

    for _ in range(BOOTSTRAPS):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        fp32_score = cls_f1(
            y_true[indices],
            pred_fp32[indices],
            labels=labels,
            average="macro",
            zero_division=0,
        )

        int8_score = cls_f1(
            y_true[indices],
            pred_int8[indices],
            labels=labels,
            average="macro",
            zero_division=0,
        )

        taxes.append(
            (fp32_score - int8_score) * 100.0
        )

    return (
        float(np.percentile(taxes, 2.5)),
        float(np.percentile(taxes, 97.5)),
    )


def evaluate_classifier():
    print()
    print("=" * 70)
    print("CLASSIFIER QUALITY CHECK")
    print("=" * 70)

    splits = build_topic_dataset()

    train_df = splits["train"]
    test_df = splits["test"].copy()

    topics = sorted(
        train_df["topic"]
        .dropna()
        .unique()
        .tolist()
    )

    label2id = {
        label: index
        for index, label in enumerate(topics)
    }

    test_df = test_df[
        ["text", "topic"]
    ].copy()

    test_df["text"] = (
        test_df["text"]
        .fillna("")
        .astype(str)
        .map(preprocess)
    )

    test_df["label"] = (
        test_df["topic"]
        .map(label2id)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(CLASSIFIER_TOKENIZER),
        local_files_only=True,
    )

    fp32 = make_session(
        CLASSIFIER_FP32
    )

    int8 = make_session(
        CLASSIFIER_INT8
    )

    y_true = []
    pred_fp32 = []
    pred_int8 = []

    for index, row in enumerate(
        test_df.itertuples(index=False),
        start=1,
    ):
        encoded = tokenizer(
            row.text,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        fp32_logits = session_predict(
            fp32,
            encoded,
        )[0]

        int8_logits = session_predict(
            int8,
            encoded,
        )[0]

        y_true.append(
            int(row.label)
        )

        pred_fp32.append(
            int(np.argmax(fp32_logits))
        )

        pred_int8.append(
            int(np.argmax(int8_logits))
        )

        if index % 250 == 0:
            print(
                f"Processed "
                f"{index}/{len(test_df)}"
            )

    labels = list(
        range(len(topics))
    )

    fp32_f1 = cls_f1(
        y_true,
        pred_fp32,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    int8_f1 = cls_f1(
        y_true,
        pred_int8,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    fp32_accuracy = cls_accuracy(
        y_true,
        pred_fp32,
    )

    int8_accuracy = cls_accuracy(
        y_true,
        pred_int8,
    )

    agreement = float(
        np.mean(
            np.asarray(pred_fp32)
            == np.asarray(pred_int8)
        )
    )

    tax_points = (
        fp32_f1 - int8_f1
    ) * 100.0

    ci_low, ci_high = (
        bootstrap_classifier_tax(
            y_true,
            pred_fp32,
            pred_int8,
            labels,
        )
    )

    print()
    print("Samples:", len(y_true))
    print(
        f"FP32 macro-F1: {fp32_f1:.6f}"
    )
    print(
        f"INT8 macro-F1: {int8_f1:.6f}"
    )
    print(
        f"Macro-F1 tax: {tax_points:.4f} points"
    )
    print(
        "95% CI for tax: "
        f"[{ci_low:.4f}, {ci_high:.4f}] points"
    )
    print(
        f"FP32 accuracy: {fp32_accuracy:.6f}"
    )
    print(
        f"INT8 accuracy: {int8_accuracy:.6f}"
    )
    print(
        f"FP32/INT8 agreement: "
        f"{agreement * 100:.2f}%"
    )

    return {
        "samples": len(y_true),
        "fp32_macro_f1": fp32_f1,
        "int8_macro_f1": int8_f1,
        "macro_f1_tax_points": tax_points,
        "tax_ci95_low": ci_low,
        "tax_ci95_high": ci_high,
        "fp32_accuracy": fp32_accuracy,
        "int8_accuracy": int8_accuracy,
        "agreement": agreement,
    }


def load_train_ner_module():
    path = Path(
        "/content/Bayan/scripts/train_ner.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "bayan_train_ner",
            path,
        )
    )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    spec.loader.exec_module(
        module
    )

    return module


def bootstrap_ner_tax(
    true_sequences,
    fp32_sequences,
    int8_sequences,
):
    rng = np.random.default_rng(SEED)

    n = len(true_sequences)
    taxes = []

    for _ in range(BOOTSTRAPS):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        truth = [
            true_sequences[i]
            for i in indices
        ]

        fp32_pred = [
            fp32_sequences[i]
            for i in indices
        ]

        int8_pred = [
            int8_sequences[i]
            for i in indices
        ]

        fp32_score = ner_f1(
            truth,
            fp32_pred,
            zero_division=0,
        )

        int8_score = ner_f1(
            truth,
            int8_pred,
            zero_division=0,
        )

        taxes.append(
            (fp32_score - int8_score) * 100.0
        )

    return (
        float(np.percentile(taxes, 2.5)),
        float(np.percentile(taxes, 97.5)),
    )


def evaluate_ner():
    print()
    print("=" * 70)
    print("NER QUALITY CHECK")
    print("=" * 70)

    train_ner = (
        load_train_ner_module()
    )

    sentences, string_labels = (
        train_ner.read_conll(
            Path(
                "/content/Bayan/"
                "data/models/bayan_ner.conll"
            )
        )
    )

    label_names = []

    for sentence_labels in string_labels:
        for label in sentence_labels:
            if label not in label_names:
                label_names.append(label)

    label2id = {
        label: index
        for index, label
        in enumerate(label_names)
    }

    id2label = {
        index: label
        for label, index
        in label2id.items()
    }

    numeric_labels = [
        [
            label2id[label]
            for label in sentence
        ]
        for sentence in string_labels
    ]

    print(
        "Applying same d3tok segmentation "
        "used during NER training..."
    )

    sentences, numeric_labels = (
        train_ner.segment_ner_data(
            sentences,
            numeric_labels,
            id2label,
            label2id,
        )
    )

    splits = train_ner.split_data(
        sentences,
        numeric_labels,
    )

    test_tokens = (
        splits["test"]["tokens"]
    )

    test_tags = (
        splits["test"]["ner_tags"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(NER_TOKENIZER),
        local_files_only=True,
    )

    fp32 = make_session(
        NER_FP32
    )

    int8 = make_session(
        NER_INT8
    )

    true_sequences = []
    fp32_sequences = []
    int8_sequences = []

    token_matches = 0
    token_total = 0

    for index, (
        tokens,
        word_labels,
    ) in enumerate(
        zip(
            test_tokens,
            test_tags,
        ),
        start=1,
    ):
        encoded = tokenizer(
            tokens,
            return_tensors="np",
            truncation=True,
            max_length=128,
            is_split_into_words=True,
        )

        word_ids = encoded.word_ids(
            batch_index=0
        )

        aligned = align_labels(
            word_ids,
            word_labels,
        )

        fp32_logits = session_predict(
            fp32,
            encoded,
        )[0]

        int8_logits = session_predict(
            int8,
            encoded,
        )[0]

        fp32_ids = np.argmax(
            fp32_logits,
            axis=-1,
        )

        int8_ids = np.argmax(
            int8_logits,
            axis=-1,
        )

        true_sequence = []
        fp32_sequence = []
        int8_sequence = []

        for true_id, fp32_id, int8_id in zip(
            aligned,
            fp32_ids,
            int8_ids,
        ):
            if true_id == -100:
                continue

            true_sequence.append(
                id2label[int(true_id)]
            )

            fp32_sequence.append(
                id2label[int(fp32_id)]
            )

            int8_sequence.append(
                id2label[int(int8_id)]
            )

            token_total += 1

            if int(fp32_id) == int(int8_id):
                token_matches += 1

        true_sequences.append(
            true_sequence
        )

        fp32_sequences.append(
            fp32_sequence
        )

        int8_sequences.append(
            int8_sequence
        )

        if index % 100 == 0:
            print(
                f"Processed "
                f"{index}/{len(test_tokens)}"
            )

    fp32_f1 = ner_f1(
        true_sequences,
        fp32_sequences,
        zero_division=0,
    )

    int8_f1 = ner_f1(
        true_sequences,
        int8_sequences,
        zero_division=0,
    )

    fp32_accuracy = ner_accuracy(
        true_sequences,
        fp32_sequences,
    )

    int8_accuracy = ner_accuracy(
        true_sequences,
        int8_sequences,
    )

    tax_points = (
        fp32_f1 - int8_f1
    ) * 100.0

    ci_low, ci_high = (
        bootstrap_ner_tax(
            true_sequences,
            fp32_sequences,
            int8_sequences,
        )
    )

    token_agreement = (
        token_matches / token_total
        if token_total
        else 0.0
    )

    print()
    print(
        "Sentences:",
        len(true_sequences),
    )

    print(
        "Evaluated labelled tokens:",
        token_total,
    )

    print(
        f"FP32 entity-F1: {fp32_f1:.6f}"
    )

    print(
        f"INT8 entity-F1: {int8_f1:.6f}"
    )

    print(
        f"Entity-F1 tax: "
        f"{tax_points:.4f} points"
    )

    print(
        "95% CI for tax: "
        f"[{ci_low:.4f}, {ci_high:.4f}] points"
    )

    print(
        f"FP32 token accuracy: "
        f"{fp32_accuracy:.6f}"
    )

    print(
        f"INT8 token accuracy: "
        f"{int8_accuracy:.6f}"
    )

    print(
        "FP32/INT8 labelled-token agreement: "
        f"{token_agreement * 100:.2f}%"
    )

    return {
        "sentences": len(
            true_sequences
        ),
        "labelled_tokens": token_total,
        "fp32_entity_f1": fp32_f1,
        "int8_entity_f1": int8_f1,
        "entity_f1_tax_points": tax_points,
        "tax_ci95_low": ci_low,
        "tax_ci95_high": ci_high,
        "fp32_token_accuracy": fp32_accuracy,
        "int8_token_accuracy": int8_accuracy,
        "token_agreement": token_agreement,
    }


def main():
    classifier = evaluate_classifier()
    ner = evaluate_ner()

    results = {
        "classifier": classifier,
        "ner": ner,
        "bootstrap_iterations": BOOTSTRAPS,
        "bootstrap_seed": SEED,
    }

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_FILE.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("QUALITY CHECK COMPLETE")
    print("=" * 70)
    print("Saved:", RESULT_FILE)


if __name__ == "__main__":
    main()
