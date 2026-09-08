"""Lab 3A: fine-tune the Bayan XLM-R topic classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )

    return parser.parse_args()


def prepare_dataframe(df, label2id):
    """Apply shared preprocessing and convert topic labels to integers."""

    df = df[["text", "topic"]].copy()

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .map(preprocess)
    )

    df["labels"] = df["topic"].map(label2id)

    return df[["text", "labels"]]


def compute_metrics(eval_pred):
    """Return macro-F1 and accuracy."""

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("BAYAN LAB 3A — XLM-R TOPIC CLASSIFIER")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Leakage-safe grouped dataset
    # ------------------------------------------------------------

    ds = build_topic_dataset()

    train_df = ds["train"]
    validation_df = ds["validation"]
    test_df = ds["test"]

    print()
    print("Dataset sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    # ------------------------------------------------------------
    # 2. Stable label mapping
    # ------------------------------------------------------------

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

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print()
    print("Labels:")
    print(label2id)

    # ------------------------------------------------------------
    # 3. Shared Bayan preprocessing
    # ------------------------------------------------------------

    train_df = prepare_dataframe(
        train_df,
        label2id,
    )

    validation_df = prepare_dataframe(
        validation_df,
        label2id,
    )

    test_df = prepare_dataframe(
        test_df,
        label2id,
    )

    train_dataset = Dataset.from_pandas(
        train_df,
        preserve_index=False,
    )

    validation_dataset = Dataset.from_pandas(
        validation_df,
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_df,
        preserve_index=False,
    )

    # ------------------------------------------------------------
    # 4. XLM-R tokenizer
    # ------------------------------------------------------------

    print()
    print(f"Loading tokenizer: {CHECKPOINT}")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    # ------------------------------------------------------------
    # 5. XLM-R sequence classifier
    # ------------------------------------------------------------

    print()
    print(f"Loading model: {CHECKPOINT}")

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(topics),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),

        num_train_epochs=3,

        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,

        fp16=True,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=50,

        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------
    # 6. Fine-tune
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    trainer.train()

    # ------------------------------------------------------------
    # 7. Validation
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validation_metrics = trainer.evaluate(
        validation_dataset,
        metric_key_prefix="validation",
    )

    print(validation_metrics)

    # ------------------------------------------------------------
    # 8. Frozen test
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FROZEN TEST")
    print("=" * 70)

    test_metrics = trainer.evaluate(
        test_dataset,
        metric_key_prefix="test",
    )

    print(test_metrics)

    # ------------------------------------------------------------
    # 9. Save rerunnable artefact
    # ------------------------------------------------------------

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    with open(
        output_dir / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "checkpoint": CHECKPOINT,
                "validation": validation_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "Validation macro-F1:",
        f"{validation_metrics['validation_macro_f1']:.4f}",
    )

    print(
        "Frozen test macro-F1:",
        f"{test_metrics['test_macro_f1']:.4f}",
    )

    print(
        "Validation accuracy:",
        f"{validation_metrics['validation_accuracy']:.4f}",
    )

    print(
        "Frozen test accuracy:",
        f"{test_metrics['test_accuracy']:.4f}",
    )

    print()
    print(
        f"Saved classifier artefact to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()