"""Lab 4: Arabic model bake-off over All-Arabic, Gulf, and MSA slices."""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess


SEED = 42

CHECKPOINTS = {
    "XLM-R incumbent": "xlm-roberta-base",
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/arabic_bakeoff",
        help="Directory for bake-off checkpoints and results.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )

    return parser.parse_args()


def prepare_dataframe(df, label2id, arabic_only=False):
    """Preprocess text and map topic labels.

    arabic_only=True keeps only Arabic rows for Arabic-centric models.
    """

    df = df.copy()

    if arabic_only:
        df = df[df["lang"].eq("ar")].copy()

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .map(preprocess)
    )

    df["labels"] = df["topic"].map(label2id)

    return df


def compute_metrics(eval_pred):
    """Return macro-F1 and accuracy."""

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

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


def tokenizer_fertility(tokenizer, texts):
    """Average model tokens per whitespace-delimited Arabic word."""

    total_words = 0
    total_tokens = 0

    for text in texts:
        words = text.split()

        if not words:
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        total_words += len(words)
        total_tokens += len(encoded["input_ids"])

    if total_words == 0:
        return 0.0

    return total_tokens / total_words


def make_dataset(df, tokenizer):
    """Convert one dataframe slice to a tokenized HF Dataset."""

    dataset = Dataset.from_pandas(
        df[["text", "labels"]],
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )


def main():
    args = parse_args()
    set_seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BAYAN LAB 4 — ARABIC MODEL BAKE-OFF")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Use the same leakage-safe grouped split as Lab 3.
    # ------------------------------------------------------------------

    splits = build_topic_dataset()

    train_raw = splits["train"]
    validation_raw = splits["validation"]
    test_raw = splits["test"]

    topics = sorted(
        train_raw["topic"]
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
    # Day-2 incumbent keeps the original bilingual Lab 3 training data.
    full_train_df = prepare_dataframe(
        train_raw,
        label2id,
        arabic_only=False,
    )

    full_validation_df = prepare_dataframe(
        validation_raw,
        label2id,
        arabic_only=False,
    )

    # Arabic-centric checkpoints train on the Arabic slice.
    arabic_train_df = prepare_dataframe(
        train_raw,
        label2id,
        arabic_only=True,
    )

    arabic_validation_df = prepare_dataframe(
        validation_raw,
        label2id,
        arabic_only=True,
    )

    # All models are evaluated on the same Arabic frozen-test slices.
    test_df = prepare_dataframe(
        test_raw,
        label2id,
        arabic_only=True,
    )

    gulf_test_df = test_df[
        test_df["dialect_region"].eq("Gulf")
    ].copy()

    msa_test_df = test_df[
        test_df["dialect_region"].eq("MSA")
    ].copy()

    print()
    print("Grouped-split sizes:")
    print(f"Day-2 bilingual train:       {len(full_train_df)}")
    print(f"Day-2 bilingual validation:  {len(full_validation_df)}")
    print(f"Arabic train:                {len(arabic_train_df)}")
    print(f"Arabic validation:           {len(arabic_validation_df)}")
    print(f"Arabic frozen test:          {len(test_df)}")
    print(f"Gulf frozen test:            {len(gulf_test_df)}")
    print(f"MSA frozen test:             {len(msa_test_df)}")

    results = {}

    # ------------------------------------------------------------------
    # 2. Fine-tune and evaluate each checkpoint identically.
    # ------------------------------------------------------------------

    for model_name, checkpoint in CHECKPOINTS.items():
        print()
        print("=" * 72)
        print(model_name)
        print(checkpoint)
        print("=" * 72)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        fertility = tokenizer_fertility(
            tokenizer,
            test_df["text"].tolist(),
        )

        if model_name == "XLM-R incumbent":
            model_train_df = full_train_df
            model_validation_df = full_validation_df
        else:
            model_train_df = arabic_train_df
            model_validation_df = arabic_validation_df

        train_dataset = make_dataset(
            model_train_df,
            tokenizer,
        )

        validation_dataset = make_dataset(
            model_validation_df,
            tokenizer,
        )

        test_dataset = make_dataset(
            test_df,
            tokenizer,
        )

        gulf_dataset = make_dataset(
            gulf_test_df,
            tokenizer,
        )

        msa_dataset = make_dataset(
            msa_test_df,
            tokenizer,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=len(topics),
            label2id=label2id,
            id2label=id2label,
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )

        batch_size = 8
        gradient_accumulation = 2

        steps_per_epoch = math.ceil(
            len(train_dataset)
            / (batch_size * gradient_accumulation)
        )

        total_steps = steps_per_epoch * args.epochs
        warmup_steps = max(
            1,
            round(total_steps * 0.10),
        )

        model_dir = output_dir / model_name.lower().replace(
            " ",
            "_",
        ).replace(
            "-",
            "_",
        )

        training_args = TrainingArguments(
            output_dir=str(model_dir / "checkpoints"),

            num_train_epochs=args.epochs,

            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=warmup_steps,

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=gradient_accumulation,

            fp16=torch.cuda.is_available(),

            eval_strategy="epoch",
            save_strategy="epoch",

            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,

            save_total_limit=1,

            logging_strategy="steps",
            logging_steps=50,

            report_to="none",
            seed=SEED,
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

        start = time.perf_counter()

        trainer.train()

        train_seconds = time.perf_counter() - start

        print()
        print("Evaluating All Arabic test slice...")
        all_metrics = trainer.evaluate(
            test_dataset,
            metric_key_prefix="all",
        )

        print("Evaluating Gulf test slice...")
        gulf_metrics = trainer.evaluate(
            gulf_dataset,
            metric_key_prefix="gulf",
        )

        print("Evaluating MSA test slice...")
        msa_metrics = trainer.evaluate(
            msa_dataset,
            metric_key_prefix="msa",
        )

        results[model_name] = {
            "checkpoint": checkpoint,
            "macro_f1_all_arabic": all_metrics["all_macro_f1"],
            "macro_f1_gulf": gulf_metrics["gulf_macro_f1"],
            "macro_f1_msa": msa_metrics["msa_macro_f1"],
            "accuracy_all_arabic": all_metrics["all_accuracy"],
            "accuracy_gulf": gulf_metrics["gulf_accuracy"],
            "accuracy_msa": msa_metrics["msa_accuracy"],
            "arabic_fertility": fertility,
            "train_seconds": train_seconds,
        }

        print()
        print("RESULT")
        print(
            f"All Arabic macro-F1: "
            f"{results[model_name]['macro_f1_all_arabic']:.4f}"
        )
        print(
            f"Gulf macro-F1: "
            f"{results[model_name]['macro_f1_gulf']:.4f}"
        )
        print(
            f"MSA macro-F1: "
            f"{results[model_name]['macro_f1_msa']:.4f}"
        )
        print(
            f"Arabic fertility: "
            f"{results[model_name]['arabic_fertility']:.3f}"
        )

        del trainer
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Save measured results.
    # ------------------------------------------------------------------

    results_path = output_dir / "results.json"

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print("ARABIC BAKE-OFF SUMMARY")
    print("=" * 72)

    print(
        f"{'Model':<22}"
        f"{'All AR':>10}"
        f"{'Gulf':>10}"
        f"{'MSA':>10}"
        f"{'Fertility':>12}"
    )

    print("-" * 64)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<22}"
            f"{metrics['macro_f1_all_arabic']:>10.4f}"
            f"{metrics['macro_f1_gulf']:>10.4f}"
            f"{metrics['macro_f1_msa']:>10.4f}"
            f"{metrics['arabic_fertility']:>12.3f}"
        )

    print()
    print(f"Saved results to: {results_path.resolve()}")


if __name__ == "__main__":
    main()