"""Lab 3B: fine-tune XLM-R for Bayan named-entity recognition."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels
from bayan.preprocessing.arabic import segment


CHECKPOINT = "xlm-roberta-base"
DATA = Path("data/models/bayan_ner.conll")
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )
    parser.add_argument(
        "--use-segmentation",
         action="store_true",
        help="Apply CAMeL Tools d3tok segmentation before NER tokenization.",
)

    return parser.parse_args()


def read_conll(path):
    """Read token/label sentences from a two-column CoNLL file."""

    sentences = []
    labels = []

    current_tokens = []
    current_labels = []

    with open(path, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)

                    current_tokens = []
                    current_labels = []

                continue

            token, label = line.rsplit(None, 1)

            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def split_data(tokens, tags):
    """Create deterministic 80/10/10 train/validation/test splits."""

    indices = list(range(len(tokens)))

    rng = random.Random(SEED)
    rng.shuffle(indices)

    total = len(indices)

    train_end = int(total * 0.80)
    validation_end = int(total * 0.90)

    train_ids = indices[:train_end]
    validation_ids = indices[train_end:validation_end]
    test_ids = indices[validation_end:]

    def select(ids):
        return {
            "tokens": [tokens[i] for i in ids],
            "ner_tags": [tags[i] for i in ids],
        }

    return {
        "train": select(train_ids),
        "validation": select(validation_ids),
        "test": select(test_ids),
    }
def segment_ner_data(tokens, tags, id2label, label2id):
    """Apply d3tok segmentation while preserving NER entity labels.

    Prefix/suffix clitics receive O. The lexical stem keeps the
    original entity label.
    """
    o_id = label2id["O"]

    segmented_sentences = []
    segmented_tags = []

    for sentence_tokens, sentence_tags in zip(tokens, tags):
        new_tokens = []
        new_tags = []

        for token, tag_id in zip(sentence_tokens, sentence_tags):
            pieces = segment(token)

            # Fallback in case CAMeL returns nothing.
            if not pieces:
                pieces = [token]

            # No actual split: preserve token and label exactly.
            if len(pieces) == 1:
                new_tokens.append(pieces[0])
                new_tags.append(tag_id)
                continue

            # Find the lexical stem: a piece that is not a clitic marker.
            stem_index = next(
                (
                    i
                    for i, piece in enumerate(pieces)
                    if not piece.endswith("+") and not piece.startswith("+")
                ),
                len(pieces) - 1,
            )

            for i, piece in enumerate(pieces):
                new_tokens.append(piece)

                if i == stem_index:
                    new_tags.append(tag_id)
                else:
                    new_tags.append(o_id)

        segmented_sentences.append(new_tokens)
        segmented_tags.append(new_tags)

    return segmented_sentences, segmented_tags

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("BAYAN LAB 3B — XLM-R NER")
    print("=" * 70)

    # 1. Read CoNLL data
    sentences, string_labels = read_conll(DATA)

    print()
    print(f"Total sentences: {len(sentences)}")
    print(
        "Total tokens:",
        sum(len(sentence) for sentence in sentences),
    )

    # 2. Stable label mapping
    label_names = []

    for sentence_labels in string_labels:
        for label in sentence_labels:
            if label not in label_names:
                label_names.append(label)

    label2id = {
        label: index
        for index, label in enumerate(label_names)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print()
    print("Labels:")
    print(label2id)

    numeric_labels = [
        [label2id[label] for label in sentence]
        for sentence in string_labels
    ]
    if args.use_segmentation:
        print()
        print("Applying CAMeL Tools d3tok segmentation...")

        sentences, numeric_labels = segment_ner_data(
            sentences,
            numeric_labels,
            id2label,
            label2id,
        )

        print("Segmentation: ENABLED")
    else:
        print()
        print("Segmentation: DISABLED")

    # 3. Train / validation / frozen test split
    splits = split_data(
        sentences,
        numeric_labels,
    )

    train_dataset = Dataset.from_dict(
        splits["train"]
    )

    validation_dataset = Dataset.from_dict(
        splits["validation"]
    )

    test_dataset = Dataset.from_dict(
        splits["test"]
    )

    print()
    print("Dataset sizes:")
    print(f"Train:      {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print(f"Test:       {len(test_dataset)}")

    # 4. Tokenizer and correct word/subword label alignment
    print()
    print(f"Loading tokenizer: {CHECKPOINT}")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            max_length=128,
            is_split_into_words=True,
        )

        aligned_labels = []

        for batch_index, word_labels in enumerate(
            batch["ner_tags"]
        ):
            word_ids = tokenized.word_ids(
                batch_index=batch_index
            )

            aligned = align_labels(
                word_ids,
                word_labels,
            )

            aligned_labels.append(aligned)

        tokenized["labels"] = aligned_labels

        return tokenized

    train_dataset = train_dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    validation_dataset = validation_dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    test_dataset = test_dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    # 5. Entity-level seqeval metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        true_predictions = []
        true_labels = []

        for prediction, label in zip(
            predictions,
            labels,
        ):
            prediction_sequence = []
            label_sequence = []

            for predicted_id, true_id in zip(
                prediction,
                label,
            ):
                if true_id == -100:
                    continue

                prediction_sequence.append(
                    id2label[int(predicted_id)]
                )

                label_sequence.append(
                    id2label[int(true_id)]
                )

            true_predictions.append(
                prediction_sequence
            )

            true_labels.append(
                label_sequence
            )
        report = classification_report(
            true_labels,
            true_predictions,
            output_dict=True,
            zero_division=0,
        )

        location_recall = report.get(
            "LOCATION",
            {},
        ).get(
            "recall",
            0.0,
        )

        return {
            "precision": precision_score(
                true_labels,
                true_predictions,
            ),
            "recall": recall_score(
                true_labels,
                true_predictions,
            ),
            "f1": f1_score(
                true_labels,
                true_predictions,
            ),
            "accuracy": accuracy_score(
                true_labels,
                true_predictions,
            ),
            "location_recall": float(location_recall),
        }

    # 6. XLM-R token classifier
    print()
    print(f"Loading model: {CHECKPOINT}")

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(
            output_dir / "checkpoints"
        ),
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
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    trainer.train()

    # 8. Validation
    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validation_metrics = trainer.evaluate(
        validation_dataset,
        metric_key_prefix="validation",
    )

    print(validation_metrics)

    # 9. Frozen test
    print()
    print("=" * 70)
    print("FROZEN TEST")
    print("=" * 70)

    test_metrics = trainer.evaluate(
        test_dataset,
        metric_key_prefix="test",
    )

    print(test_metrics)

    # 10. Save artefact
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
    ) as file:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "checkpoint": CHECKPOINT,
                "validation": validation_metrics,
                "test": test_metrics,
            },
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "Validation entity-F1:",
        f"{validation_metrics['validation_f1']:.4f}",
    )

    print(
        "Frozen test entity-F1:",
        f"{test_metrics['test_f1']:.4f}",
    )

    print(
        "Validation precision:",
        f"{validation_metrics['validation_precision']:.4f}",
    )

    print(
        "Validation recall:",
        f"{validation_metrics['validation_recall']:.4f}",
    )
    print(
        "Validation LOCATION recall:",
        f"{validation_metrics['validation_location_recall']:.4f}",
)

    print(
        "Frozen test accuracy:",
        f"{test_metrics['test_accuracy']:.4f}",
    )
    print(
     "Frozen test LOCATION recall:",
      f"{test_metrics['test_location_recall']:.4f}",
)

    print()
    print(
        f"Saved NER artefact to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()