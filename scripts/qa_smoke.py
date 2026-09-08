"""Lab 3B: run Bayan's 12-question extractive-QA smoke set."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


DATA = Path("data/models/bayan_qa.json")

CHECKPOINT = "deepset/roberta-base-squad2"

ANSWERABLE_IDS = {
    "QA-0001",
    "QA-0002",
    "QA-0003",
    "QA-0004",
    "QA-0005",
    "QA-0006",
    "QA-0007",
    "QA-0008",
    "QA-0009",
}

NULL_IDS = {
    "QA-0013",
    "QA-0014",
    "QA-0015",
}


def load_smoke_set():
    """Load the fixed 9-answerable + 3-null Bayan QA smoke cases."""

    with open(DATA, encoding="utf-8") as file:
        payload = json.load(file)

    wanted = ANSWERABLE_IDS | NULL_IDS
    examples = []

    for item in payload["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                if qa["id"] not in wanted:
                    continue

                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answers": qa["answers"],
                        "is_impossible": qa["is_impossible"],
                    }
                )

    examples.sort(key=lambda example: example["id"])

    return examples


def predict(example, tokenizer, model, device):
    """Run extractive QA and return Bayan's best-span result."""

    encoded = tokenizer(
        example["question"],
        example["context"],
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
    )

    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    sequence_ids = encoded.sequence_ids(0)

    # Only offsets belonging to the context are valid answer positions.
    offsets = [
        tuple(offset) if sequence_id == 1 else None
        for offset, sequence_id in zip(
            offset_mapping,
            sequence_ids,
        )
    ]

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)

    start_logits = (
        outputs.start_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    end_logits = (
        outputs.end_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    # RoBERTa uses the first token as the null/CLS position.
    null_score = float(
        start_logits[0] + end_logits[0]
    )

    result = best_span(
        start_logits,
        end_logits,
        offsets,
        null_score=null_score,
        null_threshold=0.0,
        max_answer_len=30,
        top_k=20,
    )

    if result["answer"] is not None:
        start_char, end_char = result["answer"]

        result["text"] = example["context"][
            start_char:end_char
        ]
    else:
        result["text"] = None

    return result


def normalise(text):
    """Simple exact-match normalisation for the smoke set."""

    if text is None:
        return None

    return " ".join(
        text.strip().lower().split()
    )


def main():
    examples = load_smoke_set()

    if len(examples) != 12:
        raise RuntimeError(
            f"Expected 12 smoke examples, found {len(examples)}"
        )

    answerable_count = sum(
        not example["is_impossible"]
        for example in examples
    )

    null_count = sum(
        example["is_impossible"]
        for example in examples
    )

    print("=" * 70)
    print("BAYAN LAB 3B — QA SMOKE TEST")
    print("=" * 70)
    print(f"Examples:   {len(examples)}")
    print(f"Answerable: {answerable_count}")
    print(f"Null:       {null_count}")
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        CHECKPOINT
    ).to(device)

    model.eval()

    answerable_correct = 0
    null_correct = 0

    for example in examples:
        result = predict(
            example,
            tokenizer,
            model,
            device,
        )

        if example["is_impossible"]:
            correct = result["answer"] is None

            if correct:
                null_correct += 1

            expected = None

        else:
            gold_answers = [
                normalise(answer["text"])
                for answer in example["answers"]
            ]

            predicted = normalise(
                result["text"]
            )

            correct = predicted in gold_answers

            if correct:
                answerable_correct += 1

            expected = [
                answer["text"]
                for answer in example["answers"]
            ]

        print(
            f"{example['id']}: "
            f"predicted={result['text']!r} "
            f"expected={expected!r} "
            f"{'PASS' if correct else 'FAIL'}"
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Answerable correct: "
        f"{answerable_correct}/9"
    )

    print(
        f"Null correct: "
        f"{null_correct}/3"
    )

    total_correct = (
        answerable_correct
        + null_correct
    )

    print(
        f"Overall: "
        f"{total_correct}/12"
    )

    if answerable_correct == 9 and null_correct == 3:
        print("QA smoke target: PASS")
    else:
        print("QA smoke target: FAIL")


if __name__ == "__main__":
    main()