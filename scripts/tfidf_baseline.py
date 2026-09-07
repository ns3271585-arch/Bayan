"""Lab 3A: TF-IDF + LinearSVC topic-classification baseline."""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess


def prepare_texts(series):
    """Apply Bayan's shared preprocessing contract."""

    return (
        series
        .fillna("")
        .astype(str)
        .map(preprocess)
        .tolist()
    )


def evaluate(model, x, y, split_name):
    """Evaluate the classifier on one split."""

    predictions = model.predict(x)

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    print("=" * 70)
    print(split_name.upper())
    print("=" * 70)

    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print()

    print(
        classification_report(
            y,
            predictions,
            digits=4,
        )
    )

    return macro_f1, accuracy


def main():
    # Build the grouped, leakage-safe splits.
    ds = build_topic_dataset()

    train_df = ds["train"]
    validation_df = ds["validation"]
    test_df = ds["test"]

    print("Grouped dataset sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    print()
    print("Unique citizens:")
    print(
        f"Train:      "
        f"{train_df['citizen_group_id'].nunique()}"
    )
    print(
        f"Validation: "
        f"{validation_df['citizen_group_id'].nunique()}"
    )
    print(
        f"Test:       "
        f"{test_df['citizen_group_id'].nunique()}"
    )

    print()

    train_texts = prepare_texts(
        train_df["text"]
    )

    validation_texts = prepare_texts(
        validation_df["text"]
    )

    test_texts = prepare_texts(
        test_df["text"]
    )

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )

    x_train = vectorizer.fit_transform(
        train_texts
    )

    x_validation = vectorizer.transform(
        validation_texts
    )

    x_test = vectorizer.transform(
        test_texts
    )

    model = LinearSVC(
        C=1.0,
        random_state=42,
    )

    model.fit(
        x_train,
        train_df["topic"],
    )

    print(
        f"TF-IDF vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print()

    validation_f1, validation_accuracy = evaluate(
        model,
        x_validation,
        validation_df["topic"],
        "Validation",
    )

    test_f1, test_accuracy = evaluate(
        model,
        x_test,
        test_df["topic"],
        "Frozen Test",
    )

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Validation macro-F1: "
        f"{validation_f1:.4f}"
    )

    print(
        f"Frozen test macro-F1: "
        f"{test_f1:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Frozen test accuracy: "
        f"{test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()