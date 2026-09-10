"""Lab 6: sliced evaluation report."""

from __future__ import annotations

import pandas as pd

from bayan.evaluation.bootstrap import bootstrap_ci


REQUIRED_COLUMNS = {
    "lang",
    "dialect_region",
    "length_bucket",
    "y_true",
    "y_pred",
    "confidence",
}

SLICE_COLUMNS = {
    "language": "lang",
    "dialect": "dialect_region",
    "class": "y_true",
    "length": "length_bucket",
}


def sliced_report(
    predictions: pd.DataFrame,
    *,
    min_slice_size: int = 30,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Build accuracy slices for language, dialect, class and length.

    Each row reports:
    - slice type and value
    - number of examples
    - accuracy
    - bootstrap confidence interval
    - mean confidence
    - whether the slice is considered small

    ``min_slice_size`` is configurable because the course specification
    requires small slices to be flagged but does not prescribe a threshold.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")

    if predictions.empty:
        raise ValueError("predictions must not be empty")

    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(
            "predictions is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if min_slice_size <= 0:
        raise ValueError("min_slice_size must be positive")

    df = predictions.copy()

    # pandas interprets the CSV value "NA" as a missing value.
    # Keep it visible as its own dialect slice instead of dropping it.
    df["dialect_region"] = df["dialect_region"].fillna("NA")

    df["correct"] = (df["y_true"] == df["y_pred"]).astype(float)

    rows = []

    overall_point, overall_lo, overall_hi = bootstrap_ci(
        df["correct"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )

    rows.append(
        {
            "slice_type": "overall",
            "slice_value": "all",
            "n": int(len(df)),
            "accuracy": overall_point,
            "ci_low": overall_lo,
            "ci_high": overall_hi,
            "mean_confidence": float(df["confidence"].mean()),
            "small_slice": len(df) < min_slice_size,
        }
    )

    for slice_number, (slice_type, column) in enumerate(
        SLICE_COLUMNS.items(),
        start=1,
    ):
        for group_number, (value, group) in enumerate(
            df.groupby(column, dropna=False, sort=True),
            start=1,
        ):
            point, lo, hi = bootstrap_ci(
                group["correct"].to_numpy(),
                n_boot=n_boot,
                seed=seed + (slice_number * 1000) + group_number,
                alpha=alpha,
            )

            rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": str(value),
                    "n": int(len(group)),
                    "accuracy": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "mean_confidence": float(group["confidence"].mean()),
                    "small_slice": len(group) < min_slice_size,
                }
            )

    return pd.DataFrame(rows)