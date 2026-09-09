"""Lab 4: audit dialect mix over the Arabic slice."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    # Keep only Arabic feedback.
    arabic = df[df["lang"].eq("ar")].copy()

    print("=" * 70)
    print("BAYAN DIALECT AUDIT")
    print("=" * 70)

    print(f"Total rows: {len(df)}")
    print(f"Arabic rows: {len(arabic)}")
    print()

    counts = arabic["dialect_region"].value_counts(dropna=False)
    percentages = (
        arabic["dialect_region"]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
    )

    print("Arabic dialect / region distribution")
    print("-" * 70)

    for region, count in counts.items():
        label = "Unknown" if pd.isna(region) else str(region)
        pct = percentages.loc[region]
        print(f"{label}: {count} ({pct:.2f}%)")

    print()
    print("Evaluation implication")
    print("-" * 70)
    print(
        "Evaluating only on MSA would not represent the Bayan Arabic "
        "distribution because most Arabic feedback belongs to the Gulf slice."
    )


if __name__ == "__main__":
    main()