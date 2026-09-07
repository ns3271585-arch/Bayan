"""Lab 3A: dataset construction and grouped split integrity."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(
    data_path: str | Path = DATA,
    random_state: int = 42,
) -> dict:
    """Build grouped train/validation/test splits with zero citizen overlap."""

    df = pd.read_csv(data_path)

    # ------------------------------------------------------------------
    # Split 1:
    # 70% train
    # 30% temporary set
    # Grouping is based on citizen_group_id.
    # ------------------------------------------------------------------
    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        first_split.split(
            df,
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # ------------------------------------------------------------------
    # Split 2:
    # Divide remaining 30% into:
    # 20% validation
    # 10% test
    #
    # validation therefore receives 2/3 of the temporary groups.
    # ------------------------------------------------------------------
    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=2 / 3,
        random_state=random_state,
    )

    validation_idx, test_idx = next(
        second_split.split(
            temp_df,
            groups=temp_df["citizen_group_id"],
        )
    )

    validation_df = temp_df.iloc[validation_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Reset row indexes for clean downstream use.
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }