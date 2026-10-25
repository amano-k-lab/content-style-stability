"""Pure functions for model-vs-human metrics (accuracy, observed consistency)."""

from __future__ import annotations

import pandas as pd


def compute_accuracy(df: pd.DataFrame) -> float:
    """Return fraction of rows where label equals prediction."""
    total = int(len(df))
    if total == 0:
        raise ZeroDivisionError("compute_accuracy requires at least one row")
    return int((df["label"] == df["prediction"]).sum()) / total


def compute_observed_consistency(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> float:
    """Return fraction of matched (by filename) records where correct/error status agrees.

    For each filename present in both df1 and df2, compares whether the two
    predictions are both correct or both wrong; returns the proportion of such
    agreements over all matched filenames.
    """
    df = pd.concat([df1, df2], ignore_index=True)
    matched = 0
    total = 0
    for filename in df["filename"].unique():
        df_filename = df[df["filename"] == filename]
        if len(df_filename) != 2:
            continue
        correct_0 = df_filename["prediction"].iloc[0] == df_filename["label"].iloc[0]
        correct_1 = df_filename["prediction"].iloc[1] == df_filename["label"].iloc[1]
        if correct_0 == correct_1:
            matched += 1
        total += 1
    if total == 0:
        raise ZeroDivisionError(
            "compute_observed_consistency requires at least one exactly matched filename pair"
        )
    return matched / total
