from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def clean_rows(
    frame: pd.DataFrame, label_column: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    features = frame.drop(columns=[label_column]).apply(pd.to_numeric, errors="coerce")
    labels = frame[label_column].reset_index(drop=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    invalid = features.isna().any(axis=1)
    valid_features = features.loc[~invalid].reset_index(drop=True)
    valid_labels = labels.loc[~invalid].reset_index(drop=True)
    combined = valid_features.copy()
    combined[label_column] = valid_labels
    grouped = combined.groupby(list(valid_features.columns), dropna=False)[label_column]
    conflicting = grouped.nunique()
    conflict_vector_count = int((conflicting > 1).sum())
    conflict_mask = grouped.transform("nunique").to_numpy() > 1
    nonconflicting_features = valid_features.loc[~conflict_mask]
    duplicate = nonconflicting_features.duplicated(keep="first")
    cleaned = combined.loc[~conflict_mask].loc[~duplicate].reset_index(drop=True)
    return cleaned, {
        "input_rows": len(frame),
        "invalid_rows_removed": int(invalid.sum()),
        "conflicting_feature_vectors": conflict_vector_count,
        "conflicting_rows_removed": int(np.asarray(conflict_mask).sum()),
        "duplicate_rows_removed": int(duplicate.sum()),
        "output_rows": len(cleaned),
    }


def split_rows(
    frame: pd.DataFrame,
    label_column: str,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> Splits:
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    train_validation, test = train_test_split(
        frame,
        test_size=test_fraction,
        random_state=seed,
        stratify=frame[label_column],
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        random_state=seed,
        stratify=train_validation[label_column],
    )
    return Splits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
