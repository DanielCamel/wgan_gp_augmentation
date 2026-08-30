from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .splitting import Splits


@dataclass
class Preprocessor:
    variance_threshold: float
    pca_variance: float

    def __post_init__(self) -> None:
        self.variance_filter = VarianceThreshold(self.variance_threshold)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_variance, svd_solver="full")
        self.label_encoder = LabelEncoder()
        self.feature_names_in_: tuple[str, ...] | None = None
        self.fit_row_count_: int | None = None

    def fit(self, train: pd.DataFrame, label_column: str) -> "Preprocessor":
        x = train.drop(columns=[label_column])
        self.feature_names_in_ = tuple(x.columns)
        self.fit_row_count_ = len(train)
        filtered = self.variance_filter.fit_transform(x)
        scaled = self.scaler.fit_transform(filtered)
        self.pca.fit(scaled)
        self.label_encoder.fit(train[label_column])
        return self

    def transform(self, frame: pd.DataFrame, label_column: str) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_names_in_ is None:
            raise RuntimeError("Preprocessor has not been fitted")
        x = frame.loc[:, self.feature_names_in_]
        filtered = self.variance_filter.transform(x)
        scaled = self.scaler.transform(filtered)
        transformed = self.pca.transform(scaled).astype(np.float32)
        labels = self.label_encoder.transform(frame[label_column]).astype(np.int64)
        return transformed, labels


def encode_splits(
    splits: Splits, prep: Preprocessor, label_column: str
) -> dict[str, np.ndarray]:
    x_train, y_train = prep.transform(splits.train, label_column)
    x_validation, y_validation = prep.transform(splits.validation, label_column)
    x_test, y_test = prep.transform(splits.test, label_column)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_test": x_test,
        "y_test": y_test,
    }


# Keeps existing joblib artifacts readable after the shorter class rename.
TrainOnlyPreprocessor = Preprocessor
