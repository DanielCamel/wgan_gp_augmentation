from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from .config import ClfConfig


@dataclass(frozen=True)
class FitResult:
    model: MLPClassifier
    history: list[dict[str, float | int]]
    best_epoch: int
    elapsed_seconds: float


def fit_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: ClfConfig,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> FitResult:
    classes = np.unique(y_train)
    model = MLPClassifier(
        hidden_layer_sizes=config.hidden_layers,
        activation="relu",
        solver="adam",
        alpha=config.alpha,
        batch_size=min(config.batch_size, len(x_train)),
        learning_rate_init=config.learning_rate,
        max_iter=1,
        shuffle=True,
        random_state=seed,
    )
    best_model: MLPClassifier | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.partial_fit(x_train, y_train, classes=classes, sample_weight=sample_weight)
        train_loss = log_loss(y_train, model.predict_proba(x_train), labels=classes)
        validation_loss = log_loss(y_validation, model.predict_proba(x_validation), labels=classes)
        history.append(
            {"epoch": epoch, "train_loss": float(train_loss), "validation_loss": float(validation_loss)}
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = float(validation_loss)
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_model is None:
        raise RuntimeError("Baseline training produced no model")
    return FitResult(best_model, history, best_epoch, time.perf_counter() - started)
