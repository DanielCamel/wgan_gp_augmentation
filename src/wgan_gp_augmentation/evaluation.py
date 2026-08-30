from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score


def score_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    minority_labels: tuple[str, ...],
) -> dict[str, object]:
    labels = np.arange(len(class_names))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fnr = {
        name: 1.0 - float(report[name]["recall"])
        for name in minority_labels
        if name in report
    }
    metrics: dict[str, object] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {name: report[name] for name in class_names},
        "minority_false_negative_rate": fnr,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }
    present = np.unique(y_true)
    if len(present) == len(class_names) and probabilities.shape[1] == len(class_names):
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro", labels=labels)
        )
    else:
        metrics["macro_roc_auc_ovr"] = None
    return metrics


def save_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
