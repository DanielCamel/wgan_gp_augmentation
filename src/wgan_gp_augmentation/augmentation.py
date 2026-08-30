from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


def add_synthetic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    synthetic_x: np.ndarray,
    synthetic_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if synthetic_x.ndim != 2 or synthetic_x.shape[1] != x_train.shape[1]:
        raise ValueError("Synthetic feature dimensions do not match training data")
    if len(synthetic_x) != len(synthetic_y):
        raise ValueError("Synthetic feature and label counts differ")
    return np.concatenate([x_train, synthetic_x]), np.concatenate([y_train, synthetic_y])


def sample_diagnostics(
    real: np.ndarray, synthetic: np.ndarray, maximum_rows: int = 1000
) -> dict[str, float | int | None]:
    """Compute descriptive checks without claiming distributional equivalence."""
    if not len(synthetic):
        return {"real_count": len(real), "synthetic_count": 0, "mean_l2": None,
                "std_l2": None, "synthetic_to_real_nn_mean": None,
                "real_pairwise_mean": None, "synthetic_pairwise_mean": None}
    real_subset = real[:maximum_rows]
    synthetic_subset = synthetic[:maximum_rows]
    nearest = pairwise_distances(synthetic_subset, real_subset).min(axis=1)
    real_pairwise = pairwise_distances(real_subset)
    synthetic_pairwise = pairwise_distances(synthetic_subset)
    real_upper = real_pairwise[np.triu_indices_from(real_pairwise, k=1)]
    synthetic_upper = synthetic_pairwise[np.triu_indices_from(synthetic_pairwise, k=1)]
    return {
        "real_count": len(real),
        "synthetic_count": len(synthetic),
        "mean_l2": float(np.linalg.norm(real.mean(axis=0) - synthetic.mean(axis=0))),
        "std_l2": float(np.linalg.norm(real.std(axis=0) - synthetic.std(axis=0))),
        "synthetic_to_real_nn_mean": float(nearest.mean()),
        "real_pairwise_mean": float(real_upper.mean()),
        "synthetic_pairwise_mean": float(synthetic_upper.mean()),
    }
