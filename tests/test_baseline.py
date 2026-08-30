import numpy as np

from wgan_gp_augmentation.baseline import fit_classifier
from wgan_gp_augmentation.config import ClfConfig


def test_baseline_trains_and_returns_probabilities() -> None:
    rng = np.random.default_rng(42)
    x = np.r_[rng.normal(-1, 0.2, (30, 3)), rng.normal(1, 0.2, (30, 3))].astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    config = ClfConfig((8,), 16, 0.01, 5, 2, 0.0001)
    result = fit_classifier(x[:48], y[:48], x[48:], y[48:], config, 42)
    probabilities = result.model.predict_proba(x[48:])
    assert probabilities.shape == (12, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
