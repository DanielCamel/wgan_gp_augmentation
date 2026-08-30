from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wgan_gp_augmentation.baseline import fit_classifier
from wgan_gp_augmentation.config import load_config
from wgan_gp_augmentation.evaluation import save_json, score_classifier
from wgan_gp_augmentation.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the real-only baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.experiment.seed if args.seed is None else args.seed
    set_seed(seed)
    data = np.load(args.prepared / "splits.npz")
    prep = joblib.load(args.prepared / "preprocessor.joblib")
    fit = fit_classifier(
        data["x_train"], data["y_train"], data["x_validation"], data["y_validation"],
        cfg.classifier, seed,
    )
    proba = fit.model.predict_proba(data["x_test"])
    pred = fit.model.predict(data["x_test"])
    class_names = prep.label_encoder.classes_.tolist()
    metrics = score_classifier(
        data["y_test"], pred, proba, class_names, cfg.experiment.minority_labels,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit.model, args.output / "model.joblib")
    np.savez_compressed(
        args.output / "predictions.npz",
        y_true=data["y_test"], y_pred=pred, probabilities=proba,
    )
    save_json(args.output / "metrics.json", metrics)
    save_json(args.output / "history.json", fit.history)
    run = {
        "condition": "unweighted_real_only_baseline",
        "seed": seed,
        "best_epoch": fit.best_epoch,
        "elapsed_seconds": fit.elapsed_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    save_json(args.output / "run.json", run)
    print(json.dumps({"macro_f1": metrics["macro_f1"], "best_epoch": fit.best_epoch}))


if __name__ == "__main__":
    main()
