from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wgan_gp_augmentation.augmentation import add_synthetic
from wgan_gp_augmentation.baseline import fit_classifier
from wgan_gp_augmentation.config import load_config
from wgan_gp_augmentation.evaluation import save_json, score_classifier
from wgan_gp_augmentation.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run augmented or weighted classifier condition.")
    parser.add_argument("--condition", choices=("augmented", "weighted"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.experiment.seed if args.seed is None else args.seed
    set_seed(seed)
    arrays = np.load(args.prepared / "splits.npz")
    x_train, y_train = arrays["x_train"], arrays["y_train"]
    sample_weight = None
    if args.condition == "augmented":
        if args.synthetic is None:
            raise ValueError("--synthetic is required for the augmented condition")
        synthetic = np.load(args.synthetic)
        x_train, y_train = add_synthetic(
            x_train, y_train, synthetic["x"], synthetic["y"]
        )
    else:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        sample_weight = sample_weight / sample_weight.mean()
    result = fit_classifier(
        x_train, y_train, arrays["x_validation"], arrays["y_validation"],
        config.classifier, seed, sample_weight=sample_weight,
    )
    preprocessor = joblib.load(args.prepared / "preprocessor.joblib")
    class_names = preprocessor.label_encoder.classes_.tolist()
    probabilities = result.model.predict_proba(arrays["x_test"])
    predictions = result.model.predict(arrays["x_test"])
    metrics = score_classifier(
        arrays["y_test"], predictions, probabilities, class_names,
        config.experiment.minority_labels,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, args.output / "model.joblib")
    np.savez_compressed(
        args.output / "predictions.npz", y_true=arrays["y_test"],
        y_pred=predictions, probabilities=probabilities,
    )
    save_json(args.output / "metrics.json", metrics)
    save_json(args.output / "history.json", result.history)
    save_json(args.output / "run.json", {
        "condition": args.condition,
        "seed": seed,
        "real_training_rows": len(arrays["x_train"]),
        "classifier_training_rows": len(x_train),
        "synthetic_training_rows": len(x_train) - len(arrays["x_train"]),
        "validation_rows": len(arrays["x_validation"]),
        "test_rows": len(arrays["x_test"]),
        "best_epoch": result.best_epoch,
        "elapsed_seconds": result.elapsed_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    })
    print(json.dumps({"condition": args.condition, "macro_f1": metrics["macro_f1"]}))


if __name__ == "__main__":
    main()
