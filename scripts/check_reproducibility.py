from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check consistency of completed experiment artifacts.")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    metadata = read_json(args.prepared / "metadata.json")
    arrays = np.load(args.prepared / "splits.npz")
    preprocessor = joblib.load(args.prepared / "preprocessor.joblib")
    assert len(arrays["x_train"]) == metadata["split_rows"]["train"]
    assert len(arrays["x_validation"]) == metadata["split_rows"]["validation"]
    assert len(arrays["x_test"]) == metadata["split_rows"]["test"]
    assert preprocessor.fit_row_count_ == len(arrays["x_train"])
    assert all(np.isfinite(arrays[name]).all() for name in ("x_train", "x_validation", "x_test"))
    expected_test = arrays["y_test"]
    minority_ids = {
        list(metadata["classes"]).index(label)
        for label in ("Bot", "Web Attack - XSS", "DoS Slowhttptest")
    }
    checked_runs = 0
    for seed in args.seeds:
        synthetic_path = args.results_root / "wgan" / f"seed_{seed}" / "synthetic_training_only.npz"
        synthetic = np.load(synthetic_path)
        assert np.isfinite(synthetic["x"]).all()
        assert set(np.unique(synthetic["y"])) <= minority_ids
        manifest = read_json(args.results_root / "wgan" / f"seed_{seed}" / "manifest.json")
        assert len(synthetic["x"]) == manifest["total_generated"]
        for condition in ("baseline", "augmented", "weighted"):
            directory = args.results_root / condition / f"seed_{seed}"
            metrics = read_json(directory / "metrics.json")
            predictions = np.load(directory / "predictions.npz")
            assert np.array_equal(predictions["y_true"], expected_test)
            assert np.isfinite(predictions["probabilities"]).all()
            assert len(metrics["confusion_matrix"]) == len(metadata["classes"])
            if condition != "baseline":
                run = read_json(directory / "run.json")
                assert run["validation_rows"] == len(arrays["x_validation"])
                assert run["test_rows"] == len(arrays["x_test"])
                expected_synthetic = len(synthetic["x"]) if condition == "augmented" else 0
                assert run["synthetic_training_rows"] == expected_synthetic
            checked_runs += 1
    aggregate = read_json(args.results_root / "aggregate.json")
    assert aggregate["seeds"] == args.seeds
    print(json.dumps({"status": "ok", "classifier_runs": checked_runs, "wgan_runs": len(args.seeds)}))


if __name__ == "__main__":
    main()
