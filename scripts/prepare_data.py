from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wgan_gp_augmentation.config import load_config
from wgan_gp_augmentation.data.loading import load_rows
from wgan_gp_augmentation.data.preprocessing import Preprocessor, encode_splits
from wgan_gp_augmentation.data.splitting import clean_rows, split_rows
from wgan_gp_augmentation.evaluation import save_json
from wgan_gp_augmentation.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe CICIDS2017 pilot splits.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    frame, load_report = load_rows(
        args.csv,
        cfg.data.label_column,
        cfg.experiment.selected_labels,
        cfg.data.chunk_size,
        cfg.data.max_rows_per_class,
        cfg.experiment.seed,
    )
    cleaned, cleaning_report = clean_rows(frame, cfg.data.label_column)
    splits = split_rows(
        cleaned,
        cfg.data.label_column,
        cfg.data.test_fraction,
        cfg.data.validation_fraction,
        cfg.experiment.seed,
    )
    prep = Preprocessor(cfg.data.variance_threshold, cfg.data.pca_variance).fit(
        splits.train, cfg.data.label_column
    )
    arrays = encode_splits(splits, prep, cfg.data.label_column)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "splits.npz", **arrays)
    joblib.dump(prep, args.output / "preprocessor.joblib")
    metadata = {
        "config": args.config.as_posix(),
        "seed": cfg.experiment.seed,
        "load": load_report.__dict__,
        "cleaning": cleaning_report,
        "split_rows": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "split_class_counts": {
            name: getattr(splits, name)[cfg.data.label_column].value_counts().sort_index().to_dict()
            for name in ("train", "validation", "test")
        },
        "feature_count_input": len(prep.feature_names_in_ or ()),
        "feature_count_after_variance_filter": int(prep.variance_filter.get_support().sum()),
        "pca_components": int(prep.pca.n_components_),
        "pca_explained_variance": float(prep.pca.explained_variance_ratio_.sum()),
        "classes": prep.label_encoder.classes_.tolist(),
        "preprocessor_fit_rows": prep.fit_row_count_,
    }
    save_json(args.output / "metadata.json", metadata)
    print(json.dumps(metadata["split_rows"], sort_keys=True))


if __name__ == "__main__":
    main()
