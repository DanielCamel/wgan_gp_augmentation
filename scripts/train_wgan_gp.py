from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wgan_gp_augmentation.augmentation import sample_diagnostics
from wgan_gp_augmentation.config import load_config
from wgan_gp_augmentation.evaluation import save_json
from wgan_gp_augmentation.training.wgan_gp import fit_wgan, sample_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one WGAN-GP per selected minority class.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.experiment.seed if args.seed is None else args.seed
    arrays = np.load(args.prepared / "splits.npz")
    preprocessor = joblib.load(args.prepared / "preprocessor.joblib")
    class_names = preprocessor.label_encoder.classes_.tolist()
    generated_x: list[np.ndarray] = []
    generated_y: list[np.ndarray] = []
    manifest: dict[str, object] = {
        "model_type": "class-specific WGAN-GP",
        "seed": seed,
        "device_available": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "classes": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for class_offset, label in enumerate(config.experiment.minority_labels):
        class_id = class_names.index(label)
        real = arrays["x_train"][arrays["y_train"] == class_id]
        count = max(0, config.wgan.target_train_count - len(real))
        class_seed = seed + 1000 * (class_offset + 1)
        result = fit_wgan(real, config.wgan, class_seed)
        synthetic = sample_generator(
            result.generator, count, config.wgan.latent_dim,
            result.feature_mean, result.feature_scale, class_seed + 1,
        )
        checkpoint = {
            "generator_state_dict": result.generator.state_dict(),
            "critic_state_dict": result.critic.state_dict(),
            "feature_mean": result.feature_mean,
            "feature_scale": result.feature_scale,
            "input_dim": real.shape[1],
            "latent_dim": config.wgan.latent_dim,
            "hidden_layers": config.wgan.hidden_layers,
            "class_name": label,
            "seed": class_seed,
        }
        safe_name = label.lower().replace(" ", "_").replace("-", "_")
        torch.save(checkpoint, args.output / f"{safe_name}.pt")
        save_json(args.output / f"{safe_name}_history.json", result.history)
        generated_x.append(synthetic)
        generated_y.append(np.full(len(synthetic), class_id, dtype=np.int64))
        manifest["classes"][label] = {
            "class_id": class_id,
            "real_training_count": len(real),
            "generated_count": len(synthetic),
            "target_training_count": config.wgan.target_train_count,
            "seed": class_seed,
            "elapsed_seconds": result.elapsed_seconds,
            "device": result.device,
            "diagnostics": sample_diagnostics(real, synthetic),
        }
    synthetic_x = np.concatenate(generated_x).astype(np.float32)
    synthetic_y = np.concatenate(generated_y)
    np.savez_compressed(args.output / "synthetic_training_only.npz", x=synthetic_x, y=synthetic_y)
    manifest["total_generated"] = len(synthetic_x)
    save_json(args.output / "manifest.json", manifest)
    print(json.dumps({"generated": len(synthetic_x), "classes": len(generated_x)}))


if __name__ == "__main__":
    main()
