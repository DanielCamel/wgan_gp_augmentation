from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wgan_gp_augmentation.evaluation import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed classifier runs.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    conditions = ("baseline", "augmented", "weighted")
    raw: dict[str, dict[int, dict[str, object]]] = {}
    for condition in conditions:
        raw[condition] = {}
        for seed in args.seeds:
            path = args.results_root / condition / f"seed_{seed}" / "metrics.json"
            raw[condition][seed] = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, object] = {"seeds": args.seeds, "conditions": {}}
    for condition in conditions:
        runs = raw[condition]
        class_names = runs[args.seeds[0]]["class_names"]
        summary["conditions"][condition] = {
            "macro_f1": mean_std([float(runs[seed]["macro_f1"]) for seed in args.seeds]),
            "macro_roc_auc_ovr": mean_std(
                [float(runs[seed]["macro_roc_auc_ovr"]) for seed in args.seeds]
            ),
            "per_class": {
                label: {
                    metric: mean_std(
                        [float(runs[seed]["per_class"][label][metric]) for seed in args.seeds]
                    )
                    for metric in ("precision", "recall", "f1-score")
                }
                for label in class_names
            },
        }
    summary["paired_macro_f1_delta"] = {
        "augmented_minus_baseline": mean_std(
            [
                float(raw["augmented"][seed]["macro_f1"])
                - float(raw["baseline"][seed]["macro_f1"])
                for seed in args.seeds
            ]
        ),
        "weighted_minus_baseline": mean_std(
            [
                float(raw["weighted"][seed]["macro_f1"])
                - float(raw["baseline"][seed]["macro_f1"])
                for seed in args.seeds
            ]
        ),
    }
    save_json(args.output, summary)
    print(json.dumps(summary["paired_macro_f1_delta"], indent=2))


if __name__ == "__main__":
    main()
