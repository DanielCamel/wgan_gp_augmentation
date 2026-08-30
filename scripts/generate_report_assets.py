from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_LABELS = {"baseline": "Baseline", "augmented": "WGAN-GP augmented", "weighted": "Weighted loss"}
MINORITY = ("Bot", "Web Attack - XSS", "DoS Slowhttptest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report tables and figures from raw results.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_results_block(readme: Path, table: str) -> None:
    start_marker, end_marker = "<!-- GENERATED_RESULTS_START -->", "<!-- GENERATED_RESULTS_END -->"
    text = readme.read_text(encoding="utf-8")
    start, end = text.index(start_marker) + len(start_marker), text.index(end_marker)
    readme.write_text(text[:start] + "\n\n" + table.strip() + "\n\n" + text[end:], encoding="utf-8")


def main() -> None:
    args = parse_args()
    aggregate = load_json(args.results_root / "aggregate.json")
    args.figures.mkdir(parents=True, exist_ok=True)
    table_dir = args.results_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for condition in CONDITION_LABELS:
        values = aggregate["conditions"][condition]
        summary_rows.append({"condition": CONDITION_LABELS[condition], "macro_f1_mean": values["macro_f1"]["mean"], "macro_f1_sample_std": values["macro_f1"]["sample_std"], "macro_roc_auc_ovr_mean": values["macro_roc_auc_ovr"]["mean"], "macro_roc_auc_ovr_sample_std": values["macro_roc_auc_ovr"]["sample_std"], "runs": values["macro_f1"]["n"]})
    pd.DataFrame(summary_rows).to_csv(table_dir / "condition_summary.csv", index=False)
    per_class_rows = []
    for condition in CONDITION_LABELS:
        for label, metrics in aggregate["conditions"][condition]["per_class"].items():
            row = {"condition": CONDITION_LABELS[condition], "class": label}
            for metric, values in metrics.items():
                metric_name = metric.replace("f1-score", "f1")
                row[f"{metric_name}_mean"] = values["mean"]
                row[f"{metric_name}_sample_std"] = values["sample_std"]
            per_class_rows.append(row)
    pd.DataFrame(per_class_rows).to_csv(table_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame([{"comparison": name, **values} for name, values in aggregate["paired_macro_f1_delta"].items()]).to_csv(table_dir / "paired_deltas.csv", index=False)
    markdown = ["| Condition | Macro F1, mean ± SD | Macro ROC-AUC, mean ± SD | Runs |", "|---|---:|---:|---:|"]
    for row in summary_rows:
        markdown.append(f"| {row['condition']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_sample_std']:.4f} | {row['macro_roc_auc_ovr_mean']:.5f} ± {row['macro_roc_auc_ovr_sample_std']:.5f} | {row['runs']} |")
    replace_results_block(args.readme, "\n".join(markdown))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(summary_rows))
    axis.bar(positions, [row["macro_f1_mean"] for row in summary_rows], yerr=[row["macro_f1_sample_std"] for row in summary_rows], capsize=5, color=["#4C78A8", "#F58518", "#54A24B"])
    axis.set_xticks(positions, [row["condition"] for row in summary_rows])
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0.93, 0.985)
    axis.set_title("Classifier performance across three seeds")
    figure.tight_layout()
    figure.savefig(args.figures / "macro_f1_conditions.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    width, positions = 0.24, np.arange(len(MINORITY))
    for offset, condition in enumerate(CONDITION_LABELS):
        recalls = [aggregate["conditions"][condition]["per_class"][label]["recall"] for label in MINORITY]
        axis.bar(positions + (offset - 1) * width, [1.0 - value["mean"] for value in recalls], width, yerr=[value["sample_std"] for value in recalls], capsize=3, label=CONDITION_LABELS[condition])
    axis.set_xticks(positions, MINORITY)
    axis.set_ylabel("False-negative rate")
    axis.set_title("Minority-class false-negative rates")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.figures / "minority_fnr.png", dpi=180)
    plt.close(figure)

    matrices: dict[str, list[np.ndarray]] = {"baseline": [], "augmented": []}
    class_names = None
    for condition in matrices:
        for seed in args.seeds:
            metrics = load_json(args.results_root / condition / f"seed_{seed}" / "metrics.json")
            class_names = metrics["class_names"]
            matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrices[condition].append(np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    image = None
    for axis, condition in zip(axes, matrices):
        image = axis.imshow(np.mean(matrices[condition], axis=0), vmin=0, vmax=1, cmap="Blues")
        axis.set_title(CONDITION_LABELS[condition])
        axis.set_xticks(range(len(class_names)), class_names, rotation=65, ha="right", fontsize=7)
        axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
        axis.set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    figure.subplots_adjust(bottom=0.28, wspace=0.16, right=0.90)
    color_axis = figure.add_axes((0.92, 0.24, 0.015, 0.56))
    figure.colorbar(image, cax=color_axis, label="Row-normalized proportion")
    figure.savefig(args.figures / "confusion_matrices.png", dpi=180)
    plt.close(figure)

    diversity_rows = []
    for seed in args.seeds:
        manifest = load_json(args.results_root / "wgan" / f"seed_{seed}" / "manifest.json")
        for label in MINORITY:
            diagnostics = manifest["classes"][label]["diagnostics"]
            diversity_rows.append({"seed": seed, "class": label, "real_pairwise_mean": diagnostics["real_pairwise_mean"], "synthetic_pairwise_mean": diagnostics["synthetic_pairwise_mean"], "synthetic_real_diversity_ratio": diagnostics["synthetic_pairwise_mean"] / diagnostics["real_pairwise_mean"]})
    diversity = pd.DataFrame(diversity_rows)
    diversity.to_csv(table_dir / "synthetic_diversity.csv", index=False)
    grouped = diversity.groupby("class")["synthetic_real_diversity_ratio"].agg(["mean", "std"])
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=5, color="#E45756")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Real-data diversity")
    axis.set_ylabel("Synthetic / real mean pairwise distance")
    axis.set_title("Synthetic diversity diagnostic")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.figures / "synthetic_diversity.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"tables": 4, "figures": 4, "seeds": args.seeds}))


if __name__ == "__main__":
    main()
