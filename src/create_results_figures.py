"""Create dissertation-ready figures directly from verified JSON/CSV evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".config" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


PRIMARY = ("real_only", "traditional", "gan", "standard_diffusion", "grounded_diffusion")
DISPLAY = {
    "real_only": "Real only", "traditional": "Traditional", "gan": "GAN",
    "standard_diffusion": "Standard\ndiffusion", "grounded_diffusion": "Grounded\ndiffusion",
    "full": "Full", "no_acoustic": "No acoustic", "no_geometric": "No geometric",
    "neither": "Neither",
}
COLOURS = ["#5B6770", "#D18F00", "#4C78A8", "#2A9D8F", "#B24C63"]


def save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def label_bars(axis, bars, digits=3):
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{bar.get_height():.{digits}f}", ha="center", va="bottom", fontsize=8,
        )


def primary_detector_figures(output: Path) -> None:
    root = Path("results/detectors/label_010_v1")
    reports = {name: json.loads((root / name / "final_report.json").read_text()) for name in PRIMARY}
    x = np.arange(len(PRIMARY)); width = 0.36
    map50 = [reports[name]["test"]["mAP50"] for name in PRIMARY]
    map95 = [reports[name]["test"]["mAP50_95"] for name in PRIMARY]
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    bars1 = axis.bar(x - width / 2, map50, width, label="mAP50", color="#4C78A8")
    bars2 = axis.bar(x + width / 2, map95, width, label="mAP50–95", color="#E07A5F")
    axis.set_xticks(x, [DISPLAY[name] for name in PRIMARY]); axis.set_ylabel("Real-test average precision")
    axis.set_ylim(0, max(map50) * 1.25); axis.grid(axis="y", alpha=.25); axis.legend(frameon=False)
    label_bars(axis, bars1); label_bars(axis, bars2)
    save(fig, output, "primary_detector_map")

    classes = ("aircraft", "human", "ship")
    matrix = np.array([[reports[name]["test"]["per_class"][class_name]["mAP50_95"] for class_name in classes] for name in PRIMARY])
    fig, axis = plt.subplots(figsize=(7.8, 5.0))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=max(.36, float(matrix.max())))
    axis.set_xticks(range(3), [name.title() for name in classes])
    axis.set_yticks(range(len(PRIMARY)), [DISPLAY[name].replace("\n", " ") for name in PRIMARY])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white" if matrix[row, column] < .16 else "black", fontsize=9)
    fig.colorbar(image, ax=axis, label="Real-test mAP50–95")
    save(fig, output, "primary_detector_per_class_map95")

    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for index, name in enumerate(PRIMARY):
        with (root / name / "train" / "results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(float(row["epoch"])) + 1 for row in rows]
        values = [float(row["metrics/mAP50-95(B)"]) for row in rows]
        axis.plot(epochs, values, label=DISPLAY[name].replace("\n", " "), color=COLOURS[index], linewidth=1.7)
    axis.set_xlabel("Training epoch"); axis.set_ylabel("Real-validation mAP50–95")
    axis.set_xlim(left=1); axis.set_ylim(bottom=0); axis.grid(alpha=.22); axis.legend(frameon=False, ncol=2)
    save(fig, output, "primary_detector_validation_curves")


def quality_figure(output: Path) -> None:
    with Path("results/quality/label_010_v1/method_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    methods = PRIMARY[1:]; labels = [DISPLAY[name].replace("\n", " ") for name in methods]
    metrics = (
        ("fid_clean_inception", "FID ↓"), ("lpips_mean", "LPIPS ↓"),
        ("aligned_ssim_mean", "Aligned SSIM ↑"), ("aligned_psnr_db_mean", "Aligned PSNR (dB) ↑"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for axis, (field, title) in zip(axes.flat, metrics):
        values = [float(rows[name][field]) for name in methods]
        bars = axis.bar(np.arange(4), values, color=COLOURS[1:])
        axis.set_xticks(np.arange(4), labels, rotation=15, ha="right")
        axis.set_title(title); axis.grid(axis="y", alpha=.22)
        label_bars(axis, bars, digits=2 if field in {"fid_clean_inception", "aligned_psnr_db_mean"} else 3)
        axis.set_ylim(0, max(values) * 1.22)
    save(fig, output, "synthetic_quality_summary")


def ablation_figure(output: Path) -> bool:
    root = Path("results/ablations/label_010_v1/detectors")
    report_path = root / "comparison_report.json"
    if not report_path.is_file():
        return False
    conditions = ("full", "no_acoustic", "no_geometric", "neither")
    reports = {name: json.loads((root / name / "final_report.json").read_text()) for name in conditions}
    x = np.arange(4); width = .36
    map50 = [reports[name]["test"]["mAP50"] for name in conditions]
    map95 = [reports[name]["test"]["mAP50_95"] for name in conditions]
    fig, axis = plt.subplots(figsize=(8.3, 4.8))
    first = axis.bar(x - width / 2, map50, width, label="mAP50", color="#4C78A8")
    second = axis.bar(x + width / 2, map95, width, label="mAP50–95", color="#E07A5F")
    axis.set_xticks(x, [DISPLAY[name] for name in conditions]); axis.set_ylabel("Real-test average precision")
    axis.set_ylim(0, max(map50) * 1.25); axis.grid(axis="y", alpha=.25); axis.legend(frameon=False)
    label_bars(axis, first); label_bars(axis, second)
    save(fig, output, "grounded_ablation_detector_map")

    classes = ("aircraft", "human", "ship")
    matrix = np.array(
        [
            [reports[name]["test"]["per_class"][class_name]["mAP50_95"] for class_name in classes]
            for name in conditions
        ]
    )
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=max(.36, float(matrix.max())))
    axis.set_xticks(range(3), [name.title() for name in classes])
    axis.set_yticks(range(4), [DISPLAY[name] for name in conditions])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, f"{matrix[row, column]:.3f}", ha="center", va="center",
                color="white" if matrix[row, column] < .16 else "black", fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Real-test mAP50–95")
    save(fig, output, "grounded_ablation_per_class_map95")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/figures/label_010_v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    primary_detector_figures(args.output); quality_figure(args.output)
    ablation_created = ablation_figure(args.output)
    report = {
        "status": "pass", "output": args.output.as_posix(),
        "primary_detector_figures": 3, "synthetic_quality_figures": 1,
        "ablation_figures_created": 2 if ablation_created else 0,
        "source_policy": "all plotted values read from verified result JSON/CSV files",
    }
    (args.output / "figure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
