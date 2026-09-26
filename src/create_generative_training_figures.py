"""Create publication-ready GAN and diffusion training figures from verified logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".config" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DEFAULT = PROJECT_ROOT / "results" / "figures" / "label_010_v1" / "generative_training"
GAN_ROOT = PROJECT_ROOT / "models" / "gan" / "label_010_v1"
DIFFUSION_ROOT = PROJECT_ROOT / "models" / "diffusion"
COLOURS = {
    "total": "#264653",
    "epsilon": "#4C78A8",
    "geometry": "#E07A5F",
    "range": "#2A9D8F",
    "speckle": "#B24C63",
    "validation": "#D18F00",
    "best": "#C1121F",
    "secondary": "#6C757D",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def rolling_mean(values: list[float], window: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.empty_like(array)
    for index in range(len(array)):
        start = max(0, index - window + 1)
        result[index] = array[start : index + 1].mean()
    return result


def style(axis: plt.Axes, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def mark_selected(axis: plt.Axes, step: int, value: float, unit: str) -> None:
    axis.axvline(step, color=COLOURS["best"], linestyle="--", linewidth=1.2, alpha=0.85)
    axis.scatter([step], [value], color=COLOURS["best"], marker="*", s=90, zorder=5)
    axis.annotate(
        f"Selected {unit} {step}\n{value:.5f}",
        xy=(step, value), xytext=(7, 9), textcoords="offset points",
        fontsize=8, color=COLOURS["best"],
    )


def save(fig: plt.Figure, output: Path, name: str, bottom: float = 0.04) -> dict[str, str]:
    fig.tight_layout(rect=(0, bottom, 1, 1))
    png = output / f"{name}.png"
    pdf = output / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png.relative_to(PROJECT_ROOT).as_posix(), "pdf": pdf.relative_to(PROJECT_ROOT).as_posix()}


def gan_figure(output: Path) -> tuple[dict[str, str], dict[str, Any]]:
    log_path = GAN_ROOT / "training_log.jsonl"
    report_path = GAN_ROOT / "training_report.json"
    rows = read_jsonl(log_path)
    report = read_json(report_path)
    epochs = [int(row["epoch"]) for row in rows]
    fields = ("train_g", "train_d", "train_adv", "train_l1", "train_fm", "validation_l1")
    if any(field not in row for row in rows for field in fields):
        raise ValueError("GAN log is missing a required training field")
    residuals = [abs(row["train_g"] - (row["train_adv"] + 10.0 * row["train_l1"] + row["train_fm"])) for row in rows]
    max_residual = max(residuals)
    if max_residual > 1e-6:
        raise ValueError(f"GAN objective reconstruction residual is {max_residual}")
    best_row = min(rows, key=lambda row: float(row["validation_l1"]))
    if abs(float(best_row["validation_l1"]) - float(report["best_validation_l1"])) > 1e-12:
        raise ValueError("GAN report best validation L1 does not match the log")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    axes[0].plot(epochs, [row["train_g"] for row in rows], label="Generator total", color=COLOURS["total"], linewidth=1.6)
    axes[0].plot(epochs, [row["train_d"] for row in rows], label="Discriminator hinge", color=COLOURS["geometry"], linewidth=1.6)
    axes[0].set_title("Adversarial objectives")
    style(axes[0], "Epoch", "Mean training loss")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(epochs, [row["train_adv"] for row in rows], label="Adversarial", color=COLOURS["geometry"], linewidth=1.5)
    axes[1].plot(epochs, [10.0 * row["train_l1"] for row in rows], label=r"$10L_{1}$", color=COLOURS["epsilon"], linewidth=1.5)
    axes[1].plot(epochs, [row["train_fm"] for row in rows], label="Feature matching", color=COLOURS["range"], linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].set_title("Generator objective components")
    style(axes[1], "Epoch", "Weighted contribution")
    axes[1].legend(frameon=False, fontsize=8)

    train_l1 = [float(row["train_l1"]) for row in rows]
    validation_l1 = [float(row["validation_l1"]) for row in rows]
    running_best = np.minimum.accumulate(validation_l1)
    axes[2].plot(epochs, train_l1, label="Training L1", color=COLOURS["epsilon"], linewidth=1.4, alpha=0.8)
    axes[2].plot(epochs, validation_l1, label="Validation L1", color=COLOURS["validation"], linewidth=1.5)
    axes[2].plot(epochs, running_best, label="Running best", color=COLOURS["secondary"], linestyle=":", linewidth=1.4)
    mark_selected(axes[2], int(best_row["epoch"]), float(best_row["validation_l1"]), "epoch")
    axes[2].set_title("Reconstruction checkpoint selection")
    style(axes[2], "Epoch", "Mean absolute error")
    axes[2].legend(frameon=False, fontsize=8)

    fig.text(
        0.5, 0.01,
        f"Validation-selected epoch {best_row['epoch']}; training stopped at epoch {report['epochs_completed']} after the fixed patience criterion.",
        ha="center", fontsize=8, color="#444444",
    )
    files = save(fig, output, "gan_training_curves", bottom=0.08)
    evidence = {
        "epochs": len(rows), "selected_epoch": int(best_row["epoch"]),
        "selected_validation_l1": float(best_row["validation_l1"]),
        "max_generator_objective_reconstruction_residual": max_residual,
        "log": log_path.relative_to(PROJECT_ROOT).as_posix(), "log_sha256": sha256(log_path),
        "report": report_path.relative_to(PROJECT_ROOT).as_posix(), "report_sha256": sha256(report_path),
    }
    return files, evidence


def split_diffusion_rows(log_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(log_path)
    training = [row for row in rows if "train_total" in row]
    validation = [row for row in rows if "validation_total" in row]
    if not training or not validation:
        raise ValueError(f"Training or validation records missing from {log_path}")
    return training, validation


def objective_residual(rows: list[dict[str, Any]], prefix: str, weights: dict[str, float]) -> float:
    residuals = []
    for row in rows:
        expected = float(row[f"{prefix}_epsilon"])
        expected += weights["geometry"] * float(row[f"{prefix}_geometry"])
        expected += weights["range"] * float(row[f"{prefix}_range"])
        expected += weights["speckle"] * float(row[f"{prefix}_speckle"])
        residuals.append(abs(float(row[f"{prefix}_total"]) - expected))
    return max(residuals)


def diffusion_sources(condition: str) -> tuple[Path, Path]:
    if condition in {"standard", "grounded"}:
        root = DIFFUSION_ROOT / condition / "label_010_v1"
        return root / "training_log.jsonl", root / "training_report.json"
    root = DIFFUSION_ROOT / "ablations" / condition / "label_010_v1"
    return root / "training_log.jsonl", root / "verified_training_report.json"


def standard_figure(output: Path) -> tuple[dict[str, str], dict[str, Any]]:
    log_path, report_path = diffusion_sources("standard")
    training, validation = split_diffusion_rows(log_path)
    report = read_json(report_path)
    weights = {"geometry": 0.0, "range": 0.0, "speckle": 0.0}
    max_residual = max(objective_residual(training, "train", weights), objective_residual(validation, "validation", weights))
    if max_residual > 1e-6:
        raise ValueError("Standard diffusion total does not reconstruct from its components")
    best_step = int(report["best_validation_step"])
    best_total = float(report["best_validation_total"])
    if min(validation, key=lambda row: row["validation_total"])["step"] != best_step:
        raise ValueError("Standard diffusion selected step does not match its log")

    steps = [int(row["step"]) for row in training]
    train_epsilon = [float(row["train_epsilon"]) for row in training]
    val_steps = [int(row["step"]) for row in validation]
    val_epsilon = [float(row["validation_epsilon"]) for row in validation]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    axes[0, 0].plot(steps, train_epsilon, color=COLOURS["epsilon"], alpha=0.28, linewidth=1.0, label="Logged value")
    axes[0, 0].plot(steps, rolling_mean(train_epsilon), color=COLOURS["epsilon"], linewidth=1.8, label="Five-record mean")
    axes[0, 0].set_title("Noise-prediction training objective")
    style(axes[0, 0], "Optimiser step", r"Training $L_{\epsilon}$")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(val_steps, val_epsilon, color=COLOURS["validation"], marker="o", linewidth=1.7, label="Validation epsilon MSE")
    mark_selected(axes[0, 1], best_step, best_total, "step")
    axes[0, 1].set_title("Validation checkpoint selection")
    style(axes[0, 1], "Optimiser step", r"Validation $L_{\epsilon}$")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(steps, [float(row["lr"]) * 1e6 for row in training], color=COLOURS["range"], linewidth=1.6)
    axes[1, 0].set_title("Executed learning-rate schedule")
    style(axes[1, 0], "Optimiser step", r"Learning rate ($\times10^{-6}$)")

    norms = [float(row["gradient_norm"]) for row in training]
    axes[1, 1].plot(steps, norms, color=COLOURS["secondary"], alpha=0.25, linewidth=1.0, label="Logged norm")
    axes[1, 1].plot(steps, rolling_mean(norms), color=COLOURS["secondary"], linewidth=1.8, label="Five-record mean")
    axes[1, 1].set_title("Gradient-norm diagnostic")
    style(axes[1, 1], "Optimiser step", "Gradient norm")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.text(0.5, 0.01, f"Best validation checkpoint: step {best_step}; final complete validation: step {report['steps_completed']}.", ha="center", fontsize=8, color="#444444")
    files = save(fig, output, "standard_diffusion_training_curves", bottom=0.06)
    evidence = {
        "training_records": len(training), "validation_records": len(validation),
        "selected_step": best_step, "selected_validation_total": best_total,
        "max_objective_reconstruction_residual": max_residual,
        "log": log_path.relative_to(PROJECT_ROOT).as_posix(), "log_sha256": sha256(log_path),
        "report": report_path.relative_to(PROJECT_ROOT).as_posix(), "report_sha256": sha256(report_path),
    }
    return files, evidence


def grounded_figure(output: Path) -> tuple[dict[str, str], dict[str, Any]]:
    log_path, report_path = diffusion_sources("grounded")
    training, validation = split_diffusion_rows(log_path)
    report = read_json(report_path)
    weights = {"geometry": 0.10, "range": 0.05, "speckle": 0.05}
    max_residual = max(objective_residual(training, "train", weights), objective_residual(validation, "validation", weights))
    if max_residual > 1e-6:
        raise ValueError("Grounded diffusion total does not reconstruct from its components")
    best_step = int(report["best_validation_step"])
    best_total = float(report["best_validation_total"])
    if min(validation, key=lambda row: row["validation_total"])["step"] != best_step:
        raise ValueError("Grounded diffusion selected step does not match its log")

    steps = [int(row["step"]) for row in training]
    val_steps = [int(row["step"]) for row in validation]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.1))
    for field, label, colour in (("train_total", "Grounded total", COLOURS["total"]), ("train_epsilon", "Epsilon", COLOURS["epsilon"])):
        values = [float(row[field]) for row in training]
        axes[0, 0].plot(steps, values, color=colour, alpha=0.22, linewidth=0.9)
        axes[0, 0].plot(steps, rolling_mean(values), color=colour, linewidth=1.8, label=label)
    axes[0, 0].set_title("Training objective")
    style(axes[0, 0], "Optimiser step", "Loss")
    axes[0, 0].legend(frameon=False, fontsize=8)

    val_total = [float(row["validation_total"]) for row in validation]
    val_epsilon = [float(row["validation_epsilon"]) for row in validation]
    axes[0, 1].plot(val_steps, val_total, color=COLOURS["total"], marker="o", linewidth=1.7, label="Grounded total")
    axes[0, 1].plot(val_steps, val_epsilon, color=COLOURS["epsilon"], marker="s", linewidth=1.5, label="Epsilon")
    mark_selected(axes[0, 1], best_step, best_total, "step")
    axes[0, 1].set_title("Validation checkpoint selection")
    style(axes[0, 1], "Optimiser step", "Validation loss")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for name, colour in (("geometry", COLOURS["geometry"]), ("range", COLOURS["range"]), ("speckle", COLOURS["speckle"])):
        values = [float(row[f"train_{name}"]) for row in training]
        axes[1, 0].plot(steps, rolling_mean(values), color=colour, linewidth=1.7, label=name.replace("geometry", "Geometric").replace("range", "Range-intensity").replace("speckle", "Log-speckle"))
    axes[1, 0].set_title("Unweighted auxiliary training terms")
    style(axes[1, 0], "Optimiser step", "Auxiliary loss")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for name, colour, weight in (("geometry", COLOURS["geometry"], .10), ("range", COLOURS["range"], .05), ("speckle", COLOURS["speckle"], .05)):
        values = [weight * float(row[f"validation_{name}"]) for row in validation]
        axes[1, 1].plot(val_steps, values, color=colour, marker="o", linewidth=1.6, label=f"{weight:.2f} × {name}")
    axes[1, 1].set_title("Weighted validation contributions")
    style(axes[1, 1], "Optimiser step", "Contribution to total")
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.text(0.5, 0.01, f"Best grounded validation checkpoint: step {best_step}. Auxiliary terms are shown both raw and with their executed weights.", ha="center", fontsize=8, color="#444444")
    files = save(fig, output, "grounded_diffusion_training_curves", bottom=0.06)
    evidence = {
        "training_records": len(training), "validation_records": len(validation),
        "selected_step": best_step, "selected_validation_total": best_total,
        "loss_weights": weights, "max_objective_reconstruction_residual": max_residual,
        "log": log_path.relative_to(PROJECT_ROOT).as_posix(), "log_sha256": sha256(log_path),
        "report": report_path.relative_to(PROJECT_ROOT).as_posix(), "report_sha256": sha256(report_path),
    }
    return files, evidence


def ablation_figure(output: Path) -> tuple[dict[str, str], dict[str, Any]]:
    specifications = {
        "full": {"source": "grounded", "title": "Full", "enabled": "geometry + acoustic"},
        "no_acoustic": {"source": "no_acoustic", "title": "No acoustic", "enabled": "geometry only"},
        "no_geometric": {"source": "no_geometric", "title": "No geometric", "enabled": "acoustic only"},
        "neither": {"source": "standard", "title": "Neither", "enabled": "epsilon only"},
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), sharey=True)
    evidence: dict[str, Any] = {}
    for axis, (condition, spec) in zip(axes.flat, specifications.items()):
        log_path, report_path = diffusion_sources(str(spec["source"]))
        _, log_validation = split_diffusion_rows(log_path)
        report = read_json(report_path)
        validation = report.get("validation_history", log_validation)
        best_step = int(report["best_validation_step"])
        best_total = float(report["best_validation_total"])
        steps = [int(row["step"]) for row in validation]
        totals = [float(row["validation_total"]) for row in validation]
        epsilon = [float(row["validation_epsilon"]) for row in validation]
        if min(validation, key=lambda row: row["validation_total"])["step"] != best_step:
            raise ValueError(f"{condition} selected step does not match its validation history")
        axis.plot(steps, totals, color=COLOURS["total"], marker="o", linewidth=1.7, label="Arm total")
        axis.plot(steps, epsilon, color=COLOURS["epsilon"], marker="s", linewidth=1.4, label="Epsilon")
        mark_selected(axis, best_step, best_total, "step")
        axis.set_title(f"{spec['title']} ({spec['enabled']})")
        style(axis, "Optimiser step", "Validation loss")
        axis.legend(frameon=False, fontsize=8)
        evidence[condition] = {
            "validation_records": len(validation), "selected_step": best_step,
            "selected_validation_total": best_total,
            "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
            "report_sha256": sha256(report_path),
        }
    fig.text(0.5, 0.012, "Checkpoint selection is within each arm. Total objectives differ across panels and are not a cross-arm performance ranking.", ha="center", fontsize=8, color="#444444")
    files = save(fig, output, "diffusion_ablation_validation_curves", bottom=0.065)
    return files, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    figures: dict[str, dict[str, str]] = {}
    evidence: dict[str, Any] = {}
    figures["gan"], evidence["gan"] = gan_figure(output)
    figures["standard_diffusion"], evidence["standard_diffusion"] = standard_figure(output)
    figures["grounded_diffusion"], evidence["grounded_diffusion"] = grounded_figure(output)
    figures["diffusion_ablation"], evidence["diffusion_ablation"] = ablation_figure(output)
    report = {
        "status": "pass",
        "purpose": "publication-ready generative-model training figures from verified append-only histories",
        "output": output.relative_to(PROJECT_ROOT).as_posix(),
        "figures": figures,
        "evidence": evidence,
        "interpretation_constraint": "validation totals select checkpoints only within objectives; unlike objective totals are not ranked across arms",
    }
    report_path = output / "generative_training_figure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
