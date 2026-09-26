"""Train and evaluate the five controlled full YOLO11n SCTD conditions.

Checkpoint selection uses the fixed real validation set. Each selected best
checkpoint is then evaluated once on the unchanged real test set. The runner is
condition-addressable so completed conditions can be retained safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".config" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".config" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "cache" / "torch"))

import numpy as np
import torch
import yaml
from ultralytics import YOLO


CONDITIONS = (
    "real_only",
    "traditional",
    "gan",
    "standard_diffusion",
    "grounded_diffusion",
)
CLASS_NAMES = {0: "aircraft", 1: "human", 2: "ship"}
RUN_PURPOSE = "full controlled YOLO11n detector condition"
COMPARISON_PURPOSE = "single-seed controlled 10% label-level YOLO11n comparison"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def metrics_dict(result: Any) -> dict[str, Any]:
    box = result.box
    class_indices = as_array(getattr(box, "ap_class_index", np.arange(len(CLASS_NAMES)))).astype(int)
    precision = as_array(getattr(box, "p", []))
    recall = as_array(getattr(box, "r", []))
    f1 = as_array(getattr(box, "f1", []))
    ap50 = as_array(getattr(box, "ap50", []))
    ap = as_array(getattr(box, "ap", []))
    position_by_class = {int(class_id): position for position, class_id in enumerate(class_indices)}

    per_class: dict[str, dict[str, float | None]] = {}
    for class_id, class_name in CLASS_NAMES.items():
        position = position_by_class.get(class_id)
        per_class[class_name] = {
            "precision": float(precision[position]) if position is not None and position < precision.size else None,
            "recall": float(recall[position]) if position is not None and position < recall.size else None,
            "f1": float(f1[position]) if position is not None and position < f1.size else None,
            "mAP50": float(ap50[position]) if position is not None and position < ap50.size else None,
            "mAP50_95": float(ap[position]) if position is not None and position < ap.size else None,
        }

    confusion = getattr(getattr(result, "confusion_matrix", None), "matrix", None)
    speed = {key: float(value) for key, value in getattr(result, "speed", {}).items()}
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": per_class,
        "confusion_matrix": as_array(confusion).tolist() if confusion is not None else None,
        "speed_ms_per_image": speed,
    }


def read_manifest_counts(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "train": sum(row["split"] == "train" for row in rows),
        "train_real": sum(row["split"] == "train" and row["origin"] == "real" for row in rows),
        "train_synthetic": sum(row["split"] == "train" and row["origin"] == "synthetic" for row in rows),
        "validation": sum(row["split"] == "val" for row in rows),
        "test": sum(row["split"] == "test" for row in rows),
    }


def read_training_summary(results_csv: Path) -> dict[str, Any]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No epochs recorded in {results_csv}")
    metric_key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[metric_key]))
    return {
        "epochs_completed": len(rows),
        "best_validation_epoch_from_csv": int(float(best["epoch"])),
        "best_epoch_validation_precision": float(best["metrics/precision(B)"]),
        "best_epoch_validation_recall": float(best["metrics/recall(B)"]),
        "best_epoch_validation_mAP50": float(best["metrics/mAP50(B)"]),
        "best_epoch_validation_mAP50_95": float(best[metric_key]),
        "final_epoch": int(float(rows[-1]["epoch"])),
    }


def load_protocol(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["detector"]
    required = {
        "model": "yolo11n.pt",
        "image_size": 640,
        "optimiser": "AdamW",
        "initial_learning_rate": 0.001,
        "batch_size": 8,
        "epochs": 100,
        "patience": 20,
        "workers": 4,
        "amp": True,
        "seed": 42,
        "deterministic": True,
        "pretrained_weights_identical_across_conditions": True,
        "detector_side_augmentation_identical_across_conditions": True,
        "validation_and_test_real_only": True,
    }
    mismatches = {key: (config.get(key), expected) for key, expected in required.items() if config.get(key) != expected}
    if mismatches:
        raise RuntimeError(f"Detector protocol differs from the frozen full protocol: {mismatches}")
    return config


def evaluate(
    checkpoint: Path,
    dataset_yaml: Path,
    split: str,
    protocol: dict[str, Any],
    condition_root: Path,
) -> tuple[dict[str, Any], Path]:
    evaluator = YOLO(str(checkpoint))
    result = evaluator.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=protocol["image_size"],
        batch=protocol["batch_size"],
        device=0,
        workers=protocol["workers"],
        project=str(condition_root.resolve()),
        name=f"evaluate_{split}",
        exist_ok=False,
        plots=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
    )
    save_dir_value = getattr(result, "save_dir", None) or getattr(evaluator.validator, "save_dir", None)
    if save_dir_value is None:
        raise RuntimeError(f"Ultralytics did not expose the {split} evaluation directory")
    save_dir = Path(save_dir_value)
    return metrics_dict(result), save_dir


def run_condition(
    condition: str,
    dataset_root: Path,
    output_root: Path,
    initial_weights: Path,
    protocol: dict[str, Any],
    skip_completed: bool,
) -> dict[str, Any]:
    condition_root = output_root / condition
    final_report_path = condition_root / "final_report.json"
    if final_report_path.is_file():
        report = json.loads(final_report_path.read_text(encoding="utf-8"))
        if skip_completed and report.get("status") == "pass":
            print(f"Skipping completed condition: {condition}")
            return report
        raise FileExistsError(f"Completed report already exists: {final_report_path}")
    if condition_root.exists():
        raise FileExistsError(
            f"Refusing incomplete existing condition directory: {condition_root}. "
            "Inspect it before moving it aside or implementing checkpoint resume."
        )

    dataset_dir = dataset_root / condition
    dataset_yaml = dataset_dir / "dataset.yaml"
    build_report = json.loads((dataset_dir / "build_report.json").read_text(encoding="utf-8"))
    if build_report.get("status") != "pass":
        raise RuntimeError(f"Dataset build report did not pass for {condition}")
    counts = read_manifest_counts(dataset_dir / "manifest.csv")
    expected_train = 26 if condition == "real_only" else 52
    if counts != {
        "train": expected_train,
        "train_real": 26,
        "train_synthetic": expected_train - 26,
        "validation": 54,
        "test": 54,
    }:
        raise RuntimeError(f"Unexpected dataset counts for {condition}: {counts}")

    condition_root.mkdir(parents=True)
    started = time.time()
    initial_hash = sha256(initial_weights)
    model = YOLO(str(initial_weights))
    model.train(
        data=str(dataset_yaml),
        epochs=protocol["epochs"],
        imgsz=protocol["image_size"],
        batch=protocol["batch_size"],
        device=0,
        workers=protocol["workers"],
        project=str(condition_root.resolve()),
        name="train",
        exist_ok=False,
        pretrained=True,
        optimizer=protocol["optimiser"],
        lr0=protocol["initial_learning_rate"],
        patience=protocol["patience"],
        seed=protocol["seed"],
        deterministic=protocol["deterministic"],
        amp=protocol["amp"],
        cache=False,
        plots=True,
        save=True,
        save_period=-1,
        verbose=True,
    )
    train_dir = Path(model.trainer.save_dir)
    best_checkpoint = train_dir / "weights" / "best.pt"
    last_checkpoint = train_dir / "weights" / "last.pt"
    results_csv = train_dir / "results.csv"
    for required_path in (best_checkpoint, last_checkpoint, results_csv, train_dir / "args.yaml"):
        if not required_path.is_file():
            raise RuntimeError(f"Missing expected training artifact: {required_path}")
    training_summary = read_training_summary(results_csv)
    training_finished = time.time()

    validation, validation_dir = evaluate(
        best_checkpoint, dataset_yaml, "val", protocol, condition_root
    )
    testing, test_dir = evaluate(
        best_checkpoint, dataset_yaml, "test", protocol, condition_root
    )
    finished = time.time()

    report = {
        "status": "pass",
        "purpose": RUN_PURPOSE,
        "condition": condition,
        "device": torch.cuda.get_device_name(0),
        "dataset_yaml": dataset_yaml.resolve().as_posix(),
        "dataset_counts": counts,
        "protocol": protocol,
        "initial_weights": initial_weights.resolve().as_posix(),
        "initial_weights_sha256": initial_hash,
        "best_checkpoint": best_checkpoint.resolve().as_posix(),
        "best_checkpoint_sha256": sha256(best_checkpoint),
        "last_checkpoint_sha256": sha256(last_checkpoint),
        "train_directory": train_dir.resolve().as_posix(),
        "validation_directory": validation_dir.resolve().as_posix(),
        "test_directory": test_dir.resolve().as_posix(),
        "training": training_summary,
        "validation": validation,
        "test": testing,
        "training_elapsed_seconds": training_finished - started,
        "total_elapsed_seconds": finished - started,
        "selection_policy": "best checkpoint selected by the fixed real validation set; test used only for final evaluation",
    }
    final_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))
    return report


def write_comparison(output_root: Path, reports: dict[str, dict[str, Any]]) -> None:
    if set(reports) != set(CONDITIONS):
        return
    initial_hashes = {report["initial_weights_sha256"] for report in reports.values()}
    protocols = {json.dumps(report["protocol"], sort_keys=True) for report in reports.values()}
    if len(initial_hashes) != 1 or len(protocols) != 1:
        raise RuntimeError("Initial weights or detector protocols differ across conditions")
    ranking = sorted(
        (
            {
                "rank": 0,
                "condition": condition,
                "test_mAP50_95": report["test"]["mAP50_95"],
                "test_mAP50": report["test"]["mAP50"],
                "test_precision": report["test"]["precision"],
                "test_recall": report["test"]["recall"],
            }
            for condition, report in reports.items()
        ),
        key=lambda row: row["test_mAP50_95"],
        reverse=True,
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    comparison = {
        "status": "pass",
        "purpose": COMPARISON_PURPOSE,
        "primary_final_metric": "real-test mAP50_95",
        "best_condition_by_test_mAP50_95": ranking[0]["condition"],
        "ranking": ranking,
        "same_initial_weights": True,
        "same_protocol": True,
        "real_validation_and_test_only": True,
        "selection_and_interpretation": (
            "Checkpoints were selected using validation only. Test metrics rank final outcomes and were not used "
            "for hyperparameter tuning. This comparison uses one deterministic seed and should not be presented "
            "as a multi-seed uncertainty estimate."
        ),
        "condition_reports": {
            condition: (output_root / condition / "final_report.json").resolve().as_posix()
            for condition in CONDITIONS
        },
        "aggregate_summary_csv": (output_root / "comparison_summary.csv").resolve().as_posix(),
        "per_class_summary_csv": (output_root / "per_class_summary.csv").resolve().as_posix(),
    }
    if set(CONDITIONS) == {"full", "no_acoustic", "no_geometric", "neither"}:
        effects = {}
        for metric in ("mAP50", "mAP50_95"):
            values = {condition: reports[condition]["test"][metric] for condition in CONDITIONS}
            effects[metric] = {
                "acoustic_with_geometry": values["full"] - values["no_acoustic"],
                "geometry_with_acoustic": values["full"] - values["no_geometric"],
                "acoustic_without_geometry": values["no_geometric"] - values["neither"],
                "geometry_without_acoustic": values["no_acoustic"] - values["neither"],
                "factorial_interaction": (
                    values["full"] - values["no_acoustic"]
                    - values["no_geometric"] + values["neither"]
                ),
            }
        comparison["factorial_effects"] = effects
        comparison["factorial_effects_interpretation"] = (
            "Signed single-seed test-score contrasts. Positive values favour the named component; "
            "they are descriptive effects, not confidence intervals or significance tests."
        )
    (output_root / "comparison_report.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    with (output_root / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    per_class_rows = []
    for condition in CONDITIONS:
        for class_name in CLASS_NAMES.values():
            metrics = reports[condition]["test"]["per_class"][class_name]
            per_class_rows.append(
                {
                    "condition": condition,
                    "class_name": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "mAP50": metrics["mAP50"],
                    "mAP50_95": metrics["mAP50_95"],
                }
            )
    with (output_root / "per_class_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    print(json.dumps(comparison, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--condition", choices=CONDITIONS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/processed/yolo_experiments/label_010_v1")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/detectors/label_010_v1")
    )
    parser.add_argument(
        "--initial-weights", type=Path, default=Path("models/cache/ultralytics/yolo11n.pt")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Full YOLO11n comparison requires CUDA")
    if not args.initial_weights.is_file():
        raise FileNotFoundError(args.initial_weights)
    protocol = load_protocol(args.config)
    reconciliation = json.loads(
        (args.dataset_root / "reconciliation_report.json").read_text(encoding="utf-8")
    )
    if reconciliation.get("status") != "pass" or not reconciliation.get(
        "cross_condition_evaluation_hashes_identical"
    ):
        raise RuntimeError("Dataset reconciliation did not pass")

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_conditions = CONDITIONS if args.all else (args.condition,)
    reports: dict[str, dict[str, Any]] = {}
    for condition in selected_conditions:
        reports[condition] = run_condition(
            condition,
            args.dataset_root,
            args.output_root,
            args.initial_weights,
            protocol,
            args.skip_completed,
        )
    for condition in CONDITIONS:
        report_path = args.output_root / condition / "final_report.json"
        if report_path.is_file() and condition not in reports:
            reports[condition] = json.loads(report_path.read_text(encoding="utf-8"))
    write_comparison(args.output_root, reports)


if __name__ == "__main__":
    main()
