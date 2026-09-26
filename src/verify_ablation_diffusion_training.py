"""Verify completed no-acoustic/no-geometric diffusion training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["no_acoustic", "no_geometric"], required=True)
    args = parser.parse_args()
    output = Path(f"models/diffusion/ablations/{args.condition}/label_010_v1")
    protocol = json.loads(Path("experiments/label_010_v1/grounded_ablation_protocol.json").read_text())
    expected_components = protocol["conditions"][args.condition]
    paths = {
        "raw_report": output / "training_report.json",
        "log": output / "training_log.jsonl",
        "checkpoint": output / "checkpoint_latest.pt",
        "weights": output / "best_model" / "diffusion_pytorch_model.safetensors",
        "config": output / "best_model" / "config.json",
    }
    missing = [path.as_posix() for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    raw_report = json.loads(paths["raw_report"].read_text(encoding="utf-8"))
    expected_training = protocol["training"]
    expected_report_fields = {
        "condition": args.condition,
        "components": expected_components,
        "max_steps": expected_training["maximum_steps"],
        "validation_every_steps": expected_training["validation_every_steps"],
        "patience_validations": expected_training["patience_validations"],
        "gradient_accumulation": expected_training["gradient_accumulation"],
        "train_images": expected_training["train_images"],
        "validation_images": expected_training["validation_images"],
    }
    mismatches = {
        field: (raw_report.get(field), expected)
        for field, expected in expected_report_fields.items()
        if raw_report.get(field) != expected
    }
    if mismatches:
        raise RuntimeError(f"Training report differs from frozen protocol: {mismatches}")
    rows = [json.loads(line) for line in paths["log"].read_text().splitlines() if line.strip()]
    training_rows = [row for row in rows if "train_total" in row]
    validation_rows = [row for row in rows if "validation_total" in row]
    if not training_rows or not validation_rows:
        raise RuntimeError("Training and validation evidence are required")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["condition"] != args.condition or checkpoint.get("components") != expected_components:
        raise RuntimeError("Checkpoint condition/components differ from frozen protocol")
    if int(checkpoint["step"]) != int(validation_rows[-1]["step"]):
        raise RuntimeError("Latest checkpoint does not match the last validation event")
    if int(validation_rows[-1]["stale_validations"]) < 2 or not raw_report["early_stopped"]:
        raise RuntimeError("The frozen two-validation stopping rule was not reached")
    running_best = float("inf")
    expected_stale = 0
    for row in validation_rows:
        expected_improved = float(row["validation_total"]) < running_best - 1e-5
        if bool(row["improved"]) != expected_improved:
            raise RuntimeError(f"Improvement flag mismatch at step {row['step']}")
        if expected_improved:
            running_best = float(row["validation_total"])
            expected_stale = 0
        else:
            expected_stale += 1
        if int(row["stale_validations"]) != expected_stale:
            raise RuntimeError(f"Stale-validation counter mismatch at step {row['step']}")
    best = min(validation_rows, key=lambda row: row["validation_total"])
    if abs(float(checkpoint["best_validation_total"]) - float(best["validation_total"])) > 1e-10:
        raise RuntimeError("Checkpoint and validation history disagree on the selected minimum")
    component_fields = {"geometric": "train_geometry", "range": "train_range", "speckle": "train_speckle"}
    for component, field in component_fields.items():
        values = [abs(float(row[field])) for row in training_rows]
        if expected_components[component] and max(values) <= 0:
            raise RuntimeError(f"Enabled {component} loss never became positive")
        if not expected_components[component] and max(values) > 1e-12:
            raise RuntimeError(f"Disabled {component} loss was non-zero")
    config = json.loads(paths["config"].read_text())
    if int(config.get("num_class_embeds", -1)) != 4:
        raise RuntimeError("Selected model lacks the expected class embeddings")
    verified = {
        **raw_report,
        "verification_status": "pass",
        "frozen_protocol": "experiments/label_010_v1/grounded_ablation_protocol.json",
        "best_validation_step": int(best["step"]),
        "validation_improvement_minimum_delta": 1e-5,
        "validation_history": validation_rows,
        "latest_checkpoint": paths["checkpoint"].as_posix(),
        "latest_checkpoint_sha256": sha256(paths["checkpoint"]),
        "best_model_weights": paths["weights"].as_posix(),
        "best_model_sha256": sha256(paths["weights"]),
        "best_model_class_embeddings": int(config["num_class_embeds"]),
        "checkpoint_reload_and_step_match": "pass",
        "validation_patience_replay": "pass",
        "disabled_loss_zero_checks": "pass",
    }
    destination = output / "verified_training_report.json"
    destination.write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
