"""Verify and document an intentionally stopped diffusion training run.

This helper is used only after stopping at a completed validation checkpoint.
It never modifies model weights or the append-only training log.
"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["standard", "grounded"], required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--configured-max-steps", type=int, default=20_000)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    output = args.output_dir or Path(
        f"models/diffusion/{args.condition}/label_010_v1"
    )
    log_path = output / "training_log.jsonl"
    checkpoint_path = output / "checkpoint_latest.pt"
    best_model_dir = output / "best_model"
    best_weights = best_model_dir / "diffusion_pytorch_model.safetensors"
    best_config = best_model_dir / "config.json"
    required = [log_path, checkpoint_path, best_weights, best_config]
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run artifacts: {missing}")

    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training_rows = [row for row in rows if "train_total" in row]
    validations = [row for row in rows if "validation_total" in row]
    if not training_rows or not validations:
        raise RuntimeError("Training and validation records are both required")

    # The checkpoint is loaded to verify it is readable and corresponds exactly
    # to the final complete validation event. Model tensors are not rewritten.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_step = int(state["step"])
    final_validation = validations[-1]
    if checkpoint_step != int(final_validation["step"]):
        raise RuntimeError(
            f"Checkpoint step {checkpoint_step} does not match final validation "
            f"step {final_validation['step']}"
        )
    if state.get("condition") != args.condition:
        raise RuntimeError(
            f"Checkpoint condition {state.get('condition')!r} != {args.condition!r}"
        )

    config = json.loads(best_config.read_text(encoding="utf-8"))
    if int(config.get("num_class_embeds", -1)) != 4:
        raise RuntimeError("Best model does not contain the expected four class embeddings")

    best_validation = min(validations, key=lambda row: row["validation_total"])
    checkpoint_best = float(state["best_validation_total"])
    if abs(checkpoint_best - float(best_validation["validation_total"])) > 1e-10:
        raise RuntimeError("Checkpoint best validation value disagrees with JSONL history")

    report = {
        "status": "pass",
        "purpose": f"full 10%-label {args.condition} diffusion training",
        "condition": args.condition,
        "steps_completed": checkpoint_step,
        "configured_max_steps": args.configured_max_steps,
        "early_stopped_by_configured_patience": False,
        "intentionally_stopped_at_validation_checkpoint": True,
        "stop_reason": args.reason,
        "best_validation_total": checkpoint_best,
        "best_validation_step": int(best_validation["step"]),
        "final_validation_total": float(final_validation["validation_total"]),
        "final_stale_validations": int(final_validation["stale_validations"]),
        "validation_history": validations,
        "train_images": 26,
        "validation_images": 54,
        "effective_batch_size": 4,
        "elapsed_seconds_to_final_logged_training_step": float(
            training_rows[-1]["elapsed_seconds"]
        ),
        "peak_cuda_memory_mb": max(
            float(row["peak_cuda_memory_mb"]) for row in training_rows
        ),
        "latest_checkpoint": checkpoint_path.as_posix(),
        "latest_checkpoint_sha256": sha256(checkpoint_path),
        "best_model": best_model_dir.as_posix(),
        "best_model_sha256": sha256(best_weights),
        "best_model_class_embeddings": int(config["num_class_embeds"]),
        "checkpoint_reload_and_step_match": "pass",
    }
    report_path = output / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
