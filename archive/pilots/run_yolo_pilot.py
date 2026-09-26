"""Run a one-epoch YOLO11n CUDA integration pilot and record real-only metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("YOLO pilot requires CUDA")
    weights = Path("models/cache/ultralytics/yolo11n.pt")
    weights.parent.mkdir(parents=True, exist_ok=True)
    output = Path("results/pilots/yolo11n_augmented_v1").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    started = time.time()
    model = YOLO(str(weights))
    train_result = model.train(
        data="data/processed/yolo_pilot_augmented_v1/dataset.yaml",
        epochs=1, imgsz=320, batch=4, device=0, workers=0,
        project=str(Path("results/pilots").resolve()), name="yolo11n_augmented_v1",
        exist_ok=False, pretrained=True, optimizer="AdamW", lr0=0.001,
        patience=1, seed=42, deterministic=True, amp=True,
        plots=False, save=True, verbose=True,
    )
    output = Path(model.trainer.save_dir)
    best = output / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("YOLO pilot did not save best.pt")
    evaluator = YOLO(str(best))
    validation = evaluator.val(
        data="data/processed/yolo_pilot_augmented_v1/dataset.yaml",
        split="val", imgsz=320, batch=4, device=0, workers=0,
        project=str(Path("results/pilots").resolve()), name="yolo11n_augmented_v1_val",
        plots=False, save_json=True, verbose=False,
    )
    testing = evaluator.val(
        data="data/processed/yolo_pilot_augmented_v1/dataset.yaml",
        split="test", imgsz=320, batch=4, device=0, workers=0,
        project=str(Path("results/pilots").resolve()), name="yolo11n_augmented_v1_test",
        plots=False, save_json=True, verbose=False,
    )
    def metrics(result):
        return {
            "precision": float(result.box.mp), "recall": float(result.box.mr),
            "mAP50": float(result.box.map50), "mAP50_95": float(result.box.map),
            "per_class_mAP50_95": [float(value) for value in result.box.maps],
        }
    report = {
        "status": "pass", "purpose": "one-epoch execution pilot; not an experimental result",
        "model": "yolo11n.pt", "device": torch.cuda.get_device_name(0),
        "epochs": 1, "image_size": 320, "batch_size": 4, "seed": 42,
        "train_samples": 30, "validation_samples": 54, "test_samples": 54,
        "validation": metrics(validation), "test": metrics(testing),
        "elapsed_seconds": time.time() - started,
        "best_checkpoint": best.as_posix(),
    }
    (output / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
