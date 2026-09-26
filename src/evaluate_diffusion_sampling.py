"""Evaluate the pre-registered DDIM/CFG grid on a fixed real validation subset.

This stage produces evidence only. It deliberately does not choose a setting;
the paired metric table and previews must be reviewed before selection is frozen.
The resulting choice is shared by standard and grounded diffusion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from diffusion_baseline import CLASSES, seed_everything
from synthesize_diffusion import sample, sha256
from validate_synthetic_annotations import boxes, target_metrics


GUIDANCE_SCALES = [1.0, 2.0, 3.0, 5.0]
STRENGTHS = [0.30, 0.45, 0.60]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen standard-diffusion sampling grid."
    )
    parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing an unrecorded CPU fallback")
    output = Path("results/full/label_010_v1/diffusion_sampling_selection")
    if output.exists():
        raise FileExistsError(output)
    candidates_dir = output / "candidates"
    candidates_dir.mkdir(parents=True)

    validation_ids = {
        line.strip()
        for line in Path("data/splits/group_aware_v1/val.txt").read_text().splitlines()
        if line.strip()
    }
    with Path("data/processed/generative_256_v1/manifest.csv").open(
        encoding="utf-8"
    ) as handle:
        validation_rows = [
            row for row in csv.DictReader(handle) if row["image_id"] in validation_ids
        ]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(validation_rows, key=lambda item: item["image_id"]):
        by_class[row["class_name"]].append(row)
    selected_rows = [row for name in CLASSES for row in by_class[name][:4]]
    if len(selected_rows) != 12 or any(len(by_class[name]) < 4 for name in CLASSES):
        raise RuntimeError("Expected four deterministic validation samples per class")
    selected_ids = [row["image_id"] for row in selected_rows]
    (output / "fixed_validation_ids.txt").write_text(
        "\n".join(selected_ids) + "\n", encoding="utf-8"
    )

    model_dir = Path("models/diffusion/standard/label_010_v1/best_model")
    model = UNet2DModel.from_pretrained(model_dir, local_files_only=True)
    model.eval().to("cuda", dtype=torch.float32)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    seed_everything(42)
    sample_rows: list[dict] = []
    summary_rows: list[dict] = []
    for strength in STRENGTHS:
        for guidance_scale in GUIDANCE_SCALES:
            key = f"strength_{strength:.2f}_cfg_{guidance_scale:.1f}"
            config_dir = candidates_dir / key
            config_dir.mkdir()
            preview: list[torch.Tensor] = []
            config_records: list[dict] = []
            for sample_index, row in enumerate(selected_rows):
                with Image.open(row["processed_image"]) as image:
                    source_pil = image.convert("RGB")
                    source = transform(source_pil).unsqueeze(0).to("cuda")
                    source_rgb = np.asarray(source_pil, dtype=np.int16)
                class_ids = torch.tensor(
                    [CLASSES.index(row["class_name"])], device="cuda"
                )
                # Common random numbers make every grid comparison paired.
                sample_seed = 42000 + sample_index
                generator = torch.Generator(device="cuda").manual_seed(sample_seed)
                generated = sample(
                    model, source, class_ids, scheduler, 50,
                    guidance_scale, strength, generator,
                )
                destination = config_dir / f"{row['image_id']}.png"
                save_image(generated.add(1).div(2), destination)
                with Image.open(destination) as image:
                    candidate_rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
                source_gray = cv2.cvtColor(
                    source_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY
                ).astype(np.float32) / 255.0
                candidate_gray = cv2.cvtColor(
                    candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY
                ).astype(np.float32) / 255.0
                (dx, dy), phase_response = cv2.phaseCorrelate(
                    source_gray, candidate_gray
                )
                phase_shift = float(np.hypot(dx, dy))
                target_ncc, min_edge, max_edge, polarity_ok = target_metrics(
                    source_gray, candidate_gray, boxes(Path(row["processed_label"]))
                )
                gates = {
                    "phase_response": float(phase_response) >= 0.10,
                    "phase_shift": phase_shift <= 8.0,
                    "target_ncc": target_ncc >= 0.25,
                    "target_edge_ratio": min_edge >= 0.35 and max_edge <= 2.85,
                    "target_contrast_polarity": polarity_ok,
                }
                record = {
                    "configuration": key,
                    "image_id": row["image_id"],
                    "class_name": row["class_name"],
                    "sample_seed": sample_seed,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "candidate_image": destination.as_posix(),
                    "approved": all(gates.values()),
                    "phase_response": float(phase_response),
                    "phase_shift_pixels": phase_shift,
                    "minimum_target_ncc": target_ncc,
                    "minimum_target_edge_ratio": min_edge,
                    "maximum_target_edge_ratio": max_edge,
                    "target_contrast_polarity_ok": polarity_ok,
                    "mean_absolute_source_difference_255": float(
                        np.abs(source_rgb - candidate_rgb).mean()
                    ),
                    "exact_pixel_copy": bool(np.array_equal(source_rgb, candidate_rgb)),
                    "failed_gates": "|".join(name for name, passed in gates.items() if not passed),
                }
                config_records.append(record)
                sample_rows.append(record)
                if sample_index in {0, 4, 8}:
                    preview.extend([source.detach().cpu()[0], generated.detach().cpu()[0]])
            save_image(
                torch.stack(preview).add(1).div(2),
                output / f"preview_{key}.png",
                nrow=2,
            )
            differences = [row["mean_absolute_source_difference_255"] for row in config_records]
            summary_rows.append(
                {
                    "configuration": key,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "samples": len(config_records),
                    "approved": sum(row["approved"] for row in config_records),
                    "approval_rate": float(np.mean([row["approved"] for row in config_records])),
                    "minimum_phase_response": min(row["phase_response"] for row in config_records),
                    "maximum_phase_shift_pixels": max(row["phase_shift_pixels"] for row in config_records),
                    "mean_minimum_target_ncc": float(np.mean([row["minimum_target_ncc"] for row in config_records])),
                    "minimum_target_ncc": min(row["minimum_target_ncc"] for row in config_records),
                    "minimum_target_edge_ratio": min(row["minimum_target_edge_ratio"] for row in config_records),
                    "maximum_target_edge_ratio": max(row["maximum_target_edge_ratio"] for row in config_records),
                    "polarity_pass_rate": float(np.mean([row["target_contrast_polarity_ok"] for row in config_records])),
                    "mean_absolute_source_difference_255": float(np.mean(differences)),
                    "minimum_absolute_source_difference_255": min(differences),
                    "maximum_absolute_source_difference_255": max(differences),
                    "exact_pixel_copies": sum(row["exact_pixel_copy"] for row in config_records),
                }
            )

    with (output / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    with (output / "configuration_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    report = {
        "status": "pass",
        "purpose": "paired validation evidence for a shared diffusion sampling choice",
        "model": model_dir.as_posix(),
        "model_sha256": sha256(model_dir / "diffusion_pytorch_model.safetensors"),
        "training_report_sha256": sha256(
            Path("models/diffusion/standard/label_010_v1/training_report.json")
        ),
        "validation_source": "real group-aware validation partition only",
        "fixed_validation_samples": 12,
        "class_allocation": {name: 4 for name in CLASSES},
        "fixed_validation_ids": selected_ids,
        "common_random_numbers": True,
        "inference_steps": 50,
        "guidance_scale_candidates": GUIDANCE_SCALES,
        "strength_candidates": STRENGTHS,
        "configurations_evaluated": len(summary_rows),
        "automatic_selection_performed": False,
        "selection_requirements": [
            "all validation candidates pass the unchanged annotation-transfer gates",
            "zero exact pixel copies",
            "review paired class previews and numeric preservation/variation trade-off",
            "freeze one setting identically for standard and grounded diffusion",
        ],
    }
    (output / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
