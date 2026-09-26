"""Generate common-random-number candidates for four grounded ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "models" / "cache" / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "cache" / "torch"))

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from diffusion_baseline import CLASSES, seed_everything
from synthesize_diffusion import sample, sha256


CONDITIONS = ("full", "no_acoustic", "no_geometric", "neither")
MODEL_DIRS = {
    "full": Path("models/diffusion/grounded/label_010_v1/best_model"),
    "no_acoustic": Path("models/diffusion/ablations/no_acoustic/label_010_v1/best_model"),
    "no_geometric": Path("models/diffusion/ablations/no_geometric/label_010_v1/best_model"),
    "neither": Path("models/diffusion/standard/label_010_v1/best_model"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--selection", type=Path,
        default=Path("results/full/label_010_v1/diffusion_sampling_selection/selection_report.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing an unrecorded CPU fallback")

    protocol = json.loads(
        Path("experiments/label_010_v1/grounded_ablation_protocol.json").read_text(encoding="utf-8")
    )
    expected_components = protocol["conditions"][args.condition]
    model_dir = MODEL_DIRS[args.condition]
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    model_weights = model_dir / "diffusion_pytorch_model.safetensors"
    if not model_weights.is_file():
        raise FileNotFoundError(model_weights)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))["selected_configuration"]
    if (
        float(selection["guidance_scale"]) != 1.0
        or float(selection["strength"]) != 0.30
        or int(selection["inference_steps"]) != 50
    ):
        raise RuntimeError(f"Sampling selection differs from the frozen ablation protocol: {selection}")

    output = args.output_dir or Path(f"synthetic/label_010_v1/ablations/{args.condition}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    train_ids = {
        line.strip()
        for line in Path("data/splits/limited_labels_group_aware_v1/train_010.txt").read_text().splitlines()
        if line.strip()
    }
    with Path("data/processed/generative_256_v1/manifest.csv").open(encoding="utf-8") as handle:
        rows = sorted(
            [row for row in csv.DictReader(handle) if row["image_id"] in train_ids],
            key=lambda row: row["image_id"],
        )
    if len(rows) != 26 or len({row["image_id"] for row in rows}) != 26:
        raise RuntimeError("Expected the 26 unique frozen real sources")

    model = UNet2DModel.from_pretrained(model_dir, local_files_only=True)
    if int(model.config.num_class_embeds) != 4:
        raise RuntimeError("Expected three SCTD classes and one classifier-free null class")
    model.eval().to("cuda", dtype=torch.float32)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    checkpoint_hash = sha256(model_weights)
    seed_everything(42)
    manifest_rows, differences, preview = [], [], []
    exact_copies = 0
    for index, row in enumerate(rows):
        source_path = Path(row["processed_image"])
        with Image.open(source_path) as image:
            source_pil = image.convert("RGB")
            source = transform(source_pil).unsqueeze(0).to("cuda")
            source_pixels = np.asarray(source_pil, dtype=np.int16)
        class_id = torch.tensor([CLASSES.index(row["class_name"])], device="cuda")
        candidate_seed = 40042 + index
        generator = torch.Generator(device="cuda").manual_seed(candidate_seed)
        generated = sample(
            model, source, class_id, scheduler, 50, 1.0, 0.30, generator
        )
        candidate_id = f"l010_ablation_{args.condition}_{row['image_id']}"
        destination = output / f"{candidate_id}.png"
        save_image(generated.add(1).div(2), destination)
        with Image.open(destination) as image:
            candidate_pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
        exact = bool(np.array_equal(source_pixels, candidate_pixels))
        difference = float(np.abs(source_pixels - candidate_pixels).mean())
        exact_copies += int(exact); differences.append(difference)
        if len(preview) < 12:
            preview.extend([source.detach().cpu()[0], generated.detach().cpu()[0]])
        manifest_rows.append({
            "candidate_id": candidate_id,
            "method": f"ablation_{args.condition}",
            "ablation_condition": args.condition,
            "geometric": expected_components["geometric"],
            "range": expected_components["range"],
            "speckle": expected_components["speckle"],
            "source_id": row["image_id"],
            "class_name": row["class_name"],
            "seed": candidate_seed,
            "source_image": source_path.as_posix(),
            "source_label": Path(row["processed_label"]).as_posix(),
            "candidate_image": destination.as_posix(),
            "candidate_sha256": sha256(destination),
            "checkpoint": model_dir.as_posix(),
            "checkpoint_sha256": checkpoint_hash,
            "guidance_scale": 1.0,
            "source_strength": 0.30,
            "inference_steps": 50,
            "mean_absolute_source_difference_255": difference,
            "exact_pixel_copy": exact,
        })

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)
    save_image(torch.stack(preview).add(1).div(2), output / "source_candidate_preview.png", nrow=2)
    report = {
        "status": "pass", "purpose": "common-random-number grounded ablation synthesis",
        "condition": args.condition, "components": expected_components,
        "generated": len(manifest_rows), "source_count": len(rows),
        "common_seed_rule": "40042 + zero-based ordered source index",
        "guidance_scale": 1.0, "source_strength": 0.30, "inference_steps": 50,
        "checkpoint": model_dir.as_posix(), "checkpoint_sha256": checkpoint_hash,
        "exact_pixel_copies": exact_copies,
        "mean_absolute_source_difference_255": float(np.mean(differences)),
        "minimum_absolute_source_difference_255": float(np.min(differences)),
        "maximum_absolute_source_difference_255": float(np.max(differences)),
    }
    (output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
