"""Generate a frozen equal-budget standard or grounded diffusion set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from diffusion_baseline import CLASSES, NULL_CLASS, seed_everything


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def sample(
    model: UNet2DModel,
    source: torch.Tensor,
    class_ids: torch.Tensor,
    training_scheduler: DDPMScheduler,
    steps: int,
    guidance_scale: float,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    scheduler = DDIMScheduler.from_config(training_scheduler.config)
    scheduler.set_timesteps(steps, device=source.device)
    start_index = min(max(round((1.0 - strength) * (steps - 1)), 0), steps - 1)
    timesteps = scheduler.timesteps[start_index:]
    noise = torch.randn(
        source.shape, generator=generator, device=source.device, dtype=source.dtype
    )
    current = scheduler.add_noise(source, noise, timesteps[0].expand(source.shape[0]))
    for timestep in timesteps:
        null_ids = torch.full_like(class_ids, NULL_CLASS)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            unconditional = model(current, timestep, class_labels=null_ids).sample
            conditional = model(current, timestep, class_labels=class_ids).sample
        prediction = unconditional.float() + guidance_scale * (
            conditional.float() - unconditional.float()
        )
        current = scheduler.step(prediction, timestep, current.float()).prev_sample
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"Non-finite DDIM sample at timestep {int(timestep)}")
    return current.clamp(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["standard", "grounded"], required=True)
    parser.add_argument("--sampling-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inference-steps", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing an unrecorded CPU fallback")

    method = "standard_diffusion" if args.condition == "standard" else "grounded_diffusion"
    # The methodology control requires standard and grounded diffusion to use
    # the same validation-selected CFG scale and source strength.
    config_path = args.sampling_config or Path(
        "results/full/label_010_v1/diffusion_sampling_selection/selection_report.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Validation-selected sampling configuration is required: {config_path}"
        )
    selection = json.loads(config_path.read_text(encoding="utf-8"))
    selected = selection.get("selected_configuration", selection)
    guidance_scale = float(selected["guidance_scale"])
    strength = float(selected["strength"])
    if guidance_scale not in {1.0, 2.0, 3.0, 5.0}:
        raise ValueError("Guidance scale is outside the pre-registered candidate set")
    if strength not in {0.30, 0.45, 0.60}:
        raise ValueError("Source strength is outside the pre-registered candidate set")

    output = args.output_dir or Path(f"synthetic/label_010_v1/{method}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model_dir = Path(f"models/diffusion/{args.condition}/label_010_v1/best_model")
    model = UNet2DModel.from_pretrained(model_dir, local_files_only=True)
    if int(model.config.num_class_embeds) != 4:
        raise RuntimeError("Expected three SCTD classes plus the classifier-free null class")
    model.eval().to("cuda", dtype=torch.float32)
    checkpoint_hash = sha256(model_dir / "diffusion_pytorch_model.safetensors")
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    with Path("experiments/label_010_v1/candidate_registry.csv").open(
        encoding="utf-8"
    ) as handle:
        registry = [row for row in csv.DictReader(handle) if row["method"] == method]
    if len(registry) != 26:
        raise RuntimeError(f"Frozen registry contains {len(registry)} {method} rows, expected 26")

    seed_everything(42)
    rows: list[dict] = []
    pixel_differences: list[float] = []
    exact_pixel_copies = 0
    preview: list[torch.Tensor] = []
    for row in registry:
        with Image.open(row["source_image"]) as image:
            source = transform(image.convert("RGB")).unsqueeze(0).to("cuda")
        class_ids = torch.tensor([CLASSES.index(row["class_name"])], device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(int(row["seed"]))
        generated = sample(
            model, source, class_ids, scheduler, args.inference_steps,
            guidance_scale, strength, generator,
        )
        destination = output / f"{row['candidate_id']}.png"
        save_image(generated.add(1).div(2), destination)
        with Image.open(row["source_image"]) as image:
            source_pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
        with Image.open(destination) as image:
            candidate_pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
        mean_difference = float(np.abs(source_pixels - candidate_pixels).mean())
        exact_copy = bool(np.array_equal(source_pixels, candidate_pixels))
        exact_pixel_copies += int(exact_copy)
        pixel_differences.append(mean_difference)
        if len(preview) < 6:
            preview.extend([source.detach().cpu()[0], generated.detach().cpu()[0]])
        rows.append(
            {
                **row,
                "candidate_image": destination.as_posix(),
                "candidate_sha256": sha256(destination),
                "checkpoint": model_dir.as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "sampling_selection_report": config_path.as_posix(),
                "guidance_scale": guidance_scale,
                "source_strength": strength,
                "inference_steps": args.inference_steps,
                "mean_absolute_source_difference_255": mean_difference,
                "exact_pixel_copy": exact_copy,
            }
        )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_image(torch.stack(preview).add(1).div(2), output / "source_candidate_preview.png", nrow=2)
    report = {
        "status": "pass",
        "method": method,
        "expected": 26,
        "generated": len(rows),
        "equal_budget_count_pass": len(rows) == 26,
        "exact_pixel_copies": exact_pixel_copies,
        "mean_absolute_source_difference_255": float(np.mean(pixel_differences)),
        "minimum_absolute_source_difference_255": float(np.min(pixel_differences)),
        "maximum_absolute_source_difference_255": float(np.max(pixel_differences)),
        "guidance_scale": guidance_scale,
        "source_strength": strength,
        "inference_steps": args.inference_steps,
        "checkpoint": model_dir.as_posix(),
        "checkpoint_sha256": rows[0]["checkpoint_sha256"],
        "sampling_selection_report": config_path.as_posix(),
    }
    (output / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
