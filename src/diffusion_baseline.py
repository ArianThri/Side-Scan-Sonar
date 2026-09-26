"""Optical-pretrained, class-conditioned, image-guided diffusion baseline.

Pilot mode verifies checkpoint transfer, classifier-free conditioning, CUDA
optimisation, DDIM source-guided sampling, adapter checkpointing, and reload.
It is deliberately too short to constitute an experimental result.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

CLASSES = ["aircraft", "human", "ship"]
NULL_CLASS = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SCTDDataset(Dataset):
    def __init__(self, manifest: Path, ids_file: Path, limit: int):
        accepted = {line.strip() for line in ids_file.read_text().splitlines() if line.strip()}
        with manifest.open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["image_id"] in accepted]
        self.rows = sorted(rows, key=lambda row: row["image_id"])[:limit]
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["processed_image"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row["class_name"]), row["image_id"]


def build_conditioned_unet(model_id: str, local_files_only: bool) -> tuple[UNet2DModel, list[str], list[str]]:
    base = UNet2DModel.from_pretrained(model_id, local_files_only=local_files_only)
    config = dict(base.config)
    # Diffusers records which constructor values came from defaults and will
    # otherwise silently restore them in from_config, ignoring our overrides.
    config.pop("_use_default_values", None)
    config["class_embed_type"] = None
    config["num_class_embeds"] = len(CLASSES) + 1
    model = UNet2DModel.from_config(config)
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    del base
    expected_missing = {"class_embedding.weight"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected transfer mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model, list(incompatible.missing_keys), list(incompatible.unexpected_keys)


@torch.no_grad()
def image_guided_sample(
    model: UNet2DModel, source: torch.Tensor, class_ids: torch.Tensor,
    training_scheduler: DDPMScheduler, steps: int, guidance_scale: float,
    strength: float, generator: torch.Generator,
) -> torch.Tensor:
    scheduler = DDIMScheduler.from_config(training_scheduler.config)
    scheduler.set_timesteps(steps, device=source.device)
    start_index = min(max(round((1.0 - strength) * (steps - 1)), 0), steps - 1)
    timesteps = scheduler.timesteps[start_index:]
    initial_timestep = timesteps[0].expand(source.shape[0])
    noise = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype)
    sample = scheduler.add_noise(source, noise, initial_timestep)
    for timestep in timesteps:
        unconditional = torch.full_like(class_ids, NULL_CLASS)
        epsilon_unconditional = model(sample, timestep, class_labels=unconditional).sample
        epsilon_conditional = model(sample, timestep, class_labels=class_ids).sample
        epsilon = epsilon_unconditional + guidance_scale * (
            epsilon_conditional - epsilon_unconditional
        )
        sample = scheduler.step(epsilon, timestep, sample).prev_sample
    return sample.clamp(-1, 1)


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Pilot requires CUDA; refusing an unrecorded CPU fallback")
    device = torch.device("cuda")
    started = time.time()
    model, missing, unexpected = build_conditioned_unet(args.model_id, args.local_files_only)
    model.enable_gradient_checkpointing()
    # Pilot adaptation exercises the new condition and output head while keeping
    # memory far below 8 GB. Full-run policy remains separately configured.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.class_embedding.parameters():
        parameter.requires_grad_(True)
    for parameter in model.conv_out.parameters():
        parameter.requires_grad_(True)
    # Keep this pilot in FP32 after direct FP16 optimisation produced a NaN.
    # Full training may use AMP only with a GradScaler and the same finite gate.
    model.to(device=device, dtype=torch.float32)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01)
    # This legacy single-model checkpoint contains no scheduler_config.json.
    # Use the standard 1,000-step linear DDPM schedule associated with the
    # original pretrained model family and persist the choice in the report.
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset = SCTDDataset(args.manifest, args.ids_file, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    step_rows = []
    last = None
    model.train()
    for step, (clean, class_ids, image_ids) in enumerate(loader):
        if step >= args.steps:
            break
        clean = clean.to(device=device, dtype=torch.float32)
        class_ids = class_ids.to(device)
        dropped = torch.rand(class_ids.shape, device=device) < args.condition_dropout
        training_classes = torch.where(dropped, torch.full_like(class_ids, NULL_CLASS), class_ids)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=device)
        noisy = scheduler.add_noise(clean, noise, timesteps)
        optimiser.zero_grad(set_to_none=True)
        prediction = model(noisy, timesteps, class_labels=training_classes).sample
        loss = F.mse_loss(prediction.float(), noise.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite diffusion loss at step {step + 1}: {loss.item()}")
        loss.backward()
        optimiser.step()
        step_rows.append({
            "step": step + 1, "image_id": image_ids[0], "timestep": int(timesteps.item()),
            "condition_dropped": bool(dropped.item()), "epsilon_mse": float(loss.detach()),
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        })
        last = (clean.detach(), class_ids.detach())
    if not step_rows or last is None:
        raise RuntimeError("No diffusion optimisation step completed")

    model.eval()
    source, class_ids = last
    sample_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    generated = image_guided_sample(
        model, source, class_ids, scheduler, args.inference_steps,
        args.guidance_scale, args.strength, sample_generator,
    )
    save_image(torch.cat([source, generated]).float().add(1).div(2), args.output_dir / "pilot_grid.png", nrow=1)
    adapter_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("class_embedding.") or key.startswith("conv_out.")
    }
    torch.save({
        "base_model": args.model_id, "adapter_state": adapter_state,
        "classes": CLASSES, "null_class": NULL_CLASS, "seed": args.seed,
        "training_steps": len(step_rows),
    }, args.output_dir / "adapter_checkpoint.pt")
    reloaded = torch.load(args.output_dir / "adapter_checkpoint.pt", map_location="cpu", weights_only=False)
    if set(reloaded["adapter_state"]) != set(adapter_state):
        raise RuntimeError("Adapter checkpoint reload invariant failed")
    report = {
        "status": "pass", "purpose": "execution pilot only; not an experimental result",
        "base_model": args.model_id, "device": torch.cuda.get_device_name(0),
        "transfer_missing_keys": missing, "transfer_unexpected_keys": unexpected,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "frozen_parameters": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        "training_steps": len(step_rows), "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale, "source_strength": args.strength,
        "training_beta_schedule": "linear", "training_timesteps": 1000,
        "pilot_precision": "float32",
        "peak_cuda_memory_mb": max(row["peak_cuda_memory_mb"] for row in step_rows),
        "elapsed_seconds": time.time() - started, "step_log": step_rows,
        "checkpoint_reload": "pass",
    }
    (args.output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/ddpm-celebahq-256")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/generative_256_v1/manifest.csv"))
    parser.add_argument("--ids-file", type=Path, default=Path("data/splits/limited_labels_group_aware_v1/train_010.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilots/standard_diffusion_v1"))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--strength", type=float, default=0.60)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
