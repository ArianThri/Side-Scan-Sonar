"""Grounded diffusion pilot with independently switchable sonar constraints."""

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
from diffusers import DDPMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

try:
    from .diffusion_baseline import (
        CLASSES, NULL_CLASS, build_conditioned_unet, image_guided_sample,
    )
except ImportError:  # Direct script execution: python src/grounded_diffusion.py
    from diffusion_baseline import (
        CLASSES, NULL_CLASS, build_conditioned_unet, image_guided_sample,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GroundedDataset(Dataset):
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
        boxes = []
        for line in Path(row["processed_label"]).read_text().splitlines():
            class_id, xc, yc, width, height = line.split()
            boxes.append([float(xc), float(yc), float(width), float(height)])
        return tensor, CLASSES.index(row["class_name"]), torch.tensor(boxes), row["image_id"]


def predicted_clean(noisy, predicted_noise, timesteps, scheduler):
    alpha = scheduler.alphas_cumprod.to(noisy.device)[timesteps]
    while alpha.ndim < noisy.ndim:
        alpha = alpha.unsqueeze(-1)
    return ((noisy - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()).clamp(-1, 1)


def luminance(images):
    rgb = images.add(1).div(2).clamp(0, 1)
    weights = images.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
    return (rgb * weights).sum(1, keepdim=True)


def region_mean(image, x1, y1, x2, y2):
    height, width = image.shape[-2:]
    x1, x2 = max(0, min(width - 1, x1)), max(1, min(width, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return image.mean()
    return image[..., y1:y2, x1:x2].mean()


def geometry_loss(predicted, clean, boxes):
    """Preserve target-to-darkest-adjacent-region contrast, without ray tracing."""
    pred_gray, clean_gray = luminance(predicted), luminance(clean)
    total = predicted.new_zeros(())
    count = 0
    height, width = predicted.shape[-2:]
    for batch_index in range(predicted.shape[0]):
        for xc, yc, bw, bh in boxes[batch_index]:
            cx, cy = int(xc * width), int(yc * height)
            box_w, box_h = max(2, int(bw * width)), max(2, int(bh * height))
            x1, x2 = cx - box_w // 2, cx + box_w // 2
            y1, y2 = cy - box_h // 2, cy + box_h // 2
            target_clean = region_mean(clean_gray[batch_index:batch_index+1], x1, y1, x2, y2)
            target_pred = region_mean(pred_gray[batch_index:batch_index+1], x1, y1, x2, y2)
            candidates = [
                (x1 - box_w, y1, x1, y2), (x2, y1, x2 + box_w, y2),
                (x1, y1 - box_h, x2, y1), (x1, y2, x2, y2 + box_h),
            ]
            clean_regions = torch.stack([
                region_mean(clean_gray[batch_index:batch_index+1], *candidate)
                for candidate in candidates
            ])
            shadow_index = int(clean_regions.argmin())
            shadow_clean = clean_regions[shadow_index]
            shadow_pred = region_mean(pred_gray[batch_index:batch_index+1], *candidates[shadow_index])
            total = total + F.l1_loss(target_pred - shadow_pred, target_clean - shadow_clean)
            count += 1
    return total / max(count, 1)


def range_profile_loss(predicted, clean):
    """Preserve the stronger recoverable mean-intensity profile; infer no range."""
    pred_gray, clean_gray = luminance(predicted), luminance(clean)
    losses = []
    for index in range(predicted.shape[0]):
        profile_x = clean_gray[index].mean(dim=1)
        profile_y = clean_gray[index].mean(dim=2)
        if profile_x.var() >= profile_y.var():
            losses.append(F.l1_loss(pred_gray[index].mean(dim=1), profile_x))
        else:
            losses.append(F.l1_loss(pred_gray[index].mean(dim=2), profile_y))
    return torch.stack(losses).mean()


def log_speckle_loss(predicted, clean, epsilon=1 / 255):
    """Match signal-weighted local high-frequency residuals in log intensity."""
    pred_intensity, clean_intensity = luminance(predicted), luminance(clean)
    pred_log = torch.log(pred_intensity + epsilon)
    clean_log = torch.log(clean_intensity + epsilon)
    pred_residual = pred_log - F.avg_pool2d(pred_log, 5, stride=1, padding=2)
    clean_residual = clean_log - F.avg_pool2d(clean_log, 5, stride=1, padding=2)
    weight = 0.25 + clean_intensity
    return ((pred_residual - clean_residual).square() * weight).mean()


def run(args):
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Grounded pilot requires CUDA")
    device = torch.device("cuda")
    model, missing, unexpected = build_conditioned_unet(args.model_id, True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.class_embedding, model.conv_out):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.to(device=device, dtype=torch.float32)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon")
    dataset = GroundedDataset(args.manifest, args.ids_file, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rows, last = [], None
    for step, (clean, class_ids, boxes, image_ids) in enumerate(loader):
        if step >= args.steps:
            break
        clean, class_ids, boxes = clean.to(device), class_ids.to(device), boxes.to(device)
        condition_dropped = torch.rand((), device=device) < 0.10
        training_classes = torch.full_like(class_ids, NULL_CLASS) if condition_dropped else class_ids
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, 1000, (1,), device=device)
        noisy = scheduler.add_noise(clean, noise, timesteps)
        optimiser.zero_grad(set_to_none=True)
        prediction = model(noisy, timesteps, class_labels=training_classes).sample
        base = F.mse_loss(prediction, noise)
        x0 = predicted_clean(noisy, prediction, timesteps, scheduler)
        geometric = geometry_loss(x0, clean, boxes) if args.geometric else base.new_zeros(())
        range_term = range_profile_loss(x0, clean) if args.range else base.new_zeros(())
        speckle = log_speckle_loss(x0, clean) if args.speckle else base.new_zeros(())
        total = base + 0.10 * geometric + 0.05 * range_term + 0.05 * speckle
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite grounded loss at step {step + 1}")
        total.backward()
        optimiser.step()
        rows.append({
            "step": step + 1, "image_id": image_ids[0], "timestep": int(timesteps.item()),
            "epsilon_mse": float(base.detach()), "geometry_loss": float(geometric.detach()),
            "range_loss": float(range_term.detach()), "log_speckle_loss": float(speckle.detach()),
            "total_loss": float(total.detach()),
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        })
        last = clean.detach(), class_ids.detach()
    if not rows or last is None:
        raise RuntimeError("No grounded optimisation step completed")
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    generated = image_guided_sample(
        model, last[0], last[1], scheduler, 4, 2.0, 0.60, generator
    )
    save_image(torch.cat([last[0], generated]).add(1).div(2), args.output_dir / "pilot_grid.png", nrow=1)
    adapter_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
        if key.startswith("class_embedding.") or key.startswith("conv_out.")
    }
    checkpoint = {
        "base_model": args.model_id, "adapter_state": adapter_state,
        "components": {"geometric": args.geometric, "range": args.range, "speckle": args.speckle},
        "seed": args.seed, "steps": len(rows),
    }
    torch.save(checkpoint, args.output_dir / "adapter_checkpoint.pt")
    reloaded = torch.load(args.output_dir / "adapter_checkpoint.pt", map_location="cpu", weights_only=False)
    if reloaded["components"] != checkpoint["components"]:
        raise RuntimeError("Grounded checkpoint reload failed")
    report = {
        "status": "pass", "purpose": "execution pilot only; not an experimental result",
        "base_model": args.model_id, "transfer_missing_keys": missing,
        "transfer_unexpected_keys": unexpected, "components": checkpoint["components"],
        "loss_weights": {"geometric": 0.10, "range": 0.05, "speckle": 0.05},
        "training_steps": len(rows), "inference_steps": 4, "guidance_scale": 2.0,
        "source_strength": 0.60, "precision": "float32",
        "peak_cuda_memory_mb": max(row["peak_cuda_memory_mb"] for row in rows),
        "elapsed_seconds": time.time() - started, "checkpoint_reload": "pass", "step_log": rows,
    }
    (args.output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/ddpm-celebahq-256")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/generative_256_v1/manifest.csv"))
    parser.add_argument("--ids-file", type=Path, default=Path("data/splits/limited_labels_group_aware_v1/train_010.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilots/grounded_diffusion_v1"))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speckle", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
