"""One-step full-backbone AMP memory/finite-loss gate for the RTX 4060."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from diffusion_baseline import SCTDDataset, build_conditioned_unet, seed_everything


def main() -> None:
    output = Path("results/resource_checks/diffusion_full_amp_v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    seed_everything(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.reset_peak_memory_stats()
    model, missing, unexpected = build_conditioned_unet("google/ddpm-celebahq-256", True)
    model.enable_gradient_checkpointing()
    model.to("cuda")
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon")
    dataset = SCTDDataset(
        Path("data/processed/generative_256_v1/manifest.csv"),
        Path("data/splits/limited_labels_group_aware_v1/train_010.txt"), 1,
    )
    clean, class_ids, _ = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    clean, class_ids = clean.cuda(), class_ids.cuda()
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([500], device="cuda")
    noisy = scheduler.add_noise(clean, noise, timesteps)
    optimiser.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        prediction = model(noisy, timesteps, class_labels=class_ids).sample
        loss = F.mse_loss(prediction.float(), noise.float())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite benchmark loss: {loss.item()}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimiser)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"Non-finite benchmark gradient norm: {gradient_norm.item()}")
    scaler.step(optimiser)
    scaler.update()
    torch.cuda.synchronize()
    report = {
        "status": "pass", "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "precision": "AMP fp16 with FP32 parameters and GradScaler",
        "gradient_checkpointing": True, "batch_size": 1, "image_size": 256,
        "loss": float(loss.detach()), "gradient_norm_before_clip": float(gradient_norm),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "transfer_missing_keys": missing, "transfer_unexpected_keys": unexpected,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
