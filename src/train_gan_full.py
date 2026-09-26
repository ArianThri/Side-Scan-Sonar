"""Resumable full GAN training for the frozen 10%-label experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan_baseline import (
    Discriminator, Generator, SCTDGenerativeDataset, degrade, seed_everything,
)


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(generator, discriminator, opt_g, opt_d, scaler_g, scaler_d, epoch, best, stale):
    return {
        "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
        "optimiser_g": opt_g.state_dict(), "optimiser_d": opt_d.state_dict(),
        "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(),
        "epoch": epoch, "best_validation_l1": best, "stale_epochs": stale,
        "seed": 42, "classes": ["aircraft", "human", "ship"],
        "pretrained_encoder": "torchvision ResNet18 IMAGENET1K_V1",
    }


@torch.no_grad()
def validate(generator, loader, device):
    generator.eval()
    total, count, preview = 0.0, 0, None
    fixed_generator = torch.Generator(device=device).manual_seed(4242)
    for real, class_ids, _ in loader:
        real, class_ids = real.to(device), class_ids.to(device)
        noise = torch.randn(real.shape[0], generator.noise_dim, device=device, generator=fixed_generator)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            fake = generator(real, class_ids, noise)
        total += F.l1_loss(fake.float(), real.float(), reduction="sum").item()
        count += real.numel()
        if preview is None:
            preview = torch.cat([real[:4], fake[:4]]).float().add(1).div(2).cpu()
    return total / count, preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("models/gan/label_010_v1"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seed_everything(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_set = SCTDGenerativeDataset(
        Path("data/processed/generative_256_v1/manifest.csv"),
        Path("data/splits/limited_labels_group_aware_v1/train_010.txt"), None,
    )
    val_set = SCTDGenerativeDataset(
        Path("data/processed/generative_256_v1/manifest.csv"),
        Path("data/splits/group_aware_v1/val.txt"), None,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    generator, discriminator = Generator(pretrained=True).to(device), Discriminator().to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=1e-4)
    scaler_g, scaler_d = torch.amp.GradScaler("cuda"), torch.amp.GradScaler("cuda")
    start_epoch, best, stale = 1, float("inf"), 0
    latest = args.output_dir / "checkpoint_latest.pt"
    if args.resume:
        state = torch.load(latest, map_location=device, weights_only=False)
        generator.load_state_dict(state["generator"]); discriminator.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["optimiser_g"]); opt_d.load_state_dict(state["optimiser_d"])
        scaler_g.load_state_dict(state["scaler_g"]); scaler_d.load_state_dict(state["scaler_d"])
        start_epoch, best, stale = state["epoch"] + 1, state["best_validation_l1"], state["stale_epochs"]
    log_path = args.output_dir / "training_log.jsonl"
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train(); discriminator.train()
        sums = {"g": 0.0, "d": 0.0, "adv": 0.0, "l1": 0.0, "fm": 0.0}; batches = 0
        for real, class_ids, _ in train_loader:
            real, class_ids = real.to(device), class_ids.to(device)
            source = degrade(real)
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fake = generator(source, class_ids)
                real_logits, _ = discriminator(source, real, class_ids)
                fake_logits, _ = discriminator(source, fake.detach(), class_ids)
                loss_d = F.relu(1-real_logits).mean() + F.relu(1+fake_logits).mean()
            if not torch.isfinite(loss_d): raise FloatingPointError(f"Non-finite D loss epoch {epoch}")
            scaler_d.scale(loss_d).backward(); scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            scaler_d.step(opt_d); scaler_d.update()
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fake = generator(source, class_ids)
                fake_logits, fake_features = discriminator(source, fake, class_ids)
                with torch.no_grad(): _, real_features = discriminator(source, real, class_ids)
                adversarial = -fake_logits.mean(); reconstruction = F.l1_loss(fake, real)
                feature_matching = F.l1_loss(fake_features, real_features)
                loss_g = adversarial + 10*reconstruction + feature_matching
            if not torch.isfinite(loss_g): raise FloatingPointError(f"Non-finite G loss epoch {epoch}")
            scaler_g.scale(loss_g).backward(); scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            scaler_g.step(opt_g); scaler_g.update()
            for key, value in (("g",loss_g),("d",loss_d),("adv",adversarial),("l1",reconstruction),("fm",feature_matching)):
                sums[key] += float(value.detach())
            batches += 1
        val_l1, preview = validate(generator, val_loader, device)
        improved = val_l1 < best - 1e-5
        if improved: best, stale = val_l1, 0
        else: stale += 1
        completed_epoch = epoch
        row = {"epoch":epoch, **{f"train_{k}":v/batches for k,v in sums.items()},
               "validation_l1":val_l1,"improved":improved,"stale_epochs":stale,
               "elapsed_seconds":time.time()-started,
               "peak_cuda_memory_mb":torch.cuda.max_memory_allocated()/1024**2}
        with log_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(row)+"\n")
        payload = checkpoint_payload(generator,discriminator,opt_g,opt_d,scaler_g,scaler_d,epoch,best,stale)
        atomic_torch_save(payload, latest)
        if improved:
            atomic_torch_save(payload, args.output_dir / "checkpoint_best.pt")
            save_image(preview, args.output_dir / "validation_preview_best.png", nrow=min(args.batch_size,4))
        if epoch % 10 == 0: print(json.dumps(row))
        if stale >= args.patience: break
    report = {"status":"pass","purpose":"full 10%-label GAN training","epochs_completed":completed_epoch,
              "early_stopped":completed_epoch < args.epochs,"best_validation_l1":best,
              "train_images":len(train_set),"validation_images":len(val_set),
              "elapsed_seconds":time.time()-started,
              "peak_cuda_memory_mb":torch.cuda.max_memory_allocated()/1024**2,
              "best_checkpoint":(args.output_dir/"checkpoint_best.pt").as_posix()}
    (args.output_dir/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
