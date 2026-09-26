"""Transfer-learning conditional GAN baseline and executable pilot.

The generator reuses an ImageNet-pretrained ResNet-18 encoder, injects class and
stochastic embeddings at the bottleneck, and decodes through U-Net skips. The
conditional PatchGAN receives source, candidate, and class maps. Pilot mode is a
real forward/backward/checkpoint/sample run, not an experimental result.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import save_image

CLASSES = ["aircraft", "human", "ship"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SCTDGenerativeDataset(Dataset):
    def __init__(self, manifest: Path, ids_file: Path, limit: int | None = None):
        accepted = {line.strip() for line in ids_file.read_text().splitlines() if line.strip()}
        with manifest.open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["image_id"] in accepted]
        rows.sort(key=lambda row: row["image_id"])
        self.rows = rows[:limit] if limit else rows
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["processed_image"]) as image:
            tensor = self.to_tensor(image.convert("RGB"))
        return tensor, CLASSES.index(row["class_name"]), row["image_id"]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, pretrained: bool = True, noise_dim: int = 64):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.resnet18(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.class_embedding = nn.Embedding(len(CLASSES), 64)
        self.noise_dim = noise_dim
        self.condition = nn.Linear(64 + noise_dim, 512)
        self.d4 = ConvBlock(512 + 256, 256)
        self.d3 = ConvBlock(256 + 128, 128)
        self.d2 = ConvBlock(128 + 64, 64)
        self.d1 = ConvBlock(64 + 64, 32)
        self.final = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1), nn.Tanh())

    def forward(self, source, class_ids, noise=None):
        e0 = self.stem(source)       # 128
        e1 = self.layer1(self.pool(e0))  # 64
        e2 = self.layer2(e1)         # 32
        e3 = self.layer3(e2)         # 16
        e4 = self.layer4(e3)         # 8
        if noise is None:
            noise = torch.randn(source.shape[0], self.noise_dim, device=source.device)
        condition = self.condition(torch.cat([self.class_embedding(class_ids), noise], dim=1))
        e4 = e4 + condition[:, :, None, None]
        x = self.d4(torch.cat([F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        x = self.d3(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        x = self.d2(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        x = self.d1(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e0], 1))
        residual = self.final(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return torch.clamp(source + 0.35 * residual, -1, 1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [9, 64, 128, 256, 512]
        layers = []
        for index in range(len(channels) - 1):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(
                channels[index], channels[index + 1], 4, stride=2, padding=1
            )))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(*layers)
        self.head = nn.utils.spectral_norm(nn.Conv2d(512, 1, 3, padding=1))

    def forward(self, source, candidate, class_ids):
        class_maps = F.one_hot(class_ids, len(CLASSES)).float()[:, :, None, None]
        class_maps = class_maps.expand(-1, -1, source.shape[2], source.shape[3])
        features = self.features(torch.cat([source, candidate, class_maps], dim=1))
        return self.head(features), features


def degrade(images: torch.Tensor) -> torch.Tensor:
    """Deterministic-shape stochastic degradation; target remains the real image."""
    noise = torch.randn_like(images) * 0.03
    brightness = torch.empty(images.shape[0], 1, 1, 1, device=images.device).uniform_(0.92, 1.08)
    return torch.clamp(images * brightness + noise, -1, 1)


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Pilot requires CUDA; refusing an unrecorded CPU fallback")
    dataset = SCTDGenerativeDataset(args.manifest, args.ids_file, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    generator = Generator(pretrained=not args.no_pretrained).to(device)
    discriminator = Discriminator().to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=1e-4)
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    log_rows = []
    last_batch = None
    for step, (real, class_ids, image_ids) in enumerate(loader):
        if step >= args.steps:
            break
        real, class_ids = real.to(device), class_ids.to(device)
        source = degrade(real)
        opt_d.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            fake = generator(source, class_ids)
            real_logits, _ = discriminator(source, real, class_ids)
            fake_logits, _ = discriminator(source, fake.detach(), class_ids)
            loss_d = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            fake = generator(source, class_ids)
            fake_logits, fake_features = discriminator(source, fake, class_ids)
            with torch.no_grad():
                _, real_features = discriminator(source, real, class_ids)
            adversarial = -fake_logits.mean()
            reconstruction = F.l1_loss(fake, real)
            feature_matching = F.l1_loss(fake_features, real_features)
            loss_g = adversarial + 10.0 * reconstruction + feature_matching
        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        row = {
            "step": step + 1, "image_ids": list(image_ids),
            "loss_g": float(loss_g.detach()), "loss_d": float(loss_d.detach()),
            "peak_cuda_memory_mb": peak_mb,
        }
        log_rows.append(row)
        last_batch = (source.detach(), fake.detach(), real.detach())

    if not log_rows or last_batch is None:
        raise RuntimeError("Pilot completed no optimisation steps")
    torch.save({
        "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
        "optimiser_g": opt_g.state_dict(), "optimiser_d": opt_d.state_dict(),
        "seed": args.seed, "steps": len(log_rows), "classes": CLASSES,
        "pretrained_encoder": not args.no_pretrained,
    }, args.output_dir / "checkpoint.pt")
    source, fake, real = last_batch
    save_image(torch.cat([source, fake, real]).add(1).div(2), args.output_dir / "pilot_grid.png", nrow=source.shape[0])
    report = {
        "status": "pass", "purpose": "execution pilot only; not an experimental result",
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "dataset_samples_available": len(dataset), "optimisation_steps": len(log_rows),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_mb": max(row["peak_cuda_memory_mb"] for row in log_rows),
        "checkpoint": (args.output_dir / "checkpoint.pt").as_posix(),
        "sample_grid": (args.output_dir / "pilot_grid.png").as_posix(),
        "step_log": log_rows,
    }
    (args.output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/generative_256_v1/manifest.csv"))
    parser.add_argument("--ids-file", type=Path, default=Path("data/splits/limited_labels_group_aware_v1/train_010.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilots/gan_v1"))
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
