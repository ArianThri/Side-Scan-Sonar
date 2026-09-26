"""Evaluate equal-budget SCTD synthetic sets without using detector test labels.

Distributional FID/MMD use the same Clean-FID Inception features for every
method.  LPIPS, SSIM, and PSNR are paired because every frozen candidate has a
known real source.  SSIM/PSNR are computed after a sub-pixel phase-correlation
translation and only on the valid overlap.  The translation is an evaluation
alignment; it never changes an image or annotation used by a detector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


METHODS = ("traditional", "gan", "standard_diffusion", "grounded_diffusion")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26:
        raise RuntimeError(f"{path} contains {len(rows)} candidates; expected 26")
    required = {"candidate_id", "source_id", "class_name", "source_image", "candidate_image"}
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    return sorted(rows, key=lambda row: row["source_id"])


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape != (256, 256, 3):
        raise ValueError(f"Expected a 256x256 RGB-compatible image: {path} ({array.shape})")
    return array


def aligned_candidate(source_rgb: np.ndarray, candidate_rgb: np.ndarray):
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    candidate_gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    (dx, dy), response = cv2.phaseCorrelate(source_gray, candidate_gray)
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        candidate_gray, matrix, (256, 256), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    valid = cv2.warpAffine(
        np.ones_like(candidate_gray), matrix, (256, 256), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0.5
    return source_gray, aligned, valid, float(dx), float(dy), float(response)


def aligned_psnr(source: np.ndarray, candidate: np.ndarray, valid: np.ndarray) -> float:
    error = float(np.mean((source[valid] - candidate[valid]) ** 2))
    return float("inf") if error <= 1e-15 else float(10.0 * math.log10(1.0 / error))


def aligned_ssim(source: np.ndarray, candidate: np.ndarray, valid: np.ndarray) -> float:
    # Wang et al. local SSIM: 11x11 Gaussian, sigma 1.5, luminance range [0,1].
    mu_source = cv2.GaussianBlur(source, (11, 11), 1.5)
    mu_candidate = cv2.GaussianBlur(candidate, (11, 11), 1.5)
    mu_source_sq = mu_source * mu_source
    mu_candidate_sq = mu_candidate * mu_candidate
    mu_cross = mu_source * mu_candidate
    sigma_source = cv2.GaussianBlur(source * source, (11, 11), 1.5) - mu_source_sq
    sigma_candidate = cv2.GaussianBlur(candidate * candidate, (11, 11), 1.5) - mu_candidate_sq
    sigma_cross = cv2.GaussianBlur(source * candidate, (11, 11), 1.5) - mu_cross
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_source_sq + mu_candidate_sq + c1) * (
        sigma_source + sigma_candidate + c2
    )
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    # A valid 11x11 neighbourhood must contain no reflected padding.
    valid_window = cv2.erode(valid.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    if not valid_window.any():
        raise RuntimeError("Alignment left no valid SSIM evaluation area")
    return float(np.mean(ssim_map[valid_window]))


def low_rank_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Exact sample FID using the low-rank covariance factors (stable for n << d)."""
    real = real_features.astype(np.float64)
    fake = fake_features.astype(np.float64)
    real_mean, fake_mean = real.mean(0), fake.mean(0)
    real_centered = (real - real_mean) / math.sqrt(real.shape[0] - 1)
    fake_centered = (fake - fake_mean) / math.sqrt(fake.shape[0] - 1)
    cross = real_centered @ fake_centered.T
    covariance_root_trace = float(np.linalg.svd(cross, compute_uv=False).sum())
    value = (
        float(np.square(real_mean - fake_mean).sum())
        + float(np.square(real_centered).sum())
        + float(np.square(fake_centered).sum())
        - 2.0 * covariance_root_trace
    )
    return max(0.0, value)


def rbf_mmd(real_features: np.ndarray, fake_features: np.ndarray):
    combined = np.concatenate([real_features, fake_features]).astype(np.float64)
    squared = np.maximum(
        np.square(combined).sum(1)[:, None]
        + np.square(combined).sum(1)[None, :]
        - 2 * combined @ combined.T,
        0.0,
    )
    positive = squared[np.triu_indices_from(squared, k=1)]
    positive = positive[positive > 0]
    bandwidth_sq = float(np.median(positive))
    if not math.isfinite(bandwidth_sq) or bandwidth_sq <= 0:
        raise RuntimeError("Cannot establish a positive median MMD bandwidth")
    kernel = np.exp(-squared / (2.0 * bandwidth_sq))
    n, m = len(real_features), len(fake_features)
    kxx, kyy, kxy = kernel[:n, :n], kernel[n:, n:], kernel[:n, n:]
    biased = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    unbiased = float(
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )
    return max(0.0, biased), unbiased, bandwidth_sq


def summarise(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()), "maximum": float(array.max()),
    }


def tensor_from_rgb(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/quality/label_010_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = Path.cwd().resolve()
    manifests = {
        method: read_manifest(root / f"synthetic/label_010_v1/{method}/manifest.csv")
        for method in METHODS
    }
    expected_sources = [row["source_id"] for row in manifests[METHODS[0]]]
    for method, rows in manifests.items():
        if [row["source_id"] for row in rows] != expected_sources:
            raise RuntimeError(f"{method} does not use the identical ordered real-source budget")
        if len({row["candidate_id"] for row in rows}) != 26:
            raise RuntimeError(f"{method} candidate identifiers are not unique")
        for row in rows:
            for field in ("source_image", "candidate_image"):
                path = (root / row[field]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
            if "candidate_sha256" in row and row["candidate_sha256"]:
                if sha256((root / row["candidate_image"]).resolve()) != row["candidate_sha256"]:
                    raise RuntimeError(f"Candidate hash mismatch: {row['candidate_id']}")

    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    cache = (root / "models/cache/metrics").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str((root / "models/cache/torch").resolve())

    # Clean-FID's Windows loader places the frozen TorchScript Inception file
    # in the current directory. Isolate it in the documented metric cache.
    from cleanfid.features import build_feature_extractor
    from cleanfid.fid import get_files_features

    previous = Path.cwd()
    os.chdir(cache)
    try:
        feature_model = build_feature_extractor("clean", device=device, use_dataparallel=False)
    finally:
        os.chdir(previous)

    source_paths = [(root / row["source_image"]).resolve() for row in manifests[METHODS[0]]]
    real_features = get_files_features(
        [str(path) for path in source_paths], feature_model, num_workers=0,
        batch_size=8, device=device, mode="clean", verbose=True,
    )
    if real_features.shape != (26, 2048):
        raise RuntimeError(f"Unexpected Clean-FID feature shape: {real_features.shape}")
    if low_rank_fid(real_features, real_features) > 1e-8:
        raise RuntimeError("Low-rank FID identity self-check failed")

    import lpips

    lpips_model = lpips.LPIPS(net="alex", version="0.1", verbose=True).to(device).eval()
    per_sample: list[dict] = []
    reports: dict[str, dict] = {}
    feature_files: dict[str, str] = {}
    np.save(args.output_dir / "real_source_inception_features.npy", real_features)

    for method, rows in manifests.items():
        candidate_paths = [(root / row["candidate_image"]).resolve() for row in rows]
        fake_features = get_files_features(
            [str(path) for path in candidate_paths], feature_model, num_workers=0,
            batch_size=8, device=device, mode="clean", verbose=True,
        )
        feature_path = args.output_dir / f"{method}_inception_features.npy"
        np.save(feature_path, fake_features)
        feature_files[method] = feature_path.as_posix()
        fid = low_rank_fid(real_features, fake_features)
        mmd_biased, mmd_unbiased, bandwidth_sq = rbf_mmd(real_features, fake_features)

        method_rows = []
        for row, source_path, candidate_path in zip(rows, source_paths, candidate_paths):
            source_rgb, candidate_rgb = load_rgb(source_path), load_rgb(candidate_path)
            source_gray, aligned, valid, dx, dy, response = aligned_candidate(source_rgb, candidate_rgb)
            with torch.inference_mode():
                lpips_value = float(lpips_model(
                    tensor_from_rgb(source_rgb, device), tensor_from_rgb(candidate_rgb, device)
                ).item())
            record = {
                "method": method, "candidate_id": row["candidate_id"],
                "source_id": row["source_id"], "class_name": row["class_name"],
                "source_image": source_path.as_posix(), "candidate_image": candidate_path.as_posix(),
                "exact_pixel_copy": bool(np.array_equal(source_rgb, candidate_rgb)),
                "mean_absolute_source_difference_255": float(
                    np.abs(source_rgb.astype(np.int16) - candidate_rgb.astype(np.int16)).mean()
                ),
                "phase_dx_pixels": dx, "phase_dy_pixels": dy,
                "phase_shift_pixels": float(math.hypot(dx, dy)), "phase_response": response,
                "valid_alignment_fraction": float(valid.mean()),
                "lpips_alex_v0_1": lpips_value,
                "aligned_luminance_ssim": aligned_ssim(source_gray, aligned, valid),
                "aligned_luminance_psnr_db": aligned_psnr(source_gray, aligned, valid),
            }
            method_rows.append(record)
            per_sample.append(record)

        by_class = defaultdict(list)
        for record in method_rows:
            by_class[record["class_name"]].append(record)
        paired_fields = (
            "mean_absolute_source_difference_255", "phase_shift_pixels", "phase_response",
            "lpips_alex_v0_1", "aligned_luminance_ssim", "aligned_luminance_psnr_db",
        )
        reports[method] = {
            "sample_count": len(method_rows), "fid_clean_inception": fid,
            "mmd2_rbf_biased": mmd_biased, "mmd2_rbf_unbiased": mmd_unbiased,
            "mmd_rbf_median_bandwidth_squared": bandwidth_sq,
            "exact_pixel_copies": sum(record["exact_pixel_copy"] for record in method_rows),
            "paired_metrics": {
                field: summarise([record[field] for record in method_rows]) for field in paired_fields
            },
            "per_class": {
                class_name: {
                    "sample_count": len(class_rows),
                    **{field: summarise([record[field] for record in class_rows]) for field in paired_fields},
                }
                for class_name, class_rows in sorted(by_class.items())
            },
        }

    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
        writer.writeheader(); writer.writerows(per_sample)
    ranking = sorted(reports, key=lambda method: (reports[method]["fid_clean_inception"], method))
    report = {
        "status": "pass", "purpose": "equal-budget synthetic quality evaluation",
        "methods": list(METHODS), "real_source_count": 26,
        "identical_ordered_source_budget": True, "image_size": [256, 256],
        "fid": {
            "feature_space": "Clean-FID clean-mode Inception-v3 pool features (2048 dimensions)",
            "estimator": "sample mean/covariance; exact low-rank Frechet computation",
            "ranking_lower_is_better": ranking,
            "small_sample_warning": "n=26 per set; FID is reported as descriptive evidence and is high-variance",
        },
        "mmd": {
            "feature_space": "same frozen Clean-FID Inception features as FID",
            "kernel": "RBF with deterministic pooled median squared-distance bandwidth",
            "primary": "biased non-negative MMD squared; unbiased estimate also retained",
        },
        "paired_metrics": {
            "lpips": "calibrated LPIPS v0.1 with ImageNet AlexNet trunk; lower is more similar",
            "ssim_psnr": "grayscale luminance after phase-correlation translation; valid overlap only",
            "interpretation_warning": "high paired similarity measures source preservation, not synthetic diversity",
        },
        "metric_cache": cache.as_posix(),
        "inception_weights": (cache / "inception-2015-12-05.pt").as_posix(),
        "inception_weights_sha256": sha256(cache / "inception-2015-12-05.pt"),
        "feature_files": feature_files, "results": reports,
    }
    (args.output_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "method_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method", "fid_clean_inception", "mmd2_rbf_biased", "mmd2_rbf_unbiased",
            "lpips_mean", "aligned_ssim_mean", "aligned_psnr_db_mean",
            "mean_absolute_difference_255", "exact_pixel_copies",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for method in METHODS:
            result = reports[method]
            writer.writerow({
                "method": method, "fid_clean_inception": result["fid_clean_inception"],
                "mmd2_rbf_biased": result["mmd2_rbf_biased"],
                "mmd2_rbf_unbiased": result["mmd2_rbf_unbiased"],
                "lpips_mean": result["paired_metrics"]["lpips_alex_v0_1"]["mean"],
                "aligned_ssim_mean": result["paired_metrics"]["aligned_luminance_ssim"]["mean"],
                "aligned_psnr_db_mean": result["paired_metrics"]["aligned_luminance_psnr_db"]["mean"],
                "mean_absolute_difference_255": result["paired_metrics"]["mean_absolute_source_difference_255"]["mean"],
                "exact_pixel_copies": result["exact_pixel_copies"],
            })
    print(json.dumps({"status": "pass", "output": args.output_dir.as_posix(), "fid_ranking": ranking}, indent=2))


if __name__ == "__main__":
    main()
