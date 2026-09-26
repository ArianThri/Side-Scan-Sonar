"""Conservative source-box transfer gate for image-guided synthetic samples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (256, 256):
        raise ValueError(f"Expected readable 256x256 image: {path}")
    return image.astype(np.float32) / 255.0


def boxes(path: Path) -> list[tuple[float, float, float, float]]:
    result = []
    for line in path.read_text().splitlines():
        _, xc, yc, width, height = line.split()
        result.append(tuple(float(value) for value in (xc, yc, width, height)))
    return result


def normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denominator) if denominator > 1e-8 else 0.0


def target_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    source_annotations,
    candidate_annotations=None,
):
    if candidate_annotations is None:
        candidate_annotations = source_annotations
    if len(source_annotations) != len(candidate_annotations):
        raise ValueError("Source and candidate annotation counts differ")
    correlations, edge_ratios, polarity_checks = [], [], []
    for source_box, candidate_box in zip(source_annotations, candidate_annotations):
        source_xc, source_yc, source_width, source_height = source_box
        candidate_xc, candidate_yc, candidate_width, candidate_height = candidate_box
        sx1 = max(0, round((source_xc - source_width / 2) * 256))
        sy1 = max(0, round((source_yc - source_height / 2) * 256))
        sx2 = min(256, round((source_xc + source_width / 2) * 256))
        sy2 = min(256, round((source_yc + source_height / 2) * 256))
        cx1 = max(0, round((candidate_xc - candidate_width / 2) * 256))
        cy1 = max(0, round((candidate_yc - candidate_height / 2) * 256))
        cx2 = min(256, round((candidate_xc + candidate_width / 2) * 256))
        cy2 = min(256, round((candidate_yc + candidate_height / 2) * 256))
        if min(sx2 - sx1, sy2 - sy1, cx2 - cx1, cy2 - cy1) < 3:
            raise ValueError("Transferred source box is too small for QC")
        source_patch = source[sy1:sy2, sx1:sx2]
        candidate_patch = candidate[cy1:cy2, cx1:cx2]
        if source_patch.shape != candidate_patch.shape:
            candidate_patch = cv2.resize(
                candidate_patch,
                (source_patch.shape[1], source_patch.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        correlations.append(normalised_correlation(source_patch, candidate_patch))
        source_edge = np.hypot(
            cv2.Sobel(source_patch, cv2.CV_32F, 1, 0), cv2.Sobel(source_patch, cv2.CV_32F, 0, 1)
        ).mean()
        candidate_edge = np.hypot(
            cv2.Sobel(candidate_patch, cv2.CV_32F, 1, 0), cv2.Sobel(candidate_patch, cv2.CV_32F, 0, 1)
        ).mean()
        edge_ratios.append(float(candidate_edge / max(source_edge, 1e-6)))
        source_margin_x, source_margin_y = max(2, sx2 - sx1), max(2, sy2 - sy1)
        candidate_margin_x = max(2, cx2 - cx1)
        candidate_margin_y = max(2, cy2 - cy1)
        srx1, sry1 = max(0, sx1 - source_margin_x), max(0, sy1 - source_margin_y)
        srx2, sry2 = min(256, sx2 + source_margin_x), min(256, sy2 + source_margin_y)
        crx1, cry1 = max(0, cx1 - candidate_margin_x), max(0, cy1 - candidate_margin_y)
        crx2, cry2 = min(256, cx2 + candidate_margin_x), min(256, cy2 + candidate_margin_y)
        source_ring = source[sry1:sry2, srx1:srx2].copy()
        candidate_ring = candidate[cry1:cry2, crx1:crx2].copy()
        source_mask = np.ones(source_ring.shape, dtype=bool)
        candidate_mask = np.ones(candidate_ring.shape, dtype=bool)
        source_mask[sy1-sry1:sy2-sry1, sx1-srx1:sx2-srx1] = False
        candidate_mask[cy1-cry1:cy2-cry1, cx1-crx1:cx2-crx1] = False
        source_contrast = float(source_patch.mean() - source_ring[source_mask].mean())
        candidate_contrast = float(candidate_patch.mean() - candidate_ring[candidate_mask].mean())
        polarity_checks.append(abs(source_contrast) < 0.03 or source_contrast * candidate_contrast > 0)
    return min(correlations), min(edge_ratios), max(edge_ratios), all(polarity_checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("results/pilots/synthetic_candidates_v1/candidate_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilots/synthetic_qc_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    approved_label_dir = args.output_dir / "approved_labels"
    approved_label_dir.mkdir()
    with args.manifest.open(encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    output_rows = []
    for row in candidates:
        source = read_gray(Path(row["source_image"]))
        candidate = read_gray(Path(row["candidate_image"]))
        (dx, dy), phase_response = cv2.phaseCorrelate(source, candidate)
        shift = float(np.hypot(dx, dy))
        source_annotations = boxes(Path(row["source_label"]))
        candidate_label = Path(row.get("candidate_label") or row["source_label"])
        target_ncc, min_edge_ratio, max_edge_ratio, polarity_ok = target_metrics(
            source, candidate, source_annotations, boxes(candidate_label)
        )
        gates = {
            "phase_response": float(phase_response) >= 0.10,
            "phase_shift": shift <= 8.0,
            "target_ncc": target_ncc >= 0.25,
            "target_edge_ratio": min_edge_ratio >= 0.35 and max_edge_ratio <= 2.85,
            "target_contrast_polarity": polarity_ok,
        }
        approved = all(gates.values())
        reasons = [name for name, passed in gates.items() if not passed]
        if approved:
            shutil.copy2(candidate_label, approved_label_dir / f"{row['candidate_id']}.txt")
        output_rows.append({
            **row, "decision": "retain_source_box" if approved else "reject",
            "approved": approved, "phase_response": f"{phase_response:.8f}",
            "phase_shift_pixels": f"{shift:.8f}", "minimum_target_ncc": f"{target_ncc:.8f}",
            "minimum_target_edge_ratio": f"{min_edge_ratio:.8f}",
            "maximum_target_edge_ratio": f"{max_edge_ratio:.8f}",
            "target_contrast_polarity_ok": polarity_ok,
            "rejection_reasons": "|".join(reasons),
        })
    with (args.output_dir / "qc_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    counts = {
        "candidate_count": len(output_rows),
        "approved": sum(str(row["approved"]).lower() == "true" for row in output_rows),
        "rejected": sum(str(row["approved"]).lower() != "true" for row in output_rows),
    }
    report = {
        "status": "pass", "purpose": "annotation-transfer gate",
        **counts,
        "thresholds": {"phase_response_min": 0.10, "phase_shift_max_pixels": 8.0,
                       "target_ncc_min": 0.25, "target_edge_ratio": [0.35, 2.85],
                       "contrast_polarity_preserved": True},
    }
    (args.output_dir / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
