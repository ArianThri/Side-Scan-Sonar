"""Create the frozen equal-budget traditional augmentation baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_labels(source: Path, destination: Path, matrix: np.ndarray) -> int:
    output_lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        class_id, xc, yc, width, height = line.split()
        xc, yc, width, height = map(float, (xc, yc, width, height))
        x1, x2 = (xc - width / 2) * 256, (xc + width / 2) * 256
        y1, y2 = (yc - height / 2) * 256, (yc + height / 2) * 256
        corners = np.array(
            [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]],
            dtype=np.float64,
        )
        transformed = corners @ matrix.T
        nx1, ny1 = transformed.min(axis=0)
        nx2, ny2 = transformed.max(axis=0)
        nx1, ny1, nx2, ny2 = np.clip([nx1, ny1, nx2, ny2], 0.0, 256.0)
        if nx2 - nx1 < 3 or ny2 - ny1 < 3:
            raise ValueError(f"Transformed box became invalid: {source}")
        output_lines.append(
            f"{class_id} {(nx1 + nx2) / 512:.8f} {(ny1 + ny2) / 512:.8f} "
            f"{(nx2 - nx1) / 256:.8f} {(ny2 - ny1) / 256:.8f}"
        )
    if not output_lines:
        raise ValueError(f"No annotations in {source}")
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return len(output_lines)


def main() -> None:
    output = Path("synthetic/label_010_v1/traditional")
    if output.exists():
        raise FileExistsError(output)
    labels_dir = output / "labels"
    labels_dir.mkdir(parents=True)
    with Path("experiments/label_010_v1/candidate_registry.csv").open(
        encoding="utf-8"
    ) as handle:
        registry = [row for row in csv.DictReader(handle) if row["method"] == "traditional"]
    if len(registry) != 26:
        raise RuntimeError(f"Frozen registry contains {len(registry)} traditional rows, expected 26")

    rows = []
    differences = []
    exact_copies = 0
    previews: dict[str, tuple[Image.Image, Image.Image]] = {}
    for row in registry:
        rng = np.random.default_rng(int(row["seed"]))
        angle_degrees = float(rng.uniform(-1.5, 1.5))
        scale = float(rng.uniform(0.99, 1.01))
        translate_x = float(rng.uniform(-3.0, 3.0))
        translate_y = float(rng.uniform(-3.0, 3.0))
        intensity_gain = float(rng.uniform(0.94, 1.06))
        gamma = float(rng.uniform(0.96, 1.04))
        matrix = cv2.getRotationMatrix2D((127.5, 127.5), angle_degrees, scale)
        matrix[:, 2] += [translate_x, translate_y]

        source_bgr = cv2.imread(row["source_image"], cv2.IMREAD_COLOR)
        if source_bgr is None or source_bgr.shape[:2] != (256, 256):
            raise ValueError(f"Expected readable 256x256 source: {row['source_image']}")
        warped = cv2.warpAffine(
            source_bgr, matrix, (256, 256), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        candidate = np.clip(
            np.power(warped.astype(np.float32) / 255.0, gamma) * intensity_gain,
            0.0,
            1.0,
        )
        candidate = np.rint(candidate * 255).astype(np.uint8)
        image_path = output / f"{row['candidate_id']}.png"
        if not cv2.imwrite(str(image_path), candidate):
            raise OSError(f"Failed to write {image_path}")
        label_path = labels_dir / f"{row['candidate_id']}.txt"
        object_count = transform_labels(Path(row["source_label"]), label_path, matrix)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        candidate_rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        difference = float(np.abs(source_rgb.astype(np.int16) - candidate_rgb.astype(np.int16)).mean())
        exact_copy = bool(np.array_equal(source_rgb, candidate_rgb))
        differences.append(difference)
        exact_copies += int(exact_copy)
        if row["class_name"] not in previews:
            previews[row["class_name"]] = (
                Image.fromarray(source_rgb), Image.fromarray(candidate_rgb)
            )
        rows.append(
            {
                **row,
                "candidate_image": image_path.as_posix(),
                "candidate_label": label_path.as_posix(),
                "candidate_sha256": sha256(image_path),
                "candidate_label_sha256": sha256(label_path),
                "object_count": object_count,
                "angle_degrees": angle_degrees,
                "scale": scale,
                "translation_x_pixels": translate_x,
                "translation_y_pixels": translate_y,
                "intensity_gain": intensity_gain,
                "gamma": gamma,
                "mean_absolute_source_difference_255": difference,
                "exact_pixel_copy": exact_copy,
                "annotation_policy": "affine-transform all four box corners then axis-align and clip",
            }
        )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if set(previews) != {"aircraft", "human", "ship"}:
        raise RuntimeError("Preview could not cover all three classes")
    preview = Image.new("RGB", (512, 256 * 3))
    for row_index, class_name in enumerate(("aircraft", "human", "ship")):
        source, candidate = previews[class_name]
        preview.paste(source, (0, row_index * 256))
        preview.paste(candidate, (256, row_index * 256))
    preview.save(output / "source_candidate_preview.png")
    report = {
        "status": "pass",
        "method": "traditional",
        "expected": 26,
        "generated": len(rows),
        "equal_budget_count_pass": len(rows) == 26,
        "exact_pixel_copies": exact_copies,
        "mean_absolute_source_difference_255": float(np.mean(differences)),
        "minimum_absolute_source_difference_255": float(np.min(differences)),
        "maximum_absolute_source_difference_255": float(np.max(differences)),
        "geometric_bounds": {
            "rotation_degrees": [-1.5, 1.5],
            "scale": [0.99, 1.01],
            "translation_pixels": [-3.0, 3.0],
        },
        "photometric_bounds": {
            "global_intensity_gain": [0.94, 1.06],
            "gamma": [0.96, 1.04],
        },
        "independent_additive_noise_used": False,
        "boxes_transformed": True,
    }
    (output / "generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
