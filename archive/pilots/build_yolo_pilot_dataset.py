"""Build a real-only evaluation / QC-approved synthetic YOLO pilot dataset."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    source_yolo = Path("data/processed/yolo_sctd_group_aware_v1")
    real_ids = {
        line.strip() for line in Path("data/splits/limited_labels_group_aware_v1/train_010.txt").read_text().splitlines()
        if line.strip()
    }
    output = Path("data/processed/yolo_pilot_augmented_v1")
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    rows = []
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for image_id in sorted(real_ids):
        image_matches = list((source_yolo / "images" / "train").glob(f"{image_id}.*"))
        if len(image_matches) != 1:
            raise RuntimeError(f"Expected one real training image for {image_id}")
        shutil.copy2(image_matches[0], output / "images" / "train" / image_matches[0].name)
        shutil.copy2(source_yolo / "labels" / "train" / f"{image_id}.txt", output / "labels" / "train" / f"{image_id}.txt")
        rows.append({"sample_id": image_id, "split": "train", "origin": "real", "method": "real", "source_id": image_id})
    for split in ("val", "test"):
        for image_path in sorted((source_yolo / "images" / split).iterdir()):
            if image_path.is_file():
                shutil.copy2(image_path, output / "images" / split / image_path.name)
                shutil.copy2(source_yolo / "labels" / split / f"{image_path.stem}.txt", output / "labels" / split / f"{image_path.stem}.txt")
                rows.append({"sample_id": image_path.stem, "split": split, "origin": "real", "method": "real", "source_id": image_path.stem})
    with Path("results/pilots/synthetic_qc_v1/qc_manifest.csv").open(encoding="utf-8") as handle:
        qc_rows = list(csv.DictReader(handle))
    approved = [row for row in qc_rows if row["approved"].lower() == "true"]
    for row in approved:
        if row["source_id"] not in real_ids:
            raise RuntimeError(f"Synthetic source is not in the limited training set: {row['source_id']}")
        candidate_id = row["candidate_id"]
        shutil.copy2(row["candidate_image"], output / "images" / "train" / f"{candidate_id}.png")
        shutil.copy2(
            Path("results/pilots/synthetic_qc_v1/approved_labels") / f"{candidate_id}.txt",
            output / "labels" / "train" / f"{candidate_id}.txt",
        )
        rows.append({"sample_id": candidate_id, "split": "train", "origin": "synthetic", "method": row["method"], "source_id": row["source_id"]})
    yaml_text = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: aircraft\n  1: human\n  2: ship\n"
    )
    (output / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        split: {
            "images": len(list((output / "images" / split).iterdir())),
            "labels": len(list((output / "labels" / split).glob("*.txt"))),
        } for split in ("train", "val", "test")
    }
    report = {
        "status": "pass", "purpose": "YOLO execution pilot only",
        "real_training_images": len(real_ids), "approved_synthetic_images": len(approved),
        "validation_real_only": True, "test_real_only": True, "counts": counts,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
