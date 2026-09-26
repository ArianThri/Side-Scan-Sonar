"""Fail-fast verification for the accepted SCTD data pipeline artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/SCTD"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/group_aware_v1"))
    parser.add_argument("--subset-dir", type=Path, default=Path("data/splits/limited_labels_group_aware_v1"))
    parser.add_argument("--yolo-dir", type=Path, default=Path("data/processed/yolo_sctd_group_aware_v1"))
    parser.add_argument("--generative-dir", type=Path, default=Path("data/processed/generative_256_v1"))
    args = parser.parse_args()

    membership = {}
    split_counts = {}
    for split, expected in (("train", 249), ("val", 54), ("test", 54)):
        split_ids = ids(args.split_dir / f"{split}.txt")
        if len(split_ids) != expected or len(set(split_ids)) != expected:
            raise AssertionError(f"{split} membership count/uniqueness failed")
        for image_id in split_ids:
            if image_id in membership:
                raise AssertionError(f"Repeated split member: {image_id}")
            membership[image_id] = split
        split_counts[split] = len(split_ids)
    if len(membership) != 357:
        raise AssertionError(f"Expected 357 split members, found {len(membership)}")

    hashes = defaultdict(list)
    for path in sorted((args.dataset_root / "JPEGImages").glob("*.jpg")):
        hashes[sha256_file(path)].append(path.stem)
    leakage = []
    for digest, group_ids in hashes.items():
        group_splits = {membership[image_id] for image_id in group_ids}
        if len(group_splits) > 1:
            leakage.append({"sha256": digest, "ids": group_ids, "splits": sorted(group_splits)})
    if leakage:
        raise AssertionError(f"Exact-duplicate leakage: {leakage[:3]}")

    subset_counts = {}
    previous = set()
    train_ids = set(ids(args.split_dir / "train.txt"))
    for level, expected in ((10, 26), (25, 63), (50, 125), (100, 249)):
        selected = set(ids(args.subset_dir / f"train_{level:03d}.txt"))
        if len(selected) != expected or not selected <= train_ids or not previous <= selected:
            raise AssertionError(f"Subset invariant failed at {level}%")
        previous = selected
        subset_counts[str(level)] = len(selected)

    yolo_box_counts = Counter()
    for split in ("train", "val", "test"):
        expected_ids = {image_id for image_id, assigned in membership.items() if assigned == split}
        image_ids = {path.stem for path in (args.yolo_dir / "images" / split).iterdir() if path.is_file()}
        label_ids = {path.stem for path in (args.yolo_dir / "labels" / split).glob("*.txt")}
        if image_ids != expected_ids or label_ids != expected_ids:
            raise AssertionError(f"YOLO coverage mismatch in {split}")
        for path in (args.yolo_dir / "labels" / split).glob("*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) not in (0, 1, 2):
                    raise AssertionError(f"Malformed YOLO row in {path}: {line}")
                values = [float(value) for value in parts[1:]]
                if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                    raise AssertionError(f"Invalid YOLO row in {path}: {line}")
                yolo_box_counts[split] += 1
    if sum(yolo_box_counts.values()) != 363:
        raise AssertionError(f"Expected 363 YOLO boxes, found {sum(yolo_box_counts.values())}")

    with (args.generative_dir / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 357 or sum(int(row["object_count"]) for row in manifest) != 363:
        raise AssertionError("Generative manifest count failed")
    manifest_ids = {row["image_id"] for row in manifest}
    if manifest_ids != set(membership):
        raise AssertionError("Generative manifest coverage failed")
    for row in manifest:
        image_path = Path(row["processed_image"])
        label_path = Path(row["processed_label"])
        with Image.open(image_path) as image:
            image.verify()
            if image.size != (256, 256):
                raise AssertionError(f"Unexpected generated input size: {image_path}")
        if sha256_file(image_path) != row["processed_sha256"] or not label_path.is_file():
            raise AssertionError(f"Generative provenance failed: {row['image_id']}")

    report = {
        "status": "pass",
        "split_counts": split_counts,
        "exact_duplicate_groups": sum(len(group) > 1 for group in hashes.values()),
        "cross_partition_exact_duplicate_groups": 0,
        "subset_counts": subset_counts,
        "yolo_box_counts": dict(yolo_box_counts),
        "generative_images": len(manifest),
        "generative_objects": sum(int(row["object_count"]) for row in manifest),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
