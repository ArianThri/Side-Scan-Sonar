"""Build and verify the five controlled 10% SCTD YOLO11n datasets.

The real-only condition contains the frozen 10% real training subset. Each
augmented condition adds exactly one QC-approved synthetic candidate for every
real source. Validation and test data are copied unchanged from the accepted
group-aware real-only YOLO dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


CONDITIONS = (
    "real_only",
    "traditional",
    "gan",
    "standard_diffusion",
    "grounded_diffusion",
)
SYNTHETIC_METHODS = CONDITIONS[1:]
CLASS_NAMES = {0: "aircraft", 1: "human", 2: "ship"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_label(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_value = float(fields[0])
        class_id = int(class_value)
        if class_value != class_id or class_id not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: invalid class ID {fields[0]}")
        x_center, y_center, width, height = map(float, fields[1:])
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{path}:{line_number}: normalised coordinate outside [0,1]")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number}: non-positive box extent")
        tolerance = 1e-5
        if (
            x_center - width / 2 < -tolerance
            or x_center + width / 2 > 1.0 + tolerance
            or y_center - height / 2 < -tolerance
            or y_center + height / 2 > 1.0 + tolerance
        ):
            raise ValueError(f"{path}:{line_number}: box extends outside the image")
        counts[class_id] += 1
    return counts


def exactly_one_image(directory: Path, image_id: str) -> Path:
    matches = [path for path in directory.glob(f"{image_id}.*") if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one image for {image_id} in {directory}, found {matches}")
    return matches[0]


def bool_field(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def preflight(
    source_yolo: Path,
    subset_path: Path,
    registry_path: Path,
    synthetic_root: Path,
    qc_root: Path,
) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, str]]]]:
    real_ids = [line.strip() for line in subset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(real_ids) != 26 or len(set(real_ids)) != 26:
        raise RuntimeError(f"Expected 26 unique frozen real IDs, found {len(real_ids)}/{len(set(real_ids))}")

    registry_rows = read_csv(registry_path)
    accepted: dict[str, list[dict[str, Any]]] = {}
    qc_rows_by_method: dict[str, list[dict[str, str]]] = {}
    for method in SYNTHETIC_METHODS:
        expected_registry = [row for row in registry_rows if row["method"] == method]
        if len(expected_registry) != 26:
            raise RuntimeError(f"Registry method {method} has {len(expected_registry)} candidates, expected 26")
        expected_by_id = {row["candidate_id"]: row for row in expected_registry}
        if len(expected_by_id) != 26:
            raise RuntimeError(f"Registry method {method} contains duplicate candidate IDs")

        qc_path = qc_root / f"{method}_qc" / "qc_manifest.csv"
        qc_rows = read_csv(qc_path)
        approved = [row for row in qc_rows if bool_field(row["approved"])]
        if len(approved) != 26:
            raise RuntimeError(f"QC method {method} has {len(approved)} approved candidates, expected 26")
        if {row["candidate_id"] for row in approved} != set(expected_by_id):
            raise RuntimeError(f"QC/registry candidate mismatch for {method}")
        if {row["source_id"] for row in approved} != set(real_ids):
            raise RuntimeError(f"QC method {method} is not one-to-one with the frozen real subset")

        method_records: list[dict[str, Any]] = []
        for row in sorted(approved, key=lambda item: item["candidate_id"]):
            expected = expected_by_id[row["candidate_id"]]
            if row["source_id"] != expected["source_id"]:
                raise RuntimeError(f"Source mismatch for {row['candidate_id']}")
            image_path = Path(row["candidate_image"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            expected_image = synthetic_root / method / f"{row['candidate_id']}.png"
            if image_path.resolve() != expected_image.resolve():
                raise RuntimeError(f"Unexpected candidate path for {row['candidate_id']}: {image_path}")
            if row.get("candidate_sha256") and sha256(image_path) != row["candidate_sha256"]:
                raise RuntimeError(f"Candidate hash mismatch for {row['candidate_id']}")
            label_path = qc_root / f"{method}_qc" / "approved_labels" / f"{row['candidate_id']}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            parse_label(label_path)
            method_records.append(
                {
                    "candidate_id": row["candidate_id"],
                    "source_id": row["source_id"],
                    "image": image_path,
                    "label": label_path,
                }
            )
        accepted[method] = method_records
        qc_rows_by_method[method] = approved

    for image_id in real_ids:
        image_path = exactly_one_image(source_yolo / "images" / "train", image_id)
        label_path = source_yolo / "labels" / "train" / f"{image_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        parse_label(label_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    return real_ids, accepted, qc_rows_by_method


def copy_sample(
    image_source: Path,
    label_source: Path,
    image_destination: Path,
    label_destination: Path,
) -> tuple[str, str]:
    shutil.copy2(image_source, image_destination)
    shutil.copy2(label_source, label_destination)
    image_hash = sha256(image_destination)
    label_hash = sha256(label_destination)
    if image_hash != sha256(image_source) or label_hash != sha256(label_source):
        raise RuntimeError(f"Copy verification failed for {image_source}")
    return image_hash, label_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-yolo", type=Path, default=Path("data/processed/yolo_sctd_group_aware_v1"))
    parser.add_argument(
        "--subset", type=Path, default=Path("data/splits/limited_labels_group_aware_v1/train_010.txt")
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("experiments/label_010_v1/candidate_registry.csv")
    )
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic/label_010_v1"))
    parser.add_argument("--qc-root", type=Path, default=Path("results/full/label_010_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/yolo_experiments/label_010_v1")
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    real_ids, accepted, _ = preflight(
        args.source_yolo, args.subset, args.registry, args.synthetic_root, args.qc_root
    )

    evaluation_source_records: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    evaluation_hashes: set[str] = set()
    for split in ("val", "test"):
        images = sorted(path for path in (args.source_yolo / "images" / split).iterdir() if path.is_file())
        if len(images) != 54:
            raise RuntimeError(f"Expected 54 {split} images, found {len(images)}")
        for image_path in images:
            label_path = args.source_yolo / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            parse_label(label_path)
            image_hash = sha256(image_path)
            evaluation_hashes.add(image_hash)
            evaluation_source_records[split].append(
                {
                    "sample_id": image_path.stem,
                    "image": image_path,
                    "label": label_path,
                    "image_sha256": image_hash,
                    "label_sha256": sha256(label_path),
                }
            )

    global_report: dict[str, Any] = {
        "status": "pass",
        "label_level": "10%",
        "real_training_budget": len(real_ids),
        "synthetic_budget_per_augmented_condition": 26,
        "conditions": {},
        "validation_real_only": True,
        "test_real_only": True,
        "cross_condition_evaluation_hashes_identical": True,
        "cross_partition_exact_image_hash_leakage": 0,
        "real_training_representation": "accepted group-aware YOLO source images at acquired resolution",
        "synthetic_training_representation": "QC-approved 256x256 shared-preprocessing candidates",
    }

    with tempfile.TemporaryDirectory(prefix=".label_010_v1_build_", dir=output.parent) as temporary:
        staging = Path(temporary)
        condition_eval_signatures: dict[str, dict[str, list[tuple[str, str, str]]]] = {}
        for condition in CONDITIONS:
            condition_root = staging / condition
            for split in ("train", "val", "test"):
                (condition_root / "images" / split).mkdir(parents=True)
                (condition_root / "labels" / split).mkdir(parents=True)

            manifest_rows: list[dict[str, Any]] = []
            train_class_counts: Counter[int] = Counter()
            train_hashes: set[str] = set()

            for image_id in real_ids:
                image_source = exactly_one_image(args.source_yolo / "images" / "train", image_id)
                label_source = args.source_yolo / "labels" / "train" / f"{image_id}.txt"
                image_destination = condition_root / "images" / "train" / image_source.name
                label_destination = condition_root / "labels" / "train" / f"{image_id}.txt"
                image_hash, label_hash = copy_sample(
                    image_source, label_source, image_destination, label_destination
                )
                if image_hash in evaluation_hashes:
                    raise RuntimeError(f"Exact train/evaluation image leakage for real source {image_id}")
                train_hashes.add(image_hash)
                train_class_counts.update(parse_label(label_destination))
                manifest_rows.append(
                    {
                        "sample_id": image_id,
                        "split": "train",
                        "origin": "real",
                        "method": "real",
                        "source_id": image_id,
                        "source_image": image_source.as_posix(),
                        "source_label": label_source.as_posix(),
                        "image_sha256": image_hash,
                        "label_sha256": label_hash,
                    }
                )

            if condition != "real_only":
                for record in accepted[condition]:
                    candidate_id = record["candidate_id"]
                    image_destination = condition_root / "images" / "train" / f"{candidate_id}.png"
                    label_destination = condition_root / "labels" / "train" / f"{candidate_id}.txt"
                    image_hash, label_hash = copy_sample(
                        record["image"], record["label"], image_destination, label_destination
                    )
                    if image_hash in evaluation_hashes:
                        raise RuntimeError(f"Exact synthetic/evaluation image leakage for {candidate_id}")
                    train_hashes.add(image_hash)
                    train_class_counts.update(parse_label(label_destination))
                    manifest_rows.append(
                        {
                            "sample_id": candidate_id,
                            "split": "train",
                            "origin": "synthetic",
                            "method": condition,
                            "source_id": record["source_id"],
                            "source_image": record["image"].as_posix(),
                            "source_label": record["label"].as_posix(),
                            "image_sha256": image_hash,
                            "label_sha256": label_hash,
                        }
                    )

            condition_eval_signatures[condition] = {}
            for split in ("val", "test"):
                signature: list[tuple[str, str, str]] = []
                for record in evaluation_source_records[split]:
                    image_source = record["image"]
                    label_source = record["label"]
                    image_destination = condition_root / "images" / split / image_source.name
                    label_destination = condition_root / "labels" / split / f"{image_source.stem}.txt"
                    image_hash, label_hash = copy_sample(
                        image_source, label_source, image_destination, label_destination
                    )
                    signature.append((record["sample_id"], image_hash, label_hash))
                    manifest_rows.append(
                        {
                            "sample_id": record["sample_id"],
                            "split": split,
                            "origin": "real",
                            "method": "real",
                            "source_id": record["sample_id"],
                            "source_image": image_source.as_posix(),
                            "source_label": label_source.as_posix(),
                            "image_sha256": image_hash,
                            "label_sha256": label_hash,
                        }
                    )
                condition_eval_signatures[condition][split] = signature

            dataset_yaml = (
                f"path: {(output / condition).as_posix()}\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "names:\n"
                "  0: aircraft\n"
                "  1: human\n"
                "  2: ship\n"
            )
            (condition_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
            fieldnames = [
                "sample_id",
                "split",
                "origin",
                "method",
                "source_id",
                "source_image",
                "source_label",
                "image_sha256",
                "label_sha256",
            ]
            with (condition_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(manifest_rows)

            synthetic_count = sum(row["origin"] == "synthetic" for row in manifest_rows)
            condition_report = {
                "status": "pass",
                "condition": condition,
                "train_images": 26 + synthetic_count,
                "real_train_images": 26,
                "synthetic_train_images": synthetic_count,
                "validation_images": 54,
                "test_images": 54,
                "train_objects": sum(train_class_counts.values()),
                "train_objects_per_class": {
                    CLASS_NAMES[class_id]: train_class_counts[class_id] for class_id in CLASS_NAMES
                },
                "train_unique_image_hashes": len(train_hashes),
                "train_evaluation_exact_hash_intersection": 0,
            }
            (condition_root / "build_report.json").write_text(
                json.dumps(condition_report, indent=2), encoding="utf-8"
            )
            global_report["conditions"][condition] = condition_report

        baseline_signature = condition_eval_signatures["real_only"]
        for condition in CONDITIONS[1:]:
            if condition_eval_signatures[condition] != baseline_signature:
                raise RuntimeError(f"Evaluation data differ for condition {condition}")

        (staging / "reconciliation_report.json").write_text(
            json.dumps(global_report, indent=2), encoding="utf-8"
        )
        staging.replace(output)

    print(json.dumps(global_report, indent=2))


if __name__ == "__main__":
    main()
