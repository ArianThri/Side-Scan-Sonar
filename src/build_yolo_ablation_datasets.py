"""Build four leakage-checked YOLO datasets for the grounded ablation study."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from build_yolo_experiment_datasets import (
    CLASS_NAMES, copy_sample, exactly_one_image, parse_label, read_csv, sha256,
)


CONDITIONS = ("full", "no_acoustic", "no_geometric", "neither")


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-yolo", type=Path, default=Path("data/processed/yolo_sctd_group_aware_v1"))
    parser.add_argument("--subset", type=Path, default=Path("data/splits/limited_labels_group_aware_v1/train_010.txt"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic/label_010_v1/ablations"))
    parser.add_argument("--qc-root", type=Path, default=Path("results/ablations/label_010_v1/synthetic_qc"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/yolo_ablation/label_010_v1"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    protocol = json.loads(
        Path("experiments/label_010_v1/grounded_ablation_protocol.json").read_text(
            encoding="utf-8"
        )
    )

    real_ids = [line.strip() for line in args.subset.read_text().splitlines() if line.strip()]
    if len(real_ids) != 26 or len(set(real_ids)) != 26:
        raise RuntimeError("Expected 26 unique frozen real IDs")

    accepted = {}
    sampling_signatures = {}
    for condition in CONDITIONS:
        manifest = read_csv(args.synthetic_root / condition / "manifest.csv")
        qc_manifest = read_csv(args.qc_root / condition / "qc_manifest.csv")
        approved = [row for row in qc_manifest if as_bool(row["approved"])]
        if len(manifest) != 26 or len(approved) != 26:
            raise RuntimeError(f"{condition}: expected 26 generated and 26 approved candidates")
        manifest_by_id = {row["candidate_id"]: row for row in manifest}
        if len(manifest_by_id) != 26 or {row["candidate_id"] for row in approved} != set(manifest_by_id):
            raise RuntimeError(f"{condition}: manifest/QC candidate mismatch")
        if {row["source_id"] for row in approved} != set(real_ids):
            raise RuntimeError(f"{condition}: source allocation differs from the frozen subset")
        expected_components = protocol["conditions"][condition]
        ordered_manifest = sorted(manifest, key=lambda item: item["source_id"])
        signature = []
        for index, row in enumerate(ordered_manifest):
            if row["ablation_condition"] != condition:
                raise RuntimeError(f"{condition}: manifest condition mismatch")
            for component in ("geometric", "range", "speckle"):
                if as_bool(row[component]) != expected_components[component]:
                    raise RuntimeError(f"{condition}: {component} switch differs from frozen protocol")
            expected_seed = 40042 + index
            if int(row["seed"]) != expected_seed:
                raise RuntimeError(f"{condition}: expected common seed {expected_seed} at index {index}")
            if (
                float(row["guidance_scale"]) != 1.0
                or float(row["source_strength"]) != 0.30
                or int(row["inference_steps"]) != 50
            ):
                raise RuntimeError(f"{condition}: sampling controls differ from frozen protocol")
            signature.append(
                (
                    row["source_id"], int(row["seed"]), float(row["guidance_scale"]),
                    float(row["source_strength"]), int(row["inference_steps"]),
                )
            )
        sampling_signatures[condition] = signature
        records = []
        for row in sorted(approved, key=lambda item: item["source_id"]):
            source = manifest_by_id[row["candidate_id"]]
            image = Path(source["candidate_image"])
            label = args.qc_root / condition / "approved_labels" / f"{row['candidate_id']}.txt"
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"Missing approved pair: {image}, {label}")
            if sha256(image) != source["candidate_sha256"]:
                raise RuntimeError(f"Candidate hash mismatch: {image}")
            parse_label(label)
            records.append({"candidate_id": row["candidate_id"], "source_id": row["source_id"], "image": image, "label": label})
        accepted[condition] = records

    first_sampling_signature = sampling_signatures[CONDITIONS[0]]
    if any(
        sampling_signatures[condition] != first_sampling_signature
        for condition in CONDITIONS[1:]
    ):
        raise RuntimeError("Ablation manifests do not share the same source/seed/sampling signature")

    evaluation = {"val": [], "test": []}
    evaluation_hashes = set()
    for split in evaluation:
        images = sorted(path for path in (args.source_yolo / "images" / split).iterdir() if path.is_file())
        if len(images) != 54:
            raise RuntimeError(f"Expected 54 {split} images")
        for image in images:
            label = args.source_yolo / "labels" / split / f"{image.stem}.txt"
            parse_label(label)
            record = {"id": image.stem, "image": image, "label": label, "image_hash": sha256(image), "label_hash": sha256(label)}
            evaluation[split].append(record); evaluation_hashes.add(record["image_hash"])

    output.parent.mkdir(parents=True, exist_ok=True)
    global_report = {
        "status": "pass", "purpose": "four-condition grounded ablation YOLO datasets",
        "conditions": {}, "real_training_budget": 26, "synthetic_budget_per_condition": 26,
        "validation_real_only": True, "test_real_only": True,
        "frozen_component_switches_match": True,
        "common_random_number_signature_identical": True,
        "cross_condition_evaluation_hashes_identical": True,
        "cross_partition_exact_image_hash_leakage": 0,
    }
    with tempfile.TemporaryDirectory(prefix=".ablation_yolo_", dir=output.parent) as temporary:
        staging = Path(temporary)
        signatures = {}
        for condition in CONDITIONS:
            root = staging / condition
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            manifest_rows, train_hashes = [], set()
            class_counts: Counter[int] = Counter()
            for image_id in real_ids:
                image = exactly_one_image(args.source_yolo / "images" / "train", image_id)
                label = args.source_yolo / "labels" / "train" / f"{image_id}.txt"
                image_hash, label_hash = copy_sample(image, label, root / "images" / "train" / image.name, root / "labels" / "train" / label.name)
                if image_hash in evaluation_hashes:
                    raise RuntimeError(f"Real train/evaluation leakage: {image_id}")
                train_hashes.add(image_hash); class_counts.update(parse_label(label))
                manifest_rows.append({"sample_id": image_id, "split": "train", "origin": "real", "method": "real", "source_id": image_id, "image_sha256": image_hash, "label_sha256": label_hash})
            for record in accepted[condition]:
                candidate_id = record["candidate_id"]
                image_hash, label_hash = copy_sample(
                    record["image"], record["label"], root / "images" / "train" / f"{candidate_id}.png", root / "labels" / "train" / f"{candidate_id}.txt"
                )
                if image_hash in evaluation_hashes:
                    raise RuntimeError(f"Synthetic train/evaluation leakage: {candidate_id}")
                train_hashes.add(image_hash); class_counts.update(parse_label(record["label"]))
                manifest_rows.append({"sample_id": candidate_id, "split": "train", "origin": "synthetic", "method": condition, "source_id": record["source_id"], "image_sha256": image_hash, "label_sha256": label_hash})
            signatures[condition] = {}
            for split, records in evaluation.items():
                split_signature = []
                for record in records:
                    image_hash, label_hash = copy_sample(
                        record["image"], record["label"], root / "images" / split / record["image"].name, root / "labels" / split / record["label"].name
                    )
                    split_signature.append((record["id"], image_hash, label_hash))
                    manifest_rows.append({"sample_id": record["id"], "split": split, "origin": "real", "method": "real", "source_id": record["id"], "image_sha256": image_hash, "label_sha256": label_hash})
                signatures[condition][split] = split_signature
            (root / "dataset.yaml").write_text(
                f"path: {(output / condition).as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: aircraft\n  1: human\n  2: ship\n",
                encoding="utf-8",
            )
            with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0])); writer.writeheader(); writer.writerows(manifest_rows)
            report = {
                "status": "pass", "condition": condition, "train_images": 52,
                "real_train_images": 26, "synthetic_train_images": 26,
                "validation_images": 54, "test_images": 54,
                "train_objects": sum(class_counts.values()),
                "train_objects_per_class": {CLASS_NAMES[index]: class_counts[index] for index in CLASS_NAMES},
                "train_unique_image_hashes": len(train_hashes),
                "train_evaluation_exact_hash_intersection": 0,
            }
            (root / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            global_report["conditions"][condition] = report
        first = signatures[CONDITIONS[0]]
        if any(signatures[condition] != first for condition in CONDITIONS[1:]):
            raise RuntimeError("Ablation validation/test content differs across conditions")
        (staging / "reconciliation_report.json").write_text(json.dumps(global_report, indent=2), encoding="utf-8")
        staging.replace(output)
    print(json.dumps(global_report, indent=2))


if __name__ == "__main__":
    main()
