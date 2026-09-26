"""Create a deterministic duplicate-group-aware, class-aware SCTD split.

Every set of byte-identical images is treated as an indivisible group. Class
targets are selected by constrained apportionment, then a dynamic programme
assigns whole groups while meeting the exact requested partition sizes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_class(path: Path) -> tuple[str, int]:
    names = [
        (obj.findtext("name") or "").strip().lower()
        for obj in ET.parse(path).getroot().findall("object")
    ]
    if not names or len(set(names)) != 1:
        raise ValueError(f"Expected one non-empty image-level class in {path}: {names}")
    return names[0], len(names)


def load_records(dataset_root: Path) -> list[dict]:
    image_dir = dataset_root / "JPEGImages"
    annotation_dir = dataset_root / "Annotations"
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    records = []
    for image_path in images:
        xml_path = annotation_dir / f"{image_path.stem}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {xml_path}")
        class_name, object_count = annotation_class(xml_path)
        records.append({
            "image_id": image_path.stem,
            "class_name": class_name,
            "object_count": object_count,
            "sha256": sha256_file(image_path),
            "image_path": image_path.as_posix(),
            "annotation_path": xml_path.as_posix(),
        })
    xml_ids = {path.stem for path in annotation_dir.glob("*.xml")}
    image_ids = {record["image_id"] for record in records}
    if xml_ids != image_ids:
        raise RuntimeError(
            f"Image/XML coverage mismatch: images_only={sorted(image_ids-xml_ids)[:10]}, "
            f"xml_only={sorted(xml_ids-image_ids)[:10]}"
        )
    return records


def apportion_class_targets(
    class_counts: dict[str, int], split_totals: dict[str, int], ratios: dict[str, float]
) -> dict[str, dict[str, int]]:
    """Use constrained largest remainders to satisfy class and split totals."""
    result = {
        class_name: {
            split: math.floor(count * ratios[split]) for split in SPLITS
        }
        for class_name, count in sorted(class_counts.items())
    }
    row_deficit = {
        class_name: class_counts[class_name] - sum(row.values())
        for class_name, row in result.items()
    }
    col_deficit = {
        split: split_totals[split] - sum(row[split] for row in result.values())
        for split in SPLITS
    }
    awarded: set[tuple[str, str]] = set()
    while sum(row_deficit.values()):
        candidates = []
        for class_name, remaining in row_deficit.items():
            if remaining <= 0:
                continue
            for split in SPLITS:
                if col_deficit[split] <= 0 or (class_name, split) in awarded:
                    continue
                ideal = class_counts[class_name] * ratios[split]
                remainder = ideal - math.floor(ideal)
                candidates.append((remainder, class_name, split))
        if not candidates:
            raise RuntimeError("No constrained apportionment satisfies requested totals")
        _, class_name, split = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        result[class_name][split] += 1
        awarded.add((class_name, split))
        row_deficit[class_name] -= 1
        col_deficit[split] -= 1
    if any(col_deficit.values()):
        raise RuntimeError(f"Column apportionment failed: {col_deficit}")
    return result


def assign_groups_exact(
    groups: list[list[dict]], train_target: int, val_target: int, seed: int
) -> dict[str, str]:
    """Assign indivisible groups to exact train/val targets using dynamic programming."""
    ordered = sorted(groups, key=lambda group: tuple(item["image_id"] for item in group))
    random.Random(seed).shuffle(ordered)
    states = {(0, 0)}
    predecessor_layers: list[dict[tuple[int, int], tuple[tuple[int, int], str]]] = []
    for group in ordered:
        size = len(group)
        next_states = set()
        predecessors = {}
        for state in states:
            for split, delta in (("train", (size, 0)), ("val", (0, size)), ("test", (0, 0))):
                candidate = (state[0] + delta[0], state[1] + delta[1])
                if candidate[0] <= train_target and candidate[1] <= val_target:
                    if candidate not in next_states:
                        next_states.add(candidate)
                        predecessors[candidate] = (state, split)
        states = next_states
        predecessor_layers.append(predecessors)
    target = (train_target, val_target)
    if target not in states:
        sizes = sorted((len(group) for group in groups), reverse=True)
        raise RuntimeError(f"Cannot meet group-aware targets {target}; group sizes={sizes}")
    choices = []
    state = target
    for predecessors in reversed(predecessor_layers):
        previous, split = predecessors[state]
        choices.append(split)
        state = previous
    choices.reverse()
    membership = {}
    for group, split in zip(ordered, choices):
        for record in group:
            membership[record["image_id"]] = split
    return membership


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/SCTD"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/group_aware_v1"))
    parser.add_argument("--train-count", type=int, default=249)
    parser.add_argument("--val-count", type=int, default=54)
    parser.add_argument("--test-count", type=int, default=54)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(args.dataset_root)
    split_totals = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    if sum(split_totals.values()) != len(records):
        raise ValueError(f"Requested totals {split_totals} do not cover {len(records)} images")
    ratios = {split: count / len(records) for split, count in split_totals.items()}
    class_counts = Counter(record["class_name"] for record in records)
    targets = apportion_class_targets(dict(class_counts), split_totals, ratios)

    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    groups_by_class = defaultdict(list)
    for group in by_hash.values():
        classes = {record["class_name"] for record in group}
        if len(classes) != 1:
            raise RuntimeError(
                f"Byte-identical group has inconsistent classes: "
                f"{[(r['image_id'], r['class_name']) for r in group]}"
            )
        groups_by_class[next(iter(classes))].append(group)

    membership = {}
    for class_index, class_name in enumerate(sorted(class_counts)):
        membership.update(assign_groups_exact(
            groups_by_class[class_name],
            targets[class_name]["train"],
            targets[class_name]["val"],
            args.seed + class_index,
        ))
    for record in records:
        record["split"] = membership[record["image_id"]]

    leakage = []
    for digest, group in by_hash.items():
        group_splits = {membership[record["image_id"]] for record in group}
        if len(group_splits) > 1:
            leakage.append((digest, sorted(group_splits)))
    if leakage:
        raise RuntimeError(f"Duplicate leakage invariant failed: {leakage[:5]}")
    actual_totals = Counter(record["split"] for record in records)
    if dict(actual_totals) != split_totals:
        raise RuntimeError(f"Split-size invariant failed: {actual_totals} != {split_totals}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        ids = sorted(record["image_id"] for record in records if record["split"] == split)
        (args.output_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    fields = ["image_id", "class_name", "object_count", "sha256", "split", "image_path", "annotation_path"]
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["image_id"]))
    summary = []
    for split in SPLITS:
        subset = [record for record in records if record["split"] == split]
        row = {"split": split, "total_images": len(subset), "total_objects": sum(r["object_count"] for r in subset)}
        for class_name in sorted(class_counts):
            selected = [r for r in subset if r["class_name"] == class_name]
            row[f"{class_name}_images"] = len(selected)
            row[f"{class_name}_objects"] = sum(r["object_count"] for r in selected)
        summary.append(row)
    with (args.output_dir / "split_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    config = {
        "method": "exact-duplicate-group-aware class-aware split",
        "version": "group_aware_v1",
        "dataset_root": args.dataset_root.as_posix(),
        "split_totals": split_totals,
        "ratios": ratios,
        "seed": args.seed,
        "hash_algorithm": "SHA-256",
        "class_targets": targets,
        "exact_duplicate_groups": sum(len(group) > 1 for group in by_hash.values()),
        "cross_partition_exact_duplicate_groups": 0,
    }
    (args.output_dir / "split_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"split_summary": summary, "config": config}, indent=2))


if __name__ == "__main__":
    main()
