"""Freeze immutable inputs and equal synthetic budgets for a full experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output = Path("experiments/label_010_v1")
    output.mkdir(parents=True, exist_ok=False)
    ids_file = Path("data/splits/limited_labels_group_aware_v1/train_010.txt")
    source_ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    with Path("data/processed/generative_256_v1/manifest.csv").open(encoding="utf-8") as handle:
        manifest = {row["image_id"]: row for row in csv.DictReader(handle)}
    missing = set(source_ids) - set(manifest)
    if missing:
        raise RuntimeError(f"Missing generative manifest rows: {sorted(missing)}")
    class_counts = Counter(manifest[image_id]["class_name"] for image_id in source_ids)
    methods = ["traditional", "gan", "standard_diffusion", "grounded_diffusion"]
    candidates = []
    for method_index, method in enumerate(methods):
        for source_index, image_id in enumerate(source_ids):
            candidates.append({
                "candidate_id": f"l010_{method}_{image_id}",
                "method": method,
                "source_id": image_id,
                "class_name": manifest[image_id]["class_name"],
                "seed": 42 + method_index * 10000 + source_index,
                "source_image": manifest[image_id]["processed_image"],
                "source_label": manifest[image_id]["processed_label"],
            })
    with (output / "candidate_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    tracked = [
        Path("data/splits/group_aware_v1/split_manifest.csv"), ids_file,
        Path("data/processed/generative_256_v1/manifest.csv"),
        Path("data/processed/generative_256_v1/preprocessing_config.json"),
        Path("configs/experiments.yaml"), Path("configs/models.yaml"),
    ]
    registry = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "label_010_v1",
        "label_level_percent": 10,
        "seed": 42,
        "real_training_images": len(source_ids),
        "class_counts": dict(sorted(class_counts.items())),
        "synthetic_ratio_to_real": 1.0,
        "synthetic_images_per_method": len(source_ids),
        "methods": methods,
        "equal_budget_verified": all(
            sum(row["method"] == method for row in candidates) == len(source_ids)
            for method in methods
        ),
        "real_validation_and_test_only": True,
        "input_sha256": {path.as_posix(): sha256(path) for path in tracked},
        "candidate_registry_sha256": sha256(output / "candidate_registry.csv"),
        "note": "First controlled full-run gate. Later label levels require separate registries.",
    }
    (output / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
