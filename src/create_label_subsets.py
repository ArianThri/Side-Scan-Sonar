"""Create nested, class-aware limited-label subsets from the fixed training split.

The subsets are nested: every image selected at a lower label percentage is also
present at all higher percentages.  Selection is deterministic within class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def image_class(annotation_dir: Path, image_id: str) -> str:
    names = {
        (obj.findtext("name") or "").strip().lower()
        for obj in ET.parse(annotation_dir / f"{image_id}.xml").getroot().findall("object")
    }
    names.discard("")
    if len(names) != 1:
        raise ValueError(f"{image_id} has {len(names)} image-level classes: {sorted(names)}")
    return next(iter(names))


def parse_levels(value: str) -> list[int]:
    levels = sorted({int(item.strip()) for item in value.split(",")})
    if not levels or any(level <= 0 or level > 100 for level in levels):
        raise argparse.ArgumentTypeError("levels must be comma-separated integers in 1..100")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create nested class-aware label subsets.")
    parser.add_argument("--train-file", type=Path, default=Path("data/splits/class_aware_v1/train.txt"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/raw/SCTD/Annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/limited_labels_v1"))
    parser.add_argument("--levels", type=parse_levels, default=parse_levels("10,25,50,100"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ids = [line.strip() for line in args.train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_class: dict[str, list[str]] = {}
    for image_id in ids:
        by_class.setdefault(image_class(args.annotation_dir, image_id), []).append(image_id)

    rng = random.Random(args.seed)
    for class_ids in by_class.values():
        class_ids.sort()
        rng.shuffle(class_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    previous: set[str] = set()
    for level in args.levels:
        selected: set[str] = set()
        for class_name, class_ids in sorted(by_class.items()):
            count = len(class_ids) if level == 100 else max(1, math.ceil(len(class_ids) * level / 100))
            chosen = class_ids[:count]
            selected.update(chosen)
            rows.append({"label_level": level, "class_name": class_name, "image_count": len(chosen)})
        if not previous.issubset(selected):
            raise RuntimeError("Nested-subset invariant failed")
        previous = selected
        (args.output_dir / f"train_{level:03d}.txt").write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")

    with (args.output_dir / "subset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label_level", "class_name", "image_count"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "subset_config.json").write_text(json.dumps({
        "method": "nested image-level class-aware sampling",
        "source_train_file": args.train_file.as_posix(),
        "levels_percent": args.levels,
        "seed": args.seed,
        "class_counts": dict(sorted(Counter({k: len(v) for k, v in by_class.items()}).items())),
        "assumption": "10, 25, 50 and 100 percent were selected because the methodology did not state label fractions."
    }, indent=2), encoding="utf-8")

    print(f"Created {len(args.levels)} nested subsets in {args.output_dir}")


if __name__ == "__main__":
    main()
