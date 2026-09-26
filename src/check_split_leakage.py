"""Report exact-image duplicate groups that cross fixed data partitions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplicate-report", type=Path, default=Path("data/audit/duplicate_report.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/class_aware_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/audit/split_leakage.csv"))
    args = parser.parse_args()
    membership = {}
    for split in ("train", "val", "test"):
        for image_id in (args.split_dir / f"{split}.txt").read_text().splitlines():
            membership[image_id.strip()] = split
    groups = defaultdict(list)
    with args.duplicate_report.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups[row["duplicate_group"]].append(row)
    rows = []
    for group, items in sorted(groups.items()):
        splits = {membership[Path(item["image_path"]).stem] for item in items}
        if len(splits) > 1:
            rows.append({"duplicate_group": group, "splits": "|".join(sorted(splits)),
                         "image_ids": "|".join(Path(item["image_path"]).stem for item in items)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["duplicate_group", "splits", "image_ids"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Cross-partition exact-duplicate groups: {len(rows)}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
