"""Create a labelled common-source preview grid for the four ablations."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CONDITIONS = ("full", "no_acoustic", "no_geometric", "neither")
HEADERS = ("Real source", "Full", "No acoustic", "No geometric", "Neither")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return {row["source_id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    root = Path("synthetic/label_010_v1/ablations")
    manifests = {condition: rows(root / condition / "manifest.csv") for condition in CONDITIONS}
    source_ids = set(manifests[CONDITIONS[0]])
    if any(set(manifests[condition]) != source_ids for condition in CONDITIONS[1:]):
        raise RuntimeError("Ablation manifests do not share identical sources")
    by_class = defaultdict(list)
    for source_id, row in manifests[CONDITIONS[0]].items():
        by_class[row["class_name"]].append(source_id)
    selected = [source_id for class_name in ("aircraft", "human", "ship") for source_id in sorted(by_class[class_name])[:2]]
    if len(selected) != 6:
        raise RuntimeError("Expected two preview sources per class")

    cell, header_height, label_height, gap = 256, 48, 28, 8
    width = len(HEADERS) * cell + (len(HEADERS) - 1) * gap
    height = header_height + len(selected) * (cell + label_height + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=18)
    for column, header in enumerate(HEADERS):
        x = column * (cell + gap)
        draw.text((x + cell / 2, 18), header, fill="black", font=font, anchor="mm")
    for row_index, source_id in enumerate(selected):
        record = manifests["full"][source_id]
        paths = [Path(record["source_image"])] + [Path(manifests[condition][source_id]["candidate_image"]) for condition in CONDITIONS]
        y = header_height + row_index * (cell + label_height + gap)
        for column, path in enumerate(paths):
            with Image.open(path) as image:
                panel = image.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(panel, (column * (cell + gap), y))
        draw.text((4, y + cell + 5), f"{record['class_name']} / {source_id}", fill="black", font=font)
    output = Path("results/ablations/label_010_v1/figures")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "common_source_ablation_preview.png"
    canvas.save(destination, dpi=(300, 300))
    report = {
        "status": "pass", "output": destination.as_posix(), "rows": len(selected),
        "columns": list(HEADERS), "sources": selected,
        "selection": "first two ordered sources per class; no test images",
    }
    (output / "preview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
