"""Extract individual generated panels from execution-pilot grids."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


def crop_grid_cell(grid: Path, column: int, row: int) -> Image.Image:
    with Image.open(grid) as image:
        left = 2 + column * 258
        top = 2 + row * 258
        cell = image.crop((left, top, left + 256, top + 256)).copy()
    if cell.size != (256, 256):
        raise RuntimeError(f"Grid extraction failed for {grid}: {cell.size}")
    return cell


def main() -> None:
    output = Path("results/pilots/synthetic_candidates_v1")
    output.mkdir(parents=True, exist_ok=False)
    with Path("data/processed/generative_256_v1/manifest.csv").open(encoding="utf-8") as handle:
        source_rows = {row["image_id"]: row for row in csv.DictReader(handle)}

    specifications = []
    gan_report = json.loads(Path("results/pilots/gan_v1/pilot_report.json").read_text())
    for column, image_id in enumerate(gan_report["step_log"][-1]["image_ids"]):
        specifications.append(("gan", image_id, Path("results/pilots/gan_v1/pilot_grid.png"), column, 1))
    for method, report_dir in (
        ("standard_diffusion", "standard_diffusion_v1"),
        ("grounded_diffusion", "grounded_diffusion_v1"),
    ):
        report = json.loads(Path(f"results/pilots/{report_dir}/pilot_report.json").read_text())
        image_id = report["step_log"][-1]["image_id"]
        specifications.append((method, image_id, Path(f"results/pilots/{report_dir}/pilot_grid.png"), 0, 1))

    rows = []
    for candidate_index, (method, image_id, grid, column, row) in enumerate(specifications, 1):
        candidate_id = f"pilot_{method}_{image_id}_{candidate_index:02d}"
        candidate_path = output / f"{candidate_id}.png"
        crop_grid_cell(grid, column, row).save(candidate_path)
        source = source_rows[image_id]
        rows.append({
            "candidate_id": candidate_id,
            "method": method,
            "source_id": image_id,
            "source_image": source["processed_image"],
            "source_label": source["processed_label"],
            "candidate_image": candidate_path.as_posix(),
            "extraction_grid": grid.as_posix(),
            "grid_column": column,
            "grid_row": row,
            "pilot_only": True,
        })
    with (output / "candidate_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "pass", "candidate_count": len(rows), "output": output.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
