"""Convert fixed SCTD partitions from Pascal VOC XML to a YOLO dataset."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

CLASSES = ["aircraft", "human", "ship"]
EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def locate_image(image_dir: Path, image_id: str) -> Path:
    matches = [image_dir / f"{image_id}{suffix}" for suffix in EXTENSIONS]
    found = [path for path in matches if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one image for {image_id}; found {found}")
    return found[0]


def convert(xml_path: Path, width: int, height: int) -> list[str]:
    lines = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing box in {xml_path}")
        xmin, ymin, xmax, ymax = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        xc, yc = ((xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height)
        bw, bh = ((xmax - xmin) / width, (ymax - ymin) / height)
        if not all(0 <= value <= 1 for value in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid normalised box in {xml_path}: {(xc, yc, bw, bh)}")
        lines.append(f"{CLASSES.index(name)} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SCTD VOC annotations to YOLO format.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/SCTD"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/class_aware_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/yolo_sctd"))
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix regenerated data with existing files in {args.output_dir}. "
            "Choose a new empty output directory."
        )

    for split in ("train", "val", "test"):
        ids = [x.strip() for x in (args.split_dir / f"{split}.txt").read_text().splitlines() if x.strip()]
        image_out, label_out = args.output_dir / "images" / split, args.output_dir / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            source = locate_image(args.dataset_root / "JPEGImages", image_id)
            with Image.open(source) as image:
                lines = convert(args.dataset_root / "Annotations" / f"{image_id}.xml", *image.size)
            shutil.copy2(source, image_out / source.name)
            (label_out / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {len(ids)} images")

    yaml = "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: aircraft\n  1: human\n  2: ship\n"
    (args.output_dir / "dataset.yaml").write_text(yaml, encoding="utf-8")


if __name__ == "__main__":
    main()
