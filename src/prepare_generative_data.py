"""Prepare one geometry-preserving input representation for every generator.

Images are converted to RGB, resized isotropically, and centred on a square
canvas. Pascal VOC boxes receive the identical scale and offset. The output is
deliberately method-neutral so GAN and diffusion comparisons cannot differ in
their source preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

CLASSES = ["aircraft", "human", "ship"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_image(directory: Path, image_id: str) -> Path:
    matches = [directory / f"{image_id}{suffix}" for suffix in IMAGE_EXTENSIONS]
    found = [path for path in matches if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one source image for {image_id}; found {found}")
    return found[0]


def read_boxes(path: Path) -> list[tuple[str, float, float, float, float]]:
    boxes = []
    for obj in ET.parse(path).getroot().findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES:
            raise ValueError(f"Unknown class {name!r} in {path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {path}")
        coords = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append((name, *coords))
    if not boxes:
        raise ValueError(f"No objects in {path}")
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/SCTD"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/group_aware_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/generative_256_v1"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fill", type=int, default=0)
    args = parser.parse_args()
    if args.size <= 0 or not 0 <= args.fill <= 255:
        raise ValueError("size must be positive and fill must be in 0..255")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")

    rows = []
    for split in ("train", "val", "test"):
        ids = [line.strip() for line in (args.split_dir / f"{split}.txt").read_text().splitlines() if line.strip()]
        image_out = args.output_dir / "images" / split
        label_out = args.output_dir / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            source = locate_image(args.dataset_root / "JPEGImages", image_id)
            xml_path = args.dataset_root / "Annotations" / f"{image_id}.xml"
            boxes = read_boxes(xml_path)
            with Image.open(source) as image:
                image = image.convert("RGB")
                width, height = image.size
                scale = min(args.size / width, args.size / height)
                resized_width = max(1, round(width * scale))
                resized_height = max(1, round(height * scale))
                resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
            left = (args.size - resized_width) // 2
            top = (args.size - resized_height) // 2
            canvas = Image.new("RGB", (args.size, args.size), color=(args.fill,) * 3)
            canvas.paste(resized, (left, top))
            output_image = image_out / f"{image_id}.png"
            canvas.save(output_image, format="PNG", optimize=False)

            label_lines = []
            transformed_boxes = []
            for name, xmin, ymin, xmax, ymax in boxes:
                txmin, tymin = xmin * scale + left, ymin * scale + top
                txmax, tymax = xmax * scale + left, ymax * scale + top
                xc = (txmin + txmax) / 2 / args.size
                yc = (tymin + tymax) / 2 / args.size
                bw = (txmax - txmin) / args.size
                bh = (tymax - tymin) / args.size
                values = (xc, yc, bw, bh)
                if not all(0 <= value <= 1 for value in values) or bw <= 0 or bh <= 0:
                    raise ValueError(f"Invalid transformed box for {image_id}: {values}")
                label_lines.append(f"{CLASSES.index(name)} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
                transformed_boxes.append({
                    "class_name": name, "xmin": txmin, "ymin": tymin,
                    "xmax": txmax, "ymax": tymax,
                })
            output_label = label_out / f"{image_id}.txt"
            output_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            rows.append({
                "image_id": image_id,
                "split": split,
                "class_name": boxes[0][0],
                "object_count": len(boxes),
                "source_image": source.as_posix(),
                "source_sha256": file_hash(source),
                "processed_image": output_image.as_posix(),
                "processed_sha256": file_hash(output_image),
                "processed_label": output_label.as_posix(),
                "original_width": width,
                "original_height": height,
                "scale": f"{scale:.12f}",
                "pad_left": left,
                "pad_top": top,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "boxes_json": json.dumps(transformed_boxes, separators=(",", ":")),
            })

    fields = list(rows[0])
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "version": "generative_256_v1",
        "method": "RGB isotropic resize with centred constant letterbox",
        "size": args.size,
        "fill_rgb": [args.fill] * 3,
        "resampling": "Pillow LANCZOS",
        "training_tensor_normalisation": "RGB uint8 [0,255] mapped to float [-1,1]",
        "box_transform": "same isotropic scale and centred offsets as image",
        "dataset_root": args.dataset_root.as_posix(),
        "split_dir": args.split_dir.as_posix(),
        "image_count": len(rows),
        "object_count": sum(row["object_count"] for row in rows),
    }
    (args.output_dir / "preprocessing_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
