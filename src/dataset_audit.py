from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_int(value: str | None) -> int | None:
    try:
        return int(float(value.strip())) if value is not None else None
    except (ValueError, AttributeError):
        return None


def parse_voc(path: Path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    xml_w = to_int(size.findtext("width")) if size is not None else None
    xml_h = to_int(size.findtext("height")) if size is not None else None
    boxes = []
    errors = []
    for index, obj in enumerate(root.findall("object"), 1):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            errors.append(f"Object {index}: missing class name or bndbox")
            continue
        coords = [to_int(box.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax")]
        if any(v is None for v in coords):
            errors.append(f"Object {index}: non-numeric bounding box")
            continue
        boxes.append((name, *coords))
    return xml_w, xml_h, boxes, errors


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SCTD images and Pascal VOC XML annotations.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/audit"))
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    images = discover(dataset_root, IMAGE_EXTENSIONS)
    xmls = discover(dataset_root, {".xml"})
    images_by_stem = defaultdict(list)
    xmls_by_stem = defaultdict(list)
    for p in images:
        images_by_stem[p.stem].append(p)
    for p in xmls:
        xmls_by_stem[p.stem].append(p)

    manifest, invalid = [], []
    class_objects, class_images = Counter(), Counter()
    hashes = defaultdict(list)

    for stem in sorted(set(images_by_stem) | set(xmls_by_stem)):
        image_list = images_by_stem.get(stem, [])
        xml_list = xmls_by_stem.get(stem, [])
        if len(image_list) != 1 or len(xml_list) != 1:
            invalid.append({
                "image_id": stem,
                "image_path": " | ".join(str(p.relative_to(dataset_root)) for p in image_list),
                "annotation_path": " | ".join(str(p.relative_to(dataset_root)) for p in xml_list),
                "issue": f"Expected one image and one XML; found {len(image_list)} image(s) and {len(xml_list)} XML(s)",
            })
        if not image_list:
            continue

        image_path = image_list[0]
        xml_path = xml_list[0] if xml_list else None
        rel_image = image_path.relative_to(dataset_root)
        file_hash = sha256_file(image_path)
        hashes[file_hash].append(image_path)

        width = height = None
        mode = ""
        image_error = ""
        try:
            with Image.open(image_path) as im:
                width, height = im.size
                mode = im.mode
        except Exception as exc:
            image_error = str(exc)
            invalid.append({"image_id": stem, "image_path": str(rel_image), "annotation_path": "", "issue": f"Image read error: {exc}"})

        xml_w = xml_h = None
        boxes = []
        if xml_path:
            try:
                xml_w, xml_h, boxes, xml_errors = parse_voc(xml_path)
                for error in xml_errors:
                    invalid.append({"image_id": stem, "image_path": str(rel_image), "annotation_path": str(xml_path.relative_to(dataset_root)), "issue": error})
            except ET.ParseError as exc:
                invalid.append({"image_id": stem, "image_path": str(rel_image), "annotation_path": str(xml_path.relative_to(dataset_root)), "issue": f"XML parse error: {exc}"})

        classes_here = set()
        valid_boxes = 0
        invalid_boxes = 0
        for index, (name, xmin, ymin, xmax, ymax) in enumerate(boxes, 1):
            classes_here.add(name)
            reasons = []
            if xmin >= xmax or ymin >= ymax:
                reasons.append("non-positive box size")
            if width is not None and height is not None and (xmin < 0 or ymin < 0 or xmax > width or ymax > height):
                reasons.append("box outside image bounds")
            if reasons:
                invalid_boxes += 1
                invalid.append({
                    "image_id": stem,
                    "image_path": str(rel_image),
                    "annotation_path": str(xml_path.relative_to(dataset_root)) if xml_path else "",
                    "issue": f"Object {index} ({name}): {'; '.join(reasons)} [{xmin}, {ymin}, {xmax}, {ymax}]",
                })
            else:
                valid_boxes += 1
                class_objects[name] += 1

        for name in classes_here:
            class_images[name] += 1

        dimension_mismatch = bool(
            width is not None and height is not None and xml_w is not None and xml_h is not None
            and (width != xml_w or height != xml_h)
        )
        if dimension_mismatch:
            invalid.append({
                "image_id": stem,
                "image_path": str(rel_image),
                "annotation_path": str(xml_path.relative_to(dataset_root)) if xml_path else "",
                "issue": f"Image/XML dimension mismatch: image={width}x{height}, xml={xml_w}x{xml_h}",
            })

        manifest.append({
            "image_id": stem,
            "image_path": str(rel_image),
            "annotation_path": str(xml_path.relative_to(dataset_root)) if xml_path else "",
            "width": width or "",
            "height": height or "",
            "mode": mode,
            "sha256": file_hash,
            "class_names": "|".join(sorted(classes_here)),
            "total_boxes": len(boxes),
            "valid_boxes": valid_boxes,
            "invalid_boxes": invalid_boxes,
            "has_xml": bool(xml_path),
            "image_read_error": image_error,
            "dimension_mismatch": dimension_mismatch,
        })

    duplicates = []
    group_number = 1
    for file_hash, paths in sorted(hashes.items()):
        if len(paths) < 2:
            continue
        group_id = f"exact_dup_{group_number:04d}"
        group_number += 1
        for path in paths:
            duplicates.append({
                "duplicate_group": group_id,
                "sha256": file_hash,
                "image_path": str(path.relative_to(dataset_root)),
                "group_size": len(paths),
            })

    class_rows = [
        {"class_name": name, "object_count": class_objects[name], "image_count": class_images[name]}
        for name in sorted(set(class_objects) | set(class_images))
    ]

    summary = {
        "dataset_root": str(dataset_root),
        "image_file_count": len(images),
        "xml_file_count": len(xmls),
        "manifest_row_count": len(manifest),
        "invalid_issue_count": len(invalid),
        "exact_duplicate_group_count": group_number - 1,
        "exact_duplicate_image_count": len(duplicates),
        "class_object_counts": dict(class_objects),
        "class_image_counts": dict(class_images),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "dataset_manifest.csv", manifest, [
        "image_id", "image_path", "annotation_path", "width", "height", "mode", "sha256",
        "class_names", "total_boxes", "valid_boxes", "invalid_boxes", "has_xml",
        "image_read_error", "dimension_mismatch"
    ])
    write_csv(output_dir / "duplicate_report.csv", duplicates, ["duplicate_group", "sha256", "image_path", "group_size"])
    write_csv(output_dir / "class_distribution.csv", class_rows, ["class_name", "object_count", "image_count"])
    write_csv(output_dir / "invalid_annotations.csv", invalid, ["image_id", "image_path", "annotation_path", "issue"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nReports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
