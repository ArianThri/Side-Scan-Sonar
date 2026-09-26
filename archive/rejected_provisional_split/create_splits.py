"""Create a reproducible class-aware train/validation/test split for SCTD.

Version 1 deliberately uses the simplest suitable method for the current dataset:
image-level stratified splitting by target class. It does not perform duplicate/
near-duplicate grouping. That can be added later only if needed.

Expected SCTD layout:
    data/raw/SCTD/JPEGImages/
    data/raw/SCTD/Annotations/

Outputs:
    data/splits/class_aware_v1/train.txt
    data/splits/class_aware_v1/val.txt
    data/splits/class_aware_v1/test.txt
    data/splits/class_aware_v1/split_manifest.csv
    data/splits/class_aware_v1/split_summary.csv
    data/splits/class_aware_v1/split_config.json
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create class-aware SCTD splits.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/SCTD"),
        help="Path containing JPEGImages/ and Annotations/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/class_aware_v1"),
        help="Directory in which split files will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("All split ratios must be > 0.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {sum(ratios):.6f}."
        )


def discover_images(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_id = path.stem
            if image_id in images:
                raise RuntimeError(f"Duplicate image stem found: {image_id}")
            images[image_id] = path
    return images


def read_annotation_class(xml_path: Path) -> tuple[str, int]:
    """Return the unique image-level target class and number of objects.

    SCTD's audit indicates each image belongs to one of aircraft/human/ship,
    although an image can contain more than one object of the same class.
    If a future file contains multiple distinct target classes, this simple
    Version-1 splitter stops rather than silently forcing a primary class.
    """
    root = ET.parse(xml_path).getroot()
    names = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is not None and name.strip():
            names.append(name.strip().lower())

    if not names:
        raise RuntimeError(f"No object class found in annotation: {xml_path}")

    unique_classes = sorted(set(names))
    if len(unique_classes) != 1:
        raise RuntimeError(
            "Version-1 class-aware splitting assumes one image-level class. "
            f"{xml_path.name} contains multiple classes: {unique_classes}. "
            "Use a multi-label stratification method for this case."
        )

    return unique_classes[0], len(names)


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    image_dir = dataset_root / "JPEGImages"
    annotation_dir = dataset_root / "Annotations"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    images = discover_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    rows = []
    missing_annotations = []

    for image_id, image_path in images.items():
        xml_path = annotation_dir / f"{image_id}.xml"
        if not xml_path.exists():
            missing_annotations.append(image_id)
            continue

        class_name, object_count = read_annotation_class(xml_path)
        rows.append(
            {
                "image_id": image_id,
                "class_name": class_name,
                "object_count": object_count,
                "image_path": image_path.as_posix(),
                "annotation_path": xml_path.as_posix(),
            }
        )

    if missing_annotations:
        preview = ", ".join(missing_annotations[:10])
        raise RuntimeError(
            f"{len(missing_annotations)} image(s) are missing XML annotations. "
            f"Examples: {preview}"
        )

    manifest = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    return manifest


def class_aware_split(
    manifest: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Two-stage image-level stratified split preserving class proportions."""
    train_df, temp_df = train_test_split(
        manifest,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=manifest["class_name"],
        shuffle=True,
    )

    # Fraction of the temporary set that should become the final test set.
    test_fraction_of_temp = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_fraction_of_temp,
        random_state=seed,
        stratify=temp_df["class_name"],
        shuffle=True,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return result.sort_values(["split", "image_id"]).reset_index(drop=True)


def verify_split(original: pd.DataFrame, split_df: pd.DataFrame) -> None:
    original_ids = set(original["image_id"])
    split_ids = set(split_df["image_id"])

    if original_ids != split_ids:
        missing = sorted(original_ids - split_ids)
        extra = sorted(split_ids - original_ids)
        raise RuntimeError(
            f"Split coverage failed. Missing={missing[:10]}, extra={extra[:10]}"
        )

    if split_df["image_id"].duplicated().any():
        duplicates = split_df.loc[
            split_df["image_id"].duplicated(keep=False), "image_id"
        ].tolist()
        raise RuntimeError(f"Images assigned more than once: {duplicates[:10]}")

    if set(split_df["split"]) != {"train", "val", "test"}:
        raise RuntimeError("Expected exactly train, val, and test splits.")


def make_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_order = ["train", "val", "test"]
    classes = sorted(split_df["class_name"].unique())

    for split in split_order:
        subset = split_df[split_df["split"] == split]
        counts = Counter(subset["class_name"])
        row = {
            "split": split,
            "total_images": len(subset),
            "total_objects": int(subset["object_count"].sum()),
        }
        for class_name in classes:
            row[f"{class_name}_images"] = counts.get(class_name, 0)
            row[f"{class_name}_objects"] = int(
                subset.loc[subset["class_name"] == class_name, "object_count"].sum()
            )
        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(
    split_df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        ids = (
            split_df.loc[split_df["split"] == split, "image_id"]
            .sort_values()
            .tolist()
        )
        (output_dir / f"{split}.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )

    split_df.to_csv(output_dir / "split_manifest.csv", index=False)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    (output_dir / "split_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

def print_report(manifest: pd.DataFrame, summary: pd.DataFrame, config: dict) -> None:
    print("\nSCTD class-aware split complete")
    print("=" * 65)

    print(f"Images: {len(manifest)}")
    print(f"Seed:   {config['seed']}")
    print(
        "Ratios: "
        f"train={config['train_ratio']:.2f}, "
        f"val={config['val_ratio']:.2f}, "
        f"test={config['test_ratio']:.2f}"
    )

    print("\nOriginal image-level class distribution:")
    print(manifest["class_name"].value_counts().sort_index().to_string())

    print("\nSplit summary:")
    print(
        f"{'Split':<8}"
        f"{'Images':>8}"
        f"{'Objects':>10}"
        f"{'Aircraft':>12}"
        f"{'Human':>9}"
        f"{'Ship':>9}"
    )
    print("-" * 56)

    for _, row in summary.iterrows():
        print(
            f"{row['split']:<8}"
            f"{int(row['total_images']):>8}"
            f"{int(row['total_objects']):>10}"
            f"{int(row['aircraft_objects']):>12}"
            f"{int(row['human_objects']):>9}"
            f"{int(row['ship_objects']):>9}"
        )


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    random.seed(args.seed)

    manifest = build_manifest(args.dataset_root)
    split_df = class_aware_split(
        manifest=manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    verify_split(manifest, split_df)
    summary = make_summary(split_df)

    config = {
        "method": "image-level class-aware stratified split",
        "version": "class_aware_v1",
        "dataset_root": args.dataset_root.as_posix(),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "duplicate_grouping": False,
        "note": (
            "Version 1 intentionally uses the simplest class-aware split. "
            "Duplicate/group-aware constraints can be added later if required."
        ),
    }

    save_outputs(split_df, summary, args.output_dir, config)
    print_report(manifest, summary, config)
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
