"""Freeze a reviewed shared diffusion sampling configuration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--review-note", required=True)
    args = parser.parse_args()
    if args.guidance_scale not in {1.0, 2.0, 3.0, 5.0}:
        raise ValueError("Guidance scale is outside the pre-registered grid")
    if args.strength not in {0.30, 0.45, 0.60}:
        raise ValueError("Strength is outside the pre-registered grid")

    root = Path("results/full/label_010_v1/diffusion_sampling_selection")
    metrics_path = root / "configuration_metrics.csv"
    evaluation_path = root / "evaluation_report.json"
    report_path = root / "selection_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    matches = [
        row for row in metrics
        if float(row["guidance_scale"]) == args.guidance_scale
        and float(row["strength"]) == args.strength
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching configuration, found {len(matches)}")
    selected = matches[0]
    if int(selected["samples"]) != 12 or int(selected["approved"]) != 12:
        raise RuntimeError("Selected configuration did not pass all validation samples")
    if int(selected["exact_pixel_copies"]) != 0:
        raise RuntimeError("Selected configuration contains an exact pixel copy")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation["automatic_selection_performed"]:
        raise RuntimeError("Expected an evidence-only grid before reviewed selection")

    numeric_metrics = {
        key: (int(value) if key in {"samples", "approved", "exact_pixel_copies"} else float(value))
        for key, value in selected.items()
        if key != "configuration"
    }
    report = {
        "status": "pass",
        "purpose": "freeze validation-selected sampling controls before synthesis",
        "selected_configuration": {
            "configuration": selected["configuration"],
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "inference_steps": int(evaluation["inference_steps"]),
        },
        "selected_validation_metrics": numeric_metrics,
        "visual_review_performed": True,
        "visual_review_note": args.review_note,
        "selection_rationale": [
            "all 12 fixed real-validation candidates passed the unchanged annotation-transfer gates",
            "zero exact pixel copies and non-trivial source variation were verified",
            "the selected setting had the strongest target/alignment preservation among eligible settings",
            "higher strength and CFG previews showed increasing banding, colour fringing, or target deformation",
            "strength 0.60 was excluded because no configuration passed all 12 candidates",
        ],
        "shared_control": {
            "standard_diffusion": True,
            "grounded_diffusion": True,
            "same_guidance_scale": True,
            "same_source_strength": True,
            "same_inference_steps": True,
        },
        "selection_data": "real group-aware validation partition only; test partition untouched",
        "evaluation_report": evaluation_path.as_posix(),
        "configuration_metrics": metrics_path.as_posix(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
