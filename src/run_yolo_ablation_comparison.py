"""Run the four-condition grounded ablation through the frozen YOLO protocol."""

from __future__ import annotations

import sys

import run_yolo_full_comparison as runner


runner.CONDITIONS = ("full", "no_acoustic", "no_geometric", "neither")
runner.__doc__ = __doc__
runner.RUN_PURPOSE = "controlled grounded-ablation YOLO11n detector condition"
runner.COMPARISON_PURPOSE = "single-seed full/no-acoustic/no-geometric/neither grounded ablation"


def main() -> None:
    if "--dataset-root" not in sys.argv:
        sys.argv.extend(["--dataset-root", "data/processed/yolo_ablation/label_010_v1"])
    if "--output-root" not in sys.argv:
        sys.argv.extend(["--output-root", "results/ablations/label_010_v1/detectors"])
    runner.main()


if __name__ == "__main__":
    main()
