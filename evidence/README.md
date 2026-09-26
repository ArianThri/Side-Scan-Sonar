# Evidence package

This directory contains lightweight, machine-readable evidence copied from the
executed workspace. It is included so dissertation tables and claims can be
traced without publishing datasets or model checkpoints.

- `dataset_audit/` — acquired-copy counts, class distribution and exact hashes.
- `splits/` — accepted group-aware split and nested subset identifiers.
- `training/` — GAN/diffusion logs and reports without weights.
- `generation/` — provenance manifests and summary reports without images.
- `annotation_qc/` — accepted QC reports plus the preserved rejected GAN run.
- `detectors/` — primary and ablation summaries, final reports and curves.
- `quality/` — FID, MMD, LPIPS, aligned SSIM/PSNR and per-sample tables.
- `resource_checks/` — memory feasibility and sampler-selection records.
- `figures/` — dissertation figures generated from accepted evidence.

Some evidence fields retain absolute paths from the original D-drive workspace.
Those strings document where the experiments ran; they are not expected to
exist on another machine. Reports must not be edited merely to make those paths
portable because that would alter the archived evidence.

