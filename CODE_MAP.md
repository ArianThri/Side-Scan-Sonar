# Code map

The accepted scripts remain together in `src/` because several executable
scripts import neighbouring modules by filename. Moving them into nested Python
packages without a tested refactor would break the executed interface.

## Dataset preparation and experiment registration

- `dataset_audit.py` — pair and validate images/XML files, class counts and exact hashes.
- `create_group_aware_splits.py` — class-aware 70:15:15 split with duplicate groups kept intact.
- `check_split_leakage.py` — independent exact-duplicate partition-leakage check.
- `create_label_subsets.py` — nested 10%, 25%, 50% and 100% training subsets.
- `convert_voc_to_yolo.py` — Pascal VOC to YOLO conversion.
- `prepare_generative_data.py` — shared 256x256 aspect-preserving preprocessing.
- `verify_data_pipeline.py` — independent processed-data and annotation checks.
- `create_experiment_registry.py` — freeze sources, seeds, budgets and provenance hashes.

## Augmentation and generative models

- `synthesize_traditional.py` — bounded geometric/photometric baseline with transformed boxes.
- `gan_baseline.py` — conditional transfer-learning GAN architecture and dataset code.
- `train_gan_full.py` — resumable AMP GAN training, validation and checkpointing.
- `synthesize_gan.py` — accepted degradation-conditioned GAN synthesis.
- `diffusion_baseline.py` — class-conditioned DDPM/CFG model construction and baseline utilities.
- `grounded_diffusion.py` — target-shadow, range-profile and log-speckle objectives.
- `train_diffusion_full.py` — standard, grounded and trained-ablation optimisation.
- `finalize_diffusion_training.py` — checkpoint/log reconciliation and selected-weight hashing.
- `evaluate_diffusion_sampling.py` — validation-only sampler grid evaluation.
- `freeze_diffusion_sampling_selection.py` — freeze the selected CFG/strength controls.
- `synthesize_diffusion.py` — equal-budget standard and grounded DDIM synthesis.

## Annotation control, detectors and quality evaluation

- `validate_synthetic_annotations.py` — localisation and target-retention gate.
- `build_yolo_experiment_datasets.py` — construct controlled five-condition detector datasets.
- `run_yolo_full_comparison.py` — train/evaluate identical YOLO11n conditions.
- `evaluate_synthetic_quality.py` — FID, MMD, LPIPS, aligned SSIM/PSNR and copy checks.

## Grounded-objective ablation

- `verify_ablation_diffusion_training.py` — verify switches, stopping and checkpoint hashes.
- `synthesize_diffusion_ablation.py` — common-random-number synthesis for four variants.
- `build_yolo_ablation_datasets.py` — build controlled ablation detector datasets.
- `run_yolo_ablation_comparison.py` — identical downstream ablation comparison.

## Resource checks and evidence figures

- `benchmark_diffusion_memory.py` — full-backbone AMP memory feasibility test.
- `create_results_figures.py` — detector, quality and ablation result plots.
- `create_generative_training_figures.py` — GAN/diffusion training curves.
- `create_diffusion_process_figure.py` — disclosed illustrative denoising-process figure.
- `create_ablation_preview.py` — matched-source ablation preview.

## Historical material

- `archive/pilots/` contains three development-pilot scripts. Pilot results are
  not the final comparative experiment.
- `archive/rejected_provisional_split/create_splits.py` created the provisional
  leakage-prone split. It is retained only for transparent provenance and must
  not be used to reproduce accepted results.
- `configs/archive/` contains the pre-execution planning configurations whose
  hashes occur in the frozen experiment registry. The root configuration files
  are the clearer executed summaries.

