# SCTD dissertation implementation

This is the curated reproducibility package for the dissertation's SCTD
synthetic-augmentation experiment. It is intentionally separate from the
execution workspace and contains code plus lightweight evidence, not the
dataset, checkpoints, generated image collections or copied detector datasets.

The executed study used the acquired SCTD copy containing **357 images, 357
Pascal VOC XML files and 363 objects**. The accepted split contains 249 training,
54 validation and 54 test images, with zero exact-duplicate groups crossing
partitions. The completed comparative detector experiment used the 10% subset:
26 real training images and 26 synthetic images per augmented condition.

## Repository map

```text
src/                         accepted executable pipeline (32 scripts)
configs/                     executed summaries
configs/archive/             original planning configs retained for provenance
experiments/label_010_v1/    frozen registry and ablation protocol
evidence/                    lightweight reports, manifests, curves and figures
archive/pilots/              development-only pilot scripts
archive/rejected_provisional_split/
                             rejected leakage-prone splitter
scripts/                     PowerShell environment helper
docs/                        appendix-ready LaTeX fragment
```

See `CODE_MAP.md` for the role of every Python file and
`IMPLEMENTATION_LOG.md` for the chronological evidence record, including
failures and corrective decisions. Paths inside historical JSON/CSV evidence
refer to the original D-drive execution workspace and are provenance strings,
not portable installation paths.

## Environment

The accepted Windows environment used Python 3.12.6, an NVIDIA RTX 4060 Laptop
GPU with 8,188 MiB VRAM, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128,
diffusers 0.39.0 and Ultralytics 8.4.120.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
. .\scripts\activate_project.ps1
```

## Dataset preparation

Follow `DATA.md`, then run:

```powershell
python src\dataset_audit.py --dataset-root data\raw\SCTD --output-dir data\audit\current
python src\create_group_aware_splits.py
python src\check_split_leakage.py --duplicate-report data\audit\current\duplicate_report.csv --split-dir data\splits\group_aware_v1 --output data\audit\current\split_leakage.csv
python src\create_label_subsets.py --train-file data\splits\group_aware_v1\train.txt --output-dir data\splits\limited_labels_group_aware_v1
python src\convert_voc_to_yolo.py --split-dir data\splits\group_aware_v1 --output-dir data\processed\yolo_sctd_group_aware_v1
python src\prepare_generative_data.py
python src\verify_data_pipeline.py
python src\create_experiment_registry.py
```

## Primary 10% experiment

The chronological log records the original commands and stopping decisions.
The principal entry points are:

```powershell
python src\synthesize_traditional.py
python src\train_gan_full.py
python src\synthesize_gan.py

python src\train_diffusion_full.py --condition standard --max-steps 20000 --validation-every 500 --patience-validations 8 --gradient-accumulation 4
python src\train_diffusion_full.py --condition grounded --max-steps 20000 --validation-every 500 --patience-validations 8 --gradient-accumulation 4
python src\evaluate_diffusion_sampling.py
python src\freeze_diffusion_sampling_selection.py
python src\synthesize_diffusion.py --condition standard
python src\synthesize_diffusion.py --condition grounded
```

The accepted standard and grounded training runs were stopped at complete
validation checkpoints after two consecutive regressions and independently
finalised. Their selected checkpoints were step 1,500 and step 1,000,
respectively. This evidence-review boundary is recorded in the corresponding
training reports; it must not be described as automatic eight-checkpoint early
stopping.

Each generated manifest is then passed separately to
`validate_synthetic_annotations.py`. After accepted outputs exist:

```powershell
python src\build_yolo_experiment_datasets.py
python src\run_yolo_full_comparison.py --all --skip-completed
python src\evaluate_synthetic_quality.py
python src\create_results_figures.py
python src\create_generative_training_figures.py
python src\create_diffusion_process_figure.py
```

## Grounded-objective ablation

The frozen protocol is
`experiments/label_010_v1/grounded_ablation_protocol.json`. Only the
no-acoustic and no-geometric variants require new training; full reuses the
grounded model and neither reuses the epsilon-only standard model.

```powershell
python src\train_diffusion_full.py --condition no_acoustic --max-steps 20000 --validation-every 500 --patience-validations 2 --gradient-accumulation 4
python src\verify_ablation_diffusion_training.py --condition no_acoustic
python src\train_diffusion_full.py --condition no_geometric --max-steps 20000 --validation-every 500 --patience-validations 2 --gradient-accumulation 4
python src\verify_ablation_diffusion_training.py --condition no_geometric

python src\synthesize_diffusion_ablation.py --condition full
python src\synthesize_diffusion_ablation.py --condition no_acoustic
python src\synthesize_diffusion_ablation.py --condition no_geometric
python src\synthesize_diffusion_ablation.py --condition neither
```

After running the unchanged annotation gate on each manifest:

```powershell
python src\build_yolo_ablation_datasets.py
python src\run_yolo_ablation_comparison.py --all --skip-completed
python src\create_ablation_preview.py
python src\create_results_figures.py
```

## Verification before publication

```powershell
python -m compileall -q src archive
python -m pip check
python -c "import json, pathlib; [json.loads(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.json')]; print('JSON OK')"
```

The observed rankings are single-seed, fixed-split results. The 25%, 50% and
100% subsets were prepared but full generative/detector comparisons were not
executed. FID at 26 samples is descriptive and high variance, and high paired
similarity must not be interpreted as downstream detector superiority.

