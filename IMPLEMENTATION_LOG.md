# Implementation execution ledger

This ledger is the factual record used to write Chapter 5. A stage is only marked
complete when its output exists and has been checked. Planned model results must
not be described in the past tense until their corresponding evidence is present.

| ID | Methodology requirement | Implementation/output | Status | Evidence or decision |
|---|---|---|---|---|
| 1 | Audit SCTD images and VOC boxes | `src/dataset_audit.py`, `data/audit/` | Complete | 357 images, 357 XML files, 363 objects, no invalid annotations |
| 2 | Detect repeated observations | SHA-256 duplicate audit | Complete | 25 exact-duplicate groups containing 55 images |
| 3 | Fixed class-aware 70:15:15 partition | `src/create_group_aware_splits.py`, `data/splits/group_aware_v1/` | Complete | 249/54/54 images, seed 42, exact duplicates assigned atomically |
| 4 | Prevent train-test duplicate leakage | `src/check_split_leakage.py`, `data/audit/current/split_leakage.csv` | Complete | Independent gate reports zero cross-partition exact-duplicate groups |
| 5 | Reduced-label training conditions | `src/create_label_subsets.py`, `data/splits/limited_labels_group_aware_v1/` | Complete | Regenerated nested 10/25/50/100% subsets containing 26/63/125/249 images |
| 6 | Detector data format | `src/convert_voc_to_yolo.py`, `data/processed/yolo_sctd_group_aware_v1/` | Complete | Regenerated 357 images and labels in a new non-stale destination |
| 7 | Traditional augmentation baseline | `src/synthesize_traditional.py`, `synthetic/label_010_v1/traditional/` | 10% full set complete | 26 seeded candidates; boxes transformed; 26/26 QC-approved |
| 8 | Transfer-learned GAN baseline | `src/train_gan_full.py`, `models/gan/label_010_v1/` | 10% full run complete | Early-stopped at epoch 56; best real-validation L1 0.00715732; 26/26 corrected candidates QC-approved |
| 9 | Standard image-guided DDPM + CFG | `src/train_diffusion_full.py`, `models/diffusion/standard/label_010_v1/` | 10% full run complete | 2,500 steps; selected step 1,500; verified best validation epsilon MSE 0.05800524; 26/26 candidates QC-approved |
| 10 | Grounded diffusion | `src/train_diffusion_full.py`, `models/diffusion/grounded/label_010_v1/` | 10% full run complete | 2,000 steps; selected step 1,000; verified weighted validation total 0.06878475; 26/26 candidates QC-approved |
| 11 | Synthetic annotation control | `src/validate_synthetic_annotations.py`, `results/full/label_010_v1/*_qc/` | 10% full review complete | Unchanged conservative gates approved 26/26 candidates for each of four methods; transformed labels supported |
| 12 | Equal synthetic budgets | `experiments/label_010_v1/`, `synthetic/label_010_v1/` | 10% full budgets complete | Each method has 26 images, 26 provenance rows and class allocation 4/3/19; hashes recorded |
| 13 | YOLOv11 downstream comparison | `src/run_yolo_pilot.py`, `results/pilots/yolo11n_augmented_v1/` | Integration pilot passed; comparison not executed | One epoch on 30 pilot samples; separate real-only val/test evaluation completed |
| 14 | FID/MMD/LPIPS/SSIM/PSNR | Synthetic quality evaluation | Not executed | SSIM/PSNR only for aligned source/output pairs |
| 15 | Precision/recall/mAP and error analysis | YOLO pilot report/predictions | Metric plumbing passed; experimental evaluation not executed | Pilot P/R/mAP and per-class mAP recorded, predictions JSON saved; no comparative claim |
| 16 | Four-way grounding ablation | `results/pilots/grounded_diffusion_*_v1/` | Execution pilots passed; full ablation not executed | All four switches ran; neither matched standard pilot epsilon losses exactly |

## Confirmed dataset scope

The original SCTD paper reports the composition of the complete published dataset.
The copy acquired and available for this implementation contains 357 paired images
and annotations. All experimental counts and conclusions in this dissertation will
therefore be explicitly limited to this available 357-image copy. The difference
will be reported as a study limitation rather than presented as an error in the
original paper.

## Decisions requiring confirmation

1. Label fractions: 10%, 25%, 50%, and 100% were selected because Chapter 3 does not specify them.
2. Synthetic budget: a 1:1 synthetic-to-real ratio was selected because Chapter 3 requires equality but gives no quantity.
3. Exact duplicates: the provisional split must not be used for final results because 12 duplicate groups cross evaluation boundaries.
4. Model checkpoints and training budget: the methodology does not name a GAN architecture, optical pretrained diffusion checkpoint, image resolution, epoch count, batch size, optimiser, learning rate, or hardware budget. These must be fixed before training.

## Reproduction order

```powershell
py src/dataset_audit.py --dataset-root data/raw/SCTD --output-dir data/audit
py src/create_splits.py
py src/check_split_leakage.py
py src/create_label_subsets.py
py src/convert_voc_to_yolo.py
```

Stop after the leakage check if its reported count is greater than zero. Do not
train or report final performance until a leakage-free split replaces the
provisional partition and all downstream subsets/conversions are regenerated.

## Execution record

### 2026-08-15 - Workspace relocation and pre-implementation verification

- Active workspace verified as `D:\Dissertation\Implementation\SCTD_Implementation_Starter`.
- `D:` capacity at inspection: 137.21 GB used and 138.85 GB free.
- GPU detected with `nvidia-smi`: NVIDIA GeForce RTX 4060 Laptop GPU, driver
  596.49, 8188 MiB total VRAM and 7956 MiB free at inspection.
- Python launcher reports Python 3.12.6.
- The project `.venv` also reports Python 3.12.6 but contains only Pillow 12.3.0
  and pip 26.2.1; pandas, scikit-learn, PyYAML and all deep-learning packages are
  absent from that environment. The global Python environment can import Pillow
  11.0.0, pandas 2.3.2, scikit-learn 1.9.0 and PyYAML 6.0.3, but cannot import
  PyTorch. No CUDA training was attempted.
- The relocated workspace is not currently a Git repository (`git status`
  returned "not a git repository").
- Read-only independent validation found 357 readable images, 357 XML files,
  no missing pairs, 363 objects, no invalid boxes, 25 exact-duplicate SHA-256
  groups containing 55 images, and 12 duplicate groups crossing the provisional
  partitions. These counts reproduce the existing audit evidence.
- The stored `data/audit/summary.json` still contains the former `C:` dataset
  path. It is retained temporarily as historical evidence and will be refreshed
  only after the leakage-free data pipeline is ready.
- Decision: all new artifacts, environments, checkpoints and results will be
  stored within this `D:` workspace. The raw dataset remains immutable.

### 2026-08-15 - Leakage-free partition and downstream regeneration

- Added `src/create_group_aware_splits.py`. It recomputes image SHA-256 hashes,
  treats each exact-duplicate group as indivisible, obtains integer class targets
  using constrained largest-remainder apportionment, and uses deterministic
  dynamic programming to meet exact partition sizes. Seed: 42.
- First execution attempt exceeded the 30-second command window because the
  dynamic programme copied its full assignment history at every state. It
  produced no final split output. The implementation was changed to compact
  predecessor layers. A further review found that the apportionment loop could
  award the same class/split remainder more than once; this was corrected before
  accepting or using the generated split.
- Accepted command: `py src/create_group_aware_splits.py`.
- Accepted split: train/validation/test = 249/54/54 images and 251/56/56
  objects. Image-level class counts are train 40 aircraft, 24 human, 185 ship;
  validation 9 aircraft, 5 human, 40 ship; test 8 aircraft, 5 human, 41 ship.
- Leakage command: `py src/check_split_leakage.py --split-dir
  data/splits/group_aware_v1 --output
  data/audit/group_aware_v1_split_leakage.csv`.
- Leakage output: `Cross-partition exact-duplicate groups: 0`.
- Updated `configs/experiments.yaml` to designate the group-aware split and its
  downstream outputs as the active experiment data.
- Updated `src/convert_voc_to_yolo.py` to refuse a non-empty output directory,
  preventing stale files from a previous split from surviving regeneration.
- Subset command: `py src/create_label_subsets.py --train-file
  data/splits/group_aware_v1/train.txt --output-dir
  data/splits/limited_labels_group_aware_v1`.
- Subset output: nested 10/25/50/100% sets containing 26/63/125/249 images.
- YOLO command: `py src/convert_voc_to_yolo.py --split-dir
  data/splits/group_aware_v1 --output-dir
  data/processed/yolo_sctd_group_aware_v1`.
- YOLO verification: train has 249 images, 249 label files and 251 box rows;
  validation 54/54/56; test 54/54/56. Total: 357 images, 357 label files and
  363 boxes.
- Refreshed the audit without overwriting the historical report using:
  `py src/dataset_audit.py --dataset-root data/raw/SCTD --output-dir
  data/audit/current`. The new summary records the correct `D:` dataset path and
  reproduces 357 images, 357 XML files, 363 objects, zero invalid issues, 25
  duplicate groups and 55 duplicated images.
- Rechecked the accepted split against the refreshed duplicate report. Output:
  `data/audit/current/split_leakage.csv`, zero leakage rows.
- One PowerShell-only summary command failed after the audit and leakage commands
  had succeeded because `ConvertTo-Json` rejected integer dictionary keys. It did
  not alter project artifacts. The summary was rerun with string keys and passed.

### 2026-08-15 - Shared generative preprocessing and model protocol

- Raw image inspection: 357 RGB images, 312 distinct dimensions, widths 174 to
  2550 pixels and heights 117 to 1731 pixels. Decision: do not stretch images to
  square because that would alter target/shadow geometry.
- Added `src/prepare_generative_data.py` and ran `py
  src/prepare_generative_data.py`. All methods now share deterministic RGB,
  aspect-preserving LANCZOS resize to a centred 256 x 256 black letterbox;
  Pascal VOC boxes receive the same scale and offsets. Training tensors will map
  uint8 RGB values to [-1, 1].
- Output `data/processed/generative_256_v1` contains 357 PNG images, 357 label
  files, 363 transformed boxes, a per-image provenance/hash manifest and the
  preprocessing configuration. The script refuses non-empty destinations.
- Visual inspection of `000002.png` found no distortion or clipping. Quantitative
  canvas inspection found median real-content coverage 71.48%; 35/357 images
  use less than half of the square and 0/357 use less than one quarter.
- Added `configs/models.yaml`. The 8 GB protocol fixes a 256-pixel conditional
  ResNet18-encoder U-Net/PatchGAN baseline, an optical-pretrained 256-pixel DDPM
  backbone with newly trained class embedding and classifier-free guidance,
  DDIM image-guided sampling, grounding switches/losses, equal 1:1 synthetic
  budgets, a four-condition ablation, and identical YOLO11n detector settings.
  Pilot limits are separate from full-run limits and no model result is claimed.
- Base `.venv` install first failed inside the network sandbox. After scoped
  network approval, `pip install -r requirements.txt` downloaded the missing
  base packages into the project environment.
- Official PyTorch guidance was consulted because wheel/CUDA compatibility is
  time-sensitive. Selected stable PyTorch 2.7.0 and torchvision 0.22.0 with the
  official CUDA 12.8 wheel for Python 3.12/Windows.
- Two foreground CUDA-wheel attempts ended before the 3.3383 GB wheel completed.
  A hidden logged install then failed with `[Errno 28] No space left on device`
  because pip used `C:` temporary storage; `C:` had only 0.39 GB free while `D:`
  had 138.43 GB. No GPU result was claimed.
- Corrective action: created project-local `.tmp/torch-install` and `.cache/pip`,
  set TEMP, TMP and PIP_CACHE_DIR only for the install process, and restarted the
  official wheel download with stdout/stderr logs in the project root.
- Added `src/verify_data_pipeline.py`, a fail-fast independent acceptance gate.
  Command: `.venv\Scripts\python.exe src\verify_data_pipeline.py`. Result:
  PASS for 249/54/54 unique memberships, 25 duplicate groups kept within one
  partition, zero leakage, nested 26/63/125/249 subsets, 357 YOLO images and
  labels with 363 valid rows, and 357 readable/hash-matched generative inputs
  with 363 objects.
- Corrected PyTorch install result: `torch==2.7.0+cu128` and
  `torchvision==0.22.0+cu128` installed successfully from the official PyTorch
  CUDA 12.8 index after redirecting temporary storage to `D:`. CUDA smoke test:
  `torch.cuda.is_available() == True`, CUDA runtime 12.8, device NVIDIA GeForce
  RTX 4060 Laptop GPU, and a GPU matrix multiplication completed.
- Installed remaining project-environment packages: diffusers 0.39.0,
  transformers 5.15.0, accelerate 1.14.0, ultralytics 8.4.120, torchmetrics
  1.9.0, LPIPS 0.1.4, clean-fid 0.1.35, tensorboard 2.21.0, safetensors 0.8.0
  and their dependencies. Install logs are `deep_dependencies.stdout.log` and
  `deep_dependencies.stderr.log`.
- First combined import test revealed Ultralytics and Matplotlib attempting to
  use settings/cache locations on `C:`; the test timed out before printing the
  requested version summary and is not counted as a pass.
- Added `scripts/activate_project.ps1`. It activates `.venv` and directs TEMP,
  pip, Torch, Hugging Face, Ultralytics, Matplotlib and XDG caches/settings to
  project-local locations on `D:` without repurposing HOME. Two subsequent
  import tests passed for PyTorch/CUDA, torchvision, diffusers, transformers,
  accelerate, Ultralytics, torchmetrics, LPIPS, clean-fid and OpenCV. The new
  Ultralytics settings file was created under `.config/ultralytics` on `D:`.

### 2026-08-15 - Transfer-learning GAN execution pilot

- Added `src/gan_baseline.py`. The generator uses the fixed ImageNet-1K V1
  ResNet18 encoder, U-Net skip decoding, a learned class embedding and stochastic
  latent injection. The output is a bounded residual transformation of the real
  source. The conditional spectral-normalised PatchGAN receives source,
  candidate and one-hot class maps. Training uses hinge adversarial, L1 content
  and discriminator feature-matching losses.
- The first command, `. scripts/activate_project.ps1; python
  src/gan_baseline.py`, failed before output creation because the sandbox blocked
  the official ResNet18 checkpoint download (`WinError 10013`). This is recorded
  as an expected network-gate failure; no random-weight fallback was accepted.
- After scoped network approval, the official 44.7 MB
  `resnet18-f37072fd.pth` checkpoint downloaded to
  `models/cache/torch/hub/checkpoints` on `D:` and passed torchvision's hash
  check.
- Accepted pilot used 12 available samples from the accepted nested 10% training
  subset, batch size 2, seed 42, FP16 CUDA and two real discriminator/generator
  optimisation steps on the RTX 4060.
- Output `results/pilots/gan_v1/pilot_report.json` reports status PASS, elapsed
  optimisation time 1.3463 seconds and peak allocated CUDA memory 398.47 MB.
  These are execution diagnostics, not model-quality or dissertation results.
- Saved and verified `checkpoint.pt` and `pilot_grid.png`. Visual inspection
  confirmed readable source/generated/target rows and preserved target layout.
  The generated row shows strong colour/residual artefacts, as expected after
  only two steps; it is not approved synthetic training data and is not evidence
  of GAN performance.
- Checkpoint reload gate passed by instantiating the same generator without a
  network download, loading the saved state dictionary strictly, and confirming
  the stored two-step/pretrained-encoder metadata.

### 2026-08-15 - Standard diffusion execution pilot

- Added `src/diffusion_baseline.py` using the fixed public
  `google/ddpm-celebahq-256` optical checkpoint. It creates a four-entry learned
  embedding for aircraft, human, ship and the classifier-free null condition,
  implements 10% condition dropout, epsilon prediction, source noising, CFG and
  accelerated DDIM reverse sampling.
- The initial sandboxed command could not access Hugging Face (`WinError 10013`)
  and produced no output directory. After scoped approval, the 454,853,117-byte
  legacy checkpoint downloaded into the project Hugging Face cache on `D:`.
  The repository supplies a PyTorch `.bin` checkpoint rather than safetensors;
  Diffusers therefore emitted its legacy unsafe-serialization warning. No
  untrusted custom code was enabled.
- The first class-conditioned construction failed its strict transfer gate:
  Diffusers retained `_use_default_values` metadata and silently ignored the
  requested `num_class_embeds`. Investigation reproduced the issue and verified
  that removing this metadata creates the required `[4, 512]` embedding. The
  loader was corrected; accepted transfer then reported only the expected
  missing key `class_embedding.weight` and zero unexpected keys.
- The legacy checkpoint contains no scheduler configuration. Decision: use and
  record the standard 1,000-step linear DDPM schedule associated with this
  checkpoint family; `configs/models.yaml` was corrected from the earlier
  unverified squared-cosine choice.
- A local-cache rerun then failed because the code still attempted to load a
  nonexistent scheduler file. The empty partial output directory was verified
  as inside the workspace and removed. Scheduler construction was made explicit.
- The first complete CUDA run used direct FP16 and produced a finite loss at step
  1 but `NaN` at step 2. It incorrectly wrote PASS because the first version
  lacked a finite-loss gate. The entire artifact directory was preserved as
  `results/pilots/standard_diffusion_v1_failed_nan`; it is rejected evidence and
  must not be used for results.
- Corrective action: added a mandatory finite-loss check and changed this small
  pilot to FP32. Full mixed-precision training is permitted only with gradient
  scaling and the same finite gate. This pilot precision override is recorded in
  `configs/models.yaml`.
- Accepted command: `. scripts/activate_project.ps1; python
  src/diffusion_baseline.py --local-files-only`.
- Accepted pilot: two finite CUDA optimisation steps on the 10% training subset,
  four DDIM inference steps, guidance scale 2.0, source strength 0.60, strict
  adapter checkpoint reload PASS, elapsed time 2.9927 seconds and peak allocated
  CUDA memory 1362.41 MB. Only the new class embedding and output head (5,507
  parameters) were trained in the execution pilot; 113,669,760 backbone
  parameters remained frozen. This does not represent the full fine-tuning run.
- Visual inspection of `results/pilots/standard_diffusion_v1/pilot_grid.png`
  confirmed readable source/output images but poor structural preservation after
  two steps. The sample is not approved synthetic data and no image-quality or
  detector conclusion is drawn from it.

### 2026-08-15 - Grounded diffusion and ablation-switch pilots

- Added `src/grounded_diffusion.py`, reusing the accepted standard diffusion
  backbone, transfer logic, class/null embeddings, 1,000-step linear schedule,
  CFG, source strength 0.60, four-step DDIM sampler, 10% training subset, seed 42
  and FP32 pilot precision.
- Geometric component: transforms each accepted YOLO box to a target region,
  inspects four immediately adjacent regions in the real structural reference,
  selects the darkest recoverable adjacent region as a shadow candidate, and
  penalises changes in target-to-shadow contrast. It does not infer altitude,
  slant range or ray-tracing geometry.
- Range-intensity component: computes horizontal and vertical mean-intensity
  profiles from the real image, selects the axis with the stronger recoverable
  profile variance per sample, and penalises profile changes. The selected axis
  is image-derived; it is not labelled as physical range when metadata are absent.
- Speckle component: converts normalised luminance to `log(x + 1/255)`, removes a
  local 5 x 5 mean, and matches signal-weighted high-frequency residuals. This
  implements signal-dependent log-domain behaviour rather than independent
  additive Gaussian noise.
- Identity unit gate initially failed to import the module as `src.*` because the
  first import path supported only direct script execution. A package-relative
  import with direct-script fallback was added. Repeated identity tests returned
  exactly 0.0 for geometric, range-profile and log-speckle losses when prediction
  equals reference.
- Full grounded pilot passed two finite optimisation steps. Active loss weights:
  geometry 0.10, range 0.05 and speckle 0.05. Peak allocated CUDA memory was
  2498.89 MB, elapsed optimisation/sampling time 1.4500 seconds, and checkpoint
  reload passed. These are execution diagnostics only.
- Ran no-acoustic (`--no-range --no-speckle`), no-geometric (`--no-geometric`),
  and neither (`--no-geometric --no-range --no-speckle`) pilots with otherwise
  identical controls. Each completed two finite steps, sampling and checkpoint
  reload.
- The neither pilot produced epsilon/total losses 0.00383010134100914 and
  0.0063086263835430145. These exactly match the two accepted standard-diffusion
  pilot epsilon losses under the same seed, establishing that disabling all
  grounding returns the training loss path to the standard condition.
- Visual inspection of the full-grounded pilot grid found a readable source and
  output but no credible quality improvement after two steps. No pilot image is
  approved for detector training and no comparative performance conclusion is
  drawn.

### 2026-08-15 - Synthetic annotation-control and YOLO11n integration pilot

- Added `src/prepare_pilot_candidates.py`. It extracts the exact 256 x 256
  generated panels from the lossless GAN, standard-diffusion and
  grounded-diffusion pilot grids and links each candidate to its source image,
  label, method and pilot provenance. Four candidates were produced: two GAN,
  one standard diffusion and one grounded diffusion. These mixed counts are for
  execution testing only and do not satisfy or replace the full equal-budget
  experiment.
- Added `src/validate_synthetic_annotations.py`. The conservative automatic gate
  measures full-image phase-correlation response/shift, normalised correlation
  inside every transferred target box, target-region edge-energy ratio, and
  preservation of target/background contrast polarity. Provisional pilot
  thresholds are response >= 0.10, shift <= 8 pixels, target NCC >= 0.25, edge
  ratio 0.35 to 2.85, and unchanged contrast polarity.
- All four pilot candidates passed: phase shifts were 0.03 to 0.42 pixels,
  minimum target NCC 0.6199 to 0.9960, and target edge ratios 0.7256 to 0.9475.
  Their source labels were copied only after the gate and each decision is in
  `results/pilots/synthetic_qc_v1/qc_manifest.csv`. This demonstrates plumbing;
  thresholds remain provisional and full generated budgets still require visual
  review/correction/rejection records.
- Added `src/build_yolo_pilot_dataset.py`. It created
  `data/processed/yolo_pilot_augmented_v1` with 26 real images from the accepted
  nested 10% training subset plus four QC-approved pilot candidates. Validation
  and test contain the unchanged 54/54 real-only partitions. Coverage check:
  train 30 images/labels, validation 54/54, test 54/54.
- Added `src/run_yolo_pilot.py` for a one-epoch CUDA integration test using the
  official pretrained YOLO11n checkpoint, image size 320, batch 4, AdamW,
  learning rate 0.001, seed 42, deterministic mode and AMP. Full experiment
  settings remain 640 pixels and 100 epochs as fixed in `configs/models.yaml`.
- Initial YOLO launch failed at the sandbox network gate while requesting the
  official 5.4 MB checkpoint. After scoped approval, it downloaded to
  `models/cache/ultralytics/yolo11n.pt` on `D:`. Ultralytics also downloaded
  `Arial.ttf` to the D-local config and `yolo26n.pt` to the D workspace solely
  for its internal AMP compatibility check; YOLO26 was not used as the detector.
- The first training attempt completed one epoch and validation but the wrapper
  then rejected the run because Ultralytics prefixed the relative project path
  with `runs/detect`; the expected checkpoint path was therefore wrong. That
  completed-but-rejected artifact is preserved at
  `runs/detect/results/pilots/yolo11n_augmented_v1`. Explicit real-only val/test
  evaluation and the final report were not completed by that attempt.
- Corrected the wrapper to use an absolute D-local project path and to read the
  authoritative `model.trainer.save_dir`. The accepted rerun completed one
  epoch, saved `best.pt`, reloaded it, evaluated the 54-image real validation set
  and separately evaluated the 54-image real test set. Total wrapper time was
  9.9014 seconds.
- Accepted one-epoch diagnostic metrics: validation precision 0.005110, recall
  0.838095, mAP50 0.099644, mAP50-95 0.047450; test precision 0.005228, recall
  0.761905, mAP50 0.046188, mAP50-95 0.018181. Test per-class mAP50-95 in fixed
  order aircraft/human/ship was 0.027919/0.000885/0.025740. These values verify
  metric plumbing only: one epoch, mixed/unequal pilot candidates and no
  real-only control mean they must not appear as comparative dissertation
  results.
- Prediction JSON files were saved under the accepted validation and test pilot
  directories. Error analysis and all controlled condition comparisons remain
  unexecuted.

### 2026-08-15 - Frozen 10%-label full-run registry and resource gate

- Added and ran `src/create_experiment_registry.py`. The immutable first full-run
  registry is `experiments/label_010_v1`: 26 real training sources (4 aircraft,
  3 human, 19 ship), seed 42 and exactly 26 registered candidates for each of
  traditional, GAN, standard diffusion and grounded diffusion. The registry
  stores SHA-256 hashes for the split manifest, label subset, preprocessing
  manifest/configuration and both experiment configurations. Equal 1:1 budget
  count and real-only validation/test flags passed.
- Decision: complete and validate the 10% condition before scaling to 25%, 50%
  and 100%. This prevents a defective generator or annotation-transfer rule from
  multiplying across all experimental conditions.
- Added and ran `src/benchmark_diffusion_memory.py` with all 113,675,267
  conditioned-UNet parameters trainable, batch 1, 256 x 256 images, gradient
  checkpointing, FP32 parameters, AMP FP16 forward pass, GradScaler, AdamW state,
  gradient clipping and a real optimizer step. Result PASS: loss 0.00824315,
  finite pre-clip gradient norm 0.412649, peak allocated/reserved CUDA memory
  2224.88/2238.00 MB. Full-backbone training is therefore feasible within 8 GB.

### 2026-08-15 - Full 10%-label GAN training and synthesis

- Added `src/train_gan_full.py` with atomic per-epoch resume checkpoints,
  separate best-checkpoint selection, FP16 AMP with independent generator and
  discriminator scalers, finite-loss gates, gradient clipping, fixed-noise
  54-image real validation, JSONL logs and early stopping (patience 25).
- Full command completed 56 of at most 200 epochs and early-stopped. Training
  used 26 real 10%-subset images, validation used all 54 real validation images,
  best validation L1 was 0.00715732, elapsed time 59.0974 seconds and peak
  allocated CUDA memory 712.20 MB. The selected checkpoint is
  `models/gan/label_010_v1/checkpoint_best.pt`.
- Added `src/synthesize_gan.py`. First synthesis produced 26 non-identical files
  and all passed localisation QC, but an added diversity gate measured only
  0.31 to 1.48/255 mean absolute source difference (mean 0.63/255). This set was
  rejected as insufficient variation and preserved under
  `synthetic/label_010_v1/gan_identity_input_rejected_low_variation` with its QC
  evidence rather than overwritten.
- Corrective decision: synthesis now follows the generator's training domain by
  applying seeded brightness (0.92 to 1.08) and normalised Gaussian degradation
  (standard deviation 0.03) before conditional restoration; stochastic latent
  noise remains independently seeded from the frozen registry.
- Corrected synthesis produced exactly 26 candidates, zero exact pixel copies,
  mean absolute source difference 2.60/255 (range 1.71 to 3.45/255), and all 26
  passed the unchanged annotation-transfer gate. Accepted artifacts are under
  `synthetic/label_010_v1/gan` and `results/full/label_010_v1/gan_qc`. This is a
  conservative GAN augmentation set; usefulness is not claimed before metrics
  and controlled detector comparison.

### 2026-08-15 - Full diffusion trainer smoke gate and active standard run

- Added `src/train_diffusion_full.py`: full-backbone training, gradient
  checkpointing, batch 1 with four-step accumulation, 10% classifier-free null
  dropout, AMP plus GradScaler, finite loss/gradient gates, norm clipping at 1.0,
  cosine learning-rate schedule with 200-step warm-up, fixed-noise 54-image real
  validation every 500 optimizer steps, atomic resumable state and safetensors
  best-model export. The configured ceiling remains 20,000 optimizer steps with
  early stopping after eight non-improving validations.
- A separate 10-step standard-condition smoke run passed training, full
  validation, best-model export and atomic optimizer checkpoint creation. Peak
  allocated CUDA memory was 2450.01 MB. Evidence is retained at
  `results/resource_checks/diffusion_standard_trainer_smoke_v1`.
- The controlled standard full run completed 2,500 optimizer steps with finite
  losses and gradients throughout. Fixed-noise validation over all 54 real
  validation images improved at steps 500/1,000/1,500 to
  0.06197066/0.05961802/0.05800524, then failed to improve at steps 2,000 and
  2,500 (0.05905258 and 0.05868024). Evidence review therefore stopped the run
  at the complete step-2,500 checkpoint rather than spending the 20,000-step
  ceiling after two consecutive regressions; the selected model remains the
  step-1,500 validation minimum. Total logged training time was 2,591.69 seconds
  and peak allocated CUDA memory was 2,450.01 MB.
- Added and ran `src/finalize_diffusion_training.py`. It independently reloaded
  the 1.3 GB latest checkpoint, required its step/condition/best score to match
  the append-only validation history, verified four class embeddings, and
  SHA-256 hashed the latest checkpoint and selected safetensors weights. All
  gates passed. The verified report is
  `models/diffusion/standard/label_010_v1/training_report.json`; selected weights
  are `models/diffusion/standard/label_010_v1/best_model`.

### 2026-08-15 - Grounded diffusion full-trainer gate and launch

- Ran a separate 10-step full-backbone grounded trainer smoke gate before the
  full run. Training and full 54-image real validation completed with finite
  epsilon, target-shadow geometry, recoverable range-profile and log-domain
  signal-dependent speckle terms. Validation components were 0.11723173,
  0.10872249, 0.09957297 and 0.41256514 respectively; their configured weighted
  total was 0.15371092. Best-model export and atomic optimizer checkpointing
  passed, with 2,450.01 MB peak allocated CUDA memory. Evidence is under
  `results/resource_checks/diffusion_grounded_trainer_smoke_v1`.
- Launched the controlled full grounded run at
  `models/diffusion/grounded/label_010_v1` using the same data, seed, backbone,
  optimizer, schedule, effective batch size and fixed validation procedure as
  the standard run. Only the three explicitly documented grounding loss terms
  differ. Final status must be appended only after termination and verification.
- During the active run, a separate non-training `--help` import check for the
  newly added sampling scripts failed after ten OpenBLAS host-memory allocation
  retries. The grounded process was using approximately 7.44 GB private host
  memory for model/optimizer/checkpoint state and continued normally; no model
  setting was changed and the import check will be rerun only after training
  releases memory. Both scripts had already passed `py_compile` syntax checks.

### 2026-08-16 - Grounded diffusion full-run completion

- The full grounded run completed 2,000 optimizer steps with finite losses and
  gradients throughout. Fixed-noise validation over all 54 real validation
  images improved from 0.07455241 at step 500 to the selected minimum
  0.06878475 at step 1,000. Steps 1,500 and 2,000 did not improve the total
  (0.07041615 and 0.06999740), so evidence review stopped the run at the second
  complete non-improving checkpoint and retained the step-1,000 weights. The
  logged training time was 1,982.20 seconds and peak allocated CUDA memory was
  2,450.01 MB.
- `src/finalize_diffusion_training.py` independently reloaded the grounded
  checkpoint, matched its condition, step and best score to the append-only
  history, verified four class embeddings, and SHA-256 hashed the latest state
  and selected safetensors weights. All gates passed. The verified report is
  `models/diffusion/grounded/label_010_v1/training_report.json`; selected weights
  are `models/diffusion/grounded/label_010_v1/best_model`.

### 2026-08-16 - Shared diffusion sampling selection and equal-budget synthesis

- Added `src/evaluate_diffusion_sampling.py`, `src/synthesize_diffusion.py` and
  `src/freeze_diffusion_sampling_selection.py`. The evaluator uses only the real
  validation partition, selects four fixed images from each class, applies
  common random seeds, and evaluates the pre-registered Cartesian grid of CFG
  scales 1/2/3/5 and source strengths 0.30/0.45/0.60 at 50 DDIM steps. It writes
  all 144 candidates, paired metrics and class previews without automatically
  choosing a setting.
- The first attempt to invoke the evaluator with `--help` began the intended
  evaluation because the initial script exposed no argument parser. This was a
  command-interface defect rather than a model failure; the correct in-scope
  grid completed successfully and `argparse` was added before future reuse.
- All configurations at strengths 0.30 and 0.45 passed the unchanged transfer
  gate for 12/12 validation images with zero exact copies. Every strength-0.60
  configuration rejected at least one image (approval 11/12, 11/12, 11/12 and
  4/12 as CFG increased). Paired visual review found increasing horizontal
  banding, colour fringing and target deformation at higher strength/CFG.
- Frozen shared choice: CFG 1.0, source strength 0.30 and 50 DDIM steps for both
  standard and grounded diffusion. Its 12/12 candidates passed, minimum target
  NCC was 0.71237689, minimum phase response 0.46953379, maximum shift 0.1485
  pixels, and source difference averaged 15.3843/255 (minimum 7.3860/255). The
  machine-validated decision and visual rationale are stored at
  `results/full/label_010_v1/diffusion_sampling_selection/selection_report.json`.
  The test split was not used.
- Standard diffusion synthesis produced exactly 26 frozen-registry candidates,
  zero exact copies and mean source difference 14.5174/255 (range
  6.1074-24.5005/255). All 26 passed the unchanged annotation-transfer gate.
  Artifacts are `synthetic/label_010_v1/standard_diffusion` and
  `results/full/label_010_v1/standard_diffusion_qc`.
- Grounded diffusion synthesis used the identical source images, registered
  seeds and sampler settings. It produced exactly 26 candidates, zero exact
  copies and mean source difference 15.2462/255 (range 6.9253-25.2868/255).
  All 26 passed the same gate. Artifacts are
  `synthetic/label_010_v1/grounded_diffusion` and
  `results/full/label_010_v1/grounded_diffusion_qc`.
- Reconciliation verified image/manifest/approved-label counts of 26/26/26 for
  GAN, standard diffusion and grounded diffusion. Selection-report SHA-256 is
  `19025eadc762121e87fbe5275c967c3bed9cb9ecf99cac74ffe35b322b289dc7`;
  standard and grounded manifest hashes are respectively
  `22936b6df64aa07cb6203e2543691d3ea47a393c445e7262bc8ce37c9ec8c7e7`
  and `fa7d593877a24e0fc3b1e01ead1db1e017ba4b5b8062e71f1e17fcee4be71367`.

### 2026-08-16 - Traditional equal-budget baseline and four-method reconciliation

- Extended `src/validate_synthetic_annotations.py` so a manifest may provide a
  transformed `candidate_label`. Source and candidate target regions are then
  compared at their respective coordinates, and the approved output receives
  the candidate label rather than incorrectly copying the source coordinates.
  The change is backward compatible: a 26-image standard-diffusion regression
  rerun produced zero differences across all recorded QC decision and metric
  fields versus the previously accepted output.
- Added and ran `src/synthesize_traditional.py`. Each frozen registry seed
  deterministically selects rotation within +/-1.5 degrees, scale 0.99-1.01,
  x/y translations within +/-3 pixels, global intensity gain 0.94-1.06 and
  gamma 0.96-1.04. No independent additive noise is used. Every YOLO box is
  transformed through all four corners, axis-aligned and clipped to the image.
- Traditional synthesis produced exactly 26 candidates, zero exact copies and
  mean source difference 18.4310/255 (range 6.7470-45.1363/255). All 26 passed
  the same alignment, target correlation, edge-ratio and contrast-polarity QC
  thresholds. Visual review of the three-class preview found mild plausible
  variation without target loss. Artifacts are
  `synthetic/label_010_v1/traditional` and
  `results/full/label_010_v1/traditional_qc`.
- Final four-method reconciliation passed. Traditional, GAN, standard diffusion
  and grounded diffusion each contain 26 images, 26 provenance rows, 26 QC
  approvals and 26 approved label files, with identical class allocation of
  4 aircraft, 3 human and 19 ship. All transformed traditional label values are
  within [0,1]. Manifest SHA-256 values are traditional
  `ce20fb20397baaef1755b7d7017b7aa4619609fcf37f43874f385e2804afa297`,
  GAN `256b8f01e88d50d28b03fe9fe732a12bb037c788c725529dcaf93b0b633a687c`,
  standard diffusion
  `22936b6df64aa07cb6203e2543691d3ea47a393c445e7262bc8ce37c9ec8c7e7`,
  and grounded diffusion
  `fa7d593877a24e0fc3b1e01ead1db1e017ba4b5b8062e71f1e17fcee4be71367`.

### 2026-08-18 - Implementation chapter evidence reconciliation

- Re-read the dissertation methodology and implementation requirements from
  `C:\Users\User\Desktop\Brunel\Term 3\Dissertation\DISS_ALL (4).pdf`, with
  particular attention to the comparative controls, diffusion grounding terms,
  equal synthetic budgets, detector protocol, metrics, and ablation design.
  The bundled Poppler wrapper could not resolve its executable path, so the PDF
  text was read with the bundled Python runtime and `pypdf`; this was a
  documentation-only fallback and did not change any experiment artifact.
- Reconciled `docs/implementation_chapter.md` against the complete execution
  ledger, accepted JSON/CSV reports, model configurations, and executed source
  code. Rewrote Chapter 5 as a 19-section evidence-based implementation chapter
  covering the D-drive environment, acquired-data scope, leakage-free split,
  nested subsets, annotation conversion, common preprocessing, frozen registry,
  traditional baseline, GAN architecture, standard diffusion architecture,
  grounded losses, synthesis QC, failure handling, near-duplicate limitation,
  YOLO integration pilot, and remaining work.
- Reported only completed evidence. The chapter explicitly labels the full
  five-condition YOLO comparison, FID, MMD, LPIPS, aligned SSIM/PSNR,
  per-class detector results, and full four-way grounding ablations as pending.
  The one-epoch YOLO run is identified only as an integration pilot.
- Disclosed two planning/execution differences instead of silently rewriting
  history: `configs/models.yaml` planned diffusion accumulation 8 while the
  executed reports record accumulation 4, and the YAML planned GAN accumulation
  2 while `src/train_gan_full.py` executed batch size 4 without accumulation.
  Executed code, append-only logs, and training reports remain authoritative.
- Added exact figure sources, suggested placement, captions, and warnings for
  19 implementation figures. A `Test-Path` verification confirmed that every
  referenced audit, split, subset, preprocessing, checkpoint-preview,
  synthesis-preview, training-report, QC, and YOLO-pilot artifact exists.
- No model was retrained, no metric was invented, and no existing experimental
  artifact was modified during this documentation pass.
- Converted the reconciled Markdown chapter into
  `docs/implementation_chapter.tex` as a LaTeX chapter fragment intended for
  `\input{docs/implementation_chapter.tex}`. Equations were retained as native
  LaTeX; evidence tables were converted to `booktabs`/`tabularx`/`longtable`;
  and four self-contained TikZ figures were added for the end-to-end pipeline,
  GAN architecture, standard diffusion architecture, and grounded objective.
- Added a commented preamble dependency list to the LaTeX file and converted
  paths and code identifiers to LaTeX-safe forms. Static verification found
  balanced braces and matching `begin`/`end` counts for every environment, with
  no remaining Markdown or Mermaid tokens. No TeX engine was installed in the
  local project environment, so final class-specific compilation must be run in
  the dissertation's LaTeX environment or Overleaf.

### 2026-08-19 - YOLO comparative-result status audit

- Inspected all YOLO-, detector-, and metric-related files under `results`,
  `models`, `data`, `experiments`, `configs`, and `src`. No full detector runs
  exist for the five required conditions: real-only, traditional, GAN,
  standard diffusion, and grounded diffusion.
- Confirmed that the only completed YOLO run is
  `results/pilots/yolo11n_augmented_v1`. Its own report identifies it as a
  one-epoch execution pilot, using 30 training images at image size 320 and
  batch size 4. Four mixed synthetic pilot candidates were combined in one
  training set, so this run cannot compare augmentation methods or identify a
  winning generative model.
- The frozen full detector protocol remains YOLO11n at image size 640, AdamW,
  learning rate 0.001, batch size 8, up to 100 epochs, patience 20, seed 42,
  identical initial weights and detector augmentation, and unchanged real-only
  validation/test sets. Therefore the absence of comparative detector results
  in Chapter 5 is an unexecuted-stage boundary, not an omitted result.

### 2026-08-19 - Five-condition YOLO dataset construction

- Added `src/build_yolo_experiment_datasets.py`. The builder preflights the
  frozen `experiments/label_010_v1/candidate_registry.csv`, the 26-image 10%
  subset, all four accepted QC manifests, every candidate hash, every approved
  label, and every YOLO coordinate before writing. It builds through a temporary
  directory and atomically publishes only after all conditions reconcile.
- Command: `.venv\Scripts\python.exe -m py_compile
  src\build_yolo_experiment_datasets.py`; result: pass. Command:
  `.venv\Scripts\python.exe src\build_yolo_experiment_datasets.py --help`;
  result: expected argument interface displayed.
- Command: `.venv\Scripts\python.exe
  src\build_yolo_experiment_datasets.py`; result: pass. Accepted output is
  `data/processed/yolo_experiments/label_010_v1`.
- The real-only condition contains 26 real images and 26 objects (4 aircraft,
  3 human, 19 ship). Traditional, GAN, standard diffusion, and grounded
  diffusion each contain the same 26 real images plus 26 approved candidates,
  with 52 objects (8 aircraft, 6 human, 38 ship). Every condition uses the same
  54 real validation and 54 real test images.
- SHA-256 reconciliation confirmed identical validation/test content across all
  five conditions and zero exact image-hash intersection between any training
  set and evaluation. The real subset has 25 unique image hashes among 26
  images because one accepted exact-duplicate group is wholly contained within
  training; it creates no partition leakage and is shared identically by every
  condition. Augmented sets have 51 unique hashes among 52 images for the same
  reason.
- Detector training retains the acquired-resolution real sources from the
  accepted YOLO dataset and adds the QC-approved 256x256 shared-preprocessing
  synthetic candidates. This preserves the real-only baseline inputs and uses
  exactly the same real representation in all augmented conditions; the
  representation difference is an explicit property of generated candidates
  rather than a condition-specific preprocessing difference.

### 2026-08-19 - Full YOLO11n runner and real-only baseline

- Added `src/run_yolo_full_comparison.py`. It asserts the frozen detector
  protocol from `configs/models.yaml`, requires a passing five-dataset
  reconciliation, loads the same initial checkpoint for every condition,
  selects `best.pt` using only the fixed real validation set, evaluates that
  checkpoint separately on real validation and real test, records aggregate and
  per-class precision/recall/F1/mAP, hashes checkpoints, and writes a final
  cross-condition ranking only when all five reports exist.
- Commands: `.venv\Scripts\python.exe -m py_compile
  src\run_yolo_full_comparison.py` and `.venv\Scripts\python.exe
  src\run_yolo_full_comparison.py --help`; results: pass. A metric-extraction
  smoke evaluation of the existing one-epoch pilot was run against all 54 real
  validation images and reproduced its aggregate metrics while adding verified
  aircraft/human/ship metrics. Smoke artifacts are
  `results/resource_checks/yolo_metric_extractor_v1`.
- Command: `.venv\Scripts\python.exe
  src\run_yolo_full_comparison.py --condition real_only`; result: pass. The run
  used the fixed 640-pixel, batch-8, AdamW, learning-rate-0.001, seed-42 protocol
  and initial checkpoint SHA-256
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da663ae6417644ee1`.
- The real-only baseline stopped after 21/100 epochs because validation fitness
  did not improve for 20 epochs; epoch 1 was retained. Training took 51.86 s
  and full training plus separate validation/test evaluation took 95.29 s.
  Real-test precision was 0.0023840, recall 0.5595238, mAP50 0.0454182 and
  mAP50-95 0.0174055. These low values are accepted evidence for the 26-image
  10% baseline and are not replaced or tuned using test data. The authoritative
  report is `results/detectors/label_010_v1/real_only/final_report.json`.

### 2026-08-19 - Traditional and GAN full YOLO11n conditions

- Command: `.venv\Scripts\python.exe
  src\run_yolo_full_comparison.py --all --skip-completed`; status at this log
  point: still running. The runner skipped the completed real-only baseline,
  completed traditional and GAN in sequence, and then began standard diffusion.
- Traditional augmentation stopped after 72/100 epochs; validation selected
  epoch 52. Training took 124.07 s and full training plus separate evaluation
  took 165.58 s. On the untouched real test set, precision was 0.7625113,
  recall 0.3015873, mAP50 0.3134163, and mAP50-95 0.1727669. Test class
  mAP50-95 was 0.0416016 for aircraft, 0.2555320 for human, and 0.2211670
  for ship. The authoritative report is
  `results/detectors/label_010_v1/traditional/final_report.json`.
- GAN augmentation completed 100/100 epochs; validation selected epoch 91.
  Training took 169.73 s and full training plus separate evaluation took
  221.42 s. On the untouched real test set, precision was 0.4782603, recall
  0.4259207, mAP50 0.3426334, and mAP50-95 0.1507039. Test class mAP50-95
  was 0.1187298 for aircraft, 0.1081968 for human, and 0.2251850 for ship.
  The authoritative report is
  `results/detectors/label_010_v1/gan/final_report.json`.
- The GAN result has higher test mAP50 than traditional but lower test
  mAP50-95. This is only an interim comparison; no winning method will be
  declared until standard and grounded diffusion complete under the same
  protocol.

### 2026-08-19 - Standard-diffusion full YOLO11n condition

- The continuing command `.venv\Scripts\python.exe
  src\run_yolo_full_comparison.py --all --skip-completed` completed the
  standard-diffusion detector and began grounded diffusion. Standard diffusion
  completed 100/100 epochs; validation selected epoch 86. Training took
  257.24 s and full training plus separate evaluation took 336.22 s.
- On the untouched real test set, precision was 0.7519135, recall 0.3492063,
  mAP50 0.3734665, and mAP50-95 0.2286528. Test class mAP50-95 was 0.0577832
  for aircraft, 0.2763478 for human, and 0.3518275 for ship. This is the
  strongest aggregate test mAP50 and mAP50-95 among completed conditions at
  this point, but grounded diffusion is still running. The authoritative
  report is
  `results/detectors/label_010_v1/standard_diffusion/final_report.json`.

### 2026-08-19 - Synthetic-quality evaluator implementation

- Added `src/evaluate_synthetic_quality.py`. It preflights the four 26-image
  manifests and hashes, enforces the identical ordered source budget, extracts
  frozen Clean-FID clean-mode Inception-v3 features for FID and RBF MMD,
  computes calibrated LPIPS v0.1 with an ImageNet AlexNet trunk, and computes
  paired luminance SSIM/PSNR after sub-pixel phase-correlation translation on
  the valid overlap only. It also retains exact-copy, mean-pixel-difference,
  alignment, and per-class evidence.
- Because n=26 is much smaller than the 2048-dimensional feature space, the
  implementation uses an algebraically exact low-rank sample-FID computation
  rather than forming and square-rooting singular 2048x2048 covariance
  matrices. FID is explicitly labelled descriptive/high-variance at this
  sample size. MMD uses the same features with a deterministic pooled median
  RBF bandwidth and records both biased non-negative and unbiased MMD-squared.
- Command: `.venv\Scripts\python.exe -m py_compile
  src\evaluate_synthetic_quality.py`; result: pass. A synthetic translation
  unit check recovered dx=3.2204 and dy=-2.1659 for an applied 3.25/-2.5-pixel
  shift, with aligned mean absolute luminance error 0.001524 and phase response
  0.986865. Full metric execution awaits the detector run and frozen pretrained
  metric-weight availability.

### 2026-08-19 - Completed five-condition YOLO11n comparison

- Command `.venv\Scripts\python.exe
  src\run_yolo_full_comparison.py --all --skip-completed` exited with code 0
  after completing every pending condition. Grounded diffusion stopped after
  85/100 epochs under the fixed 20-epoch patience; validation selected epoch
  65. Training took 169.12 s and full training plus evaluation took 210.39 s.
- Grounded diffusion produced real-test precision 0.5393930, recall 0.2638889,
  mAP50 0.2857796, and mAP50-95 0.1215365. Test class mAP50-95 was 0.0441654
  for aircraft, 0.0591333 for human, and 0.2613107 for ship. Its authoritative
  report is
  `results/detectors/label_010_v1/grounded_diffusion/final_report.json`.
- The completed single-seed ranking by the pre-declared primary real-test
  mAP50-95 is: (1) standard diffusion 0.2286528, (2) traditional 0.1727669,
  (3) GAN 0.1507039, (4) grounded diffusion 0.1215365, and (5) real-only
  0.0174055. Corresponding mAP50 values are 0.3734665, 0.3134163, 0.3426334,
  0.2857796, and 0.0454182. Thus standard diffusion is the observed winner for
  this fixed split and seed; this must not be represented as a multi-seed
  uncertainty estimate or a general significance claim.
- The runner verified the same initial checkpoint SHA-256, protocol, and
  real-only validation/test sets for all five conditions. Checkpoint selection
  used real validation only; test results were generated only at final
  evaluation and were not used for tuning. Authoritative consolidated files
  are `results/detectors/label_010_v1/comparison_report.json` and
  `results/detectors/label_010_v1/comparison_summary.csv`.

### 2026-08-19 - Full synthetic-quality results

- First command: `.venv\Scripts\python.exe
  src\evaluate_synthetic_quality.py`; result: expected network-permission
  failure while requesting the frozen Clean-FID Inception weights. The empty,
  incomplete output directory was verified and removed. The command was rerun
  with approved network access, downloaded the Clean-FID TorchScript Inception
  and ImageNet AlexNet weights into the workspace metric/model caches, and
  exited with code 0. No metric fallback or random feature network was used.
- FID (lower is better) ranked GAN 25.0966, traditional 80.4437, standard
  diffusion 149.3603, and grounded diffusion 205.7946. Biased non-negative
  Inception-feature RBF MMD-squared values were 0.0038070, 0.0134830,
  0.0238331, and 0.0323414 in the same method order. These n=26 distributional
  estimates are descriptive and high-variance.
- Paired source/candidate mean LPIPS values were 0.006636 GAN, 0.140960
  traditional, 0.140201 standard diffusion, and 0.176859 grounded diffusion.
  Mean aligned luminance SSIM was respectively 0.979608, 0.626719, 0.579402,
  and 0.520213; mean aligned PSNR was 39.0902, 21.7502, 21.5397, and 21.2663
  dB. Mean absolute source difference on the 0--255 scale was 2.5970, 18.4310,
  14.5174, and 15.2462. Every method had zero exact pixel copies.
- GAN is therefore quantitatively very source-preserving, which supports the
  user's visual observation, but the detector result shows that source
  similarity is not equivalent to augmentation utility: standard diffusion
  achieved the best real-test mAP50-95 despite a worse FID and much lower
  paired similarity than GAN. Authoritative files are
  `results/quality/label_010_v1/quality_report.json`, `method_summary.csv`, and
  `per_sample_metrics.csv`.

### 2026-08-19 - Grounded-diffusion ablation protocol freeze

- Froze `experiments/label_010_v1/grounded_ablation_protocol.json` before new
  ablation training. Conditions are full (geometric+range+speckle),
  no-acoustic (geometric only), no-geometric (range+speckle only), and neither
  (epsilon only). Full maps exactly to the existing grounded model and neither
  maps exactly to the existing standard model; only no-acoustic and
  no-geometric require new training.
- The two new variants use the same optical pretrained backbone, seed 42,
  26/54 train/validation images, AdamW at 1e-5, batch 1 with gradient
  accumulation 4, validation every 500 optimiser steps, and two consecutive
  non-improving validation checks as the stopping rule. This reproduces the
  evidence-review stopping boundary used to finalise the existing full and
  neither models.
- The four variants will be resampled with common random numbers (seed rule
  40042 plus ordered source index), the same 50-step DDIM/CFG-1.0/strength-0.30
  selection, one candidate per identical source, the unchanged annotation QC
  gates, and the unchanged YOLO protocol and real validation/test data.
- Extended `src/train_diffusion_full.py` with independently mapped
  no-acoustic/no-geometric objectives and explicit component metadata in
  checkpoints/reports. Commands: `.venv\Scripts\python.exe -m py_compile
  src\train_diffusion_full.py` and `.venv\Scripts\python.exe
  src\train_diffusion_full.py --help`; results: pass.

### 2026-08-19 - Evidence-derived dissertation figures

- Added `src/create_results_figures.py`, which reads only accepted detector
  JSON and quality CSV evidence and exports both 300-dpi PNG and vector PDF.
  Command: `.venv\Scripts\python.exe -m py_compile
  src\create_results_figures.py`; result: pass. Command:
  `.venv\Scripts\python.exe src\create_results_figures.py`; result: pass.
- Created aggregate real-test mAP bars, a per-class real-test mAP50-95
  heatmap, and a four-panel FID/LPIPS/aligned-SSIM/aligned-PSNR quality figure
  under `results/figures/label_010_v1`. All three PNGs were visually inspected
  after rendering; axes, labels, metric directions, values, and whitespace are
  readable. The ablation chart is deliberately deferred until all four
  ablation detector reports exist.

### 2026-08-19 - No-acoustic ablation training start

- Initial command: `.venv\Scripts\python.exe
  src\train_diffusion_full.py --condition no_acoustic --max-steps 20000
  --validation-every 500 --patience-validations 2
  --gradient-accumulation 4`; result: failed before model construction because
  the direct shell had not inherited the project-local Hugging Face cache
  variables. The created output directory was verified empty and removed.
  `src/train_diffusion_full.py` was corrected to set project-local `HF_HOME`,
  `HF_HUB_CACHE`, and `TORCH_HOME` before importing the model libraries; no
  weights or training state were produced by the failed attempt.
- The same command was restarted successfully from the cached optical
  checkpoint. At step 500, the first fixed-noise validation total was
  0.0686804, comprising epsilon 0.0627067 and unweighted geometric 0.0597365;
  range and speckle losses were exactly zero as required. This is the current
  selected minimum, stale validation count zero. Training remains active under
  the frozen two-validation patience.
- Step 1000 validation improved the no-acoustic total to 0.0634548 (epsilon
  0.0589132 and unweighted geometry 0.0454161), so step 1000 replaced step 500
  as the selected checkpoint and the stale-validation counter remained zero.
- Replaced the stale starter `README.md` with an evidence-aware reproduction
  guide covering the accepted data pipeline, primary detector/metric outputs,
  active ablation protocol, commands, artifact locations, and interpretation
  constraints. Pinned the non-PyTorch environment dependencies in
  `requirements.txt`; the CUDA 12.8 PyTorch/torchvision installation remains an
  explicit prior command so pip cannot silently select a different build.
- Commands: `.venv\Scripts\python.exe -m compileall -q src` and
  `.venv\Scripts\python.exe -m pip check`; results: all source files compiled
  and no broken requirements were found.
- A low-dimensional regression check first attempted Clean-FID's bundled
  `fid_from_feats`; that reference helper failed because it passes the removed
  SciPy `sqrtm(..., disp=False)` argument under installed SciPy 1.18.0. This
  does not affect the accepted evaluator, which does not call that helper.
  Repeating the test against a direct modern `scipy.linalg.sqrtm` reference on
  deterministic 26-by-5 random features gave low-rank FID 0.8809373258355748
  versus reference 0.8809373258355819, absolute difference
  7.11e-15. This independently verifies the algebraic low-rank computation.
- The no-acoustic fixed-noise validation improved again at step 1500: total
  0.0629393, epsilon 0.0579114, and unweighted geometry 0.0502794. Range and
  speckle remained exactly zero, and the stale-validation count remained zero.
  The best-model directory and atomic latest checkpoint were updated.
- `src/create_results_figures.py` was extended to produce the primary
  detector validation-history chart. The resulting
  `results/figures/label_010_v1/primary_detector_validation_curves.png` and
  vector PDF were generated and visually inspected together with the three
  previously recorded figures.
- No-acoustic step 2000 validation total was 0.0631092 (epsilon 0.0575728,
  unweighted geometry 0.0553633), which did not improve the step-1500
  selected value of 0.0629393. This was the first consecutive non-improving
  validation; the frozen stopping counter became one of two. Disabled range
  and speckle losses remained exactly zero.
- No-acoustic step 2500 validation improved narrowly but validly to total
  0.0628517 (epsilon 0.0564105, unweighted geometry 0.0644128). Because the
  improvement exceeded the frozen `1e-5` minimum-delta threshold, step 2500
  replaced step 1500 as the selected checkpoint and the stopping counter was
  reset to zero. The run was deliberately not stopped at this convenient
  boundary.
- Strengthened `src/build_yolo_ablation_datasets.py` before using it: the
  build will now reject a manifest unless its component switches match the
  frozen protocol and all four arms have an identical ordered
  source/seed/CFG/strength/inference-step signature. This supplements the
  existing 26+26 budget, QC, fixed-evaluation, and hash-leakage gates.
- A read-only PowerShell confusion-matrix extraction first failed with an
  `empty pipe element` parser error because a `foreach` block was piped
  directly. It was rerun using an assigned collection and succeeded; no file
  was changed by the failed diagnostic. Visual inspection of the authoritative
  standard-diffusion test matrix confirmed axes `Predicted` by `True`. At the
  plotted operating point it recorded 30/42 ship objects correct, 12 ship
  misses, 15 background-to-ship false positives, four aircraft-to-ship errors,
  four aircraft misses, two human-to-ship errors, and four human misses. This
  fixed-threshold evidence was added to the chapter with the explicit caveat
  that AP integrates over confidence thresholds.
- Visually inspected the standard-diffusion real-test
  `val_batch0_pred.jpg` beside `val_batch0_labels.jpg`. The batch contains
  several correctly localised ships and an obvious missed annotated target
  (`000256`), so it was selected in the figure guide as a representative
  success/error example instead of cherry-picking an all-success mosaic.
- The same read-only PowerShell `foreach`-pipeline syntax mistake recurred
  while extracting the existing full/neither diffusion report fields. It was
  immediately rerun using an assigned collection and succeeded; again, no
  artifact was changed. The accepted existing grounded/full model reports
  best total 0.0687848 at step 1000 (training stopped at 2000; weights SHA-256
  `5be69f805cc3ed5a7803e56a28e1f1c80157be5d18261b106f75c1f810920105`),
  while standard/neither reports best epsilon objective 0.0580052 at step 1500
  (stopped at 2500; weights SHA-256
  `857479a72fd1e3447bfae56d21ab087220a8ade85c1444aa92acf20682dc459e`).
- No-acoustic step 3000 validation total was 0.0631923 (epsilon 0.0576266,
  unweighted geometry 0.0556572), above the selected step-2500 value. The
  consecutive non-improvement count became one of two; disabled range and
  speckle losses were still exactly zero.
- No-acoustic step 3500 produced a new selected minimum: validation total
  0.0622829, epsilon 0.0558083, and unweighted geometry 0.0647465. Although
  the geometry term rose, the pre-declared weighted total improved, so the
  checkpoint was accepted and patience reset. Range and speckle remained
  exactly zero. Training continued rather than being stopped selectively.
- Mid-run resource command: `Get-PSDrive -Name D` plus
  `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu
  --format=csv,noheader`. Result during no-acoustic training: D: had 103.62 GB
  free; the RTX 4060 reported 8188 MiB total, 2859 MiB used, 5098 MiB free,
  50% instantaneous utilisation, and 66 C. Resource headroom was sufficient,
  so training was not truncated for storage or VRAM reasons.
- No-acoustic step 4000 validation total was 0.0643353 (epsilon 0.0581355,
  unweighted geometry 0.0619984), above the selected step-3500 minimum. The
  consecutive non-improvement counter became one of two. The selected model
  remained unchanged and both disabled losses remained exactly zero.

### 2026-08-19 - No-acoustic ablation training completion and verification

- The active no-acoustic command exited with code 0 after step 4500. Step
  4500 validation total was 0.0626777, the second consecutive non-improvement
  after the selected step-3500 minimum 0.0622829, so the frozen early-stopping
  rule was satisfied. Elapsed time was 4481.87 s and peak allocated CUDA
  memory was 2450.01 MB.
- Command: `.venv\Scripts\python.exe
  src\verify_ablation_diffusion_training.py --condition no_acoustic`;
  result: pass. The verifier matched the frozen protocol, replayed all nine
  validation decisions and patience counters using minimum delta `1e-5`,
  matched the latest checkpoint to step 4500, confirmed four class embeddings,
  and confirmed range/speckle losses were exactly zero throughout. Best weights
  SHA-256 is
  `42c96642e1f2bc85a92c34d223a833941f41d22c359bf02a74fd4ddac7c58ac3`;
  latest checkpoint SHA-256 is
  `c785256d8717078bd05023490d2937744cd9cc4a756efeb0211bd7383dd3020f`.
  Authoritative evidence is
  `models/diffusion/ablations/no_acoustic/label_010_v1/verified_training_report.json`.

### 2026-08-19 - No-geometric ablation training start

- Preflight confirmed the intended output directory did not exist and D: still
  had 103.62 GB free. Started `.venv\Scripts\python.exe
  src\train_diffusion_full.py --condition no_geometric --max-steps 20000
  --validation-every 500 --patience-validations 2
  --gradient-accumulation 4`. The cached optical DDPM provides legacy binary
  weights rather than safetensors, so Diffusers emitted its expected
  safe-serialization fallback warning while loading the same frozen local
  checkpoint; training then started and remains active.
- At no-geometric step 50 the unweighted range and speckle losses were
  positive (0.0792850 and 0.309251), while geometry was exactly zero as
  required. The first fixed validation at step 500 selected total 0.0690103,
  comprising epsilon 0.0623486, range 0.0369015, and speckle 0.0963316;
  geometry remained exactly zero and the stale-validation count was zero.
- No-geometric step 1000 validation improved the selected total to 0.0659089
  (epsilon 0.0593434, range 0.0368406, speckle 0.0944691). Geometry remained
  exactly zero, both enabled acoustic losses remained positive, and the
  stale-validation count remained zero.
- No-geometric step 1500 improved the selected validation total to 0.0644038
  (epsilon 0.0575417, range 0.0388118, speckle 0.0984309). Geometry remained
  exactly zero and patience remained reset.
- No-geometric step 2000 validation total was 0.0647566 (epsilon 0.0578560,
  range 0.0422634, speckle 0.0957485), above the selected step-1500 value.
  This was the first consecutive non-improving validation. Geometry remained
  exactly zero.

### 2026-08-19 - No-geometric ablation completion and verification

- No-geometric step 2500 validation total was 0.0644667, the second
  consecutive non-improvement after step 1500; the command exited with code 0
  and selected step 1500 total 0.0644038. Training elapsed 2441.00 s and peak
  allocated CUDA memory was 2450.01 MB.
- Command: `.venv\Scripts\python.exe
  src\verify_ablation_diffusion_training.py --condition no_geometric`;
  result: pass. The verifier matched the frozen protocol, replayed all five
  validation decisions and patience counters, matched the latest checkpoint to
  step 2500, confirmed four class embeddings, confirmed geometry was exactly
  zero throughout, and confirmed both enabled acoustic losses were positive.
  Best weights SHA-256 is
  `8114b2adfebf184e35775226e65ac682534c3e189e8d7bee9d489a2da546a046`;
  latest checkpoint SHA-256 is
  `e13ad05f94226f1cf3617430e23b35c64b3b5603f0f7013a18e98fa96a1e80f3`.
  Authoritative evidence is
  `models/diffusion/ablations/no_geometric/label_010_v1/verified_training_report.json`.
- Command `.venv\Scripts\python.exe -m compileall -q src` passed after all
  new ablation, verifier, detector-export, and figure-generator edits.
- Command `.venv\Scripts\python.exe src\run_yolo_full_comparison.py --all
  --skip-completed` skipped all five accepted primary training directories,
  reconstructed the same ranking and metrics from their immutable reports,
  and created the new machine-readable
  `results/detectors/label_010_v1/per_class_summary.csv`. No detector was
  retrained and the primary comparison values did not change.

### 2026-08-19 - Common-random-number ablation synthesis start

- Preflight confirmed none of the four intended synthesis output directories
  existed. Started `.venv\Scripts\python.exe
  src\synthesize_diffusion_ablation.py --condition full` using the verified
  all-component grounded weights, ordered frozen sources, seeds `40042+i`,
  50 DDIM steps, CFG 1.0, and strength 0.30. Diffusers emitted a conservative
  float32 casting advisory; the model is intentionally sampled in float32 and
  no precision fallback was made.

### 2026-08-19 - Common-seed synthesis, QC, and ablation dataset build

- Commands `.venv\Scripts\python.exe src\synthesize_diffusion_ablation.py
  --condition <condition>` completed with exit code 0 for full, no-acoustic,
  no-geometric, and neither. Each generated exactly 26 candidates from the
  same ordered sources using seeds `40042+i`, CFG 1.0, strength 0.30, and 50
  DDIM steps. Exact pixel copies were zero in every arm. Mean absolute source
  change on the 0--255 scale was 15.2357 full, 14.9322 no-acoustic, 14.8901
  no-geometric, and 14.5151 neither. Each generation report records the exact
  checkpoint SHA-256.
- Ran `src\validate_synthetic_annotations.py` separately on each of the four
  manifests. All commands exited with code 0 and every arm returned 26
  approved, zero rejected under the unchanged phase, shift, target-NCC, edge,
  and contrast-polarity gates. Therefore all 104 generated candidates were
  eligible for downstream use; no rejected sample was replaced or hidden.
- Command `.venv\Scripts\python.exe src\build_yolo_ablation_datasets.py`;
  result: pass. Every arm contains 52 training images (26 real plus 26
  synthetic), 52 training objects (aircraft 8, human 6, ship 38), and the same
  54 real validation plus 54 real test images. The builder confirmed frozen
  component switches, identical ordered common-random-number signatures,
  identical evaluation hashes, and zero train/evaluation exact-image leakage.
  Each training set has 51 unique hashes because the same known exact-duplicate
  pair remains wholly inside real training; it is shared by all arms and does
  not cross a partition.
- Command `.venv\Scripts\python.exe src\create_ablation_preview.py`; result:
  pass. Created
  `results/ablations/label_010_v1/figures/common_source_ablation_preview.png`
  using the first two ordered sources per class and no test images. Visual
  inspection confirmed readable headers/labels and matched source rows; the
  close visual similarity across ablations is reported rather than concealed.

### 2026-08-19 - Four-condition ablation YOLO11n run start

- Started `.venv\Scripts\python.exe src\run_yolo_ablation_comparison.py
  --all --skip-completed`. The runner accepted the reconciliation report,
  loaded the same frozen YOLO11n checkpoint as the primary comparison, and
  began the full condition with the identical image size 640, AdamW 0.001,
  batch 8, maximum 100 epochs, patience 20, AMP, seed 42, deterministic mode,
  validation-selection, and final real-test protocol.

### 2026-08-19 - Grounded-ablation detector completion and results

- Command `.venv\Scripts\python.exe src\run_yolo_ablation_comparison.py
  --all --skip-completed` completed with exit code 0. All four detector reports
  have status pass, the identical YOLO11n starting SHA-256
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da663ae6417644ee1`,
  the same detector protocol, and real-only validation and test partitions.
  Test data were used only after validation checkpoint selection.
- Full completed 100 epochs and selected epoch 90. Its real-test precision,
  recall, mAP50, and mAP50--95 were 0.3736358, 0.4563492, 0.4227801, and
  0.2692419. Best detector checkpoint SHA-256 is
  `1ff220edaad1677e2c2b094692f4f7d69f52d79fdf1f0fccd626f274836ba27e`.
- No-acoustic completed 100 epochs and selected epoch 99. Its real-test
  precision, recall, mAP50, and mAP50--95 were 0.8687077, 0.3015873,
  0.3165625, and 0.1252602. Best detector checkpoint SHA-256 is
  `c862e6f7618e74496af5211093216020e63589b86f750213f711501b30436e8d`.
- No-geometric completed 100 epochs and selected epoch 90. Its real-test
  precision, recall, mAP50, and mAP50--95 were 0.3497855, 0.3988095,
  0.3503729, and 0.1897237. Best detector checkpoint SHA-256 is
  `e1da61a70591d2da21390799e5ec4012584b1ed85adba1c196105f637daae41e`.
- Neither stopped after 27 epochs under the fixed patience-20 rule and selected
  epoch 7. Its real-test precision, recall, mAP50, and mAP50--95 were
  0.0021125, 0.6269841, 0.0719665, and 0.0222421. The high recall is paired
  with near-zero precision and is not treated as strong detection. Best
  detector checkpoint SHA-256 is
  `f4dcb05ae62ca78f18e77d4bc1fa9ce4ed7c7851c2d68d9bbf4bf8fbc6acd2d6`.
- The authoritative real-test mAP50--95 ranking is therefore full 0.2692419,
  no-geometric 0.1897237, no-acoustic 0.1252602, and neither 0.0222421. The
  primary five-way grounded result is not substituted with the ablation full
  result: the ablation intentionally resampled every arm using a distinct
  common seed sequence and answers a separate matched-component question.
- Per-class real-test mAP50--95 was: full aircraft 0.1064601, human 0.3348816,
  ship 0.3663840; no-acoustic 0.0218879, 0.1173214, 0.2365714;
  no-geometric 0.1398310, 0.1453201, 0.2840200; and neither 0.0236467,
  0.0000000, 0.0430797. The full arm was strongest for human and ship, while
  no-geometric was strongest for aircraft.
- The full-arm confusion matrix was inspected with axes predicted-by-true. At
  the plotted operating point it recorded one of eight aircraft correct, none
  of six humans correct, and 31 of 42 ships correct. Five aircraft, four
  humans, and ten ships were missed; two aircraft and two humans were assigned
  to ship. This operating-point evidence is reported separately from AP,
  which integrates over confidence thresholds.
- The saved signed factorial mAP50--95 contrasts were: acoustic with geometry
  +0.1439816, acoustic without geometry +0.1674816, geometry with acoustic
  +0.0795182, geometry without acoustic +0.1030181, and interaction -0.0234999.
  Corresponding mAP50 contrasts were +0.1062176, +0.2784064, +0.0724073,
  +0.2445961, and -0.1721888. These are descriptive one-seed contrasts rather
  than confidence intervals or significance tests.
- Authoritative outputs are
  `results/ablations/label_010_v1/detectors/comparison_report.json`,
  `comparison_summary.csv`, `per_class_summary.csv`, and the four condition
  `final_report.json` files. A read-only PowerShell extraction initially
  repeated the known direct-`foreach` pipeline parser error; it was rerun using
  an assigned collection and no experiment artifact was affected.

### 2026-08-19 - Final ablation figures and chapter evidence update

- Command `.venv\Scripts\python.exe src\create_results_figures.py`; result:
  pass. It regenerated three primary detector figures and the synthetic-quality
  summary, and created
  `results/figures/label_010_v1/grounded_ablation_detector_map.{png,pdf}` plus
  `grounded_ablation_per_class_map95.{png,pdf}` directly from accepted JSON/CSV
  evidence. Both ablation figures were visually inspected; labels, rounded
  values, ordering, and colour scales matched the machine-readable reports.
- Visually inspected
  `results/ablations/label_010_v1/detectors/full/evaluate_test/confusion_matrix.png`
  and `val_batch0_pred.jpg`. The mosaic shows multiple correctly localised ship
  detections across varied SCTD imagery, while the confusion matrix confirms
  the minority-class limitation described above; neither inspection was used
  to change a checkpoint or threshold.
- Updated `docs/implementation_chapter.tex` and
  `docs/implementation_chapter.md` with the executed ablation training evidence,
  104-candidate QC outcome, aggregate and per-class detector results, signed
  factorial contrasts, figures, operating-point error analysis, single-seed
  limitations, and the distinction between primary grounded and common-seed
  ablation full results. Updated `README.md` status and authoritative output
  paths. No unexecuted 25%, 50%, or 100% detector result was introduced.

### 2026-08-19 - Final implementation integrity verification

- Commands `.venv\Scripts\python.exe -m compileall -q src` and
  `.venv\Scripts\python.exe -m pip check` both exited 0; all Python sources
  compiled and pip reported `No broken requirements found.`
- Re-read all five primary and four ablation detector `final_report.json`
  files. Every report had status pass and all nine recorded the same initial
  YOLO11n SHA-256. Both comparison reports independently confirmed identical
  protocol, identical initial weights, and real-only validation/test use.
  The no-acoustic and no-geometric diffusion verification reports also retained
  status pass.
- Re-read the synthetic-quality report: status pass and exact pixel copies were
  zero for traditional, GAN, standard diffusion, and grounded diffusion. The
  ablation dataset reconciliation report retained status pass. No required
  primary, quality, ablation, figure, or chapter artifact was missing.
- LaTeX structural check found 603 opening and 603 closing unescaped braces,
  48 `begin` and 48 `end` environments with matching environment multisets,
  and all seven `includegraphics` paths existed. No TeX engine was installed on
  the workstation (`pdflatex`, `lualatex`, and `xelatex` were all unavailable),
  so a binary PDF compilation could not be executed locally; this is an
  environment limitation, not a fabricated successful compile.
- Final resource check: D: had 101.65 GB free. `nvidia-smi` reported the NVIDIA
  GeForce RTX 4060 Laptop GPU with 8188 MiB total, 0 MiB used by the completed
  runs, 7956 MiB free, 0% utilisation, and 40 C. All training processes had
  exited successfully.

### 2026-08-19 - GAN and diffusion training-curve evidence

- Inspected the complete append-only model histories before plotting. The GAN
  log contains 56 epoch records. Standard diffusion contains 51 training
  records and five fixed-validation records; full grounded diffusion contains
  40 training and four validation records. The verified no-acoustic and
  no-geometric reports contain nine and five validation records respectively.
  No model retraining was required.
- Added `src/create_generative_training_figures.py`. The program reads only the
  accepted JSONL histories and JSON reports, reconstructs GAN and diffusion
  total objectives from their components, verifies report-selected checkpoints
  against the logs, and writes 300-dpi PNG plus vector PDF figures. It refuses
  missing fields, mismatched selected checkpoints, or objective reconstruction
  residuals above `1e-6`.
- Command `.venv\Scripts\python.exe
  src\create_generative_training_figures.py`; result: pass. Created
  `results/figures/label_010_v1/generative_training/gan_training_curves`,
  `standard_diffusion_training_curves`,
  `grounded_diffusion_training_curves`, and
  `diffusion_ablation_validation_curves` as PNG/PDF pairs. The authoritative
  figure evidence report is `generative_training_figure_report.json`.
- GAN objective reconstruction residual was `2.27e-8`. The plotted minimum
  validation L1 is 0.00715732 at epoch 31, matching the training report; the
  early-stopped final epoch is 56. Standard diffusion objective reconstruction
  residual was exactly zero and the selected validation checkpoint is step
  1500 at 0.05800524. Full grounded residual was `5.03e-8` and the selected
  within-objective total is step 1000 at 0.06878475.
- The four-panel ablation figure uses the verified selected steps: full 1000,
  no-acoustic 3500, no-geometric 1500, and neither 1500. Each panel displays
  its own total and epsilon history. A footer and chapter caption explicitly
  prohibit treating unlike total objectives as a cross-arm performance rank.
- All four figures were visually inspected. Titles, axes, legends, selected
  checkpoint stars, annotations, and footnotes were readable and matched the
  machine-readable evidence. No terminal screenshot or manually transcribed
  curve value was used.

### 2026-08-19 - Matched diffusion-process figure

- Added `src/create_diffusion_process_figure.py` and ran
  `.venv\Scripts\python.exe src\create_diffusion_process_figure.py`; result:
  pass. This inference-only program loaded the accepted standard and full
  grounded checkpoints locally, used training source `000143.png` (ship),
  seed 20049, CFG 1.0, strength 0.30, and a requested 50-step DDIM schedule.
  Strength 0.30 starts at timestep 300 and therefore executes 16 reverse
  updates; states were captured at timesteps 300, 220, 140, 80, and 0.
- Both rows were verified to share an exactly identical initial noisy tensor.
  Standard weights SHA-256 was
  `857479a72fd1e3447bfae56d21ab087220a8ade85c1444aa92acf20682dc459e`;
  grounded weights SHA-256 was
  `5be69f805cc3ed5a7803e56a28e1f1c80157be5d18261b106f75c1f810920105`.
  Mean final source differences were 8.2973/255 and 9.4304/255 for this
  illustrative sample. These values are not substituted for the 26-image
  quality audit.
- Diffusers printed its advisory that direct `.to(float32)` can be inconsistent
  for modules that must remain float32; the reported affected-module list was
  empty for both models and inference completed with finite outputs. No dtype
  fallback or model change occurred.
- Created and visually inspected
  `matched_diffusion_denoising_process.{png,pdf}`. The figure clearly shows the
  source, common noised state, 25%, 50%, 75%, and final states for both models.
  Its report is `matched_diffusion_denoising_process_report.json` and labels it
  as explanatory evidence rather than an evaluation result.
- The first attempt to patch the Markdown chapter failed a context check
  because backslashes in the patch anchor were interpreted before matching.
  No file was changed by that failed attempt. The patch was reapplied with
  stable anchors and succeeded.
- Integrated all five new figures into `docs/implementation_chapter.tex` and
  `docs/implementation_chapter.md`, added them to the screenshot/figure guide,
  created reusable `docs/generative_training_figures.tex`, and documented the
  reproduction commands in `README.md`.
- Final checks: `compileall` passed, `pip check` reported no broken
  requirements, both new figure reports retained status pass, the matched
  process report confirmed a common initial state, all 12 LaTeX graphic paths
  existed, and the chapter had 633 balanced unescaped brace pairs plus 53
  matching begin/end environments. No TeX engine is installed, so binary PDF
  compilation remains unavailable locally.

### 2026-08-19 - Provenance correction and archived-output diffusion figure

- A provenance review established that the preceding matched-seed process
  figure was created after the experiments. Its intermediate states were
  deterministic illustrative inference states, not states archived during the
  original synthetic-data run. The standard final panel used the archived
  candidate seed 20049 and reproduced
  `l010_standard_diffusion_000143.png` exactly. The grounded row had instead
  used the same illustrative seed 20049, whereas the archived grounded
  candidate was generated with seed 30049. The earlier figure must therefore
  not be described as showing both original archived candidates.
- Revised `src/create_diffusion_process_figure.py` to read the actual seeds and
  candidate identifiers from
  `experiments/label_010_v1/candidate_registry.csv`. Standard diffusion now
  uses seed 20049 and grounded diffusion uses seed 30049. The two initial noisy
  states intentionally differ because each belongs to its own recorded
  candidate.
- Added a mandatory provenance gate: the reconstructed final tensor is
  quantised using the same RGB rule as the synthetic-image writer and compared
  pixel-for-pixel with the archived PNG. Figure generation aborts on any
  mismatch. The displayed final panels are the archived PNG files themselves;
  the initial and intermediate panels are labelled as post-hoc deterministic
  reconstructions.
- Command `.venv\Scripts\python.exe
  src\create_diffusion_process_figure.py`; result: pass. Both archived-output
  checks reported `final_pixel_identical_to_archived: true`, maximum absolute
  pixel difference 0, and mean absolute pixel difference 0.0. Verified hashes:
  standard
  `44df3f5f01f5456d13ba0fdb89642c4c03586370b6553b0c1ba7c29d8813e351`;
  grounded
  `87f9902233e8cf806d7fccd5b51538ea46c7b8c6bcab244b9cadc42b827c74c5`.
- Replaced `matched_diffusion_denoising_process.{png,pdf}` with the corrected
  archived-output version and updated its machine-readable report. Revised the
  LaTeX, Markdown, reusable-figure captions, and figure guide to state the
  separate recorded seeds and the post-hoc status of the intermediate states.
- `compileall` passed. Poppler identified the PDF as a one-page, unencrypted
  941.047 x 378.428 point document and rendered it successfully to PNG for
  visual inspection. The headings, seed labels, six columns, both rows, and
  two-line provenance footer were legible with no clipping or overlap. Final
  SHA-256 hashes were
  `4bb88559ef859faecb52781b88b09021cc0052342166d23fd5a54bc524214a60`
  for the PNG and
  `fd1f9a82c5d2114b8cbe5cf6dd5babef644d8a3fd17fe995baca5507b338abeb`
  for the PDF.

### 2026-09-26 - Curated GitHub publication package

- Created a separate, non-destructive publication folder at
  `SCTD_GitHub_Ready`; no file in the executed experiment workspace was moved,
  renamed or overwritten. The folder has not been initialised as a Git
  repository and has not been uploaded to an external service.
- Inspected all local imports with `rg -n "^(from|import) " src -g '*.py'`.
  Because accepted scripts import neighbouring modules such as
  `diffusion_baseline`, `grounded_diffusion`, and
  `run_yolo_full_comparison` by filename, the 32 accepted scripts were copied
  together under `SCTD_GitHub_Ready/src` to preserve the executed interface.
  Three development-only pilot scripts were copied to `archive/pilots` and the
  rejected leakage-prone `create_splits.py` was copied to
  `archive/rejected_provisional_split` with explicit warnings.
- Copied `requirements.txt`, the environment activation script, frozen
  experiment registry, ablation protocol and this implementation log. The
  original planning configuration files were preserved byte-for-byte in
  `configs/archive`; SHA-256 checks confirmed that both still match the hashes
  frozen in `experiments/label_010_v1/registry.json`. Added separate executed
  configuration summaries so the historical gradient-accumulation planning
  values are not misrepresented as executed settings.
- Copied only lightweight evidence: dataset-audit and split metadata, training
  logs/reports without weights, generation/QC manifests without generated
  images, detector summaries/training curves without checkpoints, quality and
  ablation tables, resource checks, and the dissertation figures. Dataset
  images/XML files, processed dataset copies, synthetic collections, model
  checkpoints, pretrained weights and numerical feature caches were excluded.
- Two initial evidence-copy commands used `Copy-Item -LiteralPath` with a `*`
  wildcard and failed because `-LiteralPath` does not expand wildcards. The
  failures affected only the experiment-registry files and figures; no source
  or destination file was deleted or overwritten. Both copies were rerun with
  `Get-ChildItem -LiteralPath ... | Copy-Item` and succeeded.
- Added a repository README, complete code map, dataset-availability notice,
  publication checklist, appendix-ready LaTeX fragment and restrictive
  `.gitignore`. A software licence and `CITATION.cff` were deliberately not
  invented because the author/licence decision has not yet been supplied.
- Verification command parsed every Python source with `ast.parse`, every JSON
  file with `json.loads`, and every YAML file with `yaml.safe_load`; result:
  PASS for 36 Python, 46 JSON and 14 YAML files. `.venv\Scripts\python.exe -m
  pip check` reported no broken requirements.
- SHA-256 comparison verified that all 36 original Python files have an
  identical packaged copy and reported zero missing or mismatched files. A
  recursive exclusion check found no `.pt`, `.pth`, `.ckpt`, `.safetensors`,
  `.onnx`, XML, or dataset-image artifacts in the package. Final package size
  before copying this updated log was 194 files and 25.35 MiB; the largest file
  was the 8.86 MiB corrected diffusion-process PNG.
- Representative packaged entry points were invoked with `--help` to exercise
  their relocated local imports. GAN training, diffusion training/synthesis,
  annotation QC and both YOLO runners all returned exit code 0. During an
  initial concurrent check the ablation runner alone returned exit code 1 with
  a traceback while the other processes were importing; an immediate isolated
  rerun returned exit code 0 and displayed the expected argument interface.
  No code or evidence file was modified in response because the isolated test
  demonstrated that the packaged import path was valid.
- The import checks created an ignored Matplotlib configuration cache and four
  Python bytecode files inside the package. A first read-only cleanup-target
  verification command repeated the PowerShell `foreach`-pipeline parser error
  (`empty pipe element`) and made no change. The verification was rerun without
  piping the loop, confirmed both exact resolved targets were inside
  `SCTD_GitHub_Ready`, and the two runtime-only directories were removed.
- Final safety scan: 195 files, 25.35 MiB, 36 Python files, zero files over
  10 MiB, zero checkpoint/dataset artifacts, and zero matches for common API
  key, access-token, client-secret or password assignment patterns outside the
  historical log/evidence exclusions. The package remains local and unversioned
  pending the user's GitHub repository and licence choices.
