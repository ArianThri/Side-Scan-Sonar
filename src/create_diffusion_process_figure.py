"""Render verified standard/grounded DDIM trajectories for Chapter 5.

The final panel in each row is the archived synthetic candidate.  The preceding
states are deterministically reconstructed from the same saved checkpoint,
source, class, sampler settings, and candidate-specific random seed.  Generation
fails unless the reconstructed final pixels exactly match the archived PNG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".config" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image
from torchvision import transforms

from diffusion_baseline import CLASSES, NULL_CLASS, seed_everything


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(-1, 1).add(1).div(2)[0]
    return array.permute(1, 2, 0).numpy()


def quantized_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Apply torchvision.save_image's RGB quantisation to a [-1, 1] tensor."""
    return (
        tensor.detach().float().cpu()[0]
        .add(1).div(2).mul(255).add(0.5).clamp(0, 255)
        .permute(1, 2, 0).to(torch.uint8).numpy()
    )


@torch.no_grad()
def trajectory(
    model: UNet2DModel,
    source: torch.Tensor,
    class_ids: torch.Tensor,
    training_scheduler: DDPMScheduler,
    inference_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[int], int]:
    scheduler = DDIMScheduler.from_config(training_scheduler.config)
    scheduler.set_timesteps(inference_steps, device=source.device)
    start_index = min(max(round((1.0 - strength) * (inference_steps - 1)), 0), inference_steps - 1)
    timesteps = scheduler.timesteps[start_index:]
    generator = torch.Generator(device=source.device).manual_seed(seed)
    noise = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype)
    current = scheduler.add_noise(source, noise, timesteps[0].expand(source.shape[0]))
    states = [current.detach().cpu()]
    labels = [int(timesteps[0])]
    capture_indices = sorted({max(0, min(len(timesteps) - 1, round((len(timesteps) - 1) * fraction))) for fraction in (.25, .50, .75, 1.0)})
    for index, timestep in enumerate(timesteps):
        null_ids = torch.full_like(class_ids, NULL_CLASS)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            unconditional = model(current, timestep, class_labels=null_ids).sample
            conditional = model(current, timestep, class_labels=class_ids).sample
        prediction = unconditional.float() + guidance_scale * (conditional.float() - unconditional.float())
        current = scheduler.step(prediction, timestep, current.float()).prev_sample
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"Non-finite DDIM state at timestep {int(timestep)}")
        if index in capture_indices:
            states.append(current.detach().cpu())
            labels.append(int(timestep))
    if len(states) != 5:
        raise RuntimeError(f"Expected initial state plus four captures, obtained {len(states)}")
    return states, labels, len(timesteps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "generative_256_v1" / "images" / "train" / "000143.png",
    )
    parser.add_argument("--class-name", choices=CLASSES, default="ship")
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument(
        "--registry", type=Path,
        default=PROJECT_ROOT / "experiments" / "label_010_v1" / "candidate_registry.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "label_010_v1" / "generative_training",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing an unrecorded CPU fallback")
    source_path = args.source.resolve()
    registry_path = args.registry.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection_path = PROJECT_ROOT / "results" / "full" / "label_010_v1" / "diffusion_sampling_selection" / "selection_report.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected_configuration", selection)
    guidance_scale = float(selected["guidance_scale"])
    strength = float(selected["strength"])
    if (guidance_scale, strength) != (1.0, 0.30):
        raise ValueError("The frozen validation-selected CFG/strength pair is not 1.0/0.30")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    with Image.open(source_path) as image:
        source_rgb = image.convert("RGB")
        source_array = np.asarray(source_rgb).copy()
        source = transform(source_rgb).unsqueeze(0).to("cuda")
    class_ids = torch.tensor([CLASSES.index(args.class_name)], device="cuda")
    training_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon")

    method_by_condition = {
        "standard": "standard_diffusion",
        "grounded": "grounded_diffusion",
    }
    candidate_rows: dict[str, dict[str, str]] = {}
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for condition, method in method_by_condition.items():
                if (
                    row["method"] == method
                    and row["source_id"] == source_path.stem
                    and row["class_name"] == args.class_name
                ):
                    candidate_rows[condition] = row
    missing = sorted(set(method_by_condition) - set(candidate_rows))
    if missing:
        raise RuntimeError(f"Candidate registry has no matching row for: {missing}")

    conditions = tuple(method_by_condition)
    rows: dict[str, list[torch.Tensor]] = {}
    timestep_labels: dict[str, list[int]] = {}
    model_hashes: dict[str, str] = {}
    trajectory_steps: dict[str, int] = {}
    final_differences: dict[str, float] = {}
    archived_outputs: dict[str, np.ndarray] = {}
    verification: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        candidate = candidate_rows[condition]
        seed = int(candidate["seed"])
        seed_everything(seed)
        model_dir = PROJECT_ROOT / "models" / "diffusion" / condition / "label_010_v1" / "best_model"
        weights = model_dir / "diffusion_pytorch_model.safetensors"
        model = UNet2DModel.from_pretrained(model_dir, local_files_only=True)
        if int(model.config.num_class_embeds) != 4:
            raise RuntimeError(f"Unexpected class embedding count for {condition}")
        model.eval().to("cuda", dtype=torch.float32)
        states, labels, step_count = trajectory(
            model, source, class_ids, training_scheduler, args.inference_steps,
            guidance_scale, strength, seed,
        )
        archived_path = (
            PROJECT_ROOT / "synthetic" / "label_010_v1" / method_by_condition[condition]
            / f"{candidate['candidate_id']}.png"
        )
        if not archived_path.is_file():
            raise FileNotFoundError(f"Archived candidate not found: {archived_path}")
        with Image.open(archived_path) as image:
            archived = np.asarray(image.convert("RGB")).copy()
        reconstructed = quantized_rgb(states[-1])
        absolute_difference = np.abs(reconstructed.astype(np.int16) - archived.astype(np.int16))
        identical = bool(np.array_equal(reconstructed, archived))
        if not identical:
            raise RuntimeError(
                f"{condition} reconstruction does not match archived candidate "
                f"(max absolute pixel difference={int(absolute_difference.max())})"
            )
        rows[condition] = states
        archived_outputs[condition] = archived
        timestep_labels[condition] = labels
        trajectory_steps[condition] = step_count
        model_hashes[condition] = sha256(weights)
        final_differences[condition] = float(
            np.abs(archived.astype(np.int16) - source_array.astype(np.int16)).mean()
        )
        verification[condition] = {
            "candidate_id": candidate["candidate_id"],
            "seed": seed,
            "archived_image": archived_path.relative_to(PROJECT_ROOT).as_posix(),
            "archived_image_sha256": sha256(archived_path),
            "final_pixel_identical_to_archived": identical,
            "max_absolute_pixel_difference": int(absolute_difference.max()),
            "mean_absolute_pixel_difference": float(absolute_difference.mean()),
        }
        del model
        torch.cuda.empty_cache()
    if timestep_labels["standard"] != timestep_labels["grounded"]:
        raise RuntimeError("The two archived-output reconstructions produced different timestep labels")

    column_titles = [
        "Source",
        f"Initial noised state\n$t={timestep_labels['standard'][0]}$",
        f"25% denoised\n$t={timestep_labels['standard'][1]}$",
        f"50% denoised\n$t={timestep_labels['standard'][2]}$",
        f"75% denoised\n$t={timestep_labels['standard'][3]}$",
        f"Final output\n$t={timestep_labels['standard'][4]}$",
    ]
    fig, axes = plt.subplots(2, 6, figsize=(13.2, 5.2))
    for row_index, condition in enumerate(conditions):
        # Render the archived PNG itself in the final column after proving that
        # the reconstructed final tensor quantises to the same RGB pixels.
        images: list[np.ndarray] = [
            display(source.detach().cpu()),
            *(display(tensor) for tensor in rows[condition][:-1]),
            archived_outputs[condition],
        ]
        for column, image in enumerate(images):
            axes[row_index, column].imshow(image)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor("#777777")
            if row_index == 0:
                axes[row_index, column].set_title(column_titles[column], fontsize=10)
        method_label = "Standard diffusion" if condition == "standard" else "Grounded diffusion"
        axes[row_index, 0].set_ylabel(
            f"{method_label}\nseed={verification[condition]['seed']}", fontsize=9,
        )
    fig.text(
        0.5, 0.028,
        f"Post-hoc deterministic reconstruction: class={args.class_name}; per-candidate archived seeds; CFG={guidance_scale:.1f}; strength={strength:.2f}; requested DDIM steps={args.inference_steps} ({trajectory_steps['standard']} executed).",
        ha="center", fontsize=8, color="#444444",
    )
    fig.text(
        0.5, 0.010,
        "Final panels are the archived synthetic PNGs and were verified pixel-identical to checkpoint reconstructions; intermediate states were reconstructed after training.",
        ha="center", fontsize=7.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.080, 1, 1))
    png = output / "matched_diffusion_denoising_process.png"
    pdf = output / "matched_diffusion_denoising_process.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report: dict[str, Any] = {
        "status": "pass",
        "purpose": "post-hoc deterministic DDIM trajectory reconstruction ending in verified archived synthetic outputs; intermediate states were not logged during original training",
        "source": source_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": sha256(source_path),
        "class_name": args.class_name,
        "guidance_scale": guidance_scale,
        "source_strength": strength,
        "requested_inference_steps": args.inference_steps,
        "executed_trajectory_steps": trajectory_steps,
        "captured_timesteps": timestep_labels["standard"],
        "common_initial_noisy_state": False,
        "reason_initial_states_differ": "Each row uses the seed recorded for its own archived candidate.",
        "archived_output_verification": verification,
        "model_weights_sha256": model_hashes,
        "mean_absolute_final_source_difference_255": final_differences,
        "figure_png": png.relative_to(PROJECT_ROOT).as_posix(),
        "figure_pdf": pdf.relative_to(PROJECT_ROOT).as_posix(),
        "sampling_selection_report": selection_path.relative_to(PROJECT_ROOT).as_posix(),
        "sampling_selection_report_sha256": sha256(selection_path),
        "candidate_registry": registry_path.relative_to(PROJECT_ROOT).as_posix(),
        "candidate_registry_sha256": sha256(registry_path),
    }
    report_path = output / "matched_diffusion_denoising_process_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
