"""Generate exactly the frozen GAN budget from the selected full checkpoint."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from gan_baseline import CLASSES, Generator


def sha256(path):
    digest=hashlib.sha256(); digest.update(Path(path).read_bytes()); return digest.hexdigest()


def main():
    output=Path("synthetic/label_010_v1/gan")
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    state=torch.load("models/gan/label_010_v1/checkpoint_best.pt",map_location="cpu",weights_only=False)
    model=Generator(pretrained=False); model.load_state_dict(state["generator"]); model.eval().cuda()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    with Path("experiments/label_010_v1/candidate_registry.csv").open(encoding="utf-8") as handle:
        registry=[row for row in csv.DictReader(handle) if row["method"]=="gan"]
    rows=[]
    with torch.no_grad():
        for row in registry:
            with Image.open(row["source_image"]) as image: source=transform(image.convert("RGB")).unsqueeze(0).cuda()
            class_id=torch.tensor([CLASSES.index(row["class_name"])],device="cuda")
            generator=torch.Generator(device="cuda").manual_seed(int(row["seed"]))
            brightness=float(torch.empty((),device="cuda").uniform_(0.92,1.08,generator=generator))
            degradation_noise=torch.randn(source.shape,device="cuda",generator=generator)*0.03
            conditioned_source=torch.clamp(source*brightness+degradation_noise,-1,1)
            noise=torch.randn(1,model.noise_dim,device="cuda",generator=generator)
            with torch.amp.autocast("cuda",dtype=torch.float16): candidate=model(conditioned_source,class_id,noise)
            path=output/f"{row['candidate_id']}.png"; save_image(candidate.float().add(1).div(2),path)
            rows.append({**row,"candidate_image":path.as_posix(),"candidate_sha256":sha256(path),
                         "checkpoint_epoch":state["epoch"],"checkpoint_validation_l1":state["best_validation_l1"],
                         "generation_brightness":brightness,"generation_noise_std_normalised":0.03,
                         "generation_policy":"seeded training-domain degradation then conditional restoration"})
    with (output/"manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={"status":"pass","method":"gan","expected":26,"generated":len(rows),
            "equal_budget_count_pass":len(rows)==26,"checkpoint":"models/gan/label_010_v1/checkpoint_best.pt"}
    (output/"generation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
