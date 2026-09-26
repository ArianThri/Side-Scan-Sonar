"""Resumable full-backbone standard/grounded diffusion training at 10% labels."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "models" / "cache" / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "cache" / "torch"))

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from diffusion_baseline import NULL_CLASS, build_conditioned_unet, seed_everything
from grounded_diffusion import (
    GroundedDataset, geometry_loss, log_speckle_loss, predicted_clean, range_profile_loss,
)


def atomic_save(payload, path):
    temporary=path.with_suffix(path.suffix+".tmp");torch.save(payload,temporary);os.replace(temporary,path)


COMPONENTS = {
    "standard": {"geometric": False, "range": False, "speckle": False},
    "grounded": {"geometric": True, "range": True, "speckle": True},
    "no_acoustic": {"geometric": True, "range": False, "speckle": False},
    "no_geometric": {"geometric": False, "range": True, "speckle": True},
}


def losses(model,clean,classes,boxes,timesteps,noise,scheduler,components):
    noisy=scheduler.add_noise(clean,noise,timesteps)
    prediction=model(noisy,timesteps,class_labels=classes).sample
    base=F.mse_loss(prediction.float(),noise.float())
    zero=base.new_zeros(())
    geometric=range_term=speckle=zero
    if any(components.values()):
        x0=predicted_clean(noisy.float(),prediction.float(),timesteps,scheduler)
        if components["geometric"]:
            geometric=geometry_loss(x0,clean.float(),boxes)
        if components["range"]:
            range_term=range_profile_loss(x0,clean.float())
        if components["speckle"]:
            speckle=log_speckle_loss(x0,clean.float())
    total=base+0.10*geometric+0.05*range_term+0.05*speckle
    return total,base,geometric,range_term,speckle


@torch.no_grad()
def validate(model,loader,scheduler,components,device):
    model.eval();sums=torch.zeros(5,device=device);count=0
    generator=torch.Generator(device=device).manual_seed(9876)
    for clean,classes,boxes,_ in loader:
        clean,classes,boxes=clean.to(device),classes.to(device),boxes.to(device)
        noise=torch.randn(clean.shape,device=device,generator=generator)
        timesteps=torch.randint(0,1000,(clean.shape[0],),device=device,generator=generator)
        with torch.amp.autocast("cuda",dtype=torch.float16):
            values=losses(model,clean,classes,boxes,timesteps,noise,scheduler,components)
        sums+=torch.stack([value.float() for value in values]);count+=1
    model.train();return [float(value/count) for value in sums]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--condition",choices=list(COMPONENTS),required=True)
    parser.add_argument("--max-steps",type=int,default=20000)
    parser.add_argument("--validation-every",type=int,default=500)
    parser.add_argument("--patience-validations",type=int,default=8)
    parser.add_argument("--gradient-accumulation",type=int,default=4)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();components=COMPONENTS[args.condition];seed_everything(42)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda")
    if args.output_dir:
        output=args.output_dir
    elif args.condition in {"standard", "grounded"}:
        output=Path(f"models/diffusion/{args.condition}/label_010_v1")
    else:
        output=Path(f"models/diffusion/ablations/{args.condition}/label_010_v1")
    if output.exists() and not args.resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    model,missing,unexpected=build_conditioned_unet("google/ddpm-celebahq-256",True)
    model.enable_gradient_checkpointing();model.to(device)
    optimiser=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),weight_decay=.01)
    lr_scheduler=get_cosine_schedule_with_warmup(optimiser,200,args.max_steps)
    scaler=torch.amp.GradScaler("cuda")
    noise_scheduler=DDPMScheduler(num_train_timesteps=1000,beta_schedule="linear",prediction_type="epsilon")
    train=GroundedDataset(Path("data/processed/generative_256_v1/manifest.csv"),Path("data/splits/limited_labels_group_aware_v1/train_010.txt"),26)
    val=GroundedDataset(Path("data/processed/generative_256_v1/manifest.csv"),Path("data/splits/group_aware_v1/val.txt"),54)
    train_loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=0)
    val_loader=DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
    step=start_epoch=0;best=float("inf");stale=0
    latest=output/"checkpoint_latest.pt"
    if args.resume:
        state=torch.load(latest,map_location=device,weights_only=False);model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"]);lr_scheduler.load_state_dict(state["lr_scheduler"]);scaler.load_state_dict(state["scaler"])
        step=state["step"];best=state["best_validation_total"];stale=state["stale_validations"]
    log_path=output/"training_log.jsonl";started=time.time();torch.cuda.reset_peak_memory_stats();optimiser.zero_grad(set_to_none=True)
    data_iterator=iter(train_loader);running=torch.zeros(5,device=device);micro_count=0
    while step<args.max_steps and stale<args.patience_validations:
        try:clean,classes,boxes,_=next(data_iterator)
        except StopIteration:data_iterator=iter(train_loader);clean,classes,boxes,_=next(data_iterator)
        clean,classes,boxes=clean.to(device),classes.to(device),boxes.to(device)
        dropped=torch.rand(classes.shape,device=device)<.10;training_classes=torch.where(dropped,torch.full_like(classes,NULL_CLASS),classes)
        noise=torch.randn_like(clean);timesteps=torch.randint(0,1000,(1,),device=device)
        with torch.amp.autocast("cuda",dtype=torch.float16):
            values=losses(model,clean,training_classes,boxes,timesteps,noise,noise_scheduler,components)
            scaled_loss=values[0]/args.gradient_accumulation
        if not all(torch.isfinite(value) for value in values):raise FloatingPointError(f"Non-finite loss before step {step+1}")
        scaler.scale(scaled_loss).backward();running+=torch.stack([value.detach().float() for value in values]);micro_count+=1
        if micro_count%args.gradient_accumulation:continue
        scaler.unscale_(optimiser);gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(gradient_norm):raise FloatingPointError(f"Non-finite gradient at step {step+1}")
        scaler.step(optimiser);scaler.update();optimiser.zero_grad(set_to_none=True);lr_scheduler.step();step+=1
        if step%50==0:
            means=(running/micro_count).tolist();row={"step":step,"train_total":means[0],"train_epsilon":means[1],"train_geometry":means[2],"train_range":means[3],"train_speckle":means[4],"lr":lr_scheduler.get_last_lr()[0],"gradient_norm":float(gradient_norm),"elapsed_seconds":time.time()-started,"peak_cuda_memory_mb":torch.cuda.max_memory_allocated()/1024**2}
            with log_path.open("a",encoding="utf-8") as handle:handle.write(json.dumps(row)+"\n")
            running.zero_();micro_count=0
            if step%500==0:print(json.dumps(row))
        if step%args.validation_every==0:
            validation=validate(model,val_loader,noise_scheduler,components,device);improved=validation[0]<best-1e-5
            if improved:best=validation[0];stale=0;model.save_pretrained(output/"best_model",safe_serialization=True)
            else:stale+=1
            validation_row={"step":step,"validation_total":validation[0],"validation_epsilon":validation[1],"validation_geometry":validation[2],"validation_range":validation[3],"validation_speckle":validation[4],"improved":improved,"stale_validations":stale}
            with log_path.open("a",encoding="utf-8") as handle:handle.write(json.dumps(validation_row)+"\n")
            state={"model":model.state_dict(),"optimiser":optimiser.state_dict(),"lr_scheduler":lr_scheduler.state_dict(),"scaler":scaler.state_dict(),"step":step,"best_validation_total":best,"stale_validations":stale,"condition":args.condition,"components":components,"seed":42}
            atomic_save(state,latest);print(json.dumps(validation_row))
    report={"status":"pass","purpose":f"full 10%-label {args.condition} diffusion training","condition":args.condition,"components":components,"loss_weights":{"epsilon":1.0,"geometric":0.10,"range":0.05,"speckle":0.05},"steps_completed":step,"max_steps":args.max_steps,"validation_every_steps":args.validation_every,"patience_validations":args.patience_validations,"gradient_accumulation":args.gradient_accumulation,"early_stopped":stale>=args.patience_validations,"best_validation_total":best,"train_images":len(train),"validation_images":len(val),"elapsed_seconds":time.time()-started,"peak_cuda_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"transfer_missing_keys":missing,"transfer_unexpected_keys":unexpected,"best_model":(output/"best_model").as_posix()}
    (output/"training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
