"""Fine-tune TryOffDiff to map casual garment photos -> canonical garment images.

The objective (standard latent diffusion, "epsilon prediction"):

    z0   = VAE.encode(clean_garment) * 0.18215          # target, in latent space
    eps  ~ N(0, I);  t ~ Uniform{0..999}
    zt   = sqrt(abar_t) * z0 + sqrt(1 - abar_t) * eps   # forward (noising) process
    loss = || eps - net(zt, t, SigLIP(casual_photo)) ||^2

The ONLY change from the original TryOffDiff training is the condition: a crumpled/folded
garment photo instead of a person wearing it.

Survives preemption (Colab disconnects, CHTC evictions): checkpoints every `save_every`
steps and when a run pauses (trainable tensors + optimizer + AMP scaler state), and resumes
from the newest checkpoint in `out_dir`. Keeps the 2 newest checkpoints plus every
`keep_every`-th one, so you can compare quality across training later. The loss of every
step is appended to loss_history.jsonl, so the full curve survives resumes. (After an
eviction, steps run since the last checkpoint are repeated and logged twice: when plotting,
keep the last entry per step.)
`smoke: true` swaps in tiny random models so the whole loop runs on a laptop CPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from ..data.synthetic import SyntheticPairs
from ..models.components import build_components
from ..models.lora import configure_trainable, trainable_state_dict

SIGLIP_NORM = T.Normalize([0.5], [0.5])


def pyramid_noise(z0: torch.Tensor, discount: float, levels: int = 4) -> torch.Tensor:
    """Gaussian noise plus upsampled coarse noise at 1/2, 1/4, ... resolution (weight discount**i).
    SD v1's top timesteps still hold 7-9% of the target, enough to read the silhouette from z_t;
    strong coarse noise hides it, so the model has to get the shape from the condition."""
    b, ch, h, w = z0.shape
    eps = torch.randn_like(z0)
    for i in range(1, levels + 1):
        lo = torch.randn(b, ch, max(h >> i, 1), max(w >> i, 1), device=z0.device)
        eps = eps + F.interpolate(lo, size=(h, w), mode="bilinear") * discount ** i
    return eps / eps.std(dim=(1, 2, 3), keepdim=True)


def _ckpts(out: Path):
    return sorted(out.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def _save(out: Path, net, opt, scaler, step: int, mode: str, cfg: dict):
    tmp = out / f"step_{step}.pt.tmp"
    torch.save({"net": trainable_state_dict(net), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "mode": mode, "cfg": cfg}, tmp)
    tmp.replace(out / f"step_{step}.pt")          # atomic: an eviction mid-save can't leave a corrupt file
    keep_every = cfg.get("keep_every", 1500)
    for old in _ckpts(out)[:-2]:                  # keep the 2 newest + every keep_every-th
        if not keep_every or int(old.stem.split("_")[1]) % keep_every:
            old.unlink()


def train(cfg: dict) -> dict:
    smoke = cfg.get("smoke", False)
    dev = cfg.get("device", "cpu" if smoke else "cuda")
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0))

    c = build_components(dev, smoke=smoke)
    mode = cfg.get("mode", "lora")
    params = configure_trainable(c.net, mode, cfg.get("lora_rank", 16), cfg.get("lora_alpha", 16))
    c.net.train()
    if cfg.get("grad_checkpointing", not smoke):
        c.net.unet.enable_gradient_checkpointing()

    # Two learning rates: the pretrained adapter is the model's whole interface to the condition,
    # so it gets a much smaller lr than the fresh LoRA weights (adapter_lr_mult: 0 = frozen).
    lr, mult = float(cfg["lr"]), float(cfg.get("adapter_lr_mult", 0.1))
    named = [(n, p) for n, p in c.net.named_parameters() if p.requires_grad]
    unet_p = [p for n, p in named if n.startswith("unet.")]
    adapter_p = [p for n, p in named if not n.startswith("unet.")]
    if mult == 0:
        for p in adapter_p:
            p.requires_grad_(False)                 # not trained -> not saved in checkpoints either
        adapter_p = []
    groups = [{"params": ps, "lr": glr, "base_lr": glr, "weight_decay": wd} for ps, glr, wd in
              [(unet_p, lr, cfg.get("weight_decay", 0.01)), (adapter_p, lr * mult, 0.0)] if ps]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups)
    warmup = int(cfg.get("warmup_updates", 100))   # optimizer updates, not micro-batches
    print(f"[optim] lora/unet lr {lr:g}, adapter lr {lr * mult:g}, warmup {warmup} updates", flush=True)
    data = SyntheticPairs(cfg["data_roots"], size=512)

    # ---- resume ---------------------------------------------------------------------
    step = 0
    if _ckpts(out):
        st = torch.load(_ckpts(out)[-1], map_location="cpu", weights_only=False)
        # Refuse to mix runs: optimizer.load_state_dict() would silently bring back the OLD lr,
        # and the weights would continue from the old run.
        old = st.get("cfg", {})
        keys = ("lr", "adapter_lr_mult", "mode", "lora_rank", "lora_alpha", "batch_size", "grad_accum",
                "data_roots", "cond_dropout", "warmup_updates", "pyramid_discount")
        diff = {k: (old.get(k), cfg.get(k)) for k in keys if str(old.get(k)) != str(cfg.get(k))}
        if diff:
            raise SystemExit(f"[resume] {out} holds a run with other settings {diff}: use a new out_dir")
        missing, unexpected = c.net.load_state_dict(st["net"], strict=False)
        assert not unexpected, unexpected[:5]
        opt.load_state_dict(st["opt"]); step = st["step"]
        resume_scaler = st.get("scaler")
        print(f"[resume] {_ckpts(out)[-1].name}", flush=True)
    else:
        resume_scaler = None
    # Re-seed from the step: with a fixed seed every resumed chunk would replay the same data
    # order, noise and timesteps, i.e. train on the same first few thousand pairs again.
    torch.manual_seed(cfg.get("seed", 0) + step)
    dl = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4),
                    drop_last=True, pin_memory=dev.startswith("cuda"), persistent_workers=False)

    accum, total = cfg.get("grad_accum", 1), cfg["max_steps"]
    use_amp = dev.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if resume_scaler and use_amp:
        scaler.load_state_dict(resume_scaler)
    this_run = cfg.get("max_steps_this_run", 10**9)
    hist_f = (out / "loss_history.jsonl").open("a")
    history, t0, start = [], time.time(), step
    while step < total:
        for batch in dl:
            x = batch["input"].to(dev, non_blocking=True)        # casual photo, [0,1]
            y = batch["target"].to(dev, non_blocking=True)       # clean garment, [0,1]
            with torch.no_grad():
                cond = c.img_enc(pixel_values=SIGLIP_NORM(x)).last_hidden_state
                drop = torch.rand(cond.shape[0], device=dev) < cfg.get("cond_dropout", 0.1)
                cond = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)   # enables CFG
                z0 = c.vae.encode(y * 2 - 1).latent_dist.sample() * c.vae_scale
            pd = float(cfg.get("pyramid_discount", 0.0))
            eps = pyramid_noise(z0, pd) if pd > 0 else torch.randn_like(z0)
            t = torch.randint(0, c.train_scheduler.config.num_train_timesteps, (z0.shape[0],), device=dev)
            zt = c.train_scheduler.add_noise(z0, eps, t)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = c.net(zt, t, cond)
            loss = F.mse_loss(pred.float(), eps)
            scaler.scale(loss / accum).backward()
            if step == start and not any(p.grad is not None for p in params):
                raise RuntimeError("no gradients reached the trainable parameters (try grad_checkpointing=false)")

            if (step + 1) % accum == 0:
                upd = (step + 1) // accum                       # 1-based optimizer-update count
                for g in opt.param_groups:                      # linear warmup; also re-applies cfg lr
                    g["lr"] = g["base_lr"] * min(1.0, upd / max(warmup, 1))
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            history.append(loss.item())
            hist_f.write(json.dumps({"step": step, "loss": history[-1]}) + "\n")

            if step % cfg.get("log_every", 50) == 0:
                rate = (step - start) / max(time.time() - t0, 1e-6)
                print(f"step {step}/{total}  loss {loss.item():.4f}  ({rate:.2f} it/s)", flush=True)
            pausing = step - start >= this_run
            if step % cfg.get("save_every", 500) == 0 or step >= total or pausing:
                hist_f.flush()
                _save(out, c.net, opt, scaler, step, mode, cfg)
            if step >= total or pausing:
                break
        if step - start >= this_run:
            print(f"[pause] stopping after {step - start} steps this run (resume later)", flush=True)
            break
    hist_f.close()
    return {"step": step, "losses": history}
