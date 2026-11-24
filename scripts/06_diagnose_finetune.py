"""Why does fine-tuning make samples worse? Three cheap checks, no sampling needed.

  1. AUDIT   what each checkpoint was really trained with (the optimizer's saved lr can
             differ from the lr you typed: see the patch notes).
  2. DRIFT   how far the adapter and the LoRA moved away from the pretrained weights.
  3. LOSS    the training objective itself (noise-prediction MSE) on held-out test pairs
             AND on training pairs, with the SAME images, timesteps and noise for every model.

How to read LOSS:
  - fine-tuned held-out loss  < zero-shot  but samples worse -> sampling-side mismatch
  - fine-tuned held-out loss  > zero-shot  and train loss < zero-shot -> training learns
    something that doesn't carry over to new garments (overfitting / shortcut)
  - fine-tuned loss is worse on BOTH train and held-out -> the optimization itself is
    damaging the model (lr too high for the adapter, wrong lr after resume, ...)

    python scripts/06_diagnose_finetune.py --test-root /content/data/synthetic/procedural_test \
        --train-root /content/data/synthetic/procedural_train \
        --ckpts $WORK/runs/proc_only/step_*.pt
"""
import argparse
import json

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from garmentcanon.data.synthetic import SyntheticPairs
from garmentcanon.models.components import build_components, load_finetuned

SIGLIP_NORM = T.Normalize([0.5], [0.5])
TS = [50, 200, 400, 600, 800, 950]          # fixed timesteps: low t = fine detail, high t = global shape

ap = argparse.ArgumentParser()
ap.add_argument("--test-root", required=True)
ap.add_argument("--train-root", default=None)
ap.add_argument("--ckpts", nargs="*", default=[])
ap.add_argument("--n", type=int, default=64, help="pairs per split")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", default="diagnosis.json")
a = ap.parse_args()
dev = "cuda"
a.ckpts = sorted(a.ckpts, key=lambda p: int(p.rsplit("step_", 1)[1].split(".")[0]))

# ---- 1. AUDIT ---------------------------------------------------------------------------
print("\n== 1. what each checkpoint was trained with ==")
for p in a.ckpts:
    st = torch.load(p, map_location="cpu", weights_only=False)
    opt_lrs = sorted({g["lr"] for g in st["opt"]["param_groups"]})
    cfg = st["cfg"]
    flag = "" if all(abs(l - float(cfg["lr"])) < 1e-12 for l in opt_lrs) else "   <-- MISMATCH"
    print(f"{p.split('/')[-1]:>14}: cfg lr {cfg['lr']}  optimizer lr {opt_lrs}{flag}  "
          f"batch {cfg['batch_size']}x{cfg.get('grad_accum', 1)}  out_dir {cfg['out_dir']}")

# ---- fixed evaluation batches (same for every model) ---------------------------------------
c = build_components(dev)
c.net.eval()                                 # dropout off, as at sampling time
base = {k: v.detach().clone() for k, v in c.net.state_dict().items() if not k.startswith("unet.")}


@torch.no_grad()
def make_batches(root):
    ds = SyntheticPairs([root])
    idx = list(range(0, len(ds), max(1, len(ds) // a.n)))[: a.n]
    g = torch.Generator(dev).manual_seed(0)
    batches = []
    for i in range(0, len(idx), a.bs):
        items = [ds[j] for j in idx[i:i + a.bs]]
        x = torch.stack([it["input"] for it in items]).to(dev)
        y = torch.stack([it["target"] for it in items]).to(dev)
        cond = c.img_enc(pixel_values=SIGLIP_NORM(x)).last_hidden_state
        z0 = c.vae.encode(y * 2 - 1).latent_dist.mean * c.vae_scale
        eps = [torch.randn(z0.shape, generator=g, device=dev) for _ in TS]
        batches.append((cond, z0, eps))
    return batches


@torch.no_grad()
def loss_per_t(batches):
    out = {t: 0.0 for t in TS}
    for cond, z0, eps_list in batches:
        for t, eps in zip(TS, eps_list):
            tt = torch.full((z0.shape[0],), t, device=dev, dtype=torch.long)
            zt = c.train_scheduler.add_noise(z0, eps, tt)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = c.net(zt, tt, cond)
            out[t] += F.mse_loss(pred.float(), eps).item() / len(batches)
    out["mean"] = sum(out[t] for t in TS) / len(TS)
    return out


splits = {"heldout": make_batches(a.test_root)}
if a.train_root:
    splits["train"] = make_batches(a.train_root)


@torch.no_grad()
def drift():
    sd = c.net.state_dict()
    ad = {k: ((sd[k].float() - v.float()).norm() / v.float().norm().clamp_min(1e-8)).item()
          for k, v in base.items() if v.dtype.is_floating_point}
    lora = []
    for _, m in c.net.unet.named_modules():
        if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
            d = m.lora_B["default"].weight @ m.lora_A["default"].weight * m.scaling["default"]
            lora.append((d.norm() / m.base_layer.weight.norm()).item())
    lora.sort()
    return {"adapter_max_rel_change": max(ad.values()) if ad else 0.0,
            "adapter_worst_tensor": max(ad, key=ad.get) if ad else None,
            "lora_median_rel_change": lora[len(lora) // 2] if lora else 0.0,
            "lora_max_rel_change": lora[-1] if lora else 0.0}


results = {}
print("\n== 2+3. drift from pretrained weights, and noise-prediction loss (lower = better) ==")
lora_added = False
for name in ["zero-shot"] + a.ckpts:
    if name != "zero-shot":
        if not lora_added:
            load_finetuned(c.net, name); lora_added = True     # adds LoRA layers once
        else:
            st = torch.load(name, map_location="cpu", weights_only=False)
            _, unexpected = c.net.load_state_dict(st["net"], strict=False)
            assert not unexpected, unexpected[:5]
        c.net.eval()
    r = {s: loss_per_t(b) for s, b in splits.items()}
    r["drift"] = drift() if name != "zero-shot" else {}
    results[name] = r
    short = name.split("/")[-1]
    line = "  ".join(f"{s} {r[s]['mean']:.4f}" for s in splits)
    print(f"{short:>14}: {line}  {r['drift']}")

print("\nheld-out loss per timestep (look for WHERE it gets worse: high t = global shape):")
print(" " * 15 + "".join(f"t={t:<7}" for t in TS))
for name, r in results.items():
    print(f"{name.split('/')[-1]:>14} " + "".join(f"{r['heldout'][t]:<9.4f}" for t in TS))
json.dump(results, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
