"""Why does the pyramid recipe get worse after step 500? One run, no training (~30 min on a T4).

Three candidate causes, and what each predicts:
  (1) LEAK RETURNS. Pyramid noise hides the silhouette left in z_t but does not remove it, so with
      more updates the model may learn to read it again.
      -> leak gap at t=950 (x0 error from pure noise minus from the right target's z_t) grows, and
         the donor test follows the donor more (donor_vs_donor falls).
  (2) NOISE MISMATCH. Training noise is pyramid noise (fine part ~0.78x normal, plus coarse
      blobs); sampling starts from plain Gaussian noise. The longer the model trains, the more it
      expects pyramid noise.
      -> loss under plain Gaussian noise stops improving or rises after 500 while loss under
         pyramid noise keeps falling; sampling step 1000 from PYRAMID start noise recovers quality
         (zero-shot from pyramid noise is the control).
  (3) OVERSHOOT. The update simply grows too big in a direction that helps the loss, not samples.
      -> neither (1) nor (2); scaling the LoRA update down (x0.5) at step 1000 recovers quality.
LoRA scale s means W = W0 + s * (B @ A): s=0 is zero-shot exactly (the adapter is frozen), s=1 the
checkpoint. It is the same as interpolating the weights between zero-shot and the checkpoint.

Sampling uses test images first..first+n-1 (default 100-119): NOT the images 0-19 used to pick
step 500.

    python scripts/10_late_drift_test.py --test-root /content/data/synthetic/procedural_test \
        --ckpts $WORK/runs/pyr_frozen_lr5e5/step_{500,750,1000}.pt --out $WORK/outputs/late_drift
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image

from garmentcanon.data.synthetic import SyntheticPairs
from garmentcanon.eval.metrics import PairedMetrics
from garmentcanon.models.components import build_components, load_finetuned
from garmentcanon.train.finetune import pyramid_noise
from garmentcanon.utils.common import image_grid

SIGLIP_NORM = T.Normalize([0.5], [0.5])
TS = [50, 200, 400, 600, 800, 950]

ap = argparse.ArgumentParser()
ap.add_argument("--test-root", required=True)
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--first", type=int, default=100)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--loss-n", type=int, default=64)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--scales", default="0.5,0.75", help="extra LoRA scales to sample at (1.0 always done)")
ap.add_argument("--pyr", type=float, default=0.8, help="pyramid_discount used in training")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=2.0)
ap.add_argument("--out", default="outputs/late_drift")
a = ap.parse_args()
dev = "cuda"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
a.ckpts = sorted(a.ckpts, key=lambda p: int(p.rsplit("step_", 1)[1].split(".")[0]))
scales = [float(s) for s in a.scales.split(",") if s]

c = build_components(dev, sampler="euler")
c.net.eval()
abar = c.train_scheduler.alphas_cumprod.to(dev)
pm = PairedMetrics(dev, which=("LPIPS", "SSIM"))
ds = SyntheticPairs([a.test_root])


@torch.no_grad()
def encode(idx):
    items = [ds[j] for j in idx]
    X = torch.stack([it["input"] for it in items]).to(dev)
    Y = torch.stack([it["target"] for it in items]).to(dev)
    C = torch.cat([c.img_enc(pixel_values=SIGLIP_NORM(X[i:i + a.bs])).last_hidden_state for i in range(0, len(X), a.bs)])
    Z = torch.cat([c.vae.encode(Y[i:i + a.bs] * 2 - 1).latent_dist.mean * c.vae_scale for i in range(0, len(X), a.bs)])
    return X, Y, C, Z


@torch.no_grad()
def pred_eps(zt, t, cond):
    outs = []
    for i in range(0, len(zt), 8):
        tt = torch.full((len(zt[i:i + 8]),), t, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16):
            outs.append(c.net(zt[i:i + 8], tt, cond[i:i + 8]).float())
    return torch.cat(outs)


# ---- fixed sets: loss pairs (spread over the test set) and sampling images (100-119) ------------
L_idx = list(range(0, len(ds), max(1, len(ds) // a.loss_n)))[: a.loss_n]
_, _, CL, ZL = encode(L_idx)
g = torch.Generator(dev).manual_seed(0)
EPS_G = [torch.randn(ZL.shape, generator=g, device=dev) for _ in TS]
torch.manual_seed(0)
EPS_P = [pyramid_noise(ZL, a.pyr) for _ in TS]

S_idx = list(range(a.first, a.first + a.n))
XS, YS, CS, _ = encode(S_idx)
sched = c.sample_scheduler
sched.set_timesteps(a.steps)
INIT = float(sched.init_noise_sigma)
NOISE_G = torch.randn((a.n, 4, 64, 64), generator=torch.Generator(dev).manual_seed(0), device=dev)
torch.manual_seed(0)
NOISE_P = pyramid_noise(torch.zeros(a.n, 4, 64, 64, device=dev), a.pyr)


@torch.no_grad()
def losses():
    r = {}
    for name, E in (("gauss", EPS_G), ("pyramid", EPS_P)):
        per = {}
        for t, eps in zip(TS, E):
            s = abar[t]
            per[t] = F.mse_loss(pred_eps(s.sqrt() * ZL + (1 - s).sqrt() * eps, t, CL), eps).item()
        per["mean"] = float(np.mean([per[t] for t in TS]))
        r[name] = per
    t, s = 950, abar[950]
    eps = EPS_G[TS.index(950)]
    x0 = lambda zt: (zt - (1 - s).sqrt() * pred_eps(zt, t, CL)) / s.sqrt()
    donor = torch.roll(ZL, 1, 0)
    xt, xn, xd = x0(s.sqrt() * ZL + (1 - s).sqrt() * eps), x0(eps.clone()), x0(s.sqrt() * donor + (1 - s).sqrt() * eps)
    r["leak950"] = {"true": F.mse_loss(xt, ZL).item(), "none": F.mse_loss(xn, ZL).item(),
                    "donor_vs_right": F.mse_loss(xd, ZL).item(), "donor_vs_donor": F.mse_loss(xd, donor).item()}
    r["leak950"]["gap"] = r["leak950"]["none"] - r["leak950"]["true"]
    return r


@torch.no_grad()
def sample_all(noise):
    imgs = []
    for i in range(0, a.n, a.bs):
        cond, lat = CS[i:i + a.bs], noise[i:i + a.bs] * INIT
        sched.set_timesteps(a.steps)
        with torch.autocast("cuda", dtype=torch.float16):
            for t in sched.timesteps:
                x_in = sched.scale_model_input(torch.cat([lat, lat]), t)
                eu, ec = c.net(x_in, t, torch.cat([torch.zeros_like(cond), cond])).chunk(2)
                lat = sched.step((eu + a.guidance * (ec - eu)).float(), t, lat).prev_sample
        imgs.append((c.vae.decode(lat.float() / c.vae_scale).sample.clamp(-1, 1) + 1) / 2)
    imgs = torch.cat(imgs)
    rows = [pm(imgs[i:i + 1], YS[i:i + 1]) | {"L1": float((imgs[i] - YS[i]).abs().mean())} for i in range(a.n)]
    return imgs, rows


def set_lora_scale(s):
    for m in c.net.unet.modules():
        if hasattr(m, "lora_A") and "default" in getattr(m, "scaling", {}):
            if not hasattr(m, "_s0"):
                m._s0 = m.scaling["default"]
            m.scaling["default"] = m._s0 * s


loss_res, samp, pics = {}, {}, {}
loss_res["zero-shot"] = losses()
for nm, nz in (("gauss", NOISE_G), ("pyramid", NOISE_P)):
    pics[f"zero-shot {nm}"], samp[f"zero-shot {nm}"] = sample_all(nz)

loaded = False
for ck in a.ckpts:
    st = torch.load(ck, map_location="cpu", weights_only=False)
    if st["cfg"].get("adapter_lr_mult", 1) != 0:
        print(f"[warn] {ck}: adapter was trained, so LoRA scale 0 is not exactly zero-shot")
    if not loaded:
        load_finetuned(c.net, ck); loaded = True
    else:
        _, unexpected = c.net.load_state_dict(st["net"], strict=False)
        assert not unexpected, unexpected[:5]
    c.net.eval(); set_lora_scale(1.0)
    tag = f"step {st['step']}"
    loss_res[tag] = losses()
    for nm, nz in (("gauss", NOISE_G), ("pyramid", NOISE_P)):
        pics[f"{tag} {nm}"], samp[f"{tag} {nm}"] = sample_all(nz)
    for s in scales:
        set_lora_scale(s)
        pics[f"{tag} x{s:g}"], samp[f"{tag} x{s:g}"] = sample_all(NOISE_G)
    set_lora_scale(1.0)

print("\n== 1. noise-prediction loss on 64 held-out pairs (lower = better), mean over t")
print(f"{'':>12}  {'plain noise':>11}  {'pyramid noise':>13}   leak@950: true / none / gap   donor->right / donor->donor")
for k, r in loss_res.items():
    l = r["leak950"]
    print(f"{k:>12}  {r['gauss']['mean']:11.4f}  {r['pyramid']['mean']:13.4f}   "
          f"{l['true']:.3f} / {l['none']:.3f} / {l['gap']:+.3f}      {l['donor_vs_right']:.3f} / {l['donor_vs_donor']:.3f}")
print("   per t, plain noise: " + "  ".join(f"t={t}" for t in TS))
for k, r in loss_res.items():
    print(f"{k:>12}  " + "  ".join(f"{r['gauss'][t]:.4f}" for t in TS))

print(f"\n== 2. sampling, test images {a.first}-{a.first + a.n - 1}, Euler {a.steps} / guidance {a.guidance}")
print("   B-A = paired difference vs 'zero-shot gauss', +- its standard error")
base = samp["zero-shot gauss"]
for k, rows in samp.items():
    line = f"{k:>20}"
    for m in ("LPIPS", "SSIM", "L1"):
        v = np.array([r[m] for r in rows]); d = v - np.array([r[m] for r in base])
        line += f"   {m} {v.mean():.4f} ({d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f})"
    print(line)

pil = lambda x: Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
cols = [k for k in pics if k.endswith("gauss") or k.endswith("pyramid") or "x0.5" in k]
image_grid([[pil(XS[i])] + [pil(pics[k][i]) for k in cols] + [pil(YS[i])] for i in range(min(8, a.n))],
           ["input"] + cols + ["target"], cell=160).save(out / "late_drift.jpg", quality=90)
json.dump({"loss": loss_res, "samples": samp}, open(out / "late_drift.json", "w"), indent=1)
print(f"\nwrote {out}/late_drift.json and late_drift.jpg")
