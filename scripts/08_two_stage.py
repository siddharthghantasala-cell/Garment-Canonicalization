"""Two-stage sampling: the SHAPE from zero-shot TryOffDiff, the DETAILS from the fine-tuned model.

Why: 07 showed the fine-tuned model reads the silhouette from the noisy latent and is better
than zero-shot when that latent carries the right silhouette (sampling from the right target
noised to the first step: LPIPS 0.249 vs zero-shot 0.298). At test time there is no target, but
zero-shot TryOffDiff gives a decent silhouette. So:
  1. sample zero-shot from pure noise           -> clean latent z_zs
  2. noise z_zs to step k of the 20-step schedule -> the fine-tuned model finishes steps k..19
k = 0 starts from the top (z_zs only leaks in, like the training latents); larger k keeps more
of the zero-shot image. No training needed.

    python scripts/08_two_stage.py --test-root data/synthetic/procedural_test \
        --ckpt /content/drive/MyDrive/garment-canon/runs/lr1e5_test/step_3000.pt --n 20 \
        --out /content/drive/MyDrive/garment-canon/outputs/two_stage

Control columns "zs-refine k=..." do the same re-noising but let the ZERO-SHOT model finish, so a
gain of "two-stage k=..." over them is due to the fine-tuned weights, not to denoising twice.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image

from garmentcanon.data.synthetic import SyntheticPairs
from garmentcanon.eval.metrics import PairedMetrics
from garmentcanon.models.components import build_components, load_finetuned
from garmentcanon.utils.common import image_grid

SIGLIP_NORM = T.Normalize([0.5], [0.5])

ap = argparse.ArgumentParser()
ap.add_argument("--test-root", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=2.0)
ap.add_argument("--starts", default="0,2,4,6,8", help="step index where the fine-tuned model takes over")
ap.add_argument("--out", default="outputs/two_stage")
a = ap.parse_args()
dev = "cuda"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
starts = [int(s) for s in a.starts.split(",")]

c = build_components(dev, sampler="euler")
c.net.eval()
sched = c.sample_scheduler
metrics = PairedMetrics(dev, which=("LPIPS", "SSIM"))

ds = SyntheticPairs([a.test_root])
idx = list(range(0, len(ds), max(1, len(ds) // a.n)))[: a.n]
items = [ds[j] for j in idx]
X = torch.stack([it["input"] for it in items]).to(dev)
Y = torch.stack([it["target"] for it in items]).to(dev)
with torch.no_grad():
    COND = torch.cat([c.img_enc(pixel_values=SIGLIP_NORM(X[i:i + a.bs])).last_hidden_state
                      for i in range(0, a.n, a.bs)])
shape = (a.n, 4, 64, 64)
EPS1 = torch.randn(shape, generator=torch.Generator(dev).manual_seed(0), device=dev)
EPS2 = torch.randn(shape, generator=torch.Generator(dev).manual_seed(1), device=dev)
chunks = [slice(i, i + a.bs) for i in range(0, a.n, a.bs)]


@torch.no_grad()
def run(cond, lat, k):
    """Euler steps k..end of the usual schedule; returns the clean latent."""
    sched.set_timesteps(a.steps)
    with torch.autocast("cuda", dtype=torch.float16):
        for t in sched.timesteps[k:]:
            if a.guidance != 1.0:
                x_in = sched.scale_model_input(torch.cat([lat, lat]), t)
                eu, ec = c.net(x_in, t, torch.cat([torch.zeros_like(cond), cond])).chunk(2)
                e = eu + a.guidance * (ec - eu)
            else:
                e = c.net(sched.scale_model_input(lat, t), t, cond)
            lat = sched.step(e.float(), t, lat).prev_sample
    return lat.float()


@torch.no_grad()
def decode(z):
    return torch.cat([(c.vae.decode(z[k] / c.vae_scale).sample.clamp(-1, 1) + 1) / 2 for k in chunks])


sched.set_timesteps(a.steps)
init, sigmas = float(sched.init_noise_sigma), sched.sigmas.clone()

Z_zs = torch.cat([run(COND[k], EPS1[k] * init, 0) for k in chunks])          # stage 1: zero-shot
imgs = {"zero-shot": decode(Z_zs)}
for s in starts:                                                              # control: zero-shot finishes
    lat = Z_zs + float(sigmas[s]) * EPS2
    imgs[f"zs-refine k={s}"] = decode(torch.cat([run(COND[k], lat[k], s) for k in chunks]))
load_finetuned(c.net, a.ckpt); c.net.eval()
imgs["finetuned"] = decode(torch.cat([run(COND[k], EPS1[k] * init, 0) for k in chunks]))
for s in starts:                                                              # stage 2
    lat = Z_zs + float(sigmas[s]) * EPS2
    imgs[f"two-stage k={s}"] = decode(torch.cat([run(COND[k], lat[k], s) for k in chunks]))

# per-image scores, so the table can say whether a difference to zero-shot is bigger than noise
per = {name: [metrics(im[i:i + 1], Y[i:i + 1]) for i in range(a.n)] for name, im in imgs.items()}
res = {name: {m: float(np.mean([r[m] for r in rows])) for m in ("LPIPS", "SSIM")} for name, rows in per.items()}
base = per["zero-shot"]
print(f"{'':>18}  LPIPS   SSIM      (lower LPIPS / higher SSIM = better; {a.n} held-out images)")
for name, r in res.items():
    print(f"{name:>18}  {r['LPIPS']:.3f}   {r['SSIM']:.3f}")
print("\ndifference to zero-shot, paired per image (diff +/- standard error; 'better on' = images that improved)."
      "\nTreat a gain as real only if |diff| > 2 x error and most images improved.")
for name, rows in per.items():
    if name == "zero-shot":
        continue
    for m, sign in (("LPIPS", -1), ("SSIM", 1)):
        d = np.array([r[m] - b[m] for r, b in zip(rows, base)])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{name:>18} {m}: diff {d.mean():+.4f} +/- {se:.4f}   better on {(sign * d > 0).sum()}/{a.n}")
json.dump(per, open(out / "two_stage_per_image.json", "w"), indent=1)
pil = lambda x: Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
rows = [[pil(X[i])] + [pil(im[i]) for im in imgs.values()] + [pil(Y[i])] for i in range(min(a.n, 12))]    # grid: first 12 images only
image_grid(rows, ["input"] + list(imgs) + ["target"], cell=160).save(out / "two_stage.jpg", quality=90)
json.dump(res, open(out / "two_stage.json", "w"), indent=1)
print(f"wrote {out}/two_stage.json and two_stage.jpg")