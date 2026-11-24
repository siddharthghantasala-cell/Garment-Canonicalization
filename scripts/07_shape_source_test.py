"""Where does the model get the garment's SHAPE from: the condition, or the noisy latent?

Background. Stable Diffusion v1's noise schedule never reaches pure noise: at t=999 the noisy
latent is still 0.068 x target + noise, at t=950 it is 0.091 x target + noise. A garment on a
white background is a strong low-frequency pattern, so in TRAINING the coarse silhouette can
be read out of z_t even at the highest timesteps. In SAMPLING the start is pure noise, so that
information does not exist. A model that learned "shape from z_t, colours/prints from the
condition" has a low teacher-forced loss and broken samples.

Three tests, zero-shot vs one fine-tuned checkpoint, on the same held-out pairs:
  A. condition shuffle: loss with the right condition vs another garment's condition, per t.
     (How much does each model use the condition at each noise level?)
  B. leak swap at t = 800..999: predict the clean latent x0 from z_t built from
       true   - the right target (what training sees)
       donor  - ANOTHER garment's target (condition stays the right one)
       none   - pure noise (what sampling sees)
     and measure the error against the right target and against the donor.
  C. sampling start: run the real sampler from pure noise, from the right target noised to
     the first sampling step, and from a donor noised the same way. Scores the outputs
     (LPIPS, silhouette IoU) and saves an image grid.

    python scripts/07_shape_source_test.py --test-root /content/data/synthetic/procedural_test \
        --ckpt $WORK/runs/lr1e5_test/step_3000.pt --out $WORK/outputs/shape_test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image

from garmentcanon.data.crumple_procedural import mask_from_white_background
from garmentcanon.data.synthetic import SyntheticPairs
from garmentcanon.eval.metrics import PairedMetrics
from garmentcanon.models.components import build_components, load_finetuned
from garmentcanon.utils.common import image_grid

SIGLIP_NORM = T.Normalize([0.5], [0.5])
TS_A = [50, 200, 400, 600, 800, 950]
TS_B = [800, 900, 950, 999]

ap = argparse.ArgumentParser()
ap.add_argument("--test-root", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--sampler", choices=["euler"], default="euler")   # test C needs Euler's sigmas
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=2.0)
ap.add_argument("--out", default="outputs/shape_test")
a = ap.parse_args()
dev = "cuda"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

c = build_components(dev, sampler=a.sampler)
c.net.eval()
abar = c.train_scheduler.alphas_cumprod.to(dev)
metrics = PairedMetrics(dev, which=("LPIPS",))

# ---- fixed data: n held-out pairs from different garments; donor = the next one -------------
ds = SyntheticPairs([a.test_root])
idx = list(range(0, len(ds), max(1, len(ds) // a.n)))[: a.n]
items = [ds[j] for j in idx]
X = torch.stack([it["input"] for it in items]).to(dev)
Y = torch.stack([it["target"] for it in items]).to(dev)
with torch.no_grad():
    COND = torch.cat([c.img_enc(pixel_values=SIGLIP_NORM(X[i:i + a.bs])).last_hidden_state
                      for i in range(0, a.n, a.bs)])
    Z0 = torch.cat([c.vae.encode(Y[i:i + a.bs] * 2 - 1).latent_dist.mean * c.vae_scale
                    for i in range(0, a.n, a.bs)])
D = lambda t: torch.roll(t, 1, 0)                  # donor: a different garment, same position in batch
g = torch.Generator(dev).manual_seed(0)
EPS = torch.randn(Z0.shape, generator=g, device=dev)


def chunks():
    for i in range(0, a.n, a.bs):
        yield slice(i, i + a.bs)


@torch.no_grad()
def eps_hat(zt, t, cond):
    tt = torch.full((zt.shape[0],), t, device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.float16):
        return c.net(zt, tt, cond).float()


def test_a():
    r = {}
    for t in TS_A:
        s = abar[t]
        zt = s.sqrt() * Z0 + (1 - s).sqrt() * EPS
        true = sum(F.mse_loss(eps_hat(zt[k], t, COND[k]), EPS[k]).item() for k in chunks()) / len(list(chunks()))
        shuf = sum(F.mse_loss(eps_hat(zt[k], t, D(COND)[k]), EPS[k]).item() for k in chunks()) / len(list(chunks()))
        r[t] = {"right_cond": true, "other_cond": shuf, "cond_effect": shuf / true}
    return r


def test_b():
    r = {}
    for t in TS_B:
        s = abar[t]
        r[t] = {}
        for src, z in [("true", Z0), ("donor", D(Z0)), ("none", None)]:
            zt = EPS.clone() if z is None else s.sqrt() * z + (1 - s).sqrt() * EPS
            x0 = torch.cat([(zt[k] - (1 - s).sqrt() * eps_hat(zt[k], t, COND[k])) / s.sqrt() for k in chunks()])
            r[t][src] = {"err_vs_right": F.mse_loss(x0, Z0).item(), "err_vs_donor": F.mse_loss(x0, D(Z0)).item()}
    return r


@torch.no_grad()
def sample(cond, lat):
    sched = c.sample_scheduler
    sched.set_timesteps(a.steps)
    with torch.autocast("cuda", dtype=torch.float16):
        for t in sched.timesteps:
            if a.guidance != 1.0:
                x_in = sched.scale_model_input(torch.cat([lat, lat]), t)
                eu, ec = c.net(x_in, t, torch.cat([torch.zeros_like(cond), cond])).chunk(2)
                e = eu + a.guidance * (ec - eu)
            else:
                e = c.net(sched.scale_model_input(lat, t), t, cond)
            lat = sched.step(e.float(), t, lat).prev_sample
    return ((c.vae.decode(lat.float() / c.vae_scale).sample.clamp(-1, 1) + 1) / 2)


def silhouette(img01: torch.Tensor) -> np.ndarray:
    a8 = (img01.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return mask_from_white_background(a8) > 0.5


def iou(m1, m2):
    return float((m1 & m2).sum() / max((m1 | m2).sum(), 1))


def test_c(tag):
    sched = c.sample_scheduler
    sched.set_timesteps(a.steps)
    sig0, init = float(sched.sigmas[0]), float(sched.init_noise_sigma)
    starts = {"pure_noise": lambda k: EPS[k] * init,                 # exactly what the runner does
              "true_leak": lambda k: Z0[k] + sig0 * EPS[k],          # = training's z_t at the first step
              "donor_leak": lambda k: D(Z0)[k] + sig0 * EPS[k]}
    imgs, r = {}, {}
    for name, f in starts.items():
        imgs[name] = torch.cat([sample(COND[k], f(k)) for k in chunks()])
        m_right = [silhouette(Y[i]) for i in range(a.n)]
        m_donor = [silhouette(D(Y)[i]) for i in range(a.n)]
        m_pred = [silhouette(imgs[name][i]) for i in range(a.n)]
        r[name] = {"LPIPS_vs_right": metrics(imgs[name], Y)["LPIPS"],
                   "IoU_vs_right": float(np.mean([iou(p, m) for p, m in zip(m_pred, m_right)])),
                   "IoU_vs_donor": float(np.mean([iou(p, m) for p, m in zip(m_pred, m_donor)]))}
    pil = lambda x: Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    rows = [[pil(X[i]), pil(imgs["pure_noise"][i]), pil(imgs["true_leak"][i]), pil(imgs["donor_leak"][i]),
             pil(D(Y)[i]), pil(Y[i])] for i in range(a.n)]
    image_grid(rows, ["input", "from noise", "from right tgt", "from donor tgt", "donor", "right target"]
               ).save(out / f"start_test_{tag}.jpg", quality=90)
    return r


results = {}
for tag in ["zero-shot", "finetuned"]:
    if tag == "finetuned":
        load_finetuned(c.net, a.ckpt); c.net.eval()
    results[tag] = {"A": test_a(), "B": test_b(), "C": test_c(tag)}
    print(f"\n######## {tag}")
    print("A. condition effect (loss with another garment's condition / loss with the right one; 1.0 = unused)")
    print("   " + "  ".join(f"t={t}: {v['cond_effect']:.2f}" for t, v in results[tag]["A"].items()))
    print("B. x0 error (latent MSE) vs the RIGHT target   [and vs the donor]  -- z_t built from:")
    for t, v in results[tag]["B"].items():
        print(f"   t={t}: " + "   ".join(f"{s} {d['err_vs_right']:.3f} [{d['err_vs_donor']:.3f}]" for s, d in v.items()))
    print("C. full sampling, by starting point")
    for s, v in results[tag]["C"].items():
        print(f"   {s:>11}: LPIPS {v['LPIPS_vs_right']:.3f}  IoU right {v['IoU_vs_right']:.2f}  IoU donor {v['IoU_vs_donor']:.2f}")
json.dump(results, open(out / "shape_test.json", "w"), indent=1)
print(f"\nwrote {out}/shape_test.json and start_test_*.jpg")
