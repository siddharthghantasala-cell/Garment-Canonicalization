"""Tier 1 pilot: how do existing models handle off-body garment photos?

Runs TryOffDiff zero-shot (and optionally the Qwen-Image-Edit baseline) on
  (a) your real photos  data/real/<id>_<state>.jpg, scored against <id>_flat.jpg after the
      flat photo is cut out and re-framed like a catalog image (data/real_photos.py)
  (b) procedurally crumpled VITON-HD test garments, scored against the clean image,
      with visible/hidden regions scored separately and results split by state
      (folded = a fold-over was applied, crumpled = not)
and writes outputs, a metrics JSON, and a gallery figure.

    python scripts/01_pilot_zero_shot.py --real data/real --viton /data/VITON-HD --count 200
"""
import argparse
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from garmentcanon.data.crumple_procedural import CrumpleParams, crumple, procedural_state
from garmentcanon.data.real_photos import STATES, frame_prediction, index_real_photos, load_rgb, prepare_flat_gt
from garmentcanon.data.synthetic import eval_transform
from garmentcanon.eval.metrics import PairedMetrics, visible_hidden_metrics
from garmentcanon.models.tryoffdiff import TryOffDiffRunner
from garmentcanon.utils.common import image_grid

ap = argparse.ArgumentParser()
ap.add_argument("--real", help="folder of your own garment photos")
ap.add_argument("--viton", help="VITON-HD root")
ap.add_argument("--pairs-file", default=None, help="TryOffDiff's cleaned test list (1,990 pairs)")
ap.add_argument("--count", type=int, default=50)
ap.add_argument("--sampler", choices=["euler", "pndm"], default="euler")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=2.0)
ap.add_argument("--qwen", action="store_true", help="also run the Qwen-Image-Edit baseline (4-bit, ~16-20 GB VRAM)")
ap.add_argument("--out", default="outputs/pilot")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

tod = TryOffDiffRunner(sampler=a.sampler)
metrics = PairedMetrics()
to_t, mask_t = eval_transform(), eval_transform(nearest=True)
qwen = None
if a.qwen:
    from garmentcanon.data.crumple_generative import GenerativeCrumpler
    qwen = GenerativeCrumpler()
QWEN_FLATTEN = ("Turn this garment into a clean e-commerce product photo: laid perfectly flat, front view, "
                "centered on a pure white background. Keep every color, print, and logo identical.")

results = defaultdict(list)
gallery = []


def run_one(source, state, key, inp: Image.Image, gt: Image.Image, vis=None, garment_mask=None, reframe=False):
    """reframe=True (real photos): re-frame each prediction by the same rule as the ground truth."""
    row = [inp]
    preds = {"tryoffdiff": tod([inp], steps=a.steps, guidance=a.guidance)[0]}
    if qwen:
        preds["qwen_edit"] = qwen.edit(inp, QWEN_FLATTEN)
    gt_t = to_t(gt)[None]
    for name, pred in preds.items():
        pred.save(out / f"{source}_{key}_{state}_{name}.jpg")
        scored = Image.fromarray(frame_prediction(np.array(pred))) if reframe else pred
        p_t = to_t(scored)[None]
        r = metrics(p_t, gt_t)
        if vis is not None:
            vis_t = mask_t(Image.fromarray((vis * 255).astype("uint8")))
            gm_t = None if garment_mask is None else \
                (mask_t(Image.fromarray(((garment_mask > 0.5) * 255).astype("uint8"))) > 0.5).float()[None]
            r |= visible_hidden_metrics(metrics, p_t, gt_t, vis_t[None], gm_t)
        results[f"{source}/{state}/{name}"].append(r)
        results[f"{source}/all/{name}"].append(r)
        row.append(pred)
    row.append(gt)
    if len(gallery) < 24:
        gallery.append(row)


if a.real:
    for gid, states in index_real_photos(a.real).items():
        gt = Image.fromarray(prepare_flat_gt(load_rgb(states["flat"]))[0])
        for st in STATES:
            if st in states:
                run_one("real", st, gid, Image.fromarray(load_rgb(states[st])), gt, reframe=True)

if a.viton:
    from garmentcanon.data.viton_hd import VitonHD
    ds = VitonHD(a.viton, "test", pairs_file=a.pairs_file)
    for i in range(min(a.count, len(ds))):
        it = ds[i]
        inp, p, vis = crumple(it["cloth"], it["cloth_mask"], CrumpleParams(seed=zlib.crc32(it["key"].encode())),
                             return_visibility=True)   # same crumple as 04_eval.py tryon's raw input
        run_one("synthetic", procedural_state(p.record), it["key"], Image.fromarray(inp),
                Image.fromarray(it["cloth"]), vis, it["cloth_mask"])

summary = {k: {m: float(np.mean([r[m] for r in v if m in r])) for m in sorted({m for r in v for m in r})} | {"n": len(v)}
           for k, v in sorted(results.items())}
(out / "pilot_metrics.json").write_text(json.dumps(summary, indent=2))
cols = ["input", "TryOffDiff"] + (["Qwen edit"] if qwen else []) + ["ground truth"]
image_grid(gallery, cols).save(out / "pilot_gallery.png")
print(json.dumps(summary, indent=2))
