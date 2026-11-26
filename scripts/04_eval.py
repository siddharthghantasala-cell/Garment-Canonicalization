"""Evaluate a checkpoint: canonicalization quality + downstream try-on.

    # 1) canonicalization metrics on synthetic test pairs (per state, visible/hidden split) and real photos
    python scripts/04_eval.py canon --ckpt runs/lora_mixed/step_6000.pt --test data/synthetic/procedural_test --real data/real
    #    (omit --ckpt to evaluate zero-shot TryOffDiff: the Tier 1 baseline)

    # 2) try-on on VITON-HD paired test. The clean and raw baselines are computed once and
    #    shared; each model gets its own folder, named by --tag (default: from the checkpoint path)
    python scripts/04_eval.py tryon --viton /data/VITON-HD --count 300                       # tag: zeroshot
    python scripts/04_eval.py tryon --ckpt runs/proc_only/step_6000.pt --viton /data/VITON-HD --count 300

    # 3) score one model against the shared baselines (writes <results>/scores_<tag>.json)
    python scripts/04_eval.py score --viton /data/VITON-HD --tag proc_only-step_6000

Results layout (try-on):
    <results>/raw_inputs/<key>.jpg      the crumpled garment photo (same for every model)
    <results>/clean/<key>.jpg           try-on with the catalog image        (upper bound)
    <results>/raw/<key>.jpg             try-on with the crumpled photo        (lower bound)
    <results>/canon_<tag>/<key>.jpg     the model's canonicalized garment
    <results>/ours_<tag>/<key>.jpg      try-on with the canonicalized garment

Add --smoke to run `canon` on tiny random models (CPU, no downloads) to test the plumbing.
"""
import argparse
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["canon", "tryon", "score"])
ap.add_argument("--ckpt")
ap.add_argument("--tag", help="name for this model's try-on folders (default: derived from --ckpt, or 'zeroshot')")
ap.add_argument("--test")
ap.add_argument("--real")
ap.add_argument("--viton")
ap.add_argument("--pairs-file", default=None, help="TryOffDiff's cleaned test list (1,990 pairs)")
ap.add_argument("--count", type=int, default=300)
ap.add_argument("--limit", type=int, default=None, help="max synthetic test pairs")
ap.add_argument("--start", type=int, default=0, help="first synthetic test pair (to split a long run into chunks)")
ap.add_argument("--sampler", choices=["euler", "pndm"], default="euler")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0, help="sampling seed (same for every image)")
ap.add_argument("--save-all", action="store_true", help="save every synthetic sample, not just the first 16")
ap.add_argument("--results", default="outputs/tryon")
ap.add_argument("--out", default="outputs/eval")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
dev = "cpu" if a.smoke else "cuda"


def default_tag(ckpt):
    if not ckpt:
        return "zeroshot"
    p = Path(ckpt)
    return f"{p.parent.name}-{p.stem}"


tag = a.tag or default_tag(a.ckpt)


def mean_dict(rows):
    keys = {k for r in rows for k in r}
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)} | {"n": len(rows)}


if a.mode == "canon":
    import torchvision.transforms.v2.functional as TF
    from garmentcanon.data.real_photos import STATES, frame_prediction, index_real_photos, load_rgb, prepare_flat_gt
    from garmentcanon.data.synthetic import SyntheticPairs, eval_transform
    from garmentcanon.eval.metrics import DistributionMetrics, PairedMetrics, visible_hidden_metrics
    from garmentcanon.models.tryoffdiff import TryOffDiffRunner
    to_t = eval_transform()
    model = TryOffDiffRunner(dev, smoke=a.smoke, ckpt=a.ckpt, sampler=a.sampler)
    pm = PairedMetrics(dev, which=("SSIM",) if a.smoke else ("DISTS", "LPIPS", "SSIM"))
    scores, out = {}, Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    if a.test:
        ds = SyntheticPairs([a.test], with_visibility=True)
        n = min(len(ds), a.start + (a.limit or len(ds)))
        dm = None if a.smoke else DistributionMetrics(dev)
        by_state = defaultdict(list)
        per_f = (out / "per_image.jsonl").open("w")      # one line per image: needed for paired tests
        for i in range(a.start, n):          # i stays the global index, so chunk files can be concatenated
            item = ds[i]
            pred_img = model([TF.to_pil_image(item["input"])], steps=a.steps, guidance=a.guidance, seed=a.seed)[0]
            if i < 16 or a.save_all:
                pred_img.save(out / "samples" / f"synthetic_{i:03d}_{item['state']}.jpg")
            pred, gt = to_t(pred_img)[None], item["target"][None]
            r = pm(pred, gt)
            if "visibility" in item:
                gm = item["target_mask"][None] if "target_mask" in item else None
                r |= visible_hidden_metrics(pm, pred, gt, item["visibility"][None], gm)
            by_state[item["state"]].append(r)
            per_f.write(json.dumps({"i": i, "key": item["key"], "state": item["state"]} | r) + "\n")
            if dm:
                dm.update(pred, gt)
        per_f.close()
        scores["synthetic"] = mean_dict([r for v in by_state.values() for r in v]) | (dm.compute() if dm else {})
        scores |= {f"synthetic/{st}": mean_dict(v) for st, v in sorted(by_state.items())}
    if a.real:
        by_state, real_rows = defaultdict(list), []
        for gid, s in index_real_photos(a.real).items():
            gt = to_t(Image.fromarray(prepare_flat_gt(load_rgb(s["flat"]))[0]))[None]
            for st in STATES:
                if st in s:
                    pred_img = model([Image.fromarray(load_rgb(s[st]))], steps=a.steps, guidance=a.guidance,
                                     seed=a.seed)[0]
                    pred_img.save(out / "samples" / f"real_{gid}_{st}.jpg")
                    pred = to_t(Image.fromarray(frame_prediction(np.array(pred_img))))[None]   # same framing as gt
                    by_state[st].append(pm(pred, gt))
                    real_rows.append({"key": f"{gid}_{st}", "state": st} | by_state[st][-1])
        (out / "per_image_real.jsonl").write_text("".join(json.dumps(r) + "\n" for r in real_rows))
        scores |= {f"real/{st}": mean_dict(v) for st, v in by_state.items()}
        if by_state:
            scores["real"] = mean_dict([r for v in by_state.values() for r in v])
    (out / "canon_scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))

elif a.mode == "tryon":
    from garmentcanon.data.crumple_procedural import CrumpleParams, crumple
    from garmentcanon.data.viton_hd import VitonHD
    from garmentcanon.models.catvton import CatVTONRunner
    from garmentcanon.models.tryoffdiff import TryOffDiffRunner
    model, vton = TryOffDiffRunner(dev, ckpt=a.ckpt, sampler=a.sampler), CatVTONRunner(dev)
    ds = VitonHD(a.viton, "test", pairs_file=a.pairs_file, paired=True)
    res = Path(a.results)
    dirs = {k: res / k for k in ("raw_inputs", "clean", "raw", f"canon_{tag}", f"ours_{tag}")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[tryon] tag={tag}  results={res}", flush=True)
    for i in range(min(a.count, len(ds))):
        it = ds[i]
        k = it["key"]
        f = {name: d / f"{k}.jpg" for name, d in dirs.items()}
        if all(p.exists() for p in f.values()):
            continue                                   # resumable, per file
        person = Image.open(it["person_path"]).convert("RGB")
        # agnostic masks: in the official VITON-HD release (same layout CatVTON's inference.py reads)
        mask = Image.open(Path(a.viton) / "test" / "agnostic-mask" / f"{k}_mask.png").convert("L")
        if f["raw_inputs"].exists():
            raw = Image.open(f["raw_inputs"]).convert("RGB")
        else:   # seed from the garment key: the same crumple whatever --count / pair list is used
            raw = Image.fromarray(crumple(it["cloth"], it["cloth_mask"],
                                          CrumpleParams(seed=zlib.crc32(k.encode())))[0])
            raw.save(f["raw_inputs"], quality=95)
        if not f["clean"].exists():
            vton(person, Image.fromarray(it["cloth"]), mask).save(f["clean"], quality=95)
        if not f["raw"].exists():
            vton(person, raw, mask).save(f["raw"], quality=95)
        if not f[f"ours_{tag}"].exists():
            canon = model([raw], steps=a.steps, guidance=a.guidance)[0]
            canon.save(f[f"canon_{tag}"], quality=95)
            vton(person, canon, mask).save(f[f"ours_{tag}"], quality=95)
        if i % 20 == 0:
            print(f"{i}/{a.count}", flush=True)

else:  # score
    from garmentcanon.eval.tryon_eval import score_tryon_folder
    table = score_tryon_folder(a.results, Path(a.viton) / "test" / "image", tag,
                               mask_dir=Path(a.viton) / "test" / "agnostic-mask", device=dev)
    print(json.dumps(table, indent=2))
