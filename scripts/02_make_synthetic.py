"""Build synthetic (casual input, clean target) pairs from VITON-HD.

    # procedural: CPU only, parallel. ~11.5k garments x 3 variants.
    # Measured ~1.25 s per crumple per CPU core at 768x1024 -> train set ~12 core-hours.
    python scripts/02_make_synthetic.py --viton /data/VITON-HD --split train --source procedural --variants 3 \
        --pairs-file <TryOffDiff cleaned train list>

    # generative: GPU, slow (tens of seconds per image). Keep it small (~500).
    # First calibrate --threshold: run with --limit 50 --threshold 0, look at the pairs and
    # their dino_sim values in manifest.jsonl, pick the cutoff, then delete that folder and rerun.
    python scripts/02_make_synthetic.py --viton /data/VITON-HD --split train --source generative --limit 500

Re-running skips garments already in the manifest, so it's safe to restart.
"""
import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from garmentcanon.data.crumple_procedural import (CrumpleParams, crumple, mask_from_white_background,
                                                  params_to_dict, procedural_state)
from garmentcanon.data.viton_hd import VitonHD

# Worker state. Set by _init_worker in every worker process, so the job function works under
# both process start methods: "fork" (Linux) and "spawn" (macOS/Windows, which re-imports this
# file in each worker and would otherwise re-run the whole script).
_DS = _ROOT = _VARIANTS = None


def _init_worker(viton, split, pairs_file, root, variants):
    global _DS, _ROOT, _VARIANTS
    _DS, _ROOT, _VARIANTS = VitonHD(viton, split, pairs_file=pairs_file), Path(root), variants


def _save_target(it, root: Path) -> np.ndarray:
    """Copy the clean garment and its mask (used to find the hidden garment region in eval)."""
    Image.fromarray(it["cloth"]).save(root / "targets" / f"{it['key']}.jpg", quality=95)
    mask = it["cloth_mask"] if it["cloth_mask"] is not None else mask_from_white_background(it["cloth"])
    Image.fromarray(((mask > 0.5) * 255).astype("uint8")).save(root / "masks" / f"{it['key']}.png")
    return mask


def procedural_job(i):
    it = _DS[i]
    root = _ROOT
    mask = _save_target(it, root)
    rows = []
    for k in range(_VARIANTS):
        seed = zlib.crc32(f"{it['key']}_{k}".encode())   # stable across runs (unlike hash())
        img, p, vis = crumple(it["cloth"], mask, CrumpleParams(seed=seed), return_visibility=True)
        name = f"{it['key']}_{k}"
        Image.fromarray(img).save(root / "inputs" / f"{name}.jpg", quality=92)
        Image.fromarray((vis * 255).astype("uint8")).save(root / "visibility" / f"{name}.png")
        rows.append({"key": it["key"], "input": f"inputs/{name}.jpg", "target": f"targets/{it['key']}.jpg",
                     "target_mask": f"masks/{it['key']}.png", "visibility": f"visibility/{name}.png",
                     "state": procedural_state(p.record), "params": params_to_dict(p)})
    return rows


def read_done(manifest: Path) -> set:
    """Keys already in the manifest. Skips a torn last line from a run that was killed mid-write."""
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viton", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--source", choices=["procedural", "generative"], default="procedural")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--pairs-file", default=None, help="e.g. TryOffDiff's cleaned pair list")
    ap.add_argument("--gen-steps", type=int, default=40, help="generative only: denoising steps per image")
    ap.add_argument("--threshold", type=float, default=0.6, help="generative only: DINOv2 similarity cutoff")
    a = ap.parse_args()

    root = Path(a.out) / f"{a.source}_{a.split}"
    for sub in ("inputs", "targets", "masks", "visibility"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    if manifest.exists() and manifest.read_bytes()[-1:] not in (b"", b"\n"):
        with manifest.open("a") as f:                 # finish a torn last line so appends start clean
            f.write("\n")
    done = read_done(manifest) | read_done(root / "rejected.jsonl")   # don't regenerate rejected garments
    ds = VitonHD(a.viton, a.split, pairs_file=a.pairs_file)
    idx = [i for i in range(len(ds)) if ds.items[i].key not in done][: a.limit]
    print(f"{len(idx)} garments to process ({len(done)} already done)", flush=True)

    if a.source == "procedural":
        init = (a.viton, a.split, a.pairs_file, str(root), a.variants)
        with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=init) as ex, manifest.open("a") as f:
            for n, rows in enumerate(ex.map(procedural_job, idx, chunksize=16)):
                f.write("".join(json.dumps(r) + "\n" for r in rows))
                f.flush()
                if n % 200 == 0:
                    print(f"{n}/{len(idx)}", flush=True)
    else:
        from garmentcanon.data.crumple_generative import FidelityFilter, GenerativeCrumpler, keep
        gen, filt = GenerativeCrumpler(), FidelityFilter()
        states = ["crumpled", "folded", "hanging"]
        kept = 0
        with manifest.open("a") as f, (root / "rejected.jsonl").open("a") as rej:
            for n, i in enumerate(idx):
                it = ds[i]
                _save_target(it, root)
                src = Image.fromarray(it["cloth"])
                st = states[n % 3]
                out = gen(src, st, seed=n, steps=a.gen_steps)
                sim = filt.similarity(src, out)
                if keep(sim, a.threshold):
                    name = f"{it['key']}_{st}.jpg"
                    out.save(root / "inputs" / name, quality=92)
                    f.write(json.dumps({"key": it["key"], "input": f"inputs/{name}",
                                        "target": f"targets/{it['key']}.jpg", "target_mask": f"masks/{it['key']}.png",
                                        "state": st, "params": {"state": st, "dino_sim": sim}}) + "\n")
                    f.flush()
                    kept += 1
                else:   # keep the image too: it's what you look at when calibrating --threshold
                    out.save(root / "inputs" / f"rejected_{it['key']}_{st}.jpg", quality=85)
                    rej.write(json.dumps({"key": it["key"], "state": st, "dino_sim": sim}) + "\n")
                    rej.flush()
                if n % 50 == 0:
                    print(f"{n}/{len(idx)}  kept {kept}", flush=True)
    print("done ->", root)


if __name__ == "__main__":
    main()
