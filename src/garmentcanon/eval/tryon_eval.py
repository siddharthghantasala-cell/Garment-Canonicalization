"""Downstream evaluation: does canonicalization make try-on better?

For each VITON-HD test pair (person P wearing garment G), run CatVTON with three
different garment inputs and compare each result to the real photo P:

    clean   G              -> upper bound (what try-on does with a perfect catalog image)
    raw     crumple(G)     -> lower bound (feeding the messy photo straight in)
    ours    model(crumple(G))

Headline number = fraction of the (raw -> clean) gap that "ours" closes:

    gap_closed = (score_raw - score_ours) / (score_raw - score_clean)     for lower-is-better metrics

1.0 means canonicalization is as good as having the catalog image; 0 means no help.

Two kinds of scores:
- full image: DISTS / LPIPS / SSIM on the whole 768 x 1024 try-on result.
- in-mask ("mask/..."): L1 and LPIPS averaged only over the agnostic mask, the region CatVTON
  repaints. Outside it, all three conditions copy the same person pixels, so full-image
  scores shrink every difference; the in-mask scores are the sharper comparison.
- KID (and FID, for reference) per condition when there are >= 50 images.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from PIL import Image

from .metrics import DistributionMetrics, PairedMetrics

_to_t = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
LOWER_IS_BETTER = {"DISTS": True, "LPIPS": True, "SSIM": False, "mask/L1": True, "mask/LPIPS": True,
                   "KID": True, "FID": True}


def gap_closed(raw: float, ours: float, clean: float, lower_is_better: bool = True) -> float:
    denom = (raw - clean) if lower_is_better else (clean - raw)
    num = (raw - ours) if lower_is_better else (ours - raw)
    return float("nan") if abs(denom) < 1e-9 else num / denom


def _load(p: Path, size) -> torch.Tensor:
    return _to_t(Image.open(p).convert("RGB").resize(size, Image.BICUBIC))[None]


def score_tryon_folder(results_dir: str | Path, gt_dir: str | Path, tag: str = "zeroshot",
                       mask_dir: str | Path | None = None, size=(768, 1024), device="cuda") -> dict:
    """Score results_dir/{clean, raw, ours_<tag>}/<key>.jpg against gt_dir/<key>.jpg.

    Only keys present in ALL three folders are scored, so the three rows are always computed
    on the same images. Writes results_dir/scores_<tag>.json.
    """
    results_dir, gt_dir = Path(results_dir), Path(gt_dir)
    conds = {"clean": "clean", "raw": "raw", "ours": f"ours_{tag}"}
    keys = None
    for folder in conds.values():
        ks = {p.stem for p in (results_dir / folder).glob("*.jpg")}
        keys = ks if keys is None else keys & ks
    keys = sorted(k for k in (keys or set()) if (gt_dir / f"{k}.jpg").exists())
    if not keys:
        raise FileNotFoundError(f"no keys common to {list(conds.values())} under {results_dir}")

    m = PairedMetrics(device)
    table = {}
    for cond, folder in conds.items():
        rows = []
        dm = DistributionMetrics(device, kid_subset=50) if len(keys) >= 50 else None
        for k in keys:
            a, b = _load(results_dir / folder / f"{k}.jpg", size), _load(gt_dir / f"{k}.jpg", size)
            r = m(a, b)
            if mask_dir is not None and (Path(mask_dir) / f"{k}_mask.png").exists():
                mk = Image.open(Path(mask_dir) / f"{k}_mask.png").convert("L").resize(size, Image.NEAREST)
                mk = (_to_t(mk)[None] > 0.5).float()
                r |= {f"mask/{n}": v for n, v in m.masked(a, b, mk).items()}
            rows.append(r)
            if dm:
                dm.update(a, b)
        names = {n for r in rows for n in r}
        table[cond] = {n: sum(r[n] for r in rows if n in r) / sum(n in r for r in rows) for n in sorted(names)}
        table[cond] |= (dm.compute() if dm else {}) | {"n": len(rows)}

    table["gap_closed"] = {n: gap_closed(table["raw"][n], table["ours"][n], table["clean"][n], low)
                           for n, low in LOWER_IS_BETTER.items()
                           if all(n in table[c] for c in conds)}
    (results_dir / f"scores_{tag}.json").write_text(json.dumps(table, indent=2))
    return table
