"""Synthetic (casual input, clean target) pairs written by scripts/02_make_synthetic.py.

Layout:
    data/synthetic/<source>_<split>/       source in {procedural, generative}
        inputs/<key>_<k>.jpg               k-th casual variant of garment <key>
        targets/<key>.jpg                  clean catalog image (copied from VITON-HD)
        masks/<key>.png                    garment mask of the target (from VITON-HD cloth-mask)
        visibility/<key>_<k>.png           procedural only: which target pixels are visible
        manifest.jsonl                     {"key", "input", "target", "state", ["visibility"], "params"}

`state` is crumpled / folded (procedural: from whether a fold was applied) or crumpled /
folded / hanging (generative: the prompt used). Older manifests without it are labeled on load.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class PadToSquare:
    """Pad the short side by repeating edge pixels, like TryOffDiff's own preprocessing."""

    def __call__(self, im):
        w, h = im.size if isinstance(im, Image.Image) else (im.shape[-1], im.shape[-2])
        s = max(w, h)
        pl, pt = (s - w) // 2, (s - h) // 2
        return TF.pad(im, [pl, pt, s - w - pl, s - h - pt], padding_mode="edge")


def eval_transform(size: int = 512, nearest: bool = False):
    """THE image -> tensor transform for scoring. Every script uses this one, so numbers from
    01_pilot and 04_eval are comparable: pad to square (edge pixels), resize, [0, 1] floats."""
    interp = T.InterpolationMode.NEAREST if nearest else T.InterpolationMode.BILINEAR
    return T.Compose([PadToSquare(), T.Resize((size, size), interpolation=interp), T.ToImage(),
                      T.ToDtype(torch.float32, scale=True)])


def row_state(r: dict) -> str:
    if "state" in r:
        return r["state"]
    p = r.get("params", {})
    if "state" in p:                                   # generative rows store the prompt state
        return p["state"]
    from .crumple_procedural import procedural_state
    return procedural_state(p.get("record", {}))


def read_manifest(root: str | Path) -> list[dict]:
    root = Path(root)
    rows, torn = [], 0
    for line in (root / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # half-written line from a run killed mid-write
                torn += 1
    if torn:
        print(f"[data] {root}: skipped {torn} unreadable manifest line(s)", flush=True)
    for r in rows:
        for k in ("input", "target", "target_mask", "visibility"):
            if k in r:
                r[k] = str(Path(r[k]) if Path(r[k]).is_absolute() else root / r[k])
        r["state"] = row_state(r)
    return rows


class SyntheticPairs(Dataset):
    def __init__(self, roots: list[str | Path], size: int = 512, with_visibility: bool = False):
        self.rows = [r for root in roots for r in read_manifest(root)]
        self.with_vis = with_visibility
        self.tf = eval_transform(size)
        self.mtf = eval_transform(size, nearest=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        out = {"input": self.tf(Image.open(r["input"]).convert("RGB")),     # casual photo -> condition
               "target": self.tf(Image.open(r["target"]).convert("RGB")),   # clean garment -> generated
               "state": r["state"], "key": r["key"]}
        if self.with_vis and "visibility" in r:
            out["visibility"] = self.mtf(Image.open(r["visibility"]).convert("L"))
        if self.with_vis and "target_mask" in r:      # garment pixels of the target, incl. white ones
            out["target_mask"] = (self.mtf(Image.open(r["target_mask"]).convert("L")) > 0.5).float()
        return out
