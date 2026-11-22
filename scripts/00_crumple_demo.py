"""Crumple a few garments and save a grid. Runs on a laptop CPU, no GPU needed.

    python scripts/00_crumple_demo.py --images assets/sample_clean.png --n 5
    python scripts/00_crumple_demo.py --viton /data/VITON-HD --count 8 --n 3
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from garmentcanon.data.crumple_procedural import CrumpleParams, crumple
from garmentcanon.utils.common import image_grid

ap = argparse.ArgumentParser()
ap.add_argument("--images", nargs="*", default=[])
ap.add_argument("--viton", help="VITON-HD root; uses test/cloth + cloth-mask")
ap.add_argument("--count", type=int, default=6, help="garments to take from VITON-HD")
ap.add_argument("--n", type=int, default=4, help="crumpled variants per garment")
ap.add_argument("--out", default="outputs/crumple_demo.png")
a = ap.parse_args()

items = [(np.array(Image.open(p).convert("RGB")), None) for p in a.images]
if a.viton:
    from garmentcanon.data.viton_hd import VitonHD
    ds = VitonHD(a.viton, "test")
    for i in range(min(a.count, len(ds))):
        it = ds[i]; items.append((it["cloth"], it["cloth_mask"]))

rows = []
for gi, (img, mask) in enumerate(items):
    row = [Image.fromarray(img)]
    for k in range(a.n):
        out, _ = crumple(img, mask, CrumpleParams(seed=gi * 1000 + k))
        row.append(Image.fromarray(out))
    rows.append(row)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
image_grid(rows, ["clean"] + [f"crumpled #{k+1}" for k in range(a.n)]).save(a.out)
print("saved", a.out)
