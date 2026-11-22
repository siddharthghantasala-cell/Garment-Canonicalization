"""Check the real-photo ground truth before trusting any real-photo number. CPU only.

For every garment with a <id>_flat photo, shows: the photo with the detected garment outline,
and the ground truth the metrics will use (garment cut out and framed like a catalog image).

    python scripts/05_prep_real.py --real data/real            # -> outputs/real_gt_preview.jpg

If an outline is wrong (garment too close to the floor's color, garment touching the photo's
edge, strong shadows), retake that flat photo: plainer contrasting background, garment fully
inside the frame with some floor around it, softer light.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from garmentcanon.data.real_photos import index_real_photos, load_rgb, prepare_flat_gt
from garmentcanon.utils.common import image_grid

ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True)
ap.add_argument("--out", default="outputs/real_gt_preview.jpg")
ap.add_argument("--save-gt", help="optional folder to also save each framed ground truth as <id>_gt.png")
a = ap.parse_args()

rows, warnings = [], []
for gid, states in index_real_photos(a.real).items():
    photo = load_rgb(states["flat"])
    gt, mask = prepare_flat_gt(photo)
    frac = float((mask > 0.5).mean())
    edge = bool(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).max() > 0.5)
    if not 0.05 < frac < 0.85 or edge:
        warnings.append(f"{gid}: garment covers {frac:.0%} of the photo"
                        + (", touches the photo's edge" if edge else "") + " -> check it")
    outlined = photo.copy()
    contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(outlined, contours, -1, (255, 0, 0), max(2, photo.shape[0] // 200))
    rows.append([Image.fromarray(outlined), Image.fromarray(gt)])
    if a.save_gt:
        Path(a.save_gt).mkdir(parents=True, exist_ok=True)
        Image.fromarray(gt).save(Path(a.save_gt) / f"{gid}_gt.png")

if not rows:
    raise SystemExit(f"no <id>_flat photos found in {a.real}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
image_grid(rows, ["flat photo + detected outline", "ground truth used for scoring"], cell=384).save(a.out)
print(f"{len(rows)} garments -> {a.out}")
for w in warnings:
    print("WARNING", w)
