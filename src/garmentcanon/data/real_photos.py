"""Your own real-world test set.

Photograph each garment in every state and name files like:

    data/real/<garment_id>_<state>.jpg      state in {flat, crumpled, folded, hanging}

e.g. hoodie01_flat.jpg, hoodie01_crumpled.jpg, hoodie01_hanging.jpg

`flat` is the ground truth: garment laid neatly, front facing up, shot from directly
above on a plain background that contrasts with the garment, even lighting. Keep the
camera height similar across garments. Photos stay on your machine; don't redistribute
retailer images.

Why the flat photo needs preprocessing
--------------------------------------
The model outputs a garment centered on a white background, framed like a VITON-HD catalog
image. A phone photo has a floor/bed behind the garment and arbitrary framing. Comparing the
two pixel by pixel would mostly measure framing, not garment quality. So:

    flat photo --segment_flat_photo()--> garment mask
               --frame_garment()------> garment cut out, scaled, centered on white (768 x 1024)

Model predictions go through frame_garment() too (with a white-background mask), so both
sides of every comparison are framed by the same rule. Check the segmentation once with
`python scripts/05_prep_real.py --real data/real` (writes a preview grid).

iPhone .heic files: `pip install pillow-heif` (installed with this package), or shoot JPG.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:                                    # lets PIL open iPhone .heic photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_OK = True
except ImportError:
    _HEIC_OK = False

STATES = ("crumpled", "folded", "hanging")
VITON_SIZE = (768, 1024)                # (width, height) of VITON-HD cloth images


def index_real_photos(root: str | Path) -> dict[str, dict[str, Path]]:
    """{garment_id: {state: path}} for every garment that has a `flat` ground truth."""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in sorted(Path(root).glob("*.*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic", ".webp"}:
            continue
        gid, _, state = p.stem.rpartition("_")
        if gid and state in (*STATES, "flat"):
            groups[gid][state] = p
    return {g: s for g, s in groups.items() if "flat" in s}


def load_rgb(path: str | Path, max_side: int = 1024) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".heic" and not _HEIC_OK:
        raise RuntimeError(f"{path.name}: install pillow-heif to read .heic files (or convert to .jpg)")
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # phones store rotation in EXIF
    im.thumbnail((max_side, max_side))
    return np.array(im)


# --------------------------------------------------------------------------------------
# Ground-truth preprocessing
# --------------------------------------------------------------------------------------
def segment_flat_photo(img: np.ndarray, border: float = 0.04) -> np.ndarray:
    """Garment mask (float32 in {0, 1}) for a garment lying on a plain background.

    Steps:
      1. Background model. Room lighting makes the floor brighter on one side, so one
         background color isn't enough. Fit a smooth surface (a quadratic in x and y, per
         Lab channel) to the pixels in a thin strip around the border (the garment must
         not touch the border; leave some floor around it). Refit once on every pixel that
         the first fit calls background, so the model also covers the image's center.
      2. Each pixel's color distance from the background model at that pixel.
      3. Threshold from the background's own noise: a pixel is garment if its distance is
         well above what background pixels show (median + 6 x robust spread, at least 12
         Lab units). This doesn't care how many colors the garment has; a threshold that
         splits the histogram in two (Otsu) would split a striped shirt into its stripes.
      4. Clean up: remove speckles, keep every sizable region (white stripes on a light
         floor can split a garment into several pieces), fill enclosed holes (white parts
         inside the garment). (GrabCut edge refinement was tried and made edges worse:
         it pulled floor into the sleeve notches.)
    """
    h, w = img.shape[:2]
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), 1.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / w - 0.5, yy / h - 0.5
    basis = np.stack([np.ones_like(xn), xn, yn, xn * yn, xn ** 2, yn ** 2], -1)     # (h, w, 6)

    b = max(2, int(border * min(h, w)))
    is_bg = np.zeros((h, w), bool)
    is_bg[:b], is_bg[-b:], is_bg[:, :b], is_bg[:, -b:] = True, True, True, True
    rng = np.random.default_rng(0)
    for _ in range(2):
        idx = np.flatnonzero(is_bg)
        idx = rng.choice(idx, min(len(idx), 20000), replace=False)
        A, Y = basis.reshape(-1, 6)[idx], lab.reshape(-1, 3)[idx]
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        dist = np.linalg.norm(lab - basis @ coef, axis=-1)
        ref = dist[is_bg]
        med = np.median(ref)
        spread = 1.4826 * np.median(np.abs(ref - med))                 # robust std
        thr = max(med + 6 * spread, 12.0)
        is_bg = dist < thr
    m = (dist >= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))     # drop speckle noise
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))    # 1-2 px gaps only: a big
    # kernel would also fill the narrow notches where sleeves meet the body
    m = _fill_holes(_significant_components(m))
    return m.astype(np.float32)


def frame_garment(img: np.ndarray, mask: np.ndarray, size: tuple[int, int] = VITON_SIZE,
                  fill: float = 0.9) -> np.ndarray:
    """Cut the garment out, scale it so its bounding box fills `fill` of the frame along the
    tighter side, and center it on white. Same rule for ground truth and predictions."""
    W, H = size
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return np.full((H, W, 3), 255, np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = img[y0:y1, x0:x1].astype(np.float32)
    a = mask[y0:y1, x0:x1].astype(np.float32)[..., None]
    crop = crop * a + 255 * (1 - a)                                   # background -> white
    s = fill * min(W / (x1 - x0), H / (y1 - y0))
    nw, nh = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.full((H, W, 3), 255, np.float32)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = crop
    return np.clip(out, 0, 255).astype(np.uint8)


def prepare_flat_gt(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat phone photo -> (catalog-framed ground truth, garment mask in the photo's frame)."""
    m = segment_flat_photo(img)
    return frame_garment(img, m), m


def frame_prediction(img: np.ndarray, thresh: int = 245) -> np.ndarray:
    """Re-frame a model output (garment on white) by the same rule as the ground truth.

    Predictions sit on white, so no cut-out is needed: find the bounding box of every
    sizable non-white region (white stripes can split a garment into several regions, so
    taking only the largest one would frame a single stripe), and frame that whole box.

    Known limit: white garment parts ON the outline (a white collar or hem) can't be told
    apart from the white background, so the box shrinks to the non-white part and the
    garment is framed slightly larger than its ground truth. Look at such garments in the
    gallery; DISTS tolerates small misalignment far better than SSIM.
    """
    m = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))      # drop speckle noise
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return frame_garment(img, np.zeros(m.shape, np.float32))
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= 0.02 * areas.max())
    ys, xs = np.nonzero(np.isin(labels, keep))
    box = np.zeros(m.shape, np.float32)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1     # keep the white inside the box as-is
    return frame_garment(img, box)


def _fill_holes(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    flood = np.pad(m.astype(np.uint8), 1)
    ff = flood.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)   # fill from the outside
    holes = (ff == 0)[1:-1, 1:-1]
    return (m.astype(bool) | holes).astype(np.uint8)


def _significant_components(m: np.ndarray, rel: float = 0.02) -> np.ndarray:
    """Every connected region at least `rel` times the size of the largest one."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return np.zeros_like(m, np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, 1 + np.flatnonzero(areas >= rel * areas.max())).astype(np.uint8)


def _largest_component(m: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return np.zeros_like(m, np.uint8)
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
