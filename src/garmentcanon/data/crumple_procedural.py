"""Procedural crumpling: turn a clean catalog garment image into a casual, off-body photo.

Why this exists
---------------
No public dataset pairs crumpled/folded garment photos with clean catalog images.
So we *make* the input side from the clean side. Because every output pixel is
sampled from the original image, logos and prints are preserved exactly, which
gives us perfectly aligned (input, target) training pairs for free.

Pipeline (each step is a separate function so you can study and ablate it):

    clean garment + mask
        -> fold_over()          partial fold along a random line (hides part of the garment)
        -> tps_warp()           thin-plate-spline warp from random control points (global distortion)
        -> wrinkle_shading()    random height field -> surface normals -> Lambertian shading
        -> random_affine()      rotation / scale / translation
        -> composite()          drop onto a procedural background (bedding, wood, solid) with a shadow
        -> photo_degrade()      color/exposure jitter, sensor noise, blur, JPEG

Limitations (be honest about these in the report):
- No true 3D: folds are 2D reflections, wrinkles are shading only.
- "Hanging" garments are not modeled here; use crumple_generative.py for those.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
@dataclass
class CrumpleParams:
    """Every knob in one place. Sampled per image; saved alongside outputs for reproducibility."""

    out_size: int = 512
    fold_prob: float = 0.6            # chance of a fold-over
    fold_max_frac: float = 0.35       # max fraction of the garment the fold can cover
    tps_points: int = 9               # control points per side (grid is tps_points x tps_points)
    tps_strength: float = 0.035       # max control-point displacement, as fraction of image size
    wrinkle_count: int = 28           # number of ridge "wrinkles"
    wrinkle_depth: float = 2.5        # height-field scale (steeper wrinkles = stronger shading)
    rot_deg: float = 25.0             # max |rotation|
    scale_range: tuple = (0.65, 0.9)  # garment size relative to frame
    background: str = "random"        # random | bedding | wood | solid
    noise_sigma: float = 4.0          # sensor noise (0-255 scale)
    jpeg_quality: tuple = (60, 95)
    seed: int | None = None
    record: dict = field(default_factory=dict)  # filled with the sampled values


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def mask_from_white_background(img: np.ndarray, thresh: int = 245) -> np.ndarray:
    """Catalog images sit on near-white backgrounds. Returns a float mask in [0, 1].

    White parts of the garment (stripes, a white collar) look like background, so after
    thresholding: close small gaps, keep every sizable region (stripes can split the
    garment into several), and fill enclosed holes.
    VITON-HD ships real masks in `cloth-mask/`; prefer those when available.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = (gray < thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        m = np.isin(labels, 1 + np.flatnonzero(areas >= 0.02 * areas.max())).astype(np.uint8)
        # fill enclosed holes: flood the background from the border; whatever it can't reach is inside
        ff = np.pad(m, 1)
        cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 1)
        m = m | (ff[1:-1, 1:-1] == 0).astype(np.uint8)
    m = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)
    return np.clip(m, 0, 1)


def _smooth_noise(h: int, w: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in [-1, 1]: random values on a coarse grid, upsampled smoothly."""
    gh, gw = max(2, h // scale), max(2, w // scale)
    coarse = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------
def fold_over(img, mask, rng, max_frac=0.35, uv=None):
    """Reflect one side of the garment across a random line and lay it on top.

    The folded flap shows the garment's *inside*, which we approximate by darkening
    and desaturating the mirrored pixels, and we add a soft shadow along the crease.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return img, mask, {}, uv
    cx, cy = xs.mean(), ys.mean()
    theta = rng.uniform(0, np.pi)
    n = np.array([np.cos(theta), np.sin(theta)])          # line normal
    # project garment pixels on the normal; place the line so only `frac` of them flip
    proj = (xs - cx) * n[0] + (ys - cy) * n[1]
    frac = rng.uniform(0.12, max_frac)
    d = np.quantile(proj, 1 - frac)                         # signed distance of the fold line

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    signed = (xx - cx) * n[0] + (yy - cy) * n[1] - d        # >0 is the side that gets folded
    # reflection of every pixel across the line: p' = p - 2*signed*n
    signed = signed.astype(np.float32)
    map_x = (xx - 2 * signed * n[0]).astype(np.float32)
    map_y = (yy - 2 * signed * n[1]).astype(np.float32)
    # the flap lands on the *other* side (signed < 0); sample it from the reflected side
    flap = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    flap_m = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    flap_m = flap_m * (signed < 0)

    # inside of the fabric: mostly the base fabric color, with the print only faintly
    # bleeding through (heavy blur), darker because it faces away from the light
    flap_f = flap.astype(np.float32)
    k = int(0.04 * max(h, w)) | 1
    bleed = cv2.GaussianBlur(flap_f, (k, k), 0)
    gray = bleed.mean(axis=2, keepdims=True)
    flap_f = 0.72 * (0.55 * bleed + 0.25 * gray + 0.20 * flap_f)

    out_m = mask * (signed < 0)                             # the folded side is removed...
    out = img.astype(np.float32)
    # crease shadow on the base layer
    crease = np.exp(-(signed ** 2) / (2 * (0.01 * max(h, w)) ** 2))
    out *= (1 - 0.35 * crease)[..., None]
    a = flap_m[..., None]
    out = out * (1 - a) + flap_f * a                        # ...and laid back on top
    out_m = np.clip(out_m + flap_m, 0, 1)
    if uv is not None:                       # what the camera now sees of the garment's FRONT:
        uv = uv.copy()
        uv[signed >= 0] = -1                 # the folded-away side is gone
        uv[flap_m > 0.5] = -1                # the flap shows the inside, not the front
    return np.clip(out, 0, 255).astype(np.uint8), out_m.astype(np.float32), {
        "fold_theta": float(theta), "fold_frac": float(frac)}, uv


def tps_warp(img, mask, rng, points=9, strength=0.06, uv=None):
    """Thin-plate-spline warp.

    TPS finds the smoothest (minimum bending energy) mapping that moves a set of
    control points to new positions. We jitter a grid of control points and let TPS
    interpolate a dense displacement field for every pixel.
    """
    h, w = mask.shape
    g = np.linspace(0, 1, points)
    src = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)                 # (N, 2) in [0,1]
    disp = rng.uniform(-strength, strength, src.shape)
    disp[(src == 0).any(1) | (src == 1).any(1)] *= 0.3                     # calmer borders
    # evaluate TPS on a coarse grid (fast), then upsample the field
    cs = 64
    cg = np.stack(np.meshgrid(np.linspace(0, 1, cs), np.linspace(0, 1, cs)), -1).reshape(-1, 2)
    field = RBFInterpolator(src, disp, kernel="thin_plate_spline", smoothing=0.0)(cg).reshape(cs, cs, 2)
    fx = cv2.resize(field[..., 0].astype(np.float32), (w, h)) * w
    fy = cv2.resize(field[..., 1].astype(np.float32), (w, h)) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x, map_y = xx + fx, yy + fy
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    out_m = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if uv is not None:
        uv = cv2.remap(uv, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1))
    return out, out_m, {"tps_strength": float(strength)}, uv


def wrinkle_heightfield(h, w, rng, count=28):
    """Sum of elongated ridges + low-frequency bumps. Higher = closer to the camera."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    H = 0.6 * _smooth_noise(h, w, 96, rng)
    for _ in range(count):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(0.15, 0.6) * max(h, w)
        width = rng.uniform(0.008, 0.03) * max(h, w)
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)      # along the ridge
        v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)     # across the ridge
        H += rng.uniform(0.4, 1.0) * np.exp(-(v ** 2) / (2 * width ** 2) - (u ** 2) / (2 * (length / 2) ** 2))
    return H


def wrinkle_shading(img, mask, rng, count=28, depth=1.0, uv=None):
    """Shade the garment as if it had wrinkles.

    height field H -> gradients -> surface normal n = (-dH/dx, -dH/dy, 1)/|.|
    Lambertian shading = max(0, n . light). Also nudges pixels along the gradient so
    prints visibly bend over the ridges.
    """
    h, w = mask.shape
    H = wrinkle_heightfield(h, w, rng, count) * depth
    gx = cv2.Sobel(H, cv2.CV_32F, 1, 0, ksize=5) / 8.0
    gy = cv2.Sobel(H, cv2.CV_32F, 0, 1, ksize=5) / 8.0
    nrm = np.stack([-gx, -gy, np.ones_like(H)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    az, el = rng.uniform(0, 2 * np.pi), rng.uniform(np.pi / 5, np.pi / 2.5)
    light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float32)
    lam = np.clip(nrm @ light, 0, 1)
    shade = lam / (lam[mask > 0.5].mean() + 1e-6)          # mean brightness stays ~1
    shade = np.clip(1 + 1.6 * (shade - 1), 0.4, 1.3)        # exaggerate contrast
    # small geometric displacement so texture follows the wrinkles
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    k = 0.004 * max(h, w)
    mx, my = xx + k * gx, yy + k * gy
    disp_img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if uv is not None:
        uv = cv2.remap(uv, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1))
    out = disp_img.astype(np.float32) * (1 + (shade[..., None] - 1) * mask[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), mask, {"light_az": float(az), "light_el": float(el)}, uv


def random_affine(img, mask, rng, size, rot_deg=25.0, scale_range=(0.65, 0.9), uv=None):
    """Fit the garment into a `size` x `size` frame with random rotation/scale/offset."""
    ys, xs = np.nonzero(mask > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    s = rng.uniform(*scale_range) * size / max(bw, bh)
    ang = rng.uniform(-rot_deg, rot_deg)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, s)
    M[:, 2] += np.array([size / 2 - cx, size / 2 - cy]) + rng.uniform(-0.06, 0.06, 2) * size
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    out_m = cv2.warpAffine(mask, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    if uv is not None:
        uv = cv2.warpAffine(uv, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=(-1, -1))
    return out, out_m, {"rot": float(ang), "scale": float(s)}, uv


def procedural_background(size, rng, kind="random"):
    kinds = ["bedding", "wood", "solid"]
    kind = rng.choice(kinds) if kind == "random" else kind
    base = rng.uniform(90, 235, 3)
    if kind == "solid":
        bg = np.ones((size, size, 3), np.float32) * base
        bg *= (1 + 0.08 * _smooth_noise(size, size, 128, rng))[..., None]
    elif kind == "wood":
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        warp = 25 * _smooth_noise(size, size, 64, rng)
        rings = np.sin((yy + warp) / rng.uniform(4, 10)) * 0.5 + 0.5
        wood = np.array([rng.uniform(110, 170), rng.uniform(70, 110), rng.uniform(40, 70)])
        bg = wood * (0.8 + 0.25 * rings)[..., None]
    else:  # bedding: soft folds + fine fabric grain
        folds = wrinkle_heightfield(size, size, rng, count=10)
        folds = (folds - folds.min()) / (np.ptp(folds) + 1e-6)
        grain = rng.normal(0, 1, (size, size)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (3, 3), 0)
        bg = base * (0.8 + 0.25 * folds + 0.03 * grain)[..., None]
    return np.clip(bg, 0, 255).astype(np.float32), str(kind)


def composite(img, mask, rng, kind="random"):
    size = mask.shape[0]
    bg, kind = procedural_background(size, rng, kind)
    # soft drop shadow, offset toward the bottom-right
    sh = cv2.GaussianBlur(mask, (0, 0), sigmaX=size * 0.015)
    off = (rng.uniform(0.005, 0.02, 2) * size).astype(int)
    sh = np.roll(sh, (off[1], off[0]), axis=(0, 1))
    bg *= (1 - 0.35 * sh)[..., None]
    a = mask[..., None]
    out = img.astype(np.float32) * a + bg * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8), {"background": kind}


def photo_degrade(img, rng, noise_sigma=4.0, jpeg_quality=(60, 95)):
    out = img.astype(np.float32)
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)               # exposure
    out *= rng.uniform(0.92, 1.08, 3)                                         # white balance
    out += rng.normal(0, noise_sigma, out.shape)                              # sensor noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.0))            # slight defocus
    q = int(rng.integers(*jpeg_quality))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out, {"jpeg_quality": q}


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def crumple(img: np.ndarray, mask: np.ndarray | None = None, params: CrumpleParams | None = None,
            return_visibility: bool = False):
    """Clean garment (H, W, 3 uint8 RGB) -> (casual photo, params-with-record[, visibility]).

    The photo is params.out_size x params.out_size. With return_visibility=True, also returns
    a float mask in the CLEAN image's frame: 1 where that part of the garment's front is still
    visible in the photo, 0 where it was folded away or hidden. Use it to score visible and
    hidden regions separately (eval/metrics.py: masked_metrics).
    """
    p = params or CrumpleParams()
    rng = np.random.default_rng(p.seed)
    if mask is None:
        mask = mask_from_white_background(img)
    mask = mask.astype(np.float32)
    if mask.max() > 1:
        mask /= 255.0
    rec = {}

    # work in a square canvas so rotations never clip the garment
    h, w = mask.shape
    side = int(max(h, w) * 1.3)
    canvas = np.full((side, side, 3), 255, np.uint8)
    cmask = np.zeros((side, side), np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    cmask[oy:oy + h, ox:ox + w] = mask
    img, mask = canvas, cmask
    src_mask = cmask[oy:oy + h, ox:ox + w].copy()
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    uv = np.stack([xx, yy], -1) if return_visibility else None   # where each pixel came from

    if rng.random() < p.fold_prob:
        img, mask, r, uv = fold_over(img, mask, rng, p.fold_max_frac, uv); rec.update(r)
    img, mask, r, uv = tps_warp(img, mask, rng, p.tps_points, p.tps_strength, uv); rec.update(r)
    img, mask, r, uv = wrinkle_shading(img, mask, rng, p.wrinkle_count, p.wrinkle_depth, uv); rec.update(r)
    img, mask, r, uv = random_affine(img, mask, rng, p.out_size, p.rot_deg, p.scale_range, uv); rec.update(r)
    img, r = composite(img, mask, rng, p.background); rec.update(r)
    img, r = photo_degrade(img, rng, p.noise_sigma, p.jpeg_quality); rec.update(r)

    p.record = rec
    if not return_visibility:
        return img, p
    return img, p, _visibility_in_source(uv, mask, src_mask, (ox, oy), rec["scale"])


def _visibility_in_source(uv, out_mask, src_mask, offset, scale):
    """Scatter the photo's visible front pixels back into the clean image's frame."""
    h, w = src_mask.shape
    vis = np.zeros((h, w), np.uint8)
    ok = (out_mask > 0.5) & (uv[..., 0] >= 0)
    u = np.round(uv[..., 0][ok] - offset[0]).astype(int)
    v = np.round(uv[..., 1][ok] - offset[1]).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    vis[v[inb], u[inb]] = 1
    # the photo is smaller than the source, so scattered points leave gaps: close them
    k = 2 * int(np.ceil(1.0 / max(scale, 1e-3))) + 1
    vis = cv2.morphologyEx(vis, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return (vis > 0).astype(np.float32) * (src_mask > 0.5)


def procedural_state(record: dict) -> str:
    """Garment state label for a procedural sample: 'folded' if a fold-over was applied, else 'crumpled'.

    The proposal reports results per state; procedural data mixes both (fold_prob=0.6), so we
    label each sample from what was actually sampled. 'hanging' only exists in generative/real data.
    """
    return "folded" if "fold_theta" in record else "crumpled"


def params_to_dict(p: CrumpleParams) -> dict:
    d = asdict(p)
    d["scale_range"] = list(d["scale_range"]); d["jpeg_quality"] = list(d["jpeg_quality"])
    return d
