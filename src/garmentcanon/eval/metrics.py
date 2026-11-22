"""Image-quality metrics, and why we use each one.

- DISTS (primary): compares deep-feature structure and texture statistics, so a correct fabric
  weave shifted by a few pixels still scores well. TryOffDiff and RAGDiffusion++ both found it
  tracks human judgment of garments best.
- LPIPS: distance between deep features ("perceptual similarity"). Lower is better.
- SSIM: local luminance/contrast/structure. Reported for comparability only: it rewards blurry
  outputs and penalizes realistic texture.
- FID / KID: compare the *distribution* of generated images to real ones (not pairwise). KID is
  unbiased and more reliable with small test sets (< ~2,000 images): prefer it here.
- Masked metrics: the same idea restricted to a region. With the crumpler's visibility mask,
  score VISIBLE regions (should be copied faithfully) and HIDDEN regions (must be guessed)
  separately. LPIPS supports this via its per-pixel distance map.

All functions take float tensors in [0, 1], shape (B, 3, H, W). Masks are (B, 1, H, W) in {0, 1}.
"""
from __future__ import annotations

import warnings

import torch


class PairedMetrics:
    def __init__(self, device: str = "cuda", which=("DISTS", "LPIPS", "SSIM")):
        self.device, self.fns = device, {}
        if "SSIM" in which:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.fns["SSIM"] = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if "LPIPS" in which:
            import lpips
            self._lpips = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
            self.fns["LPIPS"] = lambda a, b: self._lpips(a * 2 - 1, b * 2 - 1).mean()
        if "DISTS" in which:
            import piq
            self.fns["DISTS"] = piq.DISTS().to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        pred, target = pred.to(self.device), target.to(self.device)
        return {k: float(f(pred, target).mean()) for k, f in self.fns.items()}

    @torch.no_grad()
    def masked(self, pred, target, mask) -> dict[str, float]:
        """L1 and (if loaded) LPIPS averaged over the mask's pixels only."""
        pred, target, mask = pred.to(self.device), target.to(self.device), mask.to(self.device)
        denom = mask.sum().clamp_min(1.0)
        res = {"L1": float(((pred - target).abs().mean(1, keepdim=True) * mask).sum() / denom)}
        if hasattr(self, "_lpips"):
            dmap = self._lpips(pred * 2 - 1, target * 2 - 1)          # (B, 1, H, W)
            res["LPIPS"] = float((dmap * mask).sum() / denom)
        return res


def visible_hidden_metrics(m: PairedMetrics, pred, target, visibility, garment_mask=None) -> dict:
    """Split scores: visible garment region vs hidden garment region."""
    if garment_mask is None:   # garment = anything not near-white in the clean target
        garment_mask = (target.min(1, keepdim=True).values < 0.95).float()
    vis = visibility * garment_mask
    hid = (1 - visibility) * garment_mask
    out = {f"visible/{k}": v for k, v in m.masked(pred, target, vis).items()}
    if hid.sum() > 50:
        out |= {f"hidden/{k}": v for k, v in m.masked(pred, target, hid).items()}
    out["hidden_frac"] = float(hid.sum() / garment_mask.sum().clamp_min(1))
    return out


class DistributionMetrics:
    """Accumulate real/fake batches, then compute FID and KID once at the end."""

    def __init__(self, device: str = "cuda", kid_subset: int = 100):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.kid = KernelInceptionDistance(normalize=True, subset_size=kid_subset).to(device)
        self.device, self.n = device, 0

    def update(self, pred: torch.Tensor, real: torch.Tensor):
        self.n += len(pred)
        for m in (self.fid, self.kid):
            m.update(real.to(self.device), real=True)
            m.update(pred.to(self.device), real=False)

    def compute(self) -> dict[str, float]:
        if self.n < 2:
            warnings.warn("too few images for FID/KID", stacklevel=2)
            return {}
        self.kid.subset_size = min(self.kid.subset_size, self.n)
        kid_mean, kid_std = self.kid.compute()
        return {"FID": float(self.fid.compute()), "KID": float(kid_mean), "KID_std": float(kid_std)}
