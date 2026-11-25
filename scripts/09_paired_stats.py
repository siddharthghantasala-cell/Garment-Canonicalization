"""Is model B really better than model A? Paired, per-image statistics (CPU only, seconds).

Both runs score the SAME test images, so compare image by image: d_i = B_i - A_i. Image
difficulty cancels in d_i, which makes this far more sensitive than comparing two averages.

For each metric it reports:
  mean A, mean B, mean difference, a 95% bootstrap confidence interval of the mean difference
  (resample images with replacement 10,000 times), how many images B wins, and a sign-test p-value
  (if B were no better, each image would be a coin flip).
A gain is credible when the 95% interval excludes 0.

--skip-first N drops the first N images: use it when those images were used to CHOOSE the
checkpoint (step 500 was picked on images 0-19), otherwise the estimate is biased in B's favour.

    python scripts/09_paired_stats.py outputs/eval_zs_100/per_image.jsonl \
        outputs/eval_pyr5e5_500_100/per_image.jsonl --skip-first 20
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a", help="baseline per_image.jsonl (e.g. zero-shot)")
ap.add_argument("b", help="candidate per_image.jsonl")
ap.add_argument("--skip-first", type=int, default=0)
ap.add_argument("--boot", type=int, default=10000)
args = ap.parse_args()


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return {r.get("i", r["key"]): r for r in rows}


A, B = load(args.a), load(args.b)
ids = sorted(k for k in A if k in B and not (isinstance(k, int) and k < args.skip_first))
for k in ids:
    assert A[k]["key"] == B[k]["key"], f"image {k}: different garments ({A[k]['key']} vs {B[k]['key']})"
metrics = [m for m in A[ids[0]] if isinstance(A[ids[0]][m], (int, float)) and m not in ("i", "hidden_frac")
           and all(m in A[k] and m in B[k] for k in ids)]
rng = np.random.default_rng(0)


def sign_p(wins, n):
    """Two-sided sign test: chance of a split at least this lopsided from fair coin flips."""
    lo = min(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(lo + 1)) / 2 ** n)


def report(sel, title):
    print(f"\n== {title}  (n = {len(sel)} images)")
    print(f"{'metric':>16} {'A':>8} {'B':>8} {'B-A':>8}   95% CI of B-A        B better on   sign p")
    for m in metrics:
        a = np.array([A[k][m] for k in sel]); b = np.array([B[k][m] for k in sel]); d = b - a
        up = "SSIM" in m                                   # SSIM: higher is better; others lower
        boot = d[rng.integers(0, len(d), (args.boot, len(d)))].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        wins = int((d > 0).sum() if up else (d < 0).sum())
        ties = int((d == 0).sum())
        flag = "  <-- credible" if (lo > 0 if up else hi < 0) else ("  <-- worse" if (hi < 0 if up else lo > 0) else "")
        print(f"{m:>16} {a.mean():8.4f} {b.mean():8.4f} {d.mean():+8.4f}   [{lo:+.4f}, {hi:+.4f}]   "
              f"{wins:>3}/{len(d) - ties:<3}      {sign_p(wins, len(d) - ties):.3g}{flag}")


report(ids, "all images" + (f" except the first {args.skip_first}" if args.skip_first else ""))
by_state = defaultdict(list)
for k in ids:
    by_state[A[k]["state"]].append(k)
for st, sel in sorted(by_state.items()):
    report(sel, f"state = {st}")
