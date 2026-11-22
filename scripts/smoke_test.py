"""End-to-end plumbing test on a laptop CPU: no GPU, no downloads, ~5 minutes.

    python scripts/smoke_test.py

1. builds a fake mini VITON-HD (generated garments) in outputs/smoke/
2. runs the REAL scripts/02_make_synthetic.py on it (procedural crumpling + visibility masks)
3. trains with tiny random models + LoRA, stops halfway (simulated eviction), resumes
4. runs the REAL scripts/04_eval.py canon on the checkpoint (incl. visible/hidden split)
5. a separate overfit check: full fine-tuning on 2 pairs must drive the loss down

Generated images are noise (random weights); this only proves every piece connects.
Passing here means the first GPU run differs only by `smoke: false` and real weights.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from garmentcanon.train.finetune import train
from garmentcanon.utils.common import load_config

ROOT = Path("outputs/smoke")
REPO = Path(__file__).resolve().parents[1]


def make_fake_viton(root: Path, n: int = 6):
    rng = np.random.default_rng(0)
    for sub in ("cloth", "cloth-mask", "image"):
        (root / "test" / sub).mkdir(parents=True, exist_ok=True)
    lines = []
    shirt = np.array([[230, 120], [320, 90], [448, 90], [538, 120], [700, 260], [620, 360], [560, 320],
                      [560, 900], [208, 900], [208, 320], [148, 360], [68, 260]], np.int32)
    for i in range(n):
        img = np.full((1024, 768, 3), 255, np.uint8)
        m = np.zeros((1024, 768), np.uint8)
        cv2.fillPoly(m, [shirt], 255)
        base = rng.integers(20, 220, 3)
        tex = np.zeros_like(img); tex[:] = base
        period = int(rng.integers(30, 90))
        for y in range(0, 1024, period):
            tex[y:y + period // 3] = 255 - base
        cv2.circle(tex, (384, 420), int(rng.integers(50, 100)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        img[m > 0] = tex[m > 0]
        name = f"{i:05d}_00.jpg"
        cv2.imwrite(str(root / "test" / "cloth" / name), img[..., ::-1])
        cv2.imwrite(str(root / "test" / "cloth-mask" / name), m)
        cv2.imwrite(str(root / "test" / "image" / name), img[..., ::-1])   # placeholder "person"
        lines.append(f"{name} {name}")
    (root / "test_pairs.txt").write_text("\n".join(lines) + "\n")


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main():
    t0 = time.time()
    if ROOT.exists():
        shutil.rmtree(ROOT)
    viton = ROOT / "VITON-HD"
    make_fake_viton(viton)
    print("[1/5] fake VITON-HD ready")

    run([sys.executable, "scripts/02_make_synthetic.py", "--viton", str(viton), "--split", "test",
         "--source", "procedural", "--variants", "2", "--workers", "2", "--out", str(ROOT / "synthetic")])
    data = ROOT / "synthetic" / "procedural_test"
    rows = (data / "manifest.jsonl").read_text().splitlines()
    assert len(rows) == 12 and all((data / json.loads(r)["visibility"]).exists() for r in rows)
    assert all(json.loads(r)["state"] in ("crumpled", "folded") for r in rows)
    print(f"[2/5] synthetic pairs: {len(rows)} (with visibility masks)")

    cfg = load_config(REPO / "configs/finetune_lora.yaml", [
        "smoke=true", "device=cpu", f"out_dir={ROOT / 'run_lora'}", f"data_roots=[{data}]",
        "batch_size=2", "max_steps=8", "save_every=2", "log_every=2", "num_workers=0",
        "max_steps_this_run=4"])
    r1 = train(cfg)                                         # stops at step 4: simulated eviction
    assert r1["step"] == 4, r1["step"]
    cfg["max_steps_this_run"] = 10**9
    r2 = train(cfg)                                         # must resume from step_4.pt
    assert r2["step"] == 8 and len(r2["losses"]) == 4, (r2["step"], len(r2["losses"]))
    ckpt = sorted((ROOT / "run_lora").glob("step_*.pt"))[-1]
    size_mb = ckpt.stat().st_size / 1e6
    print(f"[3/5] LoRA train + resume OK  (checkpoint {ckpt.name}: {size_mb:.1f} MB, trainable tensors only)")

    run([sys.executable, "scripts/04_eval.py", "canon", "--smoke", "--ckpt", str(ckpt), "--test", str(data),
         "--limit", "3", "--steps", "4", "--out", str(ROOT / "eval")])
    scores = json.loads((ROOT / "eval" / "canon_scores.json").read_text())["synthetic"]
    assert "visible/L1" in scores and "SSIM" in scores
    print(f"[4/5] eval OK: {json.dumps({k: round(v, 3) for k, v in scores.items()})}")

    over = dict(cfg, mode="full", out_dir=str(ROOT / "run_overfit"), max_steps=60, lr=1e-3, save_every=1000,
                log_every=20, cond_dropout=0.0)
    two = ROOT / "synthetic" / "two"
    (two).mkdir(parents=True, exist_ok=True)
    (two / "manifest.jsonl").write_text("\n".join(
        json.dumps(dict(json.loads(r), input=str((data / json.loads(r)["input"]).resolve()),
                        target=str((data / json.loads(r)["target"]).resolve()),
                        visibility=str((data / json.loads(r)["visibility"]).resolve()),
                        target_mask=str((data / json.loads(r)["target_mask"]).resolve()))) for r in rows[:2]) + "\n")
    over["data_roots"] = [str(two)]
    losses = train(over)["losses"]
    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    assert last < first * 0.8, (first, last)
    print(f"[5/5] overfit check OK: loss {first:.3f} -> {last:.3f}")
    print(f"\nSMOKE TEST PASSED in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
