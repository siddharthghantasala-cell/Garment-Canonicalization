from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """YAML config + `key=value` CLI overrides (values parsed as YAML: 1e-4, true, [a,b])."""
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def image_grid(rows: list[list[Image.Image]], col_titles: list[str] | None = None, cell: int = 256) -> Image.Image:
    """Rows x columns grid with optional column titles: the qualitative gallery figure."""
    n_r, n_c = len(rows), max(len(r) for r in rows)
    top = 28 if col_titles else 0
    grid = Image.new("RGB", (n_c * cell, n_r * cell + top), "white")
    d = ImageDraw.Draw(grid)
    for j, t in enumerate(col_titles or []):
        d.text((j * cell + 6, 6), t, fill="black")
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            im = im.convert("RGB").copy(); im.thumbnail((cell, cell))
            grid.paste(im, (j * cell + (cell - im.width) // 2, top + i * cell + (cell - im.height) // 2))
    return grid
