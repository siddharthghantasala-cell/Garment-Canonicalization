"""VITON-HD loader.

Expected layout (the official release; get it from the VITON-HD GitHub / Kaggle mirror):

    VITON-HD/
      train_pairs.txt            "person.jpg cloth.jpg" per line
      test_pairs.txt
      train/ image/ cloth/ cloth-mask/ ...
      test/  image/ cloth/ cloth-mask/ ...

`cloth/` holds the clean catalog garment (our TARGET), `image/` the model wearing it
(ground truth for the downstream try-on evaluation), `cloth-mask/` the garment mask.

TryOffDiff removed duplicates and test leaks (11,552 train / 1,990 test). Their
cleaning script is in their repo; run it once and point `pairs_file` at the cleaned list.
Pair files may have two columns ("person.jpg cloth.jpg") or one (a file name per line).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class VitonItem:
    key: str
    person_path: Path
    cloth_path: Path
    cloth_mask_path: Path


class VitonHD:
    def __init__(self, root: str | Path, split: str = "test", pairs_file: str | None = None, paired: bool = True):
        self.root = Path(root)
        self.split = split
        if pairs_file:
            pf = Path(pairs_file)
        else:   # official release: test_pairs.txt; CatVTON's preprocessed copy: test_pairs_unpaired.txt
            cands = [self.root / f"{split}_pairs.txt", self.root / f"{split}_pairs_unpaired.txt"]
            pf = next((c for c in cands if c.exists()), cands[0])
        self.items: list[VitonItem] = []
        for line in pf.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            person, cloth = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], parts[0])  # 1- or 2-column lists
            if paired:            # paired setting: the person wears THIS garment -> ground truth exists
                cloth = person
            d = self.root / split
            self.items.append(VitonItem(
                key=Path(cloth).stem,
                person_path=d / "image" / person,
                cloth_path=d / "cloth" / cloth,
                cloth_mask_path=d / "cloth-mask" / cloth,
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        cloth = np.array(Image.open(it.cloth_path).convert("RGB"))
        mask = np.array(Image.open(it.cloth_mask_path).convert("L")).astype(np.float32) / 255.0 \
            if it.cloth_mask_path.exists() else None
        return {"key": it.key, "cloth": cloth, "cloth_mask": mask, "person_path": str(it.person_path)}
