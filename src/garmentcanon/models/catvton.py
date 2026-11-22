"""CatVTON wrapper: the downstream try-on model we do NOT train.

Setup (once):
    git clone https://github.com/Zheng-Chong/CatVTON third_party/CatVTON
    pip install -r third_party/CatVTON/requirements.txt   # skip detectron2/densepose: we use dataset masks

Matches the repo's app.py / inference.py: CatVTONPipeline(base_ckpt, attn_ckpt, "mix", ...),
person resized-and-cropped and garment resized-and-padded to 768x1024, mask blurred with
VaeImageProcessor(blur_factor=9), 50 steps, guidance 2.5. VITON-HD masks come from CatVTON's
preprocessed release: test/agnostic-mask/<person>_mask.png.

License note: CatVTON is CC BY-NC-SA 4.0 (non-commercial). Fine for a course project; a
commercial app would need a different try-on model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

CATVTON_DIR = Path(__file__).resolve().parents[3] / "third_party" / "CatVTON"
SIZE = (768, 1024)  # (width, height)


class CatVTONRunner:
    def __init__(self, device: str = "cuda", base_ckpt: str = "booksforcharlie/stable-diffusion-inpainting",
                 attn_repo: str = "zhengchong/CatVTON", dtype: str = "bf16"):
        if not CATVTON_DIR.exists():
            raise FileNotFoundError(f"clone CatVTON into {CATVTON_DIR} first (see docstring)")
        sys.path.insert(0, str(CATVTON_DIR))
        from diffusers.image_processor import VaeImageProcessor
        from huggingface_hub import snapshot_download
        from model.pipeline import CatVTONPipeline
        from utils import init_weight_dtype, resize_and_crop, resize_and_padding

        self._crop, self._pad = resize_and_crop, resize_and_padding
        self.pipe = CatVTONPipeline(base_ckpt=base_ckpt, attn_ckpt=snapshot_download(attn_repo),
                                    attn_ckpt_version="mix", weight_dtype=init_weight_dtype(dtype),
                                    use_tf32=True, device=device)
        self.mask_proc = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                           do_convert_grayscale=True)
        self.device = device

    @torch.no_grad()
    def __call__(self, person: Image.Image, garment: Image.Image, mask: Image.Image,
                 steps: int = 50, guidance: float = 2.5, seed: int = 42) -> Image.Image:
        person = self._crop(person.convert("RGB"), SIZE)
        garment = self._pad(garment.convert("RGB"), SIZE)
        mask = self.mask_proc.blur(self._crop(mask, SIZE), blur_factor=9)
        g = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(image=person, condition_image=garment, mask=mask,
                         num_inference_steps=steps, guidance_scale=guidance, generator=g)[0]
