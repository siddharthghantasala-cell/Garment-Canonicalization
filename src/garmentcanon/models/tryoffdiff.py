"""Sampling: casual garment photo(s) -> canonical garment image(s).

How it works (LEARNING.md §1):
  1. SigLIP encodes the condition image into 1024 tokens x 768 dims.
  2. Start from Gaussian noise in Stable Diffusion's latent space (4 x 64 x 64 for 512 px).
  3. Each step, the network predicts the noise given the tokens. Classifier-free guidance:
     also predict with an all-zeros condition and extrapolate away from it.
  4. Decode the final latent with the SD VAE into a 512 x 512 image.

Published settings disagree between paper versions, so try both in the sanity check:
  - arXiv v1 (2411.18350v1): PNDM, 50 steps, guidance 2.0      -> sampler="pndm", steps=50
  - MGT follow-up (2504.13078): Euler, 20 steps, guidance 1.5  -> sampler="euler", guidance=1.5
Both: pad to square, resize 512, SigLIP-B/16-512, ~10 GB VRAM. Our defaults (Euler, 20,
2.0) sit between them; pick whichever setting gives the best sanity image and use it everywhere.
"""
from __future__ import annotations

import torch
import torchvision.transforms.v2 as T
from PIL import Image

from ..data.synthetic import PadToSquare
from .components import Components, build_components, load_finetuned

SIGLIP_NORM = T.Normalize([0.5], [0.5])


def condition_transform(size: int = 512):
    return T.Compose([PadToSquare(), T.Resize((size, size)), T.ToImage(), T.ToDtype(torch.float32, scale=True)])


class TryOffDiffRunner:
    def __init__(self, device: str = "cuda", smoke: bool = False, ckpt: str | None = None,
                 sampler: str = "euler", comps: Components | None = None):
        self.device = device
        self.c = comps or build_components(device, smoke=smoke, sampler=sampler)
        if ckpt:
            load_finetuned(self.c.net, ckpt)
        self.c.net.eval()
        self.tf = condition_transform()

    @torch.no_grad()
    def encode_condition(self, images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.tf(im.convert("RGB")) for im in images]).to(self.device)
        return self.c.img_enc(pixel_values=SIGLIP_NORM(x)).last_hidden_state      # (B, 1024, 768)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], steps: int = 20, guidance: float = 2.0,
                 seed: int = 0) -> list[Image.Image]:
        c = self.c
        cond = self.encode_condition(images)
        g = torch.Generator(self.device).manual_seed(seed)
        sched = c.sample_scheduler
        sched.set_timesteps(steps)
        lat = torch.randn((len(images), 4, 64, 64), generator=g, device=self.device) * sched.init_noise_sigma
        use_amp = self.device.startswith("cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            for t in sched.timesteps:
                if guidance and guidance != 1.0:
                    x_in = sched.scale_model_input(torch.cat([lat, lat]), t)
                    eps_u, eps_c = c.net(x_in, t, torch.cat([torch.zeros_like(cond), cond])).chunk(2)
                    eps = eps_u + guidance * (eps_c - eps_u)          # classifier-free guidance
                else:
                    eps = c.net(sched.scale_model_input(lat, t), t, cond)
                lat = sched.step(eps.float(), t, lat).prev_sample
        img = c.vae.decode(lat.float() / c.vae_scale).sample
        img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in img]
