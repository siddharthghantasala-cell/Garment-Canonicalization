"""TryOffDiff network: Stable Diffusion's U-Net + a small adapter for image conditioning.

Uses the layer names of `TryOffDiff` in the authors' repo (github.com/rizavelioglu/tryoffdiff,
tryoffdiff/modeling/model.py), so their released weights load directly. The U-Net is passed
in, so tests can use a tiny random U-Net instead of downloading 3.4 GB.

Two adapter variants exist in the TryOffDiff papers: the BMVC 2025 paper writes the condition
as C(I) = (LN o Linear o SigLIP)(I) (no transformer layer), while this code originally always
included one self-attention layer first. `use_transformer` selects the variant;
components.build_components() picks it from the keys actually present in the weight file, so
either kind of checkpoint loads.

Shapes, for one 512x512 condition image:

    SigLIP tokens        (B, 1024, 768)   1024 = (512/16)^2 patches, 768 dims each
    transformer          (B, 1024, 768)   one self-attention layer: patches exchange context
    permute + proj       (B,  768,  77)   Linear(1024 -> 77) mixes *tokens*, not features...
    permute + norm       (B,   77,  768)  ...giving exactly the (77 x 768) shape of SD-v1's
                                          CLIP *text* embedding, so the U-Net needs no changes
    U-Net cross-attn     reads those 77 tokens at every resolution, predicts the noise
"""
from __future__ import annotations

import torch
from torch import nn


class TryOffDiff(nn.Module):
    def __init__(self, unet: nn.Module, n_tokens_in: int = 1024, n_tokens_out: int = 77, dim: int = 768,
                 use_transformer: bool = True):
        super().__init__()
        self.unet = unet
        self.transformer = (nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
                            if use_transformer else nn.Identity())
        self.proj = nn.Linear(n_tokens_in, n_tokens_out)
        self.norm = nn.LayerNorm(dim)

    def adapt(self, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.transformer(cond_emb)                 # (B, 1024, 768)  (identity if no transformer)
        x = self.proj(x.permute(0, 2, 1))              # (B, 768, 77)
        return self.norm(x.permute(0, 2, 1))           # (B, 77, 768)

    def forward(self, noisy_latents: torch.Tensor, t: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        return self.unet(noisy_latents, t, encoder_hidden_states=self.adapt(cond_emb)).sample
