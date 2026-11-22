"""Build every model component in one place: real pretrained weights, or tiny random stand-ins.

`smoke=True` builds the same architecture types at toy size with random weights and no
downloads, so the full train/sample/eval pipeline can be tested on a laptop CPU. Shapes that
matter to the pipeline are kept real: 512x512 images, 4x64x64 latents, 1024 SigLIP tokens of
768 dims, 77 context tokens. Smoke outputs are noise; only the plumbing is being tested.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .tryoffdiff_arch import TryOffDiff

SIGLIP_ID = "google/siglip-base-patch16-512"
SD_ID = "CompVis/stable-diffusion-v1-4"
TRYOFFDIFF_REPO = "rizavelioglu/tryoffdiff"


@dataclass
class Components:
    img_enc: nn.Module          # SigLIP vision tower (frozen)
    vae: nn.Module              # SD-v1 VAE (frozen)
    net: TryOffDiff             # what we fine-tune
    train_scheduler: object     # adds noise during training
    sample_scheduler: object    # removes noise at inference
    vae_scale: float            # SD-v1 latent scaling factor (0.18215)


def _tiny_unet():
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=768,
        attention_head_dim=8, norm_num_groups=16)


def _tiny_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(16, 16, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=1, norm_num_groups=8, sample_size=512, scaling_factor=0.18215)


def _tiny_siglip():
    from transformers import SiglipVisionConfig, SiglipVisionModel
    cfg = SiglipVisionConfig(hidden_size=768, intermediate_size=256, num_hidden_layers=1,
                             num_attention_heads=8, image_size=512, patch_size=16)
    return SiglipVisionModel(cfg)


def _clean_state(state) -> dict:
    """Accept the common ways a checkpoint gets saved: a plain state dict, a whole nn.Module,
    a dict wrapping the weights ({'state_dict': ...} / {'model': ...}), and keys prefixed by
    DataParallel/Accelerate ('module.') or torch.compile ('_orig_mod.')."""
    if isinstance(state, nn.Module):
        state = state.state_dict()
    for k in ("state_dict", "model", "net"):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]
    out = {}
    for k, v in state.items():
        for pre in ("module.", "_orig_mod."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def build_components(device: str = "cuda", smoke: bool = False, weights: str | None = None,
                     sampler: str = "euler") -> Components:
    """sampler: 'euler' (TryOffDiff paper / HF demo) or 'pndm' (authors' predict.py for v1)."""
    from diffusers import DDPMScheduler, EulerDiscreteScheduler, PNDMScheduler

    if smoke:
        torch.manual_seed(0)
        img_enc, vae, unet = _tiny_siglip(), _tiny_vae(), _tiny_unet()
        train_sched = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
        sample_sched = (EulerDiscreteScheduler if sampler == "euler" else PNDMScheduler).from_config(train_sched.config)
        net = TryOffDiff(unet)
    else:
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import SiglipVisionModel
        img_enc = SiglipVisionModel.from_pretrained(SIGLIP_ID)
        vae = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae")
        train_sched = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
        Sched = EulerDiscreteScheduler if sampler == "euler" else PNDMScheduler
        try:      # the authors' scheduler config if their repo has one, else Stable Diffusion's
            sample_sched = Sched.from_pretrained(snapshot_download(TRYOFFDIFF_REPO, allow_patterns=["scheduler/*"]),
                                                 subfolder="scheduler")
        except (OSError, ValueError, EnvironmentError):
            print("[weights] no scheduler/ in the TryOffDiff repo; using Stable Diffusion's config", flush=True)
            sample_sched = Sched.from_config(train_sched.config)
        # Which weight file: a local path, or a file name inside the HF repo. Override with the
        # TRYOFFDIFF_WEIGHTS environment variable if the repo's file is named differently
        # (list them: python -c "from huggingface_hub import list_repo_files as l; print(l('rizavelioglu/tryoffdiff'))").
        weights = weights or os.environ.get("TRYOFFDIFF_WEIGHTS", "tryoffdiff.pth")
        path = weights if Path(weights).exists() else hf_hub_download(TRYOFFDIFF_REPO, weights)
        state = _clean_state(torch.load(path, map_location="cpu", weights_only=False))
        has_tf = any(k.startswith("transformer.") for k in state)
        print(f"[weights] TryOffDiff adapter: {'transformer + ' if has_tf else ''}Linear + LayerNorm "
              f"({len(state)} tensors)", flush=True)
        net = TryOffDiff(UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet"), use_transformer=has_tf)
        missing, unexpected = net.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys in TryOffDiff weights: {unexpected[:5]}"
        assert not missing, f"missing keys in TryOffDiff weights: {missing[:5]}"

    img_enc.eval().requires_grad_(False).to(device)
    vae.eval().requires_grad_(False).to(device)
    net.to(device)
    return Components(img_enc, vae, net, train_sched, sample_sched, 0.18215)


def load_finetuned(net: TryOffDiff, ckpt_path: str):
    """Load one of our checkpoints (LoRA, full, or adapter) on top of the base TryOffDiff weights."""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if st.get("mode") == "lora":
        from .lora import configure_trainable
        configure_trainable(net, "lora", st["cfg"].get("lora_rank", 16), st["cfg"].get("lora_alpha", 16), verbose=False)
    missing, unexpected = net.load_state_dict(st["net"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return st.get("step")
