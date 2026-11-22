"""Choosing what to train.

Full fine-tuning updates all ~860M U-Net weights: best quality, most memory.
LoRA freezes the U-Net and learns low-rank updates W + B@A (rank r) on the attention
projections only, typically a few million parameters. The small adapter that maps SigLIP
tokens into the U-Net (transformer, proj, norm) is always trained fully: it's tiny, and it's
exactly the part that has to learn "this input is a crumpled garment, not a person".

Checkpoints store ONLY the trainable tensors (see trainable_state_dict), so a LoRA checkpoint
is tens of MB instead of 3.4 GB. Loading = base TryOffDiff weights + these tensors on top.
"""
from __future__ import annotations

import torch.nn as nn


def configure_trainable(net: nn.Module, mode: str = "lora", rank: int = 16, alpha: int = 16,
                        verbose: bool = True) -> list[nn.Parameter]:
    for p in net.parameters():
        p.requires_grad_(False)

    if mode == "full":
        for p in net.parameters():
            p.requires_grad_(True)
    elif mode == "lora":
        from peft import LoraConfig
        cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                         target_modules=["to_q", "to_k", "to_v", "to_out.0"])
        net.unet.add_adapter(cfg)               # injects LoRA layers; only they are trainable in the U-Net
        for name, mod in net.named_children():  # train everything outside the U-Net (the adapter)
            if name != "unet":
                for p in mod.parameters():
                    p.requires_grad_(True)
    elif mode == "adapter":                     # cheapest ablation: U-Net fully frozen
        for name, mod in net.named_children():
            if name != "unet":
                for p in mod.parameters():
                    p.requires_grad_(True)
    else:
        raise ValueError(mode)

    params = [p for p in net.parameters() if p.requires_grad]
    if verbose:
        n_train = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in net.parameters())
        print(f"[trainable] {mode}: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M parameters "
              f"({100*n_train/n_total:.2f}%)")
    return params


def trainable_state_dict(net: nn.Module) -> dict:
    names = {n for n, p in net.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in net.state_dict().items() if k in names}
