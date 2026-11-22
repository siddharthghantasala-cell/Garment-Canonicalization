"""Generative crumpling with an open image-editing model (Qwen-Image-Edit, Apache 2.0).

More realistic than procedural crumpling (real fabric physics "learned" from photos),
but the model may change colors, logos, or garment shape. So every output is scored
against the source garment with DINOv2 features and dropped below a threshold.

Memory: Qwen-Image-Edit is a ~20B image transformer plus a ~7B Qwen2.5-VL text encoder,
about 54 GB of weights in bf16, which fits neither an L4 (24 GB) nor an A100-40GB. So by
default both big components are loaded in 4-bit (bitsandbytes NF4, diffusers'
PipelineQuantizationConfig), and on GPUs under 30 GB the pipeline also offloads idle
components to CPU (enable_model_cpu_offload). quant="none" loads full bf16 (A100-80GB only).

Speed: each image costs `steps` denoising passes, doubled when true_cfg_scale > 1 (the
official examples use 4.0 with negative_prompt=" "). Expect tens of seconds per image:
keep the generative set small (~500) and time 10 images before launching a big batch.
"""
from __future__ import annotations

import torch
from PIL import Image

PROMPTS = {
    "crumpled": "Show this exact garment crumpled casually on an unmade bed, photographed from above with a phone. "
                "Keep every color, print, and logo identical.",
    "folded": "Show this exact garment folded in half on a wooden table, photographed with a phone. "
              "Keep every color, print, and logo identical.",
    "hanging": "Show this exact garment hanging on a plastic hanger against a closet wall, photographed with a phone. "
               "Keep every color, print, and logo identical.",
}


class GenerativeCrumpler:
    def __init__(self, model_id: str = "Qwen/Qwen-Image-Edit", device: str = "cuda", quant: str = "4bit",
                 offload: str = "auto", true_cfg_scale: float = 4.0):
        from diffusers import QwenImageEditPipeline
        kw = {}
        if quant == "4bit":
            from diffusers.quantizers import PipelineQuantizationConfig
            kw["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                              "bnb_4bit_compute_dtype": torch.bfloat16},
                components_to_quantize=["transformer", "text_encoder"])
        elif quant != "none":
            raise ValueError(quant)
        self.pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        if offload == "on" or (offload == "auto" and gpu_gb < 30):
            self.pipe.enable_model_cpu_offload()     # moves each component to the GPU only while it runs
        else:
            self.pipe.to(device)
        self.device, self.true_cfg_scale = device, true_cfg_scale

    @torch.no_grad()
    def edit(self, image: Image.Image, prompt: str, seed: int = 0, steps: int = 40) -> Image.Image:
        g = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(image=image.convert("RGB"), prompt=prompt, negative_prompt=" ",
                         true_cfg_scale=self.true_cfg_scale, num_inference_steps=steps, generator=g).images[0]

    def __call__(self, garment: Image.Image, state: str, seed: int = 0, steps: int = 40) -> Image.Image:
        return self.edit(garment, PROMPTS[state], seed=seed, steps=steps)


class FidelityFilter:
    """Cosine similarity of DINOv2 global features between source and edited garment."""

    def __init__(self, model_id: str = "facebook/dinov2-base", device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def similarity(self, a: Image.Image, b: Image.Image) -> float:
        x = self.proc(images=[a, b], return_tensors="pt").to(self.device)
        f = self.model(**x).pooler_output
        f = torch.nn.functional.normalize(f, dim=-1)
        return float((f[0] * f[1]).sum())


def keep(similarity: float, threshold: float = 0.6) -> bool:
    """Threshold is a guess: calibrate it by eyeballing ~50 pairs and picking the cutoff."""
    return similarity >= threshold
