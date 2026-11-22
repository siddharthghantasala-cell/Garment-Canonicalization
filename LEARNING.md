# Learning roadmap: understand every piece you build

The rule for this project: **don't run a file you can't explain.** Each concept below has a
one-line "backbone", an analogy, where it lives in the code, what to read, and a check you
should be able to pass before moving on. The order follows the build order and the timeline.

## The concept map (top-down)

```
                    Garment canonicalization
                   casual photo -> catalog image
                              |
      +-----------------------+------------------------+
      |                       |                        |
 1. GENERATIVE MODEL     2. DATA                  3. EVALUATION
 (how images are made)   (what it learns from)    (how we know it works)
      |                       |                        |
 1a Diffusion            2a Domain gap            3a Perceptual metrics
 1b Latent diffusion     2b Image warping / TPS   3b Distribution metrics
 1c Conditioning         2c Shading & normals     3c Downstream evaluation
 1d Guidance (CFG)       2d Generative synthesis
 1e Fine-tuning & LoRA
      |
 4. THE TASK FAMILY: try-on (CatVTON) <-> try-off (TryOffDiff) <-> ours
```

---

## 1. The generative model

### 1a. Diffusion models
- **Backbone:** learn to remove a little noise from an image; run that backwards from pure noise and you get a new image.
- **Analogy:** a sculptor who is great at one move, "make this block slightly less blocky", repeated 1,000 times.
- **In code:** `train/finetune.py` (the loss: predict the noise `eps`), `models/tryoffdiff.py` (the sampling loop).
- **Read:** Lilian Weng, "What are Diffusion Models?" (blog); Ho et al., *DDPM* (2020), sections 2–3. Deeper: Calvin Luo, "Understanding Diffusion Models: A Unified Perspective" (2022).
- **Check:** Why does the network predict the *noise* rather than the clean image? Implement `add_noise` yourself (2 lines, see the docstring in `finetune.py`) and compare to `DDPMScheduler.add_noise`.

### 1b. Latent diffusion (Stable Diffusion)
- **Backbone:** run diffusion on a compressed 64×64×4 "latent" from a VAE instead of 512×512×3 pixels: 48× fewer numbers.
- **Analogy:** edit the blueprint, not the building; the VAE is the architect who turns blueprints into buildings.
- **In code:** `vae.encode(...) * 0.18215` in `finetune.py`, `vae.decode(lat / 0.18215)` in `tryoffdiff.py`.
- **Read:** Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022), sections 3.1–3.3.
- **Check:** Encode and decode one garment with the VAE and look at what's lost. (Fine logos often blur, which is a known limit of SD-v1's VAE and a reason RAGDiffusion++ switched backbones.)

### 1c. Conditioning: telling the model *which* image to make
- **Backbone:** the U-Net's cross-attention layers read a sequence of "context" tokens; normally text tokens, in TryOffDiff they're image tokens from SigLIP.
- **Analogy:** the sculptor keeps glancing at a reference photo pinned to the wall. Cross-attention is the glance.
- **In code:** `img_enc(...).last_hidden_state` → 1024 tokens × 768 dims → TryOffDiff's adapter → U-Net. **This is the exact interface your project changes:** same model, new kind of reference photo.
- **Read:** the TryOffDiff paper, section 3, then `models/tryoffdiff_arch.py` (its docstring traces every tensor shape). For SigLIP: Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (2023), just the idea.
- **Check:** What are the shapes at each step from input photo to U-Net context? Write them down.

### 1d. Classifier-free guidance (CFG)
- **Backbone:** predict noise with and without the condition, then extrapolate toward the conditioned one: `eps_u + w·(eps_c − eps_u)`.
- **Analogy:** "do what the reference says, and a bit more than you would on your own."
- **In code:** the `guidance` line in `tryoffdiff.py`; `cond_dropout` in the config (training must sometimes see a blank condition, or CFG has nothing to subtract).
- **Read:** Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022), the first 3 pages.
- **Check:** Run the pilot with guidance 1.0, 2.0, 4.0. What changes?

### 1e. Fine-tuning and LoRA
- **Backbone:** LoRA freezes a big weight matrix W and learns a small low-rank update B·A, so you train ~1% of the parameters.
- **Analogy:** instead of rewriting a textbook, you add sticky notes in the margins.
- **In code:** `models/lora.py` (what's frozen, what's trained, and the printed parameter count).
- **Read:** Hu et al., *LoRA* (ICLR 2022), sections 1 and 4.1.
- **Check:** Why train the SigLIP→U-Net adapter fully but the U-Net only with LoRA? (Hint: which part has to learn something genuinely new?)

---

## 2. The data

### 2a. Domain gap (sim-to-real)
- **Backbone:** a model trained on one distribution degrades on another; your whole project is about one such gap (catalog/worn → casual off-body).
- **Analogy:** a driver who learned in empty parking lots, now on a highway in the rain.
- **In code:** comparing `real/*` vs `synthetic` rows in `04_eval.py canon`.
- **Read:** the Street TryOn paper's introduction (CVPRW 2024): the same problem, for try-on.
- **Check:** Name three ways your synthetic crumples differ from real ones. Those are the hypotheses your real-photo test set tests.

### 2b. Image warping and thin-plate splines
- **Backbone:** a warp is a function telling each output pixel where to sample from; TPS is the smoothest such function that moves given control points to given targets.
- **Analogy:** pin a rubber sheet at a few points, drag the pins, and the rest of the sheet follows smoothly.
- **In code:** `tps_warp()` and `fold_over()` in `data/crumple_procedural.py` (both build `map_x, map_y` and call `cv2.remap`).
- **Read:** OpenCV `remap` docs; Bookstein, *Principal Warps* (1989), only the intro and figures.
- **Check:** Draw a grid on the clean shirt, crumple it, and look at how the grid bends. Change `tps_strength` and predict the result before you look.

### 2c. Shading from surface normals
- **Backbone:** brightness ≈ max(0, normal · light direction) (Lambertian); wrinkles change normals, so they change brightness.
- **Analogy:** a crumpled paper under a desk lamp: the facets facing the lamp are bright.
- **In code:** `wrinkle_heightfield()` → `wrinkle_shading()`.
- **Read:** any graphics intro on Lambertian shading (Scratchapixel has a good one).
- **Check:** Why compute the normal as (−∂H/∂x, −∂H/∂y, 1)?

### 2d. Generative data synthesis
- **Backbone:** use a pretrained image-editing model to render realistic inputs, then filter out outputs that changed the garment.
- **In code:** `data/crumple_generative.py` (prompts + DINOv2 similarity filter).
- **Check:** Look at 50 generated pairs and pick the filter threshold yourself. Record how often logos changed: that number goes in your report.

---

## 3. Evaluation

### 3a. Perceptual metrics: LPIPS and DISTS
- **Backbone:** compare images in the feature space of a pretrained network instead of pixel by pixel; DISTS also compares texture statistics, so a shifted but correct weave still scores well.
- **Analogy:** two photos of the same sweater taken a step apart look identical to you, but every pixel differs.
- **In code:** `eval/metrics.py`.
- **Read:** Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS, CVPR 2018); Ding et al., *DISTS* (TPAMI 2020), the abstract and figure 1.
- **Check:** Shift a garment image 4 px and compute SSIM, LPIPS, DISTS against the original. Which one barely moves?

### 3b. Distribution metrics: FID and KID
- **Backbone:** fit statistics to Inception features of many real and many generated images, then measure how far apart the two clouds are.
- **In code:** `DistributionMetrics` in `eval/metrics.py`.
- **Read:** Heusel et al. (FID, 2017) and Bińkowski et al. (KID, 2018), the definitions only.
- **Check:** Why is KID preferred for your small test sets?

### 3c. Downstream evaluation
- **Backbone:** judge a component by whether it improves the system it feeds: here, try-on quality against the real photo, with clean and raw inputs as upper and lower bounds.
- **In code:** `eval/tryon_eval.py` (`gap_closed`).
- **Check:** Explain to a friend in two sentences why "gap closed" is a better headline than raw LPIPS.

---

## 4. The task family
- **Try-on (CatVTON):** garment + person → person wearing garment. Read the CatVTON paper's method section; notice it needs no pose or parsing inputs.
- **Try-off (TryOffDiff):** person wearing garment → garment. Read the whole TryOffDiff paper: it's short and it's your starting point.
- **Ours:** off-body casual garment → garment. Be able to say exactly how it differs from both.
- **Context:** skim RAGDiffusion++ (retrieval priors, RL post-training) and ClothAtlas (robotic cloth state); you'll cite both.

## Suggested pacing (matches the timeline)
| Week | Learn | Build |
| --- | --- | --- |
| 1 | 1a, 1b, 2b, 2c | run `00_crumple_demo.py`, read every line of `crumple_procedural.py` |
| 2 | 1c, 1d, 3a, 3b | read `tryoffdiff_arch.py` + `tryoffdiff.py`, run the pilot, explain every metric number |
| 3 | 1e, 2d | generate synthetic data, first fine-tuning run |
| 4 | 2a, 3c | ablation runs, real-photo evaluation |
| 5–6 | 4 (write-up) | try-on evaluation, figures, report |

When you want to go deep on any of these, ask for a top-down lesson on it.
