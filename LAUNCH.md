# First GPU run: checklist

Everything below has passed the CPU smoke test (`python scripts/smoke_test.py`) with tiny random
models. A real run changes only two things: real weights (downloaded from Hugging Face on first
use, ~6 GB total) and real data. Do the steps in order; each one checks the one before it.
(CHTC instead of Colab: follow `chtc/README.md`, but do step 2 on Colab first.)

## 0. Environment (Colab or CHTC)
```bash
pip install -e ".[gpu,dev]"
git clone https://github.com/Zheng-Chong/CatVTON third_party/CatVTON   # only needed for try-on eval
python -m pytest -q && python scripts/smoke_test.py                     # same result as on CPU: PASSED
```
Don't `pip install -r third_party/CatVTON/requirements.txt` into this environment without
reading it first: if it pins older library versions, it can break diffusers/peft here. Try
the try-on eval without it; install single missing packages only if an import fails.

## 1. Data
- **VITON-HD.** The official release has everything: train/test splits, `cloth-mask/` and
  `agnostic-mask/` (checked Nov 21: all 2,032 test and 11,647 train pairs complete). It unzips to
  `VITON-HD/zalando-hd-resized/`; pass that inner folder as `--viton` (or move its contents up
  one level so `VITON-HD/` directly holds `train/`, `test/` and the pair lists).
- **Clean pair lists** (optional but recommended): TryOffDiff's script removes duplicates and
  test leaks (11,552 / 1,990 pairs). Pass them with `--pairs-file` to `02`, `01` and `04 tryon`,
  so your numbers are comparable to the TryOffDiff paper. One- or two-column lists both work.
- **Your photos** in `data/real/` as `<id>_<state>.jpg` (see `src/garmentcanon/data/real_photos.py`).
  Then check the ground truth the metrics will use (CPU, seconds):
  ```bash
  python scripts/05_prep_real.py --real data/real        # look at outputs/real_gt_preview.jpg
  ```
  Retake any flat photo whose red outline is wrong.

## 2. Sanity check: reproduce TryOffDiff itself (~2 min, L4 is enough). THE GATE.
Before touching crumpled garments, confirm the original task works: person photo in, garment out.
The papers report different inference settings, so try both:
```python
import gc, torch
from PIL import Image
from garmentcanon.models.tryoffdiff import TryOffDiffRunner
person = Image.open("VITON-HD/test/image/00006_00.jpg")
for sampler, steps, g in [("euler", 20, 1.5), ("euler", 20, 2.0), ("pndm", 50, 2.0)]:
    r = TryOffDiffRunner(sampler=sampler)
    r([person], steps=steps, guidance=g)[0].save(f"sanity_{sampler}{steps}_g{g}.jpg")
    del r; gc.collect(); torch.cuda.empty_cache()      # free the model before loading the next one
```
- Each image should look like that model's shirt laid flat. Pick the better setting and pass it
  everywhere afterwards (`--sampler/--steps/--guidance` on `01` and `04`).
- The log line `[weights] TryOffDiff adapter: ...` says which adapter the weight file contains;
  the loader adapts to it.
- **If loading fails:** list the repo's files and point to the right one:
  ```bash
  python -c "from huggingface_hub import list_repo_files as l; print(l('rizavelioglu/tryoffdiff'))"
  export TRYOFFDIFF_WEIGHTS=<file name from that list>     # or a local path
  ```
- If the images are wrong, stop and debug here, before training anything.

## 3. Build data, then the Tier 1 benchmark
```bash
python scripts/02_make_synthetic.py --viton VITON-HD --split train --source procedural --variants 3 --workers 8
python scripts/02_make_synthetic.py --viton VITON-HD --split test  --source procedural --variants 1
python scripts/01_pilot_zero_shot.py --viton VITON-HD --real data/real --count 200 <sampler args>
```
- Data generation is CPU work (~1.25 s per crumple per core; train set ~12 core-hours): run it on
  a CPU machine, not paid GPU time. It works on macOS/Linux/Windows and is safe to restart.
- Look at `outputs/pilot/pilot_gallery.png` before reading any numbers. Results are split by
  state (`synthetic/crumpled`, `synthetic/folded`, `real/<state>`) and include `.../all`.

## 4. First training run
```bash
python scripts/03_finetune.py configs/finetune_lora.yaml \
    data_roots='[data/synthetic/procedural_train]' out_dir=runs/proc_only max_steps=6000
```
- **Watch the loss.** It should start near 0.1–0.2 (TryOffDiff is already a good denoiser) and
  drift down slowly. Diffusion loss is noisy; judge samples, not single loss values. The full
  curve is in `runs/<name>/loss_history.jsonl`.
- **Sample early.** At step 500, run `04_eval.py canon --ckpt runs/proc_only/step_500.pt --limit 20`
  and look at the images. Catching a bug at step 500 costs minutes; at step 6000, an afternoon.
- **Memory.** batch_size 8 + gradient checkpointing targets an A100-40GB; on an L4 use
  `batch_size=4 grad_accum=2`. (`max_steps` counts micro-batches.)
- **Checkpoints.** The 2 newest plus every `keep_every` (1500) steps are kept.
- **Time.** Not measured yet. Record it/s from the log on your first run and update this line.

## 5. Evaluate
```bash
# canonicalization quality (zero-shot: omit --ckpt)
python scripts/04_eval.py canon --ckpt runs/proc_only/step_6000.pt --test data/synthetic/procedural_test --real data/real
# try-on: baselines (clean/raw) computed once, each model in its own folder
python scripts/04_eval.py tryon --viton VITON-HD --count 300                                  # tag: zeroshot
python scripts/04_eval.py tryon --ckpt runs/proc_only/step_6000.pt --viton VITON-HD --count 300   # tag: proc_only-step_6000
python scripts/04_eval.py score --viton VITON-HD --tag zeroshot
python scripts/04_eval.py score --viton VITON-HD --tag proc_only-step_6000
```
Compare against the zero-shot numbers: that difference is your Tier 2 result. In the try-on
scores, the `mask/...` rows (inside the repainted region) are the sharper comparison.

## Optional: Qwen-Image-Edit (baseline + generative data)
Loads in 4-bit by default (~16–20 GB peak with CPU offload on smaller GPUs). Use an A100 for
`01 --qwen`, since TryOffDiff is loaded at the same time. Time 10 images first. Calibrate the
filter: `02 --source generative --limit 50 --threshold 0`, look at the images and `dino_sim`
values, pick the cutoff, delete that folder, rerun with `--threshold <cutoff> --limit 500`.

## Known limits of the smoke test
- It can't check that the real Hugging Face weights load. `build_components` asserts there are
  no missing or unexpected keys, so a mismatch fails loudly at startup.
- It can't check sample *quality*, CatVTON (needs real weights), LPIPS/DISTS/FID (their
  backbones download weights), or Qwen-Image-Edit.
