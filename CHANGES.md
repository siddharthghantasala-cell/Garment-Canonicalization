# Changes from the Nov 17, 2026 review

Each entry: what was wrong → what happens now. "Tested" means it ran here (CPU, no torch);
"reviewed" means it needs torch/GPU and was checked by reading.

## Would have crashed or given wrong numbers
1. **Real-photo ground truth didn't match the model's framing.** A phone photo has a floor
   behind the garment and arbitrary framing; outputs are catalog-framed on white. → New
   `segment_flat_photo` (smooth background model + noise-based threshold, robust to uneven
   lighting and striped garments) and `frame_garment` (cut out, scale, center on white).
   Predictions are re-framed by the same rule. Tested: overlap with the true garment 0.98–0.99
   on simulated photos. Check yours with `scripts/05_prep_real.py`.
2. **01 and 04 scored real photos differently** (pad-then-resize vs squash). → One
   `eval_transform()` used everywhere. Reviewed.
3. **Try-on eval re-scored the old model.** It skipped a garment if `ours/<key>.jpg` existed,
   so a second checkpoint with the default folder silently reused the first one's results; and
   it recomputed CatVTON for clean/raw every time (3× cost). → Baselines computed once;
   each model gets `ours_<tag>/`; scoring only uses keys present in all three folders. Reviewed.
4. **Qwen-Image-Edit couldn't fit on an L4 or A100-40GB** (full bf16 ≈ 54 GB despite the
   "4-bit" comment). → 4-bit NF4 via diffusers' `PipelineQuantizationConfig` + CPU offload on
   GPUs < 30 GB; official `true_cfg_scale`/`negative_prompt` usage. Reviewed.
5. **Data generation crashed on macOS** (no `__main__` guard; macOS starts workers with
   "spawn"). → Restructured with a worker initializer. Tested under spawn: original crashed with
   0 rows, fixed version wrote all rows.
6. **CHTC files couldn't work:** copied from /staging inside the job (CHTC documents
   `osdf:///` transfer), never shipped the synthetic data to the training job, lost progress on
   eviction. → Rewritten: osdf inputs/outputs, CPU data-generation job, exit-driven
   checkpointing (exit 85) for training, eval job. Reviewed (not submitted).
7. **Weight-file assumptions (defensive, not a confirmed bug):** the adapter's layer names were
   matched to the authors' GitHub code in an earlier session, which includes a transformer
   layer; the BMVC paper's text describes only Linear+LayerNorm. The repo code is the better
   evidence for the released file, so the original design is probably right. The loader now
   also detects the adapter from the file's keys, accepts common save formats (`module.`
   prefixes, wrapped dicts, whole modules), falls back to SD's scheduler config, and takes
   `TRYOFFDIFF_WEIGHTS` to override the file name. It changes nothing if the original
   assumption holds. Reviewed + unit tests (skip without torch).

## Smaller fixes
8. Trainer lost up to `save_every - 1` steps when pausing for `max_steps_this_run` → saves on pause. Reviewed.
9. AMP GradScaler state wasn't checkpointed → saved and restored. Checkpoint writes are atomic. Reviewed.
10. Loss history was overwritten on each resume → appended to `loss_history.jsonl`. Reviewed.
11. Only the 2 newest checkpoints kept → plus every `keep_every` (default 1500). Reviewed.
12. Procedural samples were all labeled "crumpled" → `state` = folded/crumpled from what was
    sampled; per-state results in 01 and 04. Tested.
13. Hidden-region scores ignored white parts of the garment (mask came from "not near-white")
    → real garment masks saved with the data and used. Tested (data side).
14. `mask_from_white_background` kept only the largest region, so white stripes cut garments
    apart → keeps every sizable region and fills enclosed holes. Tested.
15. A half-written manifest line (run killed mid-write) broke restart and loading → skipped
    with a warning. Rejected generative samples are recorded so reruns don't regenerate them. Tested.
16. `pytest -q` errored on a CPU-only install (torch imported before the skip) → skips. Tested.
17. `.heic` photos couldn't be opened → `pillow-heif` added and registered. Reviewed.
18. Eval used VITON-HD's full test list → `--pairs-file` on 01/04 for TryOffDiff's cleaned list;
    1- and 2-column lists accepted. Tested.
19. Try-on scores diluted by the unchanged part of the image → added scores inside the
    agnostic mask (`mask/L1`, `mask/LPIPS`) and KID. Reviewed.
20. Pilot and try-on crumpled the same garment differently (seed = list position) → seed =
    hash of the garment ID everywhere. Reviewed.
21. Inference settings: papers report PNDM/50/2.0 (v1) and Euler/20/1.5 (follow-up) →
    `--sampler/--steps/--guidance` on 01 and 04; LAUNCH step 2 tries both.

## Still unverified (needs a GPU)
- Real TryOffDiff weights load and produce good garments (LAUNCH step 2).
- CatVTON runs with this environment's library versions.
- Qwen-Image-Edit 4-bit memory and speed; CHTC submit files on the real cluster.
