# Running on Google Colab

Use Colab for interactive work (debugging, looking at outputs); use CHTC for long training.

```python
# Cell 1: GPU check + Drive (outputs and checkpoints survive disconnects there)
!nvidia-smi
from google.colab import drive; drive.mount('/content/drive')
WORK = '/content/drive/MyDrive/garment-canon'

# Cell 2: code
!git clone <your repo url> /content/garment-canon   # or upload the zip and unzip it
%cd /content/garment-canon
!pip install -q -e ".[gpu]"
!git clone https://github.com/Zheng-Chong/CatVTON third_party/CatVTON

# Cell 3: data (keep the VITON-HD archive in Drive; unzip to local disk for speed)
!cp "$WORK/VITON-HD.zip" /content/ && unzip -q /content/VITON-HD.zip -d /content/

# Cell 4: check your real-photo ground truth (CPU, seconds), then the sanity check (LAUNCH.md step 2)
!python scripts/05_prep_real.py --real $WORK/real --out $WORK/outputs/real_gt_preview.jpg

# Cell 5: run things, writing outputs to Drive. SAMPLER = the setting that won the sanity check.
SAMPLER = "--sampler euler --steps 20 --guidance 2.0"
!python scripts/00_crumple_demo.py --viton /content/VITON-HD --count 6 --out $WORK/outputs/crumple_demo.png
!python scripts/01_pilot_zero_shot.py --viton /content/VITON-HD --real $WORK/real --count 200 --out $WORK/outputs/pilot {SAMPLER}
!python scripts/03_finetune.py configs/finetune_lora.yaml out_dir=$WORK/runs/proc_only \
    data_roots='[/content/data/synthetic/procedural_train]' batch_size=4 grad_accum=2
```

GPU choice: **L4** for inference and debugging, **A100** for real training runs and for
`--qwen`. Generate the synthetic data on a CPU machine (or CHTC's `make_synthetic.sub`) and
copy it in, rather than paying GPU time for it (~12 core-hours for the train set).
If the session drops mid-training, rerun the same cell: the trainer resumes from Drive.
