"""Fine-tune TryOffDiff on synthetic pairs.

    python scripts/03_finetune.py configs/finetune_lora.yaml
    python scripts/03_finetune.py configs/finetune_lora.yaml data_roots='[data/synthetic/procedural_train]' out_dir=runs/proc_only

Chunked runs (CHTC exit-driven checkpointing): with max_steps_this_run=N pause_exit_code=85,
each run trains N steps, saves a checkpoint, and exits with code 85 ("checkpoint saved, run me
again") until max_steps is reached, then exits 0. HTCondor restarts it automatically.
"""
import sys

from garmentcanon.train.finetune import train
from garmentcanon.utils.common import load_config

cfg = load_config(sys.argv[1], sys.argv[2:])
print(cfg, flush=True)
result = train(cfg)
if result["step"] < cfg["max_steps"] and cfg.get("pause_exit_code"):
    print(f"[checkpoint] step {result['step']}/{cfg['max_steps']}: exiting with {cfg['pause_exit_code']}", flush=True)
    sys.exit(int(cfg["pause_exit_code"]))
print(f"[done] step {result['step']}/{cfg['max_steps']}", flush=True)
