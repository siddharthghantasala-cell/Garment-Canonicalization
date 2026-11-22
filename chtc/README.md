# Running on CHTC (HTCondor)

CHTC runs **batch jobs**: you write a submit file (`.sub`), `condor_submit` it, and HTCondor
runs it on a free machine. Every job starts on a fresh machine, so everything it needs is
shipped in by HTCondor: listed in `transfer_input_files`, never read from `/staging` directly.

| Submit file | What it does | Machine |
| --- | --- | --- |
| `make_synthetic.sub` | procedural pairs for train + test -> tarballs in /staging | 16 CPUs, no GPU |
| `pilot.sub` | Tier 1 zero-shot pilot (`01_pilot_zero_shot.py`) | GPU >= 16 GB |
| `finetune.sub` | LoRA fine-tuning, checkpoints itself, restarts by itself | GPU >= 40 GB |
| `eval.sub` | canonicalization metrics + try-on + scores for one model | GPU >= 24 GB |

## How files move (why the files look the way they do)
- **Large files (> 1 GB) live in /staging** and reach jobs through CHTC's documented
  syntax `transfer_input_files = osdf:///chtc/staging/...`; outputs go back with
  `transfer_output_remaps`. CHTC warns that changed files served this way may not update
  (caching), so **give a file a new name whenever you change its contents** (e.g.
  `synthetic_procedural_train_v2.tar.gz`, then update the `.sub`).
- **Small things (< 1 GB) live in your work folder** on the submit server and move with plain
  HTCondor transfer: the code tarball, `real/` photos, `tryon/` results, LoRA checkpoints.
- **Fine-tuning uses exit-driven checkpointing** (CHTC guide "Checkpointing Jobs"): the trainer
  trains `CHUNK` steps, saves into `run_<RUN>/`, and exits with code 85. HTCondor saves that
  folder and restarts the job; the trainer resumes. It exits 0 at `max_steps`. If the machine
  is lost mid-chunk, the job restarts elsewhere from the last saved chunk.
- LoRA checkpoints are ~0.1 GB each; `keep_every=3000` keeps the run folder under ~0.5 GB.
  These files assume LoRA or adapter mode: a full fine-tune's checkpoints (~10 GB with
  optimizer state) are too big for this route.

## One-time setup
1. Log in to your submit server: `ssh <netid>@<submit-server>.chtc.wisc.edu`.
2. **Find your staging path.** Check CHTC's "Manage large data in /staging" page or run
   `ls -d /staging/*/$USER /staging/$USER 2>/dev/null`. If it prints `/staging/s/sghantasala`,
   then in every `.sub` set `STAGING = osdf:///chtc/staging/s/sghantasala`.
3. **VITON-HD.** The official release already has everything (incl. `agnostic-mask/`). Make
   sure the folder you pack directly holds `train/`, `test/` and the pair lists (the zip unpacks
   to `zalando-hd-resized/`: rename that folder to `VITON-HD`), then:
   ```bash
   tar -czf VITON-HD.tar.gz VITON-HD          # the archive must unpack to VITON-HD/
   scp VITON-HD.tar.gz <netid>@transfer.chtc.wisc.edu:/staging/<your path>/
   ```
4. **Work folder** on the submit server:
   ```bash
   mkdir -p ~/cs566 && cd ~/cs566
   git clone <your repo> garment-canon            # or scp + unzip
   git clone https://github.com/Zheng-Chong/CatVTON garment-canon/third_party/CatVTON
   mkdir -p logs real tryon run_none extra
   cp <your photos> real/                          # <id>_<state>.jpg, see data/real_photos.py
   tar -czf garment-canon.tar.gz --exclude=.git garment-canon
   ```
   **Re-create `garment-canon.tar.gz` after every code change.**

## Order of jobs
```bash
condor_submit garment-canon/chtc/make_synthetic.sub   # ~1 h; 2 tarballs appear in /staging
condor_submit garment-canon/chtc/pilot.sub            # Tier 1 numbers + gallery -> outputs/pilot/
condor_submit garment-canon/chtc/finetune.sub         # -> run_proc_only/ when done
condor_submit garment-canon/chtc/eval.sub             # RUN = none first (zero-shot), then RUN = proc_only
condor_q                                              # watch your jobs
tail -f logs/finetune_proc_only_*.out                 # live training log
```
Do the TryOffDiff sanity check (LAUNCH.md step 2) on Colab before submitting `pilot.sub`:
a broken setup fails the same way on CHTC, just slower to find out.

## GPU options used (from CHTC's GPU guide)
- `request_gpus = 1`, `+WantGPULab = true`
- `+GPUJobLength = "short" | "medium" | "long"`: shorter jobs start sooner
- `gpus_minimum_memory = <MB>`

Verify these option names in CHTC's "Use GPUs" page before the first submit; they occasionally change.

## If something goes wrong
- Job held: `condor_q -hold` shows the reason (often a missing input file or a wrong STAGING path).
- Job log empty: look at `logs/*.err` first.
- Training restarts from the beginning every time: check that `run_<RUN>/` holds `step_*.pt`
  files after the first chunk (the `.log` file shows a checkpoint transfer after exit code 85).
