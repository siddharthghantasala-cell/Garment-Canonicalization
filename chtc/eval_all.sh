#!/bin/bash
# Full evaluation of one model, run from the job's top folder (called by run_job.sh).
# Usage: eval_all.sh <RUN|none> <STEP> [sampler args...]
set -euo pipefail
RUN=$1; STEP=$2; shift 2
if [ "$RUN" = none ]; then
  CKPT_ARG=(); TAG=zeroshot
else
  CKPT_ARG=(--ckpt "run_$RUN/step_$STEP.pt"); TAG="$RUN-step_$STEP"
fi
S=garment-canon/scripts/04_eval.py
REAL_ARG=(); [ -n "$(ls -A real 2>/dev/null)" ] && REAL_ARG=(--real real)

python $S canon "${CKPT_ARG[@]}" --test data/synthetic/procedural_test "${REAL_ARG[@]}" \
       --out "outputs/eval_$TAG" "$@"
python $S tryon "${CKPT_ARG[@]}" --tag "$TAG" --viton VITON-HD --count 300 --results tryon "$@"
python $S score --tag "$TAG" --viton VITON-HD --results tryon
