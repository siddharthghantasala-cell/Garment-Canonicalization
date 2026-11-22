#!/bin/bash
# Runs on the CHTC execute node inside the PyTorch container.
# Usage (from a submit file):  run_job.sh <script.py | script.sh> [args...]
#
# Before this runs, HTCondor has already copied the job's inputs into this directory
# (see the .sub files): garment-canon.tar.gz (code), data tarballs from /staging
# (VITON-HD.tar.gz, synthetic_*.tar.gz), small folders from the submit directory, and, when
# a fine-tuning job restarts, its checkpoint folder run_<name>/.
#
# Exit code = the script's exit code. 85 means "checkpoint saved, run me again" (fine-tuning);
# HTCondor then restarts this script in the SAME directory, so setup runs only once.
#
# Env vars set by the submit file:
#   EXTRAS  pip extras to install (gpu | dev). Default gpu.
#   PACK    "<name>.tar.gz:<dir>" -> after success, pack <dir> into <name>.tar.gz (data jobs)
set -uo pipefail
cd "${_CONDOR_SCRATCH_DIR:-$PWD}"
export HOME=$PWD                      # pip --user installs into ./.local; the container's HOME may not be writable
export HF_HOME=$PWD/hf_cache PYTHONUNBUFFERED=1

if [ ! -f .setup_done ]; then
  set -e
  tar -xzf garment-canon.tar.gz
  for t in *.tar.gz; do
    [ "$t" = garment-canon.tar.gz ] || { echo "unpacking $t"; tar -xzf "$t"; }
  done
  python -m pip install --quiet --user -e "garment-canon[${EXTRAS:-gpu}]"
  touch .setup_done
  set +e
fi
nvidia-smi 2>/dev/null || echo "(no GPU on this job)"

case "$1" in
  *.py) python "$@" ;;
  *)    bash "$@" ;;
esac
code=$?

if [ $code -eq 0 ] && [ -n "${PACK:-}" ]; then
  name=${PACK%%:*}; dir=${PACK#*:}
  echo "packing $dir -> $name"
  tar -czf "$name" "$dir" || code=1
fi
exit $code
