#!/bin/bash -l
# Rebuttal (ReVg W1): pretrain NeurDuo-EEG-Small without the slow stream.
# Architecture and recipe: configs/pretrain/small_no_slow_stream.json (use_slow_stream = false; the
# memory writer and memory retrieval stay; backbone 3.94M). Same tokenizer, manifests and
# hyperparameters as the paper's Small run; 4 GPUs x 32 sequences = effective batch 128, 16 epochs.
# The same recipe on 4 GPUs took 7.5 h for the full model. Resumes from latest.pt if it exists, so
# the job can simply be submitted again after a time-out.
#
# Usage: sbatch scripts/pretrain/small_no_slow_stream.sh
#   run dir: /projects/u6xc/haiping/neurduo/runs/pretrain/small_no_slow_stream/
#            checkpoints/{best,latest,step_*}.pt
#   log:     logs/pretrain/   (logs -> /projects/u6xc/haiping/neurduo/logs)
# Smoke test (own run dir, stop it by hand after a few hundred steps):
#   TAG=_smoke sbatch --time=00:30:00 scripts/pretrain/small_no_slow_stream.sh
#SBATCH --job-name=pre-small-no-slow-stream
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/pretrain/%x-%j.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
NAME=small_no_slow_stream${TAG:-}
OUT=${OUT:-/projects/u6xc/haiping/neurduo/runs/pretrain/$NAME}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"

echo "node=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
git -C "$REPO" status --short
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

RESUME=()
if [ -f "$OUT/checkpoints/latest.pt" ]; then
    RESUME=(--resume_from "$OUT/checkpoints/latest.pt")
    echo "resuming from $OUT/checkpoints/latest.pt"
fi

export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$REPO"
NPROC_PER_NODE=4 PYTHON=python CONFIG=configs/pretrain/small_no_slow_stream.json \
    bash scripts/pretrain/pretrain.sh --output_dir "$OUT" "${RESUME[@]}" ${EXTRA:-}
echo "done $(date '+%F %T')"
