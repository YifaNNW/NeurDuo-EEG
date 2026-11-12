#!/bin/bash -l
# Reproduction check: fine-tune NeurDuo-Base on CHB-MIT (15 channels) with the learning-rate curve
# of the paper's runs. Those runs (harness_v20, August 2026) used --grad_accum 4 with a scheduler
# that counted micro-batches, so warm-up lasted until epoch 12 of 20 and the lr was still 9.7e-5 at
# the end. The current code warms up in 3 epochs and decays to zero, and with it Base reaches
# AUC-PR 0.269 (chbmit_base.sh) against 0.374 (seed 42) to 0.392 (seed 1) in the paper-era runs.
# Everything else is identical to chbmit_base.sh; only --sched_stretch 4 is added.
#
# Usage: sbatch scripts/finetune/chbmit_base_sched4.sh
#   run dir: /projects/u6xc/haiping/neurduo/runs/finetune/base_sched4/chbmit/
#   log:     logs/finetune/chbmit/   (logs -> /projects/u6xc/haiping/neurduo/logs)
#SBATCH --job-name=ft-chbmit-base-sched4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/finetune/chbmit/%x-%A_%a.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
DATA=${DATA:-$REPO/finetune_data/chbmit}
CKPT=${CKPT:-$REPO/ckpt/release/neurduo_eeg_base.pt}
NAME=base_sched4${TAG:-}
OUT=${OUT:-/projects/u6xc/haiping/neurduo/runs/finetune/$NAME}
MODEL_DIR=${MODEL_DIR:-$OUT/chbmit/models}
FOLD=${SLURM_ARRAY_TASK_ID}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"

echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${FOLD} commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
git -C "$REPO" status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"
python scripts/finetune/run_finetune.py --dataset chbmit --data_dir "$DATA" \
    --checkpoint "$CKPT" --output_dir "$OUT" --fold "$FOLD" --seed 42 \
    --sched_stretch 4 --resume --model_dir "$MODEL_DIR" ${EXTRA:-}
echo "done $(date '+%F %T')"
