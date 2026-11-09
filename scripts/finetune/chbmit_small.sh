#!/bin/bash -l
# Fine-tune NeurDuo-Small on CHB-MIT (15 channels, 20 x 30 s = 10 min context, whole-sequence
# normalisation, state carried across units), 5 folds x 1 seed, with the paper protocol of
# harness_accumfix/submit_table3_5.sh (seed 42; paper AUC-PR 0.413 +- 0.174 over folds). The paper split
# each step into 4 accumulated micro-batches for memory; here one GPU takes the same 8 sequences
# (160 units) per step at once (~11 GB), so the effective batch is unchanged.
# Also saves the selected model and the validation
# predictions, which the rebuttal perturbation test and validation-chosen thresholds need.
#
# Usage: sbatch scripts/finetune/chbmit_small.sh
#   run dir: /projects/u6xc/haiping/neurduo/runs/finetune/small/chbmit/
#            fold{k}_seed42.json, predictions/, models/fold{k}_seed42.pt
#   log:     logs/finetune/chbmit/   (logs -> /projects/u6xc/haiping/neurduo/logs)
# Runs stay in the personal folder; copy a finished model to /projects/u6xc/neurduo/ckpt/finetune/
# only once it is final (docs/PATHS.md).
# Smoke test: EXTRA="--epochs 2 --warmup_epochs 1" TAG=_smoke sbatch --array=0 --time=01:00:00 ...
#SBATCH --job-name=ft-chbmit-small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/finetune/chbmit/%x-%A_%a.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
DATA=${DATA:-$REPO/finetune_data/chbmit}
CKPT=${CKPT:-$REPO/ckpt/release/neurduo_eeg_small.pt}
NAME=small${TAG:-}
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
    --resume --model_dir "$MODEL_DIR" ${EXTRA:-}
echo "done $(date '+%F %T')"
