#!/bin/bash -l
# Fine-tune NeurDuo-Base on two-channel CHB-MIT (dataset chbmit_2ch: T7 and P8 of the 15-channel
# data, readout budget 12000; 20 x 30 s = 10 min context, whole-sequence normalisation, state
# carried across units), 5 folds x 1 seed. Same arguments as chbmit_base.sh (paper protocol of
# harness_accumfix/submit_table3_5.sh: seed 42, 20 epochs); only the dataset differs. One step takes
# 8 sequences (160 units) at once, no gradient accumulation.
# Saves the selected model and the validation predictions, which the validation-chosen thresholds
# of the rebuttal need (scripts/rebuttal/false_alarm/).
#
# Usage: sbatch scripts/finetune/chbmit_2ch_base.sh
#   run dir: /projects/u6xc/haiping/neurduo/runs/finetune/base/chbmit_2ch/
#            fold{k}_seed42.json, predictions/, models/fold{k}_seed42.pt
#   log:     logs/finetune/chbmit_2ch/   (logs -> /projects/u6xc/haiping/neurduo/logs)
# Smoke test: EXTRA="--epochs 2 --warmup_epochs 1" TAG=_smoke sbatch --array=0 --time=01:00:00 ...
#SBATCH --job-name=ft-chbmit2ch-base
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/finetune/chbmit_2ch/%x-%A_%a.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
DATA=${DATA:-$REPO/finetune_data/chbmit}
CKPT=${CKPT:-$REPO/ckpt/release/neurduo_eeg_base.pt}
NAME=base${TAG:-}
OUT=${OUT:-/projects/u6xc/haiping/neurduo/runs/finetune/$NAME}
MODEL_DIR=${MODEL_DIR:-$OUT/chbmit_2ch/models}
FOLD=${SLURM_ARRAY_TASK_ID}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"

echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${FOLD} commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
git -C "$REPO" status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"
python scripts/finetune/run_finetune.py --dataset chbmit_2ch --data_dir "$DATA" \
    --checkpoint "$CKPT" --output_dir "$OUT" --fold "$FOLD" --seed 42 \
    --resume --model_dir "$MODEL_DIR" ${EXTRA:-}
echo "done $(date '+%F %T')"
