#!/bin/bash -l
# Fine-tune NeurDuo-Large on CHB-MIT (15 channels, 20 x 30 s = 10 min context, whole-sequence
# normalisation, state carried across units), 5 folds x 1 seed. Same arguments as chbmit_small.sh,
# which follow the paper protocol of harness_accumfix/submit_table3_5.sh (seed 42, 20 epochs); only
# the checkpoint differs. One step takes 8 sequences (160 units) at once; if that does not fit the
# GPU, pass EXTRA="--grad_accum 2" (same effective batch, as in the paper).
# Saves the selected model and the validation predictions, which the validation-chosen thresholds
# of the rebuttal need (scripts/rebuttal/false_alarm/).
#
# Usage: sbatch scripts/finetune/chbmit_large.sh
#   run dir: /projects/u6xc/haiping/neurduo/runs/finetune/large/chbmit/
#            fold{k}_seed42.json, predictions/, models/fold{k}_seed42.pt
#   log:     logs/finetune/chbmit/   (logs -> /projects/u6xc/haiping/neurduo/logs)
# Smoke test: EXTRA="--epochs 2 --warmup_epochs 1" TAG=_smoke sbatch --array=0 --time=02:00:00 ...
#SBATCH --job-name=ft-chbmit-large
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/finetune/chbmit/%x-%A_%a.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
DATA=${DATA:-$REPO/finetune_data/chbmit}
CKPT=${CKPT:-$REPO/ckpt/release/neurduo_eeg_large.pt}
NAME=large${TAG:-}
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
