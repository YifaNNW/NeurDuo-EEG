#!/bin/bash -l
# Rebuttal (ReVg Q1/W1): CHB-MIT perturbation test on the fine-tuned NeurDuo-Small models
# (runs/finetune/small/chbmit/models/fold{k}_seed42.pt). Inference only, one fold per array task.
# Test A: replace 4 s / 30 s of background EEG; test B: plant a 30 s seizure of the same patient;
# lags 0-540 s before the last unit (9:30-10:00). See scripts/rebuttal/perturb_chbmit.py.
#
# Usage: sbatch scripts/rebuttal/perturb_chbmit.sh
#   out: /projects/u6xc/haiping/neurduo/runs/rebuttal/perturb/small/chbmit/fold{k}.npz
#   log: logs/rebuttal/   (logs -> /projects/u6xc/haiping/neurduo/logs)
# Smoke test (fold 0, 8 windows):
#   TAG=_smoke EXTRA="--max_windows 8" sbatch --array=0 --time=00:30:00 --job-name=perturb-smoke scripts/rebuttal/perturb_chbmit.sh
# Summary: python scripts/rebuttal/summarize_perturb.py --in_dir <out dir>
#SBATCH --job-name=perturb-chbmit-small
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/rebuttal/%x-%A_%a.log
set -eo pipefail

ENV=/projects/u6xc/haiping/miniforge3/envs/neurduo
REPO=$HOME/code/NeurDuo-EEG
OUT=/projects/u6xc/haiping/neurduo/runs/rebuttal/perturb/small${TAG:-}/chbmit
FOLD=${SLURM_ARRAY_TASK_ID}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"

echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${FOLD} commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
git -C "$REPO" status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"
python scripts/rebuttal/perturb_chbmit.py --fold "$FOLD" --data_dir finetune_data/chbmit \
    --out_dir "$OUT" ${EXTRA:-}
echo "done $(date '+%F %T')"
