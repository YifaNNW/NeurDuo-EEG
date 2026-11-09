#!/bin/bash -l
# Rebuttal (ReVg Q3): plain causal per-channel Mamba trained from scratch on CHB-MIT.
# Architecture: configs/pretrain/small_pure_mamba.json (fast Mamba stack only: no channel attention,
# slow stream or memory; backbone 2.90M, readout 256), random initialisation, no pretrained weights.
# Everything else follows scripts/finetune/chbmit_small.sh: 15 channels, 20 x 30 s units, state carried
# across units, 5 folds x seed 42, 8 sequences (160 units) per step on one GPU, no gradient accumulation.
#
# Usage: sbatch scripts/finetune/chbmit_small_pure_mamba_scratch.sh
#   run dir: /projects/u6xc/haiping/neurduo/runs/finetune/small_pure_mamba_scratch/chbmit/
#            fold{k}_seed42.json, predictions/, models/fold{k}_seed42.pt
#   log:     logs/finetune/chbmit/   (logs -> /projects/u6xc/haiping/neurduo/logs)
# Aggregate: python scripts/finetune/aggregate.py --dataset chbmit \
#            --output_dir /projects/u6xc/haiping/neurduo/runs/finetune/small_pure_mamba_scratch
#SBATCH --job-name=ft-chbmit-puremamba-scratch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/finetune/chbmit/%x-%A_%a.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
DATA=${DATA:-$REPO/finetune_data/chbmit}
NAME=small_pure_mamba_scratch${TAG:-}
OUT=${OUT:-/projects/u6xc/haiping/neurduo/runs/finetune/$NAME}
FOLD=${SLURM_ARRAY_TASK_ID}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"

echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${FOLD} commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
git -C "$REPO" status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"
python scripts/finetune/run_finetune.py --dataset chbmit --data_dir "$DATA" \
    --no_pretrained --model_config configs/pretrain/small_pure_mamba.json \
    --output_dir "$OUT" --fold "$FOLD" --seed 42 \
    --resume --model_dir "$OUT/chbmit/models" ${EXTRA:-}
echo "done $(date '+%F %T')"
