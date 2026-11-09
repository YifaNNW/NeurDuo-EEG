#!/bin/bash -l
# Rebuttal (ReVg Q3): plain causal per-channel Mamba trained from scratch on Sleep-EDF.
# Architecture: configs/pretrain/small_pure_mamba.json (fast Mamba stack only: no channel attention,
# slow stream or memory; backbone 2.90M, readout 256), random initialisation, no pretrained weights.
# Protocol of the paper's Sleep-EDF NeurDuo row (Tables 3 and 6, carried state; harness_v20 tag
# _ctx20cont_mamba): Fpz and Pz, 20 x 30 s units, state carried across units, Mamba context head,
# per-unit normalisation (the dataset spec's default 'window'), 5 subject folds x seed 42, 20 epochs,
# warm-up 3, model chosen by validation kappa, 8 sequences (160 units) per step on one GPU, no gradient
# accumulation (the paper used none on Sleep-EDF either).
# Normalisation: the paper's Sleep-EDF macro-F1 0.7553 +- 0.0081 is already in the notes on 2026-08-21
# (notes/eeg_fm_status_2026-07-13.md:7681), before harness_v20/submit_sleepedf_seqnorm.sh (2026-09-01)
# proposed whole-sequence normalisation, and the paper still reports that number on 2026-09-14. So the
# paper's Sleep-EDF rows use per-unit normalisation; only CHB-MIT uses whole-sequence normalisation.
#
# Usage: sbatch scripts/finetune/sleepedf_small_pure_mamba_scratch.sh
#   run dir: /projects/u6xc/haiping/neurduo/runs/finetune/small_pure_mamba_scratch/sleepedf/
#            fold{k}_seed42.json, predictions/, models/fold{k}_seed42.pt
#   log:     logs/finetune/sleepedf/   (logs -> /projects/u6xc/haiping/neurduo/logs)
# Aggregate: python scripts/finetune/aggregate.py --dataset sleepedf \
#            --output_dir /projects/u6xc/haiping/neurduo/runs/finetune/small_pure_mamba_scratch
#SBATCH --job-name=ft-sleepedf-puremamba-scratch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=/projects/u6xc/haiping/neurduo/logs/finetune/sleepedf/%x-%A_%a.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
DATA=${DATA:-$REPO/finetune_data/sleepedf}
NAME=small_pure_mamba_scratch${TAG:-}
OUT=${OUT:-/projects/u6xc/haiping/neurduo/runs/finetune/$NAME}
FOLD=${SLURM_ARRAY_TASK_ID}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"

echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${FOLD} commit=$(git -C "$REPO" rev-parse --short HEAD) start=$(date '+%F %T')"
git -C "$REPO" status --short
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"
python scripts/finetune/run_finetune.py --dataset sleepedf --data_dir "$DATA" \
    --no_pretrained --model_config configs/pretrain/small_pure_mamba.json \
    --output_dir "$OUT" --fold "$FOLD" --seed 42 \
    --resume --model_dir "$OUT/sleepedf/models" ${EXTRA:-}
echo "done $(date '+%F %T')"
