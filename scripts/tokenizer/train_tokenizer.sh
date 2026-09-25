#!/usr/bin/env bash
# Launch spectral tokenizer training with torch.distributed.run on one node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

exec "${PYTHON:-python}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-1}" \
    "${SCRIPT_DIR}/run_train_tokenizer.py" \
    --config "${CONFIG:-${PROJECT_ROOT}/configs/tokenizer.json}" \
    "$@"
