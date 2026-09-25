#!/usr/bin/env bash
# Launch backbone pretraining with torch.distributed.run on one or more nodes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NNODES="${NNODES:-1}"
if [[ "${NNODES}" -gt 1 ]]; then
    LAUNCH=(--nnodes="${NNODES}" --rdzv_backend=c10d
            --rdzv_endpoint="${MASTER_ADDR:?set MASTER_ADDR for multi-node runs}:${MASTER_PORT:-29500}"
            --rdzv_id="${RDZV_ID:-neurduo}")
else
    LAUNCH=(--standalone --nnodes=1)
fi

exec "${PYTHON:-python}" -m torch.distributed.run \
    "${LAUNCH[@]}" --nproc_per_node="${NPROC_PER_NODE:-1}" \
    "${SCRIPT_DIR}/run_pretrain.py" \
    --config "${CONFIG:-${PROJECT_ROOT}/configs/pretrain/small.json}" \
    "$@"
