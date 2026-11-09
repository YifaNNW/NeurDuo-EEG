#!/bin/bash -l
# Smoke-test the neurduo conda env on a GPU node: CUDA, mamba-ssm kernels,
# causal-conv1d, and a forward/backward pass through the NeurDuo backbone.
# Usage: sbatch scripts/env/check_env_gpu.sbatch   (log goes to /projects/u6xc/haiping/logs/)
#SBATCH --job-name=check-env
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=/projects/u6xc/haiping/logs/%x-%j.log
set -eo pipefail

ENV=${ENV:-/projects/u6xc/haiping/miniforge3/envs/neurduo}
CKPT=${CKPT:-/projects/u6xc/neurduo/ckpt/release/neurduo_eeg_small.pt}
REPO=${REPO:-$HOME/code/NeurDuo-EEG}
source /projects/u6xc/haiping/miniforge3/bin/activate "$ENV"

echo "node=$(hostname) env=$ENV python=$(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO"
python - "$CKPT" <<'EOF'
import sys, torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
assert torch.cuda.is_available()

import causal_conv1d, mamba_ssm
from mamba_ssm import Mamba
print('mamba_ssm', mamba_ssm.__version__, 'causal_conv1d', causal_conv1d.__version__)
m = Mamba(d_model=64).cuda()
x = torch.randn(2, 128, 64, device='cuda', requires_grad=True)
m(x).sum().backward()
print('Mamba fwd/bwd ok, grad norm %.4f' % x.grad.norm().item())

import eeg_fm
print('eeg_fm imported from', eeg_fm.__file__)
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print('checkpoint keys', list(payload))
print('ENV OK')
EOF
