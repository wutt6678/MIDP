#!/usr/bin/env bash
# Run the CelebA attribute classification test with LLaVA-1.5-13B in the
# `midp` conda env.
# Usage:
#   ./run_llava.sh                                  # default config, GPU 3
#   ./run_llava.sh 0                                # run on GPU 0
#   ./run_llava.sh 1 --limit 50 --attributes Smiling,Male   # quick subset
#   ./run_llava.sh 0 --set output.name=my_run       # custom output prefix
set -euo pipefail

# First positional arg (optional): GPU id. Defaults to 3.
GPU_ID=3
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    GPU_ID="$1"
    shift
fi

source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate midp

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python run_celeba_test.py --config configs/llava15_13b_celeba.yaml "$@"
