#!/usr/bin/env bash
# Run the CelebA attribute classification test with the `midp` conda env.
# Usage:
#   ./run.sh                                        # default config (instruct model), GPU 3
#   ./run.sh 0                                      # run on GPU 0
#   ./run.sh 0 --config configs/mllama_llama32_11b_base_celeba.yaml
#   ./run.sh 1 --limit 50 --attributes Smiling,Male # quick subset run
#   ./run.sh --demo                                 # sanity-check generation only
#   ./run.sh --limit 0                              # full dataset (slow)
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

python run_celeba_test.py "$@"
