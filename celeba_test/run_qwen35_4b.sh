#!/usr/bin/env bash
# Run the CelebA attribute classification test for Qwen3.5-9B with the
# isolated `midp_qwen` conda env (see setup_qwen_env.sh).
# Usage:
#   ./run_qwen35_4b.sh                                        # GPU 3, default qwen config
#   ./run_qwen35_4b.sh 0                                      # run on GPU 0
#   ./run_qwen35_4b.sh 0 --limit 50 --attributes Smiling,Male # quick subset run
#   ./run_qwen35_4b.sh 0 --demo                               # sanity-check generation only
#   ./run_qwen35_4b.sh 0 --limit 0                            # full dataset (slow)
set -euo pipefail

# First positional arg (optional): GPU id. Defaults to 3.
GPU_ID=3
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    GPU_ID="$1"
    shift
fi

source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate midp_qwen

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python run_celeba_test.py --config configs/qwen35_4b_celeba.yaml "$@"
