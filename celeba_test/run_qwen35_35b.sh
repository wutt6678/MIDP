#!/usr/bin/env bash
# Run the CelebA attribute classification test for Qwen3.5-35B-A3B with the
# isolated `midp_qwen` conda env (see setup_qwen_env.sh).
#
# The model is a 35B MoE (~70 GB in bf16) sharded via device_map=auto across
# the GPUs listed in the first argument (comma-separated). Defaults to 0,1.
#
# Usage:
#   ./run_qwen35_35b.sh                                  # GPUs 0,1, default config
#   ./run_qwen35_35b.sh 1,3                              # shard across GPUs 1 and 3
#   ./run_qwen35_35b.sh 0,1 --limit 50 --attributes Smiling,Male   # quick subset
#   ./run_qwen35_35b.sh 0,1 --demo                       # sanity-check generation
#   ./run_qwen35_35b.sh 0,1 --limit 0                    # full dataset (very slow)
set -euo pipefail

# First positional arg (optional): comma-separated GPU ids. Defaults to 0,1.
GPU_IDS="0,1"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    GPU_IDS="$1"
    shift
fi

source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate midp_qwen

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

python run_celeba_test.py --config configs/qwen35_35b_a3b_celeba.yaml "$@"
