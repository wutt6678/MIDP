#!/bin/bash
# E2C-v2 budget verification — GPU 0 shard: train D then M_shuffled in parallel
# with M training on GPU 1. Outputs go to the same budget_verify dirs.
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

RESOLVED_CONFIG=e2c_v2/manifests/e2c_budget_verify_resolved.yaml
OUTBASE=e2c_v2/outputs/budget_verify

for COND in D M_shuffled; do
    mkdir -p $OUTBASE/$COND/eval
    echo "=== [GPU0] Training $COND ==="
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_train_route.py \
        --condition $COND --config $RESOLVED_CONFIG \
        --manifest-dir e2c_v2/manifests --dataset-dir e2c_v2/data/experimental \
        --output-dir $OUTBASE/$COND --image-base-dir e2c/data/processed \
        --device cuda:0 --checkpoint-steps 300
    echo "[GPU0] $COND training complete"
done
echo "=== [GPU0] shard complete ==="
