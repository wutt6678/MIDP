#!/bin/bash
# E2C-v2 budget verification — eval shard for GPU 0: evaluates M then D.
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

RESOLVED_CONFIG=e2c_v2/manifests/e2c_budget_verify_resolved.yaml
OUTBASE=e2c_v2/outputs/budget_verify

for COND in M D; do
    echo "=== [GPU0] Evaluating $COND ==="
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_eval_route.py \
        --condition $COND --adapter-dir $OUTBASE/$COND/adapter_final \
        --config $RESOLVED_CONFIG --probe-dir e2c_v2/data/experimental/probes \
        --output-dir $OUTBASE/$COND/eval \
        --image-base-dir e2c/data/processed --device cuda:0
    echo "[GPU0] $COND eval complete"
done
echo "=== [GPU0] eval shard complete ==="
