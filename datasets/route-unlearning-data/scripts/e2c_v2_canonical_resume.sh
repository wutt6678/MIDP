#!/bin/bash
# E2C-v2 canonical run — resume: M_shuffled training, evals, analysis.
# M and D training already complete.
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

RESOLVED_CONFIG=e2c_v2/manifests/e2c_canonical_resolved.yaml
CODE_SHA=e0d2c52ec260ea0cb76985ae72e9dcb01e9fe8de
OUTBASE=e2c_v2/outputs/${CODE_SHA}

echo "=== Training M_shuffled ==="
/scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
    python scripts/e2c_train_route.py \
    --condition M_shuffled --config $RESOLVED_CONFIG \
    --manifest-dir e2c_v2/manifests --dataset-dir e2c_v2/data/experimental \
    --output-dir $OUTBASE/M_shuffled --image-base-dir e2c/data/processed \
    --device cuda:0 --checkpoint-steps 100
echo "M_shuffled training complete"

echo "=== Evaluating all conditions ==="
for COND in M D M_shuffled; do
    echo "=== Evaluating $COND ==="
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_eval_route.py \
        --condition $COND --adapter-dir $OUTBASE/$COND/adapter_final \
        --config $RESOLVED_CONFIG --probe-dir e2c_v2/data/experimental/probes \
        --output-dir $OUTBASE/$COND/eval \
        --image-base-dir e2c/data/processed --device cuda:0
    echo "$COND eval complete"
done

echo "=== Running route analysis ==="
/scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
    python scripts/e2c_analyze_routes.py \
    --eval-dir-m $OUTBASE/M/eval \
    --eval-dir-d $OUTBASE/D/eval \
    --eval-dir-ms $OUTBASE/M_shuffled/eval \
    --manifest-dir e2c_v2/manifests \
    --output-dir e2c_v2/reports \
    --seed 17 --bootstrap-resamples 2000

echo ""
echo "=== CANONICAL RUN COMPLETE ==="
echo "Code SHA: $CODE_SHA"
echo "Results: e2c_v2/reports/"
cat e2c_v2/reports/e2c_route_validation.json 2>/dev/null || echo "no validation json"
