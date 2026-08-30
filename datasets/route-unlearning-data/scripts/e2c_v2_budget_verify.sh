#!/bin/bash
# E2C-v2 budget verification run — re-runs canonical conditions with 900 steps
# (3x the frozen 300) to test whether the canonical failure is training-budget
# underfitting vs. an unfixable pipeline defect. Runs on GPU 1.
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

RESOLVED_CONFIG=e2c_v2/manifests/e2c_budget_verify_resolved.yaml
OUTBASE=e2c_v2/outputs/budget_verify

echo "=== Budget verification: 900 steps, GPU 1 ==="

for COND in M D M_shuffled; do
    mkdir -p $OUTBASE/$COND/eval
    echo "=== Training $COND ==="
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_train_route.py \
        --condition $COND --config $RESOLVED_CONFIG \
        --manifest-dir e2c_v2/manifests --dataset-dir e2c_v2/data/experimental \
        --output-dir $OUTBASE/$COND --image-base-dir e2c/data/processed \
        --device cuda:0 --checkpoint-steps 300
    echo "$COND training complete"
done

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
    --output-dir $OUTBASE/reports \
    --seed 17 --bootstrap-resamples 2000

echo ""
echo "=== BUDGET VERIFY COMPLETE ==="
cat $OUTBASE/reports/e2c_route_validation.json 2>/dev/null || echo "no validation json"
