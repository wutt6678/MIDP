#!/bin/bash
# E2C-v2 budget verification — eval shard for GPU 3: waits for M_shuffled
# training summary, evaluates M_shuffled, then runs route analysis (CPU).
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

RESOLVED_CONFIG=e2c_v2/manifests/e2c_budget_verify_resolved.yaml
OUTBASE=e2c_v2/outputs/budget_verify

echo "=== [GPU3] Waiting for M_shuffled training summary ==="
while [ ! -f "$OUTBASE/M_shuffled/training_summary.json" ]; do
    echo "$(date +%H:%M:%S) waiting..."
    sleep 120
done
echo "M_shuffled training done."

echo "=== [GPU3] Evaluating M_shuffled ==="
/scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
    python scripts/e2c_eval_route.py \
    --condition M_shuffled --adapter-dir $OUTBASE/M_shuffled/adapter_final \
    --config $RESOLVED_CONFIG --probe-dir e2c_v2/data/experimental/probes \
    --output-dir $OUTBASE/M_shuffled/eval \
    --image-base-dir e2c/data/processed --device cuda:0
echo "[GPU3] M_shuffled eval complete"

echo "=== Waiting for M and D evals (GPU 0 shard) ==="
while [ ! -f "$OUTBASE/M/eval/eval_summary.json" ] || [ ! -f "$OUTBASE/D/eval/eval_summary.json" ]; do
    echo "$(date +%H:%M:%S) waiting for M/D evals..."
    sleep 60
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
