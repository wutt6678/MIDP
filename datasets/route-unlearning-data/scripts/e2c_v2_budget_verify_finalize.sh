#!/bin/bash
# E2C-v2 budget verification — finalize shard: waits until all 3 condition
# trainings (M on GPU1, D/M_shuffled on GPU0) have written summaries, then
# runs probe evaluation for all conditions and the route analysis.
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

RESOLVED_CONFIG=e2c_v2/manifests/e2c_budget_verify_resolved.yaml
OUTBASE=e2c_v2/outputs/budget_verify

echo "=== Waiting for all trainings to complete ==="
while true; do
    DONE=0
    for COND in M D M_shuffled; do
        [ -f "$OUTBASE/$COND/training_summary.json" ] && DONE=$((DONE+1))
    done
    echo "$(date +%H:%M:%S) trainings complete: $DONE/3"
    [ $DONE -eq 3 ] && break
    sleep 300
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
