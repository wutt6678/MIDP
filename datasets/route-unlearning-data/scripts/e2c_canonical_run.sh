#!/bin/bash
# Canonical route training: all 3 conditions on GPU 0
# Config: LR=2e-5, 500 steps, seed=17
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

CONFIG=e2c/configs/e2c_calibration_lr2e-5_steps500.yaml
CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo "uncommitted")
OUTBASE=e2c/outputs/${CODE_SHA}

echo "=== E2C Canonical Route Training ==="
echo "Code SHA: ${CODE_SHA}"
echo "Config: ${CONFIG}"

for COND in M D M_shuffled; do
    mkdir -p $OUTBASE/$COND/eval
    echo "=== Training $COND ==="
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_train_route.py \
        --condition $COND --config $CONFIG \
        --manifest-dir e2c/manifests --dataset-dir e2c/data/splits \
        --output-dir $OUTBASE/$COND --image-base-dir e2c/data/processed \
        --device cuda:0 --checkpoint-steps 100
    echo "$COND training complete"
done

echo "=== All training complete. Starting evaluations ==="
for COND in M D M_shuffled; do
    echo "=== Evaluating $COND ==="
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_eval_route.py \
        --condition $COND --adapter-dir $OUTBASE/$COND/adapter_final \
        --config $CONFIG --probe-dir e2c/data/splits \
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
    --manifest-dir e2c/manifests \
    --output-dir e2c/reports \
    --seed 17 --bootstrap-resamples 2000

echo "=== CANONICAL RUN COMPLETE ==="
echo "Code SHA: ${CODE_SHA}"
echo "Results: e2c/reports/"
cat e2c/reports/e2c_route_validation.json 2>/dev/null || echo "no validation json"
