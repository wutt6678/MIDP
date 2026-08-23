#!/bin/bash
# E2C-v2 canonical run — requires successful preflight + calibration decision
# before proceeding with canonical M/D/M-shuffled training.
set -e
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

PREFLIGHT_REPORT=e2c_v2/reports/e2c_v2_preflight_report.json
CALIBRATION_DECISION=e2c_v2/manifests/e2c_calibration_decision.json

echo "=== E2C-v2 Canonical Run ==="
echo "Checking prerequisites..."

# Gate 1: Preflight must pass
if [ ! -f "$PREFLIGHT_REPORT" ]; then
    echo "FATAL: Preflight report not found: $PREFLIGHT_REPORT"
    echo "Run: python scripts/e2c_build_dataset_v2.py --mode preflight"
    exit 1
fi

PREFLIGHT_PASS=$(python3 -c "import json; print(json.load(open('$PREFLIGHT_REPORT'))['overall_pass'])")
if [ "$PREFLIGHT_PASS" != "True" ]; then
    echo "FATAL: Preflight did not pass. Cannot proceed."
    echo "Review: $PREFLIGHT_REPORT"
    exit 1
fi
echo "  Preflight: PASS"

# Gate 2: Calibration decision must be FREEZE_CANONICAL_CONFIG
if [ ! -f "$CALIBRATION_DECISION" ]; then
    echo "FATAL: Calibration decision not found: $CALIBRATION_DECISION"
    echo "Complete GPU calibration first, then write the decision artifact."
    exit 1
fi

DECISION=$(python3 -c "import json; print(json.load(open('$CALIBRATION_DECISION'))['decision'])")
if [ "$DECISION" != "FREEZE_CANONICAL_CONFIG" ]; then
    echo "FATAL: Calibration decision is '$DECISION', not FREEZE_CANONICAL_CONFIG"
    echo "Cannot proceed with canonical training."
    exit 1
fi
echo "  Calibration: FREEZE_CANONICAL_CONFIG"

# Gate 3: Resolved canonical config must exist
RESOLVED_CONFIG=e2c_v2/manifests/e2c_canonical_resolved.yaml
if [ ! -f "$RESOLVED_CONFIG" ]; then
    echo "FATAL: Resolved canonical config not found: $RESOLVED_CONFIG"
    echo "Write the frozen config from calibration before training."
    exit 1
fi
CONFIG_SHA=$(sha256sum "$RESOLVED_CONFIG" | cut -d' ' -f1)
echo "  Resolved config SHA: $CONFIG_SHA"

# Record code SHA
CODE_SHA=$(git rev-parse HEAD 2>/dev/null || echo "uncommitted")
OUTBASE=e2c_v2/outputs/${CODE_SHA}

echo ""
echo "Code SHA: $CODE_SHA"
echo "Config: $RESOLVED_CONFIG ($CONFIG_SHA)"
echo ""

# Train all 3 conditions
for COND in M D M_shuffled; do
    mkdir -p $OUTBASE/$COND/eval
    echo "=== Training $COND ==="
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_train_route.py \
        --condition $COND --config $RESOLVED_CONFIG \
        --manifest-dir e2c_v2/manifests --dataset-dir e2c_v2/data/experimental \
        --output-dir $OUTBASE/$COND --image-base-dir e2c/data/processed \
        --device cuda:0 --checkpoint-steps 100
    echo "$COND training complete"
done

echo "=== All training complete. Evaluating ==="
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
echo "Config SHA: $CONFIG_SHA"
echo "Results: e2c_v2/reports/"
cat e2c_v2/reports/e2c_route_validation.json 2>/dev/null || echo "no validation json"
