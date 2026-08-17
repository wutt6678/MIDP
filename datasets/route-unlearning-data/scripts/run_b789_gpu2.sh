#!/usr/bin/env bash
# Run B7 (MMUnlearner), B8 (MANU), B9 (R²MU-adapted) sequentially on GPU 2.
set -euo pipefail

PYTHON=/scratch/wutiantong/miniconda3/envs/midp/bin/python
RUNNER=/scratch/wutiantong/MIDP/datasets/route-unlearning-data/scripts/run_mllmu_baseline.py
CFGDIR=/scratch/wutiantong/MIDP/datasets/route-unlearning-data/configs/experiments/mllmu_baselines
COMMON_CFG=$CFGDIR/common.yaml
LOGDIR=/scratch/wutiantong/MIDP/datasets/route-unlearning-data/outputs/experiments/mllmu_baselines/logs

mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=2

echo "[$(date -Iseconds)] Starting B7: MMUnlearner on GPU 2"
$PYTHON "$RUNNER" --method mmunlearner --config $CFGDIR/mmunlearner.yaml --common-config $COMMON_CFG 2>&1 | tee "$LOGDIR/b7_mmunlearner.log"
echo "[$(date -Iseconds)] B7 finished"

echo "[$(date -Iseconds)] Starting B8: MANU on GPU 2"
$PYTHON "$RUNNER" --method manu --config $CFGDIR/manu.yaml --common-config $COMMON_CFG 2>&1 | tee "$LOGDIR/b8_manu.log"
echo "[$(date -Iseconds)] B8 finished"

echo "[$(date -Iseconds)] Starting B9: R²MU-adapted on GPU 2"
$PYTHON "$RUNNER" --method r2mu_adapted --config $CFGDIR/r2mu_adapted.yaml --common-config $COMMON_CFG 2>&1 | tee "$LOGDIR/b9_r2mu_adapted.log"
echo "[$(date -Iseconds)] B9 finished"

echo "[$(date -Iseconds)] All B7/B8/B9 runs complete"
