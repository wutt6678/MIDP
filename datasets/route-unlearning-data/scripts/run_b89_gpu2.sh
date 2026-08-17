#!/usr/bin/env bash
# Run B8 (MANU) and B9 (R²MU-adapted) sequentially on GPU 2.
# B7 (MMUnlearner) already completed successfully.
set -euo pipefail

PYTHON=/scratch/wutiantong/miniconda3/envs/midp/bin/python
RUNNER=/scratch/wutiantong/MIDP/datasets/route-unlearning-data/scripts/run_mllmu_baseline.py
CFGDIR=/scratch/wutiantong/MIDP/datasets/route-unlearning-data/configs/experiments/mllmu_baselines
COMMON_CFG=$CFGDIR/common.yaml
LOGDIR=/scratch/wutiantong/MIDP/datasets/route-unlearning-data/outputs/experiments/mllmu_baselines/logs

mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=2

echo "[$(date -Iseconds)] Starting B8: MANU on GPU 2"
$PYTHON "$RUNNER" --method manu --config $CFGDIR/manu.yaml --common-config $COMMON_CFG 2>&1 | tee "$LOGDIR/b8_manu.log"
echo "[$(date -Iseconds)] B8 finished"

echo "[$(date -Iseconds)] Starting B9: R²MU-adapted on GPU 2"
$PYTHON "$RUNNER" --method r2mu_adapted --config $CFGDIR/r2mu_adapted.yaml --common-config $COMMON_CFG 2>&1 | tee "$LOGDIR/b9_r2mu_adapted.log"
echo "[$(date -Iseconds)] B9 finished"

echo "[$(date -Iseconds)] All B8/B9 runs complete"
