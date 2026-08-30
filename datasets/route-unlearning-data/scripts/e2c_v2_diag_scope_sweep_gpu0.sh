#!/bin/bash
# E2C-v2 scope ablation on GPU 0: S1 -> S2 -> S3 sequentially (M1-only, 1000 steps).
set -u
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

OUTBASE=e2c_v2/outputs/diag_scope
mkdir -p $OUTBASE

for S in S1 S2 S3; do
    echo "$(date +%H:%M:%S) === launching $S on GPU 0 ===" | tee -a $OUTBASE/sweep.log
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_v2_diag_scope_ablation.py --scope $S --device cuda:0 \
        > $OUTBASE/${S}.log 2>&1
    RC=$?
    echo "$(date +%H:%M:%S) $S exited rc=$RC" | tee -a $OUTBASE/sweep.log
    if [ $RC -ne 0 ]; then
        echo "$S FAILED; stopping ablation" | tee -a $OUTBASE/sweep.log
        exit 1
    fi
done
echo "$(date +%H:%M:%S) SCOPE ABLATION COMPLETE" | tee -a $OUTBASE/sweep.log
