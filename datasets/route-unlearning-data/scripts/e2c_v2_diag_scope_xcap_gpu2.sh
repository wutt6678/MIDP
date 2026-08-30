#!/bin/bash
# E2C-v2 scope x capacity experiment (GPU 2), M1-only, N=10, 1000 steps:
#   S1_r8_fullscope   merger only, rank 8           (cross-modal access)
#   S0_R16_fullscope  language attention, rank 16   (capacity control)
#   S2_r8_fullscope   merger + language attn, rank 8 (access + readout)
# S0 rank-8 baseline already exists: diag_i2n_capacity/n10.
set -u
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

OUTBASE=e2c_v2/outputs/diag_scope_xcap
mkdir -p $OUTBASE

run() {
    local SCOPE=$1 RANK=$2 TAG=$3
    echo "$(date +%H:%M:%S) === launching $SCOPE rank=$RANK ($TAG) on GPU 2 ===" | tee -a $OUTBASE/sweep.log
    CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_v2_diag_scope_ablation.py \
        --scope $SCOPE --rank $RANK --run-tag $TAG \
        --out-base $OUTBASE --device cuda:0 \
        > $OUTBASE/${SCOPE}${TAG}.log 2>&1
    RC=$?
    echo "$(date +%H:%M:%S) $SCOPE$TAG exited rc=$RC" | tee -a $OUTBASE/sweep.log
    if [ $RC -ne 0 ]; then
        echo "$SCOPE$TAG FAILED; stopping" | tee -a $OUTBASE/sweep.log
        exit 1
    fi
}

run S1 8 _r8_fullscope
run S0 16 _R16_fullscope
run S2 8 _r8_fullscope

echo "$(date +%H:%M:%S) SCOPE X CAPACITY COMPLETE" | tee -a $OUTBASE/sweep.log
