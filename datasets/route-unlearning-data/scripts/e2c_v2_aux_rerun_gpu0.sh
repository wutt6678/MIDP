#!/bin/bash
# E2C-v2 aux-head reruns on GPU 0 (all GPUs free), sequential:
#   1. S1-ID (merger scope, aux head @ layer 1, lam=1.0) -- primary
#   2. S0-ID (language-attn scope, aux head @ layer 1, lam=1.0) -- supporting
#   3. geometry comparison (pre/post) on the S1-ID adapter
set -u
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

OUTBASE=e2c_v2/outputs/diag_aux_idhead
LOG=$OUTBASE/rerun.log
PY=/scratch/wutiantong/miniconda3/bin/conda

run_job() {
    local NAME=$1; shift
    echo "$(date +%H:%M:%S) === $NAME start ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY run -n midp-qwen35 python scripts/e2c_v2_diag_aux_idhead.py "$@" \
        > $OUTBASE/${NAME}.run.log 2>&1
    local RC=$?
    echo "$(date +%H:%M:%S) === $NAME exit rc=$RC ===" | tee -a $LOG
    if [ $RC -ne 0 ]; then
        tail -20 $OUTBASE/${NAME}.run.log | tee -a $LOG
        echo "$(date +%H:%M:%S) $NAME FAILED; continuing to next job" | tee -a $LOG
    fi
}

run_job S1-ID --scope S1 --aux-layer 1 --lam-id 1.0 --device cuda:0
run_job S0-ID --scope S0 --aux-layer 1 --lam-id 1.0 --device cuda:0

echo "$(date +%H:%M:%S) === geometry comparison on S1-ID adapter ===" | tee -a $LOG
S1AD=$OUTBASE/S1-ID_lam1_auxL1/adapter_final
if [ -d "$S1AD" ]; then
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY run -n midp-qwen35 python scripts/e2c_v2_diag_visual_separability.py \
        --adapter-dir $S1AD --label S1-ID \
        --out-dir e2c_v2/outputs/diag_merger_geometry/S1-ID --device cuda:0 \
        > $OUTBASE/geometry_S1ID.run.log 2>&1
    echo "$(date +%H:%M:%S) geometry exit rc=$?" | tee -a $LOG
else
    echo "$(date +%H:%M:%S) S1-ID adapter missing; geometry skipped" | tee -a $LOG
fi
echo "$(date +%H:%M:%S) RERUN CHAIN COMPLETE" | tee -a $LOG
