#!/bin/bash
# E2C-v2 scope ablation sequential runner (GPU 0):
# waits for the currently running S3 job (stride-4 variant, writes to
# diag_scope/S3) to finish, then runs S2 with the same stride-4 scope.
set -u
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

OUTBASE=e2c_v2/outputs/diag_scope
LOG=$OUTBASE/seq_runner.log

echo "$(date +%H:%M:%S) waiting for running S3 job to finish..." | tee -a $LOG
while [ ! -f "$OUTBASE/S3/scope_summary.json" ]; do
    # Also bail out if the S3 process died without a summary
    if ! pgrep -f "diag_scope_ablation.py --scope S3" > /dev/null; then
        echo "$(date +%H:%M:%S) S3 process gone but no summary; S3.log tail:" | tee -a $LOG
        tail -5 $OUTBASE/S3.log | tee -a $LOG
        echo "$(date +%H:%M:%S) continuing to S2 anyway" | tee -a $LOG
        break
    fi
    sleep 300
done
echo "$(date +%H:%M:%S) S3 done, launching S2 (stride-4 scope) on GPU 0" | tee -a $LOG

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
    python scripts/e2c_v2_diag_scope_ablation_stride4.py --scope S2 --device cuda:0 \
    --out-base $OUTBASE \
    > $OUTBASE/S2_stride4.log 2>&1
RC=$?
echo "$(date +%H:%M:%S) S2 exited rc=$RC" | tee -a $LOG
echo "$(date +%H:%M:%S) SEQUENTIAL RUN COMPLETE" | tee -a $LOG
