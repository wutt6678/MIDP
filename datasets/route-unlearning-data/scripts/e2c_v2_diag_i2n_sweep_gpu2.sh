#!/bin/bash
# E2C-v2 diagnostic sweep on GPU 2: sequential N = 2,4,6,8,10 (M1-only I2N).
set -u
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

OUTBASE=e2c_v2/outputs/diag_i2n_capacity
mkdir -p $OUTBASE

for N in 2 4 6 8 10; do
    echo "$(date +%H:%M:%S) === launching N=$N on GPU 2 ===" | tee -a $OUTBASE/sweep_gpu2.log
    CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        /scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
        python scripts/e2c_v2_diag_i2n_capacity.py --n $N --device cuda:0 \
        > $OUTBASE/n${N}.log 2>&1
    RC=$?
    echo "$(date +%H:%M:%S) N=$N exited rc=$RC" | tee -a $OUTBASE/sweep_gpu2.log
    if [ $RC -ne 0 ]; then
        echo "N=$N FAILED; stopping sweep" | tee -a $OUTBASE/sweep_gpu2.log
        exit 1
    fi
done

echo "$(date +%H:%M:%S) all jobs done, aggregating" | tee -a $OUTBASE/sweep_gpu2.log
/scratch/wutiantong/miniconda3/bin/conda run -n midp-qwen35 \
    python scripts/e2c_v2_diag_i2n_aggregate.py --out-base $OUTBASE \
    | tee -a $OUTBASE/sweep_gpu2.log
echo "$(date +%H:%M:%S) SWEEP COMPLETE" | tee -a $OUTBASE/sweep_gpu2.log
