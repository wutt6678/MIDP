#!/bin/bash
# E2C-v3 causal mediation: run each condition as a separate process
# to avoid CUDA deadlocks from multiple model loads in one process.
set -u
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

OUT=e2c_v3/outputs
LOG=$OUT/sequential.log
PY=/scratch/wutiantong/miniconda3/bin/conda
SCRIPT=scripts/e2c_v3_causal_mediation.py

run_cond() {
    local COND=$1
    echo "$(date +%H:%M:%S) === $COND start ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY run -n midp-qwen35 python -u $SCRIPT \
        --device cuda:0 --out-base $OUT --condition "$COND" \
        > $OUT/${COND}.log 2>&1
    local RC=$?
    echo "$(date +%H:%M:%S) === $COND exit rc=$RC ===" | tee -a $LOG
    if [ $RC -ne 0 ]; then
        tail -20 $OUT/${COND}.log | tee -a $LOG
        echo "$(date +%H:%M:%S) $COND FAILED; stopping" | tee -a $LOG
        exit 1
    fi
}

mkdir -p $OUT

run_cond "M-latent"
run_cond "D"
run_cond "M-latent-shuffled"

# Merge results from all conditions into summary
python3 -c "
import json, sys
results = {}
for name, path in [('M-latent', '$OUT/M_latent'),
                    ('D', '$OUT/D'),
                    ('M-latent-shuffled', '$OUT/M_latent_shuffled')]:
    # Look for results in the condition directory
    import glob
    traces = glob.glob(path + '/training_trace.jsonl')
    if traces:
        trace = [json.loads(l) for l in open(traces[0])]
        results[name] = {'final_loss': trace[-1]['loss'] if trace else None}
print(json.dumps(results, indent=2))
" | tee $OUT/summary_check.json

echo "$(date +%H:%M:%S) ALL CONDITIONS COMPLETE" | tee -a $LOG
