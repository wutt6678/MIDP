#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R12 — Multi-GPU Re-run Missing Methods Only
#
# Runs only missing methods in parallel across GPUs.
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   bash datasets/route-unlearning-data/scripts/r12_rerun_multigpu.sh
# --------------------------------------------------------------------------- #
set -euo pipefail

# -- Load shared environment ----------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/r12r14_env.sh"

# -- Prerequisites --------------------------------------------------------- #
verify_code_sha || exit 1

SMOKE_ROOT="${RUN_ROOT}/r12_smoke"
mkdir -p "${SMOKE_ROOT}"

echo ""
echo "================================================================"
echo "  R12 RE-RUN MISSING — Multi-GPU"
echo "  output root: ${SMOKE_ROOT}"
echo "================================================================"
echo ""

# -- GPU assignment -------------------------------------------------------- #
GPU_LIST=(1 2 3)  # Avoid GPU 0 which has most memory pressure

# -- Check which methods are missing --------------------------------------- #
missing_methods=()
for method in mmunlearner r2mu_adapted; do
    if [ -f "${SMOKE_ROOT}/${method}/eval/eval_results.json" ]; then
        echo "  ${method}: EXISTS (skipping)"
    else
        missing_methods+=("$method")
        echo "  ${method}: MISSING (will run)"
    fi
done

if [ ${#missing_methods[@]} -eq 0 ]; then
    echo ""
    echo "All methods already have results. Nothing to re-run."
    exit 0
fi

echo ""
echo "Will run ${#missing_methods[@]} methods on GPUs: ${GPU_LIST[*]}"
echo ""

# -- Function to run on specific GPU --------------------------------------- #
run_on_gpu() {
    local method="$1"
    local gpu_id="$2"
    local log_file="${SMOKE_ROOT}/${method}.log"
    
    echo "[GPU ${gpu_id}] Starting ${method}..."
    
    CUDA_VISIBLE_DEVICES="$gpu_id" python "${SUITE_DIR}/scripts/run_mllmu_baseline_suite.py" \
        --only "$method" \
        --expected-code-sha "$CODE_SHA" \
        --runtime-output-root "$SMOKE_ROOT" \
        > "$log_file" 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU ${gpu_id}] ✓ ${method} complete"
    else
        echo "[GPU ${gpu_id}] ✗ ${method} FAILED"
    fi
    return $status
}

# -- Run in parallel ------------------------------------------------------- #
pids=()
gpu_idx=0

for method in "${missing_methods[@]}"; do
    gpu=${GPU_LIST[$gpu_idx]}
    run_on_gpu "$method" "$gpu" &
    pids+=($!)
    echo "  Launched PID ${pids[-1]}: ${method} on GPU ${gpu}"
    gpu_idx=$(( (gpu_idx + 1) % ${#GPU_LIST[@]} ))
done

# -- Wait ------------------------------------------------------------------ #
echo ""
echo "Waiting for ${#pids[@]} methods..."

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

echo ""
echo "================================================================"
if [ $failed -eq 0 ]; then
    echo "  R12 RE-RUN — all methods passed ✓"
else
    echo "  R12 RE-RUN — ${failed} method(s) failed ✗"
fi
echo "================================================================"
