#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R12 — Multi-GPU Smoke Validation
#
# Runs all 5 methods in parallel across multiple GPUs.
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   bash datasets/route-unlearning-data/scripts/r12_smoke_multigpu.sh
# --------------------------------------------------------------------------- #
set -euo pipefail

# -- Load shared environment ----------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/r12r14_env.sh"

# -- Prerequisites --------------------------------------------------------- #
verify_code_sha || exit 1

SMOKE_ROOT="${RUN_ROOT}/r12_smoke"
mkdir -p "${SMOKE_ROOT}"

# -- Preflight ------------------------------------------------------------- #
run_preflight "${SMOKE_ROOT}/preflight" || exit 1
check_preflight_hashes "${SMOKE_ROOT}/preflight/suite_preflight_report.json" || exit 1

echo ""
echo "================================================================"
echo "  R12 SMOKE — Multi-GPU Execution"
echo "  output root: ${SMOKE_ROOT}"
echo "================================================================"
echo ""

# -- GPU assignment -------------------------------------------------------- #
# Available GPUs: 0,1,2,3 — assign methods to different GPUs
GPU_LIST=(0 1 2 3)
METHODS=(prompting ga mmunlearner manu r2mu_adapted)

# Function to run a single method on a specific GPU
run_on_gpu() {
    local method="$1"
    local gpu_id="$2"
    local output_root="$3"
    local log_file="${output_root}/${method}.log"
    
    echo "[GPU ${gpu_id}] Starting ${method}..."
    
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    cd "$SUITE_DIR"
    
    python scripts/run_mllmu_baseline_suite.py \
        --only "$method" \
        --expected-code-sha "$CODE_SHA" \
        --runtime-output-root "$output_root" \
        > "$log_file" 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU ${gpu_id}] ✓ ${method} complete"
    else
        echo "[GPU ${gpu_id}] ✗ ${method} FAILED (exit code: $status)"
    fi
    return $status
}

# -- Run methods in parallel ----------------------------------------------- #
pids=()
gpu_idx=0

for method in "${METHODS[@]}"; do
    gpu=${GPU_LIST[$gpu_idx]}
    
    # Check if already done
    if [ -f "${SMOKE_ROOT}/${method}/eval/eval_results.json" ] || \
       [ -f "${SMOKE_ROOT}/${method}/eval/prune_05/eval_results.json" ]; then
        echo "[GPU ${gpu}] ${method} already exists, skipping"
    else
        run_on_gpu "$method" "$gpu" "$SMOKE_ROOT" &
        pids+=($!)
        echo "  Launched PID ${pids[-1]}: ${method} on GPU ${gpu}"
    fi
    
    gpu_idx=$(( (gpu_idx + 1) % ${#GPU_LIST[@]} ))
done

# -- Wait for all background jobs ------------------------------------------ #
echo ""
echo "Waiting for ${#pids[@]} methods to complete..."
echo ""

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=$((failed + 1))
    fi
done

echo ""
echo "================================================================"
if [ $failed -eq 0 ]; then
    echo "  R12 SMOKE — all methods finished successfully"
else
    echo "  R12 SMOKE — ${failed} method(s) failed"
fi
echo "================================================================"
echo ""
echo "Check results in: ${SMOKE_ROOT}"
echo "Logs: ${SMOKE_ROOT}/*.log"
