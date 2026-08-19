#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R13 — Re-run Missing Methods (GPU-aware)
#
# kl: ~20 GB → GPU 1
# npo: ~20 GB → GPU 2
# mmunlearner: ~37 GB → GPU 1 (needs most memory)
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   bash datasets/route-unlearning-data/scripts/r13_rerun_multigpu.sh
# --------------------------------------------------------------------------- #
set -euo pipefail

# -- Load shared environment ----------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/r12r14_env.sh"

# -- Prerequisites --------------------------------------------------------- #
verify_code_sha || exit 1

FULL_ROOT="${RUN_ROOT}/r13_full"
mkdir -p "${FULL_ROOT}"

echo ""
echo "================================================================"
echo "  R13 RE-RUN MISSING — GPU-aware"
echo "  output root: ${FULL_ROOT}"
echo "================================================================"
echo ""

# -- Check which methods are missing --------------------------------------- #
missing_methods=()
for method in kl manu; do
    if [ -f "${FULL_ROOT}/${method}/eval/eval_results.json" ]; then
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

# -- Run all on GPU 3 ------------------------------------------------------ #
GPU_ID=3

# Oracle adapter path for NPO (from previous npo_oracle run)
ORACLE_ADAPTER="/scratch/wutiantong/MIDP/datasets/route-unlearning-data/runtime_outputs/mllmu_baselines/0e2b8083317694936ce8af8dc6857d9160d39f2f/r13_full/npo_oracle/checkpoints/adapter_final"

failed=0
for method in "${missing_methods[@]}"; do
    log_file="${FULL_ROOT}/${method}.log"
    
    echo "=== Running ${method} on GPU ${GPU_ID} ==="
    
    # Build command with optional oracle adapter path
    cmd="CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run_mllmu_baseline_suite.py"
    cmd+=" --only $method"
    cmd+=" --expected-code-sha $CODE_SHA"
    cmd+=" --runtime-output-root $FULL_ROOT"
    
    # Add oracle adapter path for NPO
    if [ "$method" = "npo" ] && [ -d "$ORACLE_ADAPTER" ]; then
        cmd+=" --oracle-adapter-path $ORACLE_ADAPTER"
        echo "  Using oracle adapter: $ORACLE_ADAPTER"
    fi
    
    (
        cd "$SUITE_DIR" || exit 1
        eval "$cmd" 2>&1 | tee "$log_file"
    )
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "  ✓ ${method} complete"
    else
        echo "  ✗ ${method} FAILED"
        failed=$((failed + 1))
    fi
    echo ""
done

echo "================================================================"
if [ $failed -eq 0 ]; then
    echo "  R13 RE-RUN — all methods passed ✓"
else
    echo "  R13 RE-RUN — ${failed} method(s) failed ✗"
fi
echo "================================================================"
