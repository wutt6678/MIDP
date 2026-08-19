#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R12 — Re-run Missing Methods (GPU-aware)
#
# mmunlearner needs ~37 GB → GPU 1 (38 GB free)
# r2mu_adapted needs ~20 GB → GPU 2 or 3 (29 GB free)
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
echo "  R12 RE-RUN MISSING — GPU-aware"
echo "  output root: ${SMOKE_ROOT}"
echo "================================================================"
echo ""

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

# -- Run all on GPU 2 ------------------------------------------------------ #
GPU_ID=1

failed=0
for method in "${missing_methods[@]}"; do
    log_file="${SMOKE_ROOT}/${method}.log"
    
    echo "=== Running ${method} on GPU ${GPU_ID} ==="
    
    (
        cd "$SUITE_DIR" || exit 1
        CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_mllmu_baseline_suite.py \
            --only "$method" \
            --expected-code-sha "$CODE_SHA" \
            --runtime-output-root "$SMOKE_ROOT" \
            2>&1 | tee "$log_file"
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
    echo "  R12 RE-RUN — all methods passed ✓"
else
    echo "  R12 RE-RUN — ${failed} method(s) failed ✗"
fi
echo "================================================================"
