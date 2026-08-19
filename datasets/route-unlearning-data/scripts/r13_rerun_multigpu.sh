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
for method in kl npo mmunlearner; do
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

# -- GPU assignment -------------------------------------------------------- #
# kl and npo need ~20 GB each → GPUs 2,3 (29 GB free)
# mmunlearner needs ~37 GB → GPU 1 (38 GB free)
declare -A GPU_MAP
GPU_MAP[kl]=2
GPU_MAP[npo]=3
GPU_MAP[mmunlearner]=1

# Run mmunlearner first (needs most memory), then others
ordered_methods=()
for m in mmunlearner kl npo; do
    for missing in "${missing_methods[@]}"; do
        if [ "$m" = "$missing" ]; then
            ordered_methods+=("$m")
        fi
    done
done

failed=0
for method in "${ordered_methods[@]}"; do
    gpu=${GPU_MAP[$method]}
    log_file="${FULL_ROOT}/${method}.log"
    
    echo "=== Running ${method} on GPU ${gpu} ==="
    
    (
        cd "$SUITE_DIR" || exit 1
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_mllmu_baseline_suite.py \
            --only "$method" \
            --expected-code-sha "$CODE_SHA" \
            --runtime-output-root "$FULL_ROOT" \
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
    echo "  R13 RE-RUN — all methods passed ✓"
else
    echo "  R13 RE-RUN — ${failed} method(s) failed ✗"
fi
echo "================================================================"
