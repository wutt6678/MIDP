#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R13 — Re-run Failed/Missing Methods Only
#
# Runs only the methods that failed or are missing from the previous R13 run.
# Preserves existing results.
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   bash datasets/route-unlearning-data/scripts/r13_rerun_failed.sh
# --------------------------------------------------------------------------- #
set -euo pipefail

# -- Load shared environment ----------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/r12r14_env.sh"

# -- GPU override for R13 -------------------------------------------------- #
export CUDA_VISIBLE_DEVICES=0

# -- Prerequisites --------------------------------------------------------- #
verify_code_sha || exit 1

FULL_ROOT="${RUN_ROOT}/r13_full"
mkdir -p "${FULL_ROOT}"

echo ""
echo "================================================================"
echo "  R13 RE-RUN FAILED METHODS"
echo "  output root: ${FULL_ROOT}"
echo "================================================================"
echo ""

# ----------------------------------------------------------------------- #
# Check which methods are missing
# ----------------------------------------------------------------------- #
missing_methods=()

for method in kl npo mmunlearner; do
    if [ ! -f "${FULL_ROOT}/${method}/eval/eval_results.json" ]; then
        missing_methods+=("$method")
        echo "  ${method}: MISSING (will run)"
    else
        echo "  ${method}: EXISTS (skipping)"
    fi
done

if [ ${#missing_methods[@]} -eq 0 ]; then
    echo ""
    echo "All methods already have results. Nothing to re-run."
    exit 0
fi

echo ""
echo "Will run: ${missing_methods[*]}"
echo ""

# ----------------------------------------------------------------------- #
# Run missing methods
# ----------------------------------------------------------------------- #
for method in "${missing_methods[@]}"; do
    run_method "$method" "${FULL_ROOT}" "R13 ${method}" || exit 1
done

# ----------------------------------------------------------------------- #
# Post-rerun validation
# ----------------------------------------------------------------------- #
echo ""
echo "================================================================"
echo "  R13 RE-RUN — complete"
echo "================================================================"
echo ""
echo "Verify suite_summary.json now shows:"
echo '  {"execution_scope": "full",'
echo '   "missing_comparison_methods": [],'
echo '   "eval_complete": true}'
echo ""
echo "If all pass → proceed to R14:"
echo "  bash datasets/route-unlearning-data/scripts/r14_freeze.sh"
