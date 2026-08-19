#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R12 — Re-run Failed/Missing Methods Only
#
# Runs only the methods that failed or are missing from the previous R12 run.
# Preserves existing results.
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   bash datasets/route-unlearning-data/scripts/r12_rerun_failed.sh
# --------------------------------------------------------------------------- #
set -euo pipefail

# -- Load shared environment ----------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/r12r14_env.sh"

# -- GPU selection (default GPU 2) ----------------------------------------- #
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# -- Prerequisites --------------------------------------------------------- #
verify_code_sha || exit 1

SMOKE_ROOT="${RUN_ROOT}/r12_smoke"
mkdir -p "${SMOKE_ROOT}"

echo ""
echo "================================================================"
echo "  R12 RE-RUN FAILED METHODS"
echo "  output root: ${SMOKE_ROOT}"
echo "================================================================"
echo ""

# ----------------------------------------------------------------------- #
# Check which methods are missing
# ----------------------------------------------------------------------- #
missing_methods=()

for method in mmunlearner r2mu_adapted; do
    if [ ! -f "${SMOKE_ROOT}/${method}/eval/eval_results.json" ]; then
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
    run_method "$method" "${SMOKE_ROOT}" "R12 ${method}" || exit 1
done

# ----------------------------------------------------------------------- #
# Post-rerun validation
# ----------------------------------------------------------------------- #
echo ""
echo "================================================================"
echo "  R12 RE-RUN — complete"
echo "================================================================"
echo ""
echo "For each re-run method verify:"
echo "  - 500/500 probes, 0 inference errors"
echo "  - strict_validation_pass = true"
echo "  - R²MU: answer_tokens_excluded=true"
echo ""
echo "If all pass → proceed to R13:"
echo "  bash datasets/route-unlearning-data/scripts/r13_full.sh"
