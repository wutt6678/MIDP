#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R13 — Run Missing 'manu' Method
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   bash datasets/route-unlearning-data/scripts/r13_run_manu.sh
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
echo "  R13 — Running 'manu' on GPU 3"
echo "  output root: ${FULL_ROOT}"
echo "================================================================"
echo ""

# -- Check if manu already exists ------------------------------------------ #
if [ -f "${FULL_ROOT}/manu/eval/eval_results.json" ]; then
    echo "  manu: EXISTS (nothing to do)"
    exit 0
fi

# -- Run manu -------------------------------------------------------------- #
log_file="${FULL_ROOT}/manu.log"

echo "=== Running manu on GPU 3 ==="

(
    cd "$SUITE_DIR" || exit 1
    CUDA_VISIBLE_DEVICES=3 python scripts/run_mllmu_baseline_suite.py \
        --only manu \
        --expected-code-sha "$CODE_SHA" \
        --runtime-output-root "$FULL_ROOT" \
        2>&1 | tee "$log_file"
)

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "  ✓ manu complete"
    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "  ✗ manu FAILED"
    echo "================================================================"
    exit 1
fi
