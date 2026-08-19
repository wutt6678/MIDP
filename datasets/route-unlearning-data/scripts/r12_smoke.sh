#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R12 — Real-Qwen Smoke Validation
#
# Proves that repaired baselines execute end-to-end on real Qwen3.5-9B.
# Uses reduced budgets; labeled evidence_scope=smoke.
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   source datasets/route-unlearning-data/scripts/r12r14_env.sh
#   bash datasets/route-unlearning-data/scripts/r12_smoke.sh
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
echo "  R12 SMOKE — starting"
echo "  output root: ${SMOKE_ROOT}"
echo "================================================================"
echo ""

# ----------------------------------------------------------------------- #
# 1. Prompting (eval-only, no training)
# ----------------------------------------------------------------------- #
run_method prompting "${SMOKE_ROOT}" "[1/5] Prompting smoke" || exit 1

# ----------------------------------------------------------------------- #
# 2. GA (ordinary LoRA training + common evaluator)
# ----------------------------------------------------------------------- #
run_method ga "${SMOKE_ROOT}" "[2/5] GA smoke" || exit 1

# ----------------------------------------------------------------------- #
# 3. MMUnlearner (exact-k mask on real LoRA tensors)
# ----------------------------------------------------------------------- #
run_method mmunlearner "${SMOKE_ROOT}" "[3/5] MMUnlearner smoke" || exit 1

# ----------------------------------------------------------------------- #
# 4. MANU (structural pruning + restore verification, 5% rate)
# ----------------------------------------------------------------------- #
run_method manu "${SMOKE_ROOT}" "[4/5] MANU smoke" || exit 1

# ----------------------------------------------------------------------- #
# 5. R²MU-adapted (pre-answer representation extraction)
# ----------------------------------------------------------------------- #
run_method r2mu_adapted "${SMOKE_ROOT}" "[5/5] R²MU-adapted smoke" || exit 1

# ----------------------------------------------------------------------- #
# Post-smoke validation
# ----------------------------------------------------------------------- #
echo ""
echo "================================================================"
echo "  R12 SMOKE — all methods finished"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Inspect each method's eval/ directory for:"
echo "     - strict_validation_pass = true"
echo "     - exact_pair_count = 500"
echo "     - inference_errors = 0"
echo "     - method = <expected>"
echo "     - model_revision = ${MODEL_REVISION}"
echo ""
echo "  2. For MANU, verify:"
echo "     - all_restores_verified = true"
echo ""
echo "  3. For R²MU, verify:"
echo "     - answer_tokens_excluded = true"
echo "     - boundary_diagnostics.n_prefix_boundaries_valid > 0"
echo ""
echo "  4. If all pass → proceed to R13:"
echo "     bash datasets/route-unlearning-data/scripts/r13_full.sh"
echo ""
echo "  5. If any fail → STOP. Do not proceed to R13."
