#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R13 — Full Baseline Evidence Generation
#
# Runs ALL comparison methods at canonical full budgets.
# Output becomes input to R14 final comparison.
#
# Prerequisites:
#   - R12 smoke passed for all mandatory methods
#   - All method configs are at canonical full budgets (NOT smoke)
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   source datasets/route-unlearning-data/scripts/r12r14_env.sh
#   bash datasets/route-unlearning-data/scripts/r13_full.sh
# --------------------------------------------------------------------------- #
set -euo pipefail

# -- Load shared environment ----------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/r12r14_env.sh"

# -- Prerequisites --------------------------------------------------------- #
verify_code_sha || exit 1

FULL_ROOT="${RUN_ROOT}/r13_full"
mkdir -p "${FULL_ROOT}"

# -- Preflight ------------------------------------------------------------- #
run_preflight "${FULL_ROOT}/preflight" || exit 1
check_preflight_hashes "${FULL_ROOT}/preflight/suite_preflight_report.json" || exit 1

echo ""
echo "================================================================"
echo "  R13 FULL — starting"
echo "  output root: ${FULL_ROOT}"
echo "================================================================"
echo ""

# ----------------------------------------------------------------------- #
# Canonical-config audit (Section 41)
# ----------------------------------------------------------------------- #
echo "=== Canonical-config audit ==="
echo "Verify each method's resolved config before running."
echo "Inspect configs/experiments/mllmu_baselines/*.yaml"
echo ""
echo "  GA:        training.max_optimizer_steps = 125"
echo "  GD:        training.max_optimizer_steps = 125"
echo "  KL:        training.max_optimizer_steps = 125"
echo "  NPO:       training.max_optimizer_steps = 125"
echo "  Prompting: eval-only (500 probes)"
echo "  MMU:       saliency_n_samples = 32, training = 125 steps"
echo "  MANU:      5% + 10% prune rates"
echo "  R²MU:      num_optimizer_steps = 125"
echo "  MIDP-CM:   evidence_mode = historical_bound"
echo ""
read -p "Have you verified all configs are canonical? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborting. Verify configs before running R13."
    exit 1
fi

# ----------------------------------------------------------------------- #
# R13 run order (Section 19)
# ----------------------------------------------------------------------- #

run_method prompting    "${FULL_ROOT}" "[1/10] Prompting"          || exit 1
run_method ga           "${FULL_ROOT}" "[2/10] GA"                 || exit 1
run_method gd           "${FULL_ROOT}" "[3/10] GD"                 || exit 1
run_method kl           "${FULL_ROOT}" "[4/10] KL"                 || exit 1
run_method npo_oracle   "${FULL_ROOT}" "[5/10] NPO oracle"         || exit 1
run_method npo          "${FULL_ROOT}" "[6/10] NPO"                || exit 1
run_method mmunlearner  "${FULL_ROOT}" "[7/10] MMUnlearner"        || exit 1
run_method manu         "${FULL_ROOT}" "[8/10] MANU (5% + 10%)"    || exit 1
run_method r2mu_adapted "${FULL_ROOT}" "[9/10] R²MU-adapted"       || exit 1
run_method midp_cm      "${FULL_ROOT}" "[10/10] MIDP-CM"           || exit 1

# ----------------------------------------------------------------------- #
# Post-R13 validation
# ----------------------------------------------------------------------- #
echo ""
echo "================================================================"
echo "  R13 FULL — all methods finished"
echo "================================================================"
echo ""
echo "Verify suite_summary.json shows:"
echo '  {"execution_scope": "full",'
echo '   "missing_comparison_methods": [],'
echo '   "eval_complete": true,'
echo '   "research_suite_complete": true}'
echo ""
echo "For each method verify:"
echo "  - 500/500 probes, 0 inference errors"
echo "  - strict_validation_pass = true"
echo "  - exact_pairing_pass = true"
echo "  - 2/2/2/94 identity grouping"
echo "  - MANU: all_restores_verified=true for both 5% and 10%"
echo "  - R²MU: answer_tokens_excluded=true"
echo "  - MMUnlearner: exact mask cardinality confirmed"
echo ""
echo "If all pass → proceed to R14:"
echo "  bash datasets/route-unlearning-data/scripts/r14_freeze.sh"
