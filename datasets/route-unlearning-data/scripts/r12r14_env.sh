#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R12–R14 Shared Environment Setup
#
# Source this file before running any experiment script:
#   conda activate midp-qwen35
#   cd /path/to/MIDP
#   source datasets/route-unlearning-data/scripts/r12r14_env.sh
#
# NOTE: This file does NOT set -e.  It is sourced into your interactive
# shell, so set -e here would kill your terminal on any failed command.
# --------------------------------------------------------------------------- #

# -- Frozen contract ------------------------------------------------------- #
# Default SHA (can be overridden by .frozen_env for local development)
export CODE_SHA="a19f66c3df92572d50eae8982c91ba7ef68f7b6b"
export MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
export PROCESSED_DATASET_SHA="7200df4ec361ee52ad8a183b1181271980f35fb3f79690931f17481080c0d8c1"
export ROUTE_PROBE_SHA="aeca4ee889e429ad717afb4d83c265b3990aebd5c1464b8afb4b4a2ad4dfd864"
export SELECTION_MANIFEST_SHA="a7ff1fceb715fb30b34809d98eb6e7e25a4a21a88a8b188c024d124b49b19655"

# Load local overrides (gitignored)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/.frozen_env" ]; then
    source "${SCRIPT_DIR}/.frozen_env"
fi

# -- GPU selection --------------------------------------------------------- #
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# -- Paths ----------------------------------------------------------------- #
# Resolve the route-unlearning-data project root from this file's location.
export SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ROOT="${SUITE_DIR}/runtime_outputs/mllmu_baselines/${CODE_SHA}"

# -- Hardware info (for reports) ------------------------------------------- #
export HARDWARE="$(nvidia-smi --query-gpu=name,memory.total \
  --format=csv,noheader 2>/dev/null | head -1 | xargs -0 echo || echo 'N/A')"
export CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo unknown)"
export PYTORCH_VERSION="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
export TRANSFORMERS_VERSION="$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo unknown)"
export PEFT_VERSION="$(python -c 'import peft; print(peft.__version__)' 2>/dev/null || echo unknown)"

# -- Helpers --------------------------------------------------------------- #
verify_code_sha() {
    echo "=== Verifying code SHA ==="
    local actual
    actual="$(git rev-parse HEAD 2>/dev/null)" || {
        echo "FATAL: not inside the MIDP git repository"
        echo "  Run: cd /scratch/wutiantong/MIDP"
        return 1
    }
    if [ "$actual" != "$CODE_SHA" ]; then
        echo "FATAL: HEAD=$actual, expected=$CODE_SHA"
        return 1
    fi
    echo "  HEAD = ${actual}  ✓"

    local dirty
    dirty="$(git status --porcelain 2>/dev/null | grep -v '^??' || true)"
    if [ -n "$dirty" ]; then
        echo "FATAL: tracked files are modified:"
        echo "$dirty"
        return 1
    fi
    echo "  working tree clean (untracked files OK)  ✓"
}

run_preflight() {
    local output_root="$1"
    echo "=== Running preflight → ${output_root} ==="
    (
        cd "$SUITE_DIR" || return 1
        python scripts/run_mllmu_baseline_suite.py \
            --preflight-only \
            --expected-code-sha "$CODE_SHA" \
            --runtime-output-root "$output_root"
    ) || return 1
    echo "  preflight PASSED  ✓"
}

run_method() {
    local method="$1"
    local output_root="$2"
    local label="${3:-$method}"
    echo "=== Running ${label} ==="
    (
        cd "$SUITE_DIR" || return 1
        python scripts/run_mllmu_baseline_suite.py \
            --only "$method" \
            --expected-code-sha "$CODE_SHA" \
            --runtime-output-root "$output_root" \
            2>&1 | tee "${output_root}/${method}.log"
    ) || { echo "FAILED: ${label}"; return 1; }
    echo "  ${label} complete  ✓"
}

check_preflight_hashes() {
    local report="$1"
    echo "=== Checking preflight hash evidence ==="
    if [ ! -f "$report" ]; then
        echo "FATAL: preflight report not found: $report"
        return 1
    fi
    python3 -c "
import json, sys
with open('${report}') as f:
    r = json.load(f)
required = {
    'processed_dataset_sha256_match': '${PROCESSED_DATASET_SHA}',
    'route_probe_sha256_match': '${ROUTE_PROBE_SHA}',
}
checks = {c['name']: c for c in r['checks']}
ok = True
for name, expected_sha in required.items():
    c = checks.get(name)
    if c is None:
        print(f'  SKIP: {name} (not in report — file may be absent in CI)')
        continue
    if not c['pass']:
        print(f'  FAIL: {name} — {c[\"detail\"]}')
        ok = False
    else:
        print(f'  PASS: {name}')
if not r['preflight_passed']:
    print('FATAL: preflight did not pass')
    sys.exit(1)
if ok:
    print('  All hash checks PASSED')
" || return 1
}

echo "R12–R14 environment loaded."
echo "  CODE_SHA   = ${CODE_SHA:0:16}..."
echo "  RUN_ROOT   = ${RUN_ROOT}"
echo "  SUITE_DIR  = ${SUITE_DIR}"
