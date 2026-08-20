#!/usr/bin/env bash
# Qwen3.5-4B Real Unlearning Experiment Runner
#
# Runs real unlearning with forget/retain loss using the full UnlearningTrainer.
#
# Usage:
#   bash scripts/r17_4b_real_unlearning.sh              # Full run (50 steps)
#   bash scripts/r17_4b_real_unlearning.sh --smoke      # Smoke test (1 step)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments/unlearning/qwen35_4b/real_v1"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_4B_DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

CONDA_ENV="midp-qwen35"
PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"

SMOKE_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            SMOKE_FLAG="--smoke"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Qwen3.5-4B Real Unlearning"
echo "=========================================="
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Smoke mode: ${SMOKE_FLAG:-no}"
echo "=========================================="

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python not found: ${PYTHON}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Start time: ${START_TIME}"

cd "${PROJECT_ROOT}"
${PYTHON} scripts/run_4b_real_unlearning.py ${SMOKE_FLAG} 2>&1 | tee "${OUTPUT_DIR}/real_unlearning.log"

END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "End time: ${END_TIME}"

if [[ ! -f "${OUTPUT_DIR}/real_unlearning_report.json" ]]; then
    echo "ERROR: Report not found. Unlearning may have failed."
    exit 1
fi

echo "=========================================="
echo "Real unlearning complete!"
echo "Report: ${OUTPUT_DIR}/real_unlearning_report.json"
echo "Log: ${OUTPUT_DIR}/real_unlearning.log"
echo "=========================================="

${PYTHON} -c "
import json
from pathlib import Path

report_path = Path('${OUTPUT_DIR}/real_unlearning_report.json')
with open(report_path) as f:
    r = json.load(f)

print('\\nReal Unlearning Summary:')
print(f\"  Experiment ID: {r['experiment_id']}\")
print(f\"  Target identities: {r['identities']['target_ids']}\")
print(f\"  Retain identities: {r['identities']['retain_ids']}\")
print(f\"  Forget samples: {r['training_samples']['forget']}\")
print(f\"  Retain samples: {r['training_samples']['retain']}\")
print(f\"  Training steps: {r['training'].get('num_steps', 'N/A')}\")
print(f\"  Adapter saved: {r['adapter_path']}\")
"
