#!/usr/bin/env bash
# Qwen3.5-4B Unlearning Canary Runner
#
# Runs a real end-to-end unlearning canary with actual training and evaluation.
#
# Usage:
#   bash scripts/r16_4b_canary.sh              # Full canary (50 steps, 500 probes)
#   bash scripts/r16_4b_canary.sh --smoke      # Smoke canary (1 step, 10 probes)
#
# This verifies:
# - Real target/forget examples loaded
# - Real retain examples loaded
# - Loss remains finite
# - LoRA gradients nonzero
# - LoRA parameters change
# - Checkpoint saved and reloaded
# - Post-eval produces 500/500 matched probes
# - Inference errors = 0
# - Per-family deltas reported

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments/unlearning/qwen35_4b/canary_v1"

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
echo "Qwen3.5-4B Unlearning Canary"
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
${PYTHON} scripts/run_4b_unlearning_canary.py ${SMOKE_FLAG} 2>&1 | tee "${OUTPUT_DIR}/canary.log"

END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "End time: ${END_TIME}"

if [[ ! -f "${OUTPUT_DIR}/canary_report.json" ]]; then
    echo "ERROR: Canary report not found. Canary may have failed."
    exit 1
fi

echo "=========================================="
echo "Canary complete!"
echo "Report: ${OUTPUT_DIR}/canary_report.json"
echo "Log: ${OUTPUT_DIR}/canary.log"
echo "=========================================="

${PYTHON} -c "
import json
from pathlib import Path

report_path = Path('${OUTPUT_DIR}/canary_report.json')
with open(report_path) as f:
    r = json.load(f)

print('\\nCanary Summary:')
print(f\"  Target identities: {r['identities']['target_ids']}\")
print(f\"  Retain identities: {r['identities']['retain_ids']}\")
print(f\"  Control identities: {r['identities']['control_ids']}\")
print(f\"  Identity counts: {r['identity_counts']}\")
print(f\"  Training steps: {r['training']['num_steps']}\")
print(f\"  Final loss: {r['training']['final_loss']:.4f}\")
print(f\"  LoRA tensors changed: {r['training']['lora_tensors_changed']}/{r['training']['lora_tensors_total']}\")
print(f\"  Post-eval probes: {r['post_evaluation']['num_probes']}\")
print(f\"  Inference errors: {r['post_evaluation']['inference_errors']}\")
print('\\nFamily Deltas:')
for family, deltas in r['post_evaluation']['family_deltas'].items():
    print(f\"  {family}: delta={deltas['delta']:.4f}\")
print('\\nRequirements Met:')
for req, met in r['requirements_met'].items():
    status = '✓' if met else '✗'
    print(f\"  {status} {req}\")
"
