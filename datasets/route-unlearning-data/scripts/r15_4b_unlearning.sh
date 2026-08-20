#!/usr/bin/env bash
# Qwen3.5-4B Unlearning Experiment Runner
#
# This script runs the full unlearning pipeline for Qwen3.5-4B.
#
# Usage:
#   bash scripts/r15_4b_unlearning.sh              # Full production run
#   bash scripts/r15_4b_unlearning.sh --smoke      # Smoke test (1 step, 10 probes)
#   bash scripts/r15_4b_unlearning.sh --resume     # Resume previous run
#
# Prerequisites:
#   - Conda environment: midp-qwen35
#   - GPU available (CUDA_DEVICE=0 by default)
#   - Baseline completed: outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/
#
# Output:
#   outputs/experiments/unlearning/qwen35_4b/v1/
#     - adapter/                    (trained LoRA adapter)
#     - post_eval_results.jsonl     (post-unlearning evaluation)
#     - experiment_manifest.json    (full provenance)
#     - preservation_report.json    (baseline vs post comparison)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/experiments/unlearning_4b_v1.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments/unlearning/qwen35_4b/v1"

# GPU binding (r13_full.sh convention)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN35_4B_DEVICE="cuda:${CUDA_VISIBLE_DEVICES}"

# Conda environment
CONDA_ENV="midp-qwen35"
PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"

# Parse arguments
SMOKE_FLAG=""
RESUME_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)
            SMOKE_FLAG="--smoke"
            shift
            ;;
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Qwen3.5-4B Unlearning Experiment"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Smoke mode: ${SMOKE_FLAG:-no}"
echo "Resume mode: ${RESUME_FLAG:-no}"
echo "=========================================="

# Verify prerequisites
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_binding.json" ]]; then
    echo "ERROR: Baseline binding not found. Run baseline generation first."
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python not found: ${PYTHON}"
    echo "Please ensure conda environment '${CONDA_ENV}' is installed."
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Record start time
START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Start time: ${START_TIME}"

# Run unlearning pipeline
cd "${PROJECT_ROOT}"
${PYTHON} scripts/run_4b_unlearning.py \
    --config "${CONFIG}" \
    ${SMOKE_FLAG} \
    ${RESUME_FLAG} \
    2>&1 | tee "${OUTPUT_DIR}/unlearning.log"

# Record end time
END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "End time: ${END_TIME}"

# Verify output
if [[ ! -f "${OUTPUT_DIR}/experiment_manifest.json" ]]; then
    echo "ERROR: Experiment manifest not found. Unlearning may have failed."
    exit 1
fi

echo "=========================================="
echo "Unlearning complete!"
echo "Manifest: ${OUTPUT_DIR}/experiment_manifest.json"
echo "Log: ${OUTPUT_DIR}/unlearning.log"
echo "=========================================="

# Print summary
${PYTHON} -c "
import json
from pathlib import Path

manifest_path = Path('${OUTPUT_DIR}/experiment_manifest.json')
with open(manifest_path) as f:
    m = json.load(f)

print('\\nExperiment Summary:')
print(f\"  Experiment ID: {m['experiment_id']}\")
print(f\"  Training steps: {m['training']['num_steps']}\")
print(f\"  Post-eval probes: {m['post_evaluation']['num_probes']}\")
print(f\"  Visual accuracy: {m['preservation_report']['post_visual_accuracy']:.4f}\")
print(f\"  Name-only fuzzy match: {m['preservation_report']['post_name_only_fuzzy']:.4f}\")
print(f\"  Adapter SHA-256: {m['adapter']['sha256'][:16]}...\")
print(f\"  Code commit: {m['code_provenance']['experiment_code_commit'][:8]}\")
"
