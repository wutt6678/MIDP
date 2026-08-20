#!/usr/bin/env bash
# Common unlearning runner for all models.
# Usage: bash run_unlearning_all.sh --model MODEL_KEY [--canary] [--gpu GPU]
#
# Examples:
#   bash run_unlearning_all.sh --model qwen35_4b --canary --gpu 0
#   bash run_unlearning_all.sh --model phi4_mm --gpu 0
#   bash run_unlearning_all.sh --model glm46v_flash --gpu 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
MODEL=""
CANARY=""
GPU="0"
SKIP_POST_EVAL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --canary) CANARY="--canary"; shift ;;
        --gpu) GPU="$2"; shift 2 ;;
        --skip-post-eval) SKIP_POST_EVAL="--skip-post-eval"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "Usage: $0 --model MODEL_KEY [--canary] [--gpu GPU]"
    echo ""
    echo "Available models:"
    echo "  qwen35_4b       Qwen3.5-4B (midp-qwen35 env)"
    echo "  glm46v_flash    GLM-4.6V-Flash (midp env)"
    echo "  internvl35_8b_hf InternVL-3.5-8B (midp env)"
    echo "  phi4_mm         Phi-4-MM (midp-phi4mm env)"
    echo "  gemma3_12b      Gemma-3-12B (midp-qwen35 env)"
    exit 1
fi

# Environment mapping
case "${MODEL}" in
    qwen35_4b)    ENV_NAME="midp-qwen35" ;;
    glm46v_flash) ENV_NAME="midp" ;;
    internvl35_8b_hf) ENV_NAME="midp" ;;
    phi4_mm)      ENV_NAME="midp-phi4mm" ;;
    gemma3_12b)   ENV_NAME="midp-qwen35" ;;
    *)
        echo "ERROR: Unknown model '${MODEL}'"
        echo "Available: qwen35_4b, glm46v_flash, internvl35_8b_hf, phi4_mm, gemma3_12b"
        exit 1
        ;;
esac

PROFILE="${PROJECT_DIR}/configs/models/unlearning/${MODEL}.yaml"
METHOD="${PROJECT_DIR}/configs/methods/candidate_margin_v1.yaml"
SELECTION="${PROJECT_DIR}/configs/experiments/common/frozen_identity_selection_v1.yaml"

echo "=============================================="
echo "Model:    ${MODEL}"
echo "Env:      ${ENV_NAME}"
echo "GPU:      ${GPU}"
echo "Profile:  ${PROFILE}"
echo "Canary:   ${CANARY:-no}"
echo "=============================================="

# Check profile exists
if [[ ! -f "${PROFILE}" ]]; then
    echo "ERROR: Profile not found: ${PROFILE}"
    exit 1
fi

# Activate environment
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate "${ENV_NAME}"

echo "Python: $(which python)"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"

# Run
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "${PROJECT_DIR}"

python scripts/run_model_unlearning.py \
    --model-profile "${PROFILE}" \
    --method-config "${METHOD}" \
    --selection "${SELECTION}" \
    --seed 17 \
    --device "cuda:0" \
    ${CANARY} \
    ${SKIP_POST_EVAL}

echo ""
echo "Done. Output: outputs/experiments/unlearning/${MODEL}/candidate_margin/"
