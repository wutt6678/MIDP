#!/usr/bin/env bash
# Create an isolated conda env (midp_qwen) capable of loading Qwen3.5-9B.
# Kept separate from `midp` so the pinned Llama-3.2-Vision pipeline is untouched.
set -euo pipefail

source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh

ENV_NAME=midp_qwen

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env '$ENV_NAME' already exists; reusing it."
else
    echo "[setup] creating env '$ENV_NAME' (python 3.11)..."
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

echo "[setup] installing torch + torchvision (latest CUDA build)..."
pip install --upgrade torch torchvision

echo "[setup] installing latest transformers + eval deps..."
pip install --upgrade transformers accelerate datasets huggingface_hub \
    pillow pyyaml scikit-learn pyarrow pandas tqdm

echo "[setup] verifying qwen3_5 support..."
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")
print("AutoConfig OK -> model_type:", cfg.model_type,
      "| architectures:", cfg.architectures)
PY

echo "[setup] done. env '$ENV_NAME' ready."
