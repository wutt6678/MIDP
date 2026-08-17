"""Reference and oracle model loaders for KL and NPO baselines.

This module provides functions to load:
- Frozen reference model for KL Minimization (pre-unlearning model)
- Oracle model for NPO (retain-only fine-tuned model)

Both reference and oracle models are loaded in eval mode with
requires_grad=False for all parameters.

Public API
----------
.. autofunction:: load_frozen_reference_model
.. autofunction:: load_oracle_model
.. autofunction:: verify_reference_model_fingerprint
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_frozen_reference_model(
    model_id: str,
    revision: str,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
) -> tuple[Any, Any]:
    """Load a frozen reference model for KL Minimization.

    This is the pre-unlearning model (same base model, no adapter).
    Used as the reference distribution that the current model should
    stay close to on retain data.

    Parameters
    ----------
    model_id:
        HuggingFace model ID (e.g., "Qwen/Qwen3.5-9B").
    revision:
        Model revision hash.
    dtype:
        Torch dtype string.
    device:
        Device for the model.

    Returns
    -------
    model:
        The frozen reference model in eval mode.
    processor:
        The processor for tokenization and image processing.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = getattr(torch, dtype)
    snapshot_download(model_id, revision=revision)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        revision=revision,
        device_map=device,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    # Freeze all parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        f"Loaded frozen reference model {model_id} revision {revision} "
        f"({sum(p.numel() for p in model.parameters())} params, all frozen)"
    )
    return model, processor


def load_oracle_model(
    model_id: str,
    revision: str,
    adapter_path: str | Path,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: list[str] | None = None,
) -> tuple[Any, Any]:
    """Load a frozen oracle model for NPO.

    The oracle is a retain-only fine-tuned model. It must be loaded
    with the same LoRA configuration used during oracle training.

    Parameters
    ----------
    model_id:
        HuggingFace model ID.
    revision:
        Model revision hash.
    adapter_path:
        Path to the trained LoRA adapter directory.
    dtype:
        Torch dtype string.
    device:
        Device for the model.
    lora_rank:
        LoRA rank used during oracle training.
    lora_alpha:
        LoRA alpha used during oracle training.
    lora_target_modules:
        LoRA target modules used during oracle training.

    Returns
    -------
    model:
        The frozen oracle model with LoRA adapter in eval mode.
    processor:
        The processor for tokenization and image processing.
    """
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = getattr(torch, dtype)
    snapshot_download(model_id, revision=revision)

    # Load base model
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        revision=revision,
        device_map=device,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device,
    )

    # Freeze all parameters (base + adapter)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        f"Loaded frozen oracle model from {adapter_path} "
        f"(all params frozen)"
    )
    return model, processor


def verify_reference_model_fingerprint(
    reference_model: Any,
    expected_fingerprint: str,
) -> bool:
    """Verify that the reference model matches the expected fingerprint.

    Parameters
    ----------
    reference_model:
        The loaded reference model.
    expected_fingerprint:
        Expected model fingerprint/hash.

    Returns
    -------
    match:
        True if fingerprints match.
    """
    # Compute a simple fingerprint from model config
    config = reference_model.config
    fingerprint = f"{config.model_type}_{config.hidden_size}_{config.num_hidden_layers}"

    match = fingerprint == expected_fingerprint
    if not match:
        logger.warning(
            f"Reference model fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {fingerprint}"
        )
    return match


def write_oracle_manifest(
    output_path: str | Path,
    model_id: str,
    revision: str,
    retain_dataset_sha256: str,
    selection_manifest_sha256: str,
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
    adapter_sha256: str,
    seed: int,
    optimizer_steps: int,
) -> None:
    """Write the oracle manifest with full provenance.

    Parameters
    ----------
    output_path:
        Path to write the manifest JSON.
    model_id:
        Base model ID.
    revision:
        Model revision.
    retain_dataset_sha256:
        SHA256 of the retain dataset.
    selection_manifest_sha256:
        SHA256 of the identity selection manifest.
    training_config:
        Training configuration dict.
    lora_config:
        LoRA configuration dict.
    adapter_sha256:
        SHA256 of the trained adapter.
    seed:
        Training seed.
    optimizer_steps:
        Number of optimizer steps.
    """
    manifest = {
        "base_model": {
            "model_id": model_id,
            "revision": revision,
        },
        "retain_dataset_sha256": retain_dataset_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "training_config": training_config,
        "lora_config": lora_config,
        "adapter_sha256": adapter_sha256,
        "seed": seed,
        "optimizer_steps": optimizer_steps,
        "provenance": {
            "oracle_trained_on_retain_only": True,
            "oracle_trained_on_target": False,
            "oracle_trained_on_control": False,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    logger.info(f"Wrote oracle manifest: {output_path}")
