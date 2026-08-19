#!/usr/bin/env python3
"""Generate Qwen3.5-9B module and LoRA target inventories (P0-R13).

Loads the real frozen-revision model and produces:
1. qwen35_9b_module_inventory.json — full module tree metadata
2. qwen35_9b_lora_target_inventory.json — LoRA targets with classification
3. qwen35_9b_layer_inventory.json — per-layer attention type inventory
4. Structural metadata verification against the profile

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python scripts/generate_qwen35_9b_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "model_inventories"
PROFILE_PATH = PROJECT_ROOT / "configs" / "models" / "unlearning" / "qwen35_9b.yaml"

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def load_model():
    """Load the real Qwen3.5-9B model at the pinned revision."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info(f"Loading {MODEL_ID} @ {MODEL_REVISION} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"Model loaded: {type(model).__name__}")
    return model, processor


def generate_module_inventory(model) -> dict:
    """Generate full module inventory from named_modules()."""
    all_modules = []
    for name, module in model.named_modules():
        if name:  # skip root
            all_modules.append({
                "name": name,
                "type": type(module).__name__,
                "n_params": sum(p.numel() for p in module.parameters(recurse=False)),
            })

    # Key structural paths
    model_class = type(model).__name__

    # Language model submodel
    lang_model = getattr(model.model, "language_model", None)
    lang_model_class = type(lang_model).__name__ if lang_model else None

    # Vision submodel
    visual = getattr(model.model, "visual", None)
    visual_class = type(visual).__name__ if visual else None

    # Language layers
    lang_layers = getattr(lang_model, "layers", None) if lang_model else None
    n_lang_layers = len(lang_layers) if lang_layers is not None else None

    # Language config
    text_config = getattr(model.config, "text_config", None)
    hidden_size = text_config.hidden_size if text_config else None
    intermediate_size = text_config.intermediate_size if text_config else None
    num_hidden_layers = text_config.num_hidden_layers if text_config else None

    return {
        "model_class": model_class,
        "language_submodel_class": lang_model_class,
        "vision_submodel_class": visual_class,
        "language_layer_path": "model.language_model.layers",
        "language_layer_count": n_lang_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers_from_config": num_hidden_layers,
        "total_module_count": len(all_modules),
        "modules": all_modules,
    }


def generate_layer_inventory(model) -> list[dict]:
    """Generate per-layer attention type inventory."""
    lang_layers = model.model.language_model.layers
    inventory = []
    for i, layer in enumerate(lang_layers):
        self_attn = getattr(layer, "self_attn", None)
        attn_type = type(self_attn).__name__ if self_attn else "unknown"

        has_q = hasattr(self_attn, "q_proj") if self_attn else False
        has_k = hasattr(self_attn, "k_proj") if self_attn else False
        has_v = hasattr(self_attn, "v_proj") if self_attn else False
        has_o = hasattr(self_attn, "o_proj") if self_attn else False

        inventory.append({
            "layer_index": i,
            "attention_type": attn_type,
            "has_q_proj": has_q,
            "has_k_proj": has_k,
            "has_v_proj": has_v,
            "has_o_proj": has_o,
        })
    return inventory


def generate_lora_inventory(model, profile) -> dict:
    """Generate LoRA target inventory using the adapter."""
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    adapter = Qwen35Adapter(profile)
    targets = adapter.resolve_lora_targets(model)

    # Classify
    language_count = 0
    vision_count = 0
    projector_count = 0
    connector_count = 0
    for name in targets:
        if "language_model" in name or "text_model" in name:
            language_count += 1
        elif "visual" in name or "vision" in name:
            vision_count += 1
        elif "projector" in name:
            projector_count += 1
        elif "connector" in name:
            connector_count += 1

    total_params = sum(p.numel() for p in model.parameters())
    selected_params = 0
    named = dict(model.named_modules())
    for name in targets:
        mod = named.get(name)
        if mod is not None:
            selected_params += sum(p.numel() for p in mod.parameters())

    inventory = {
        "model_key": profile.key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "adapter_family": profile.adapter_name,
        "scope_regex": profile.lora_scope_regex,
        "target_leaf_names": list(profile.lora_target_leaf_names),
        "selected_module_count": len(targets),
        "selected_modules": sorted(targets),
        "language_module_count": language_count,
        "vision_module_count": vision_count,
        "projector_module_count": projector_count,
        "connector_module_count": connector_count,
        "selected_parameter_count": selected_params,
        "total_parameter_count": total_params,
    }

    _ser = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    inventory["inventory_sha256"] = hashlib.sha256(_ser.encode()).hexdigest()
    return inventory


def verify_structural(model, profile):
    """Verify structural metadata against profile."""
    from route_data.models.trainable.qwen35 import Qwen35Adapter
    from route_data.models.trainable.registry import validate_structural_metadata

    adapter = Qwen35Adapter(profile)
    errors = validate_structural_metadata(adapter, model)
    return errors


def verify_pad_token(processor):
    """Verify pad token ID from processor."""
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    return pad_id


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_model()

    # Load profile
    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        load_profile_from_yaml,
    )
    profile = load_profile_from_yaml(str(PROFILE_PATH))
    profile_sha = compute_profile_sha256(str(PROFILE_PATH))

    logger.info("=" * 60)
    logger.info("P0-R13: Qwen3.5-9B Real Model Inventory")
    logger.info("=" * 60)

    # 1. Module inventory
    logger.info("\n--- Module Inventory ---")
    module_inv = generate_module_inventory(model)
    logger.info(f"Model class: {module_inv['model_class']}")
    logger.info(f"Language submodel: {module_inv['language_submodel_class']}")
    logger.info(f"Vision submodel: {module_inv['vision_submodel_class']}")
    logger.info(f"Language layers: {module_inv['language_layer_count']}")
    logger.info(f"Hidden size: {module_inv['hidden_size']}")
    logger.info(f"Intermediate size: {module_inv['intermediate_size']}")
    logger.info(f"Total modules: {module_inv['total_module_count']}")

    module_path = OUTPUT_DIR / "qwen35_9b_module_inventory.json"
    with open(module_path, "w") as f:
        json.dump(module_inv, f, indent=2)
    logger.info(f"Written: {module_path}")

    # 2. Layer inventory (hybrid attention)
    logger.info("\n--- Layer Inventory (Hybrid Attention) ---")
    layer_inv = generate_layer_inventory(model)
    full_attn = sum(1 for l in layer_inv if l["has_q_proj"] and l["has_k_proj"] and l["has_v_proj"] and l["has_o_proj"])
    attn_types = {}
    for l in layer_inv:
        t = l["attention_type"]
        attn_types[t] = attn_types.get(t, 0) + 1

    logger.info(f"Total language layers: {len(layer_inv)}")
    logger.info(f"Full-attention layers (q/k/v/o): {full_attn}")
    logger.info(f"Attention types: {attn_types}")
    lora_targetable = sum(1 for l in layer_inv if l["has_q_proj"] and l["has_v_proj"])
    logger.info(f"LoRA-targetable layers: {lora_targetable}")

    layer_path = OUTPUT_DIR / "qwen35_9b_layer_inventory.json"
    with open(layer_path, "w") as f:
        json.dump(layer_inv, f, indent=2)
    logger.info(f"Written: {layer_path}")

    # 3. LoRA target inventory
    logger.info("\n--- LoRA Target Inventory ---")
    lora_inv = generate_lora_inventory(model, profile)
    logger.info(f"Selected modules: {lora_inv['selected_module_count']}")
    logger.info(f"  language: {lora_inv['language_module_count']}")
    logger.info(f"  vision: {lora_inv['vision_module_count']}")
    logger.info(f"  projector: {lora_inv['projector_module_count']}")
    logger.info(f"  connector: {lora_inv['connector_module_count']}")
    logger.info(f"Selected params: {lora_inv['selected_parameter_count']:,}")
    logger.info(f"Total params: {lora_inv['total_parameter_count']:,}")
    logger.info(f"Inventory SHA: {lora_inv['inventory_sha256']}")

    lora_path = OUTPUT_DIR / "qwen35_9b_lora_target_inventory.json"
    with open(lora_path, "w") as f:
        json.dump(lora_inv, f, indent=2)
    logger.info(f"Written: {lora_path}")

    # 4. Structural verification
    logger.info("\n--- Structural Verification ---")
    struct_errors = verify_structural(model, profile)
    if struct_errors:
        logger.error("STRUCTURAL VALIDATION FAILED:")
        for e in struct_errors:
            logger.error(f"  - {e}")
    else:
        logger.info("Structural validation PASSED")

    # 5. Pad token verification
    logger.info("\n--- Pad Token Verification ---")
    pad_id = verify_pad_token(processor)
    logger.info(f"Pad token ID from processor: {pad_id}")

    # 6. Language layer path verification
    logger.info("\n--- Language Layer Path ---")
    try:
        layers = model.model.language_model.layers
        logger.info(f"model.model.language_model.layers: {len(layers)} layers — OK")
    except AttributeError as e:
        logger.error(f"Path model.model.language_model.layers FAILED: {e}")

    # 7. Summary
    logger.info("\n" + "=" * 60)
    logger.info("P0-R13 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Profile SHA: {profile_sha}")
    logger.info(f"Model revision: {MODEL_REVISION}")
    logger.info(f"Structural errors: {len(struct_errors)}")
    logger.info(f"LoRA targets: {lora_inv['selected_module_count']}")
    logger.info(f"Vision targets: {lora_inv['vision_module_count']}")
    logger.info(f"Pad token: {pad_id}")
    logger.info(f"LoRA inventory SHA: {lora_inv['inventory_sha256']}")

    all_pass = (
        len(struct_errors) == 0
        and lora_inv["vision_module_count"] == 0
        and lora_inv["projector_module_count"] == 0
        and lora_inv["connector_module_count"] == 0
        and lora_inv["selected_module_count"] > 0
        and pad_id is not None
        and module_inv["language_layer_count"] == 32
        and module_inv["hidden_size"] == 4096
        and module_inv["intermediate_size"] == 12288
    )
    logger.info(f"\nP0-R13 OVERALL: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
