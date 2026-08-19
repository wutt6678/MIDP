#!/usr/bin/env python3
"""Generic model inventory generator for trainable model adapters.

Works with any model profile registered in the adapter registry.
Supports two modes:

- ``--discovery``: Observe runtime topology without requiring the
  profile to already contain final structural metadata.
- ``--verify``: Require exact runtime equality with the frozen profile.

Usage::

    # Discovery mode (while placeholders still exist)
    python scripts/generate_trainable_model_inventory.py \\
        --model-profile configs/models/unlearning/qwen35_4b.yaml \\
        --device cuda:0 \\
        --discovery

    # Verification mode (after profile is frozen)
    python scripts/generate_trainable_model_inventory.py \\
        --model-profile configs/models/unlearning/qwen35_9b.yaml \\
        --device cuda:0 \\
        --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "model_inventories"


def json_dumps_safe(obj: Any) -> str:
    """Deterministic JSON serialization for SHA-256."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_profile(profile_path: str):
    from route_data.models.trainable.registry import (
        compute_profile_sha256,
        load_profile_from_yaml,
    )
    profile = load_profile_from_yaml(profile_path)
    profile_sha = compute_profile_sha256(profile_path)
    return profile, profile_sha


def create_adapter(profile):
    from route_data.models.trainable.registry import create_adapter
    return create_adapter(profile.key, profile=profile)


def load_model_and_processor(adapter, device: str):
    from route_data.eval.unlearning_harness import load_base_model_via_adapter
    model, processor = load_base_model_via_adapter(
        adapter, device=device, training=False,
    )
    model.eval()
    return model, processor


def inventory_modules(model) -> dict:
    """Full named_modules inventory."""
    modules = []
    for name, module in model.named_modules():
        if name:
            modules.append({
                "name": name,
                "type": type(module).__name__,
                "n_params": sum(
                    p.numel() for p in module.parameters(recurse=False)
                ),
            })
    return {
        "total_module_count": len(modules),
        "modules": modules,
    }


def inventory_language_layers(model, adapter) -> dict:
    """Per-layer attention type and projection inventory."""
    try:
        layers = adapter.language_layers(model)
    except (RuntimeError, AttributeError) as exc:
        return {"error": str(exc), "layers": []}

    lang_cfg = adapter.language_config(model)
    inventory = []
    for i, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        linear_attn = getattr(layer, "linear_attn", None)

        attn_type = type(self_attn).__name__ if self_attn else "none"
        linear_attn_type = type(linear_attn).__name__ if linear_attn else None

        has_q = hasattr(self_attn, "q_proj") if self_attn else False
        has_k = hasattr(self_attn, "k_proj") if self_attn else False
        has_v = hasattr(self_attn, "v_proj") if self_attn else False
        has_o = hasattr(self_attn, "o_proj") if self_attn else False

        mlp = getattr(layer, "mlp", None)
        mlp_type = type(mlp).__name__ if mlp else None
        mlp_projections = {}
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                mlp_projections[proj_name] = hasattr(mlp, proj_name)

        inventory.append({
            "layer_index": i,
            "attention_type": attn_type,
            "linear_attention_type": linear_attn_type,
            "has_q_proj": has_q,
            "has_k_proj": has_k,
            "has_v_proj": has_v,
            "has_o_proj": has_o,
            "mlp_type": mlp_type,
            "mlp_projections": mlp_projections,
        })
    return {
        "language_layer_count": len(layers),
        "hidden_size": getattr(lang_cfg, "hidden_size", None),
        "intermediate_size": getattr(lang_cfg, "intermediate_size", None),
        "num_hidden_layers_config": getattr(lang_cfg, "num_hidden_layers", None),
        "layers": inventory,
    }


def inventory_processor(processor, adapter) -> dict:
    """Processor key inventory and token IDs."""
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)

    # Resolve candidate token IDs
    yes_ids = []
    no_ids = []
    try:
        yes_ids = adapter.candidate_token_ids(processor, "Yes")
    except Exception:
        pass
    try:
        no_ids = adapter.candidate_token_ids(processor, "No")
    except Exception:
        pass

    return {
        "processor_type": type(processor).__name__,
        "tokenizer_type": type(tokenizer).__name__,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "bos_token_id": bos_id,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
    }


def inventory_lora_candidates(model, adapter) -> dict:
    """LoRA target candidate inventory with classification."""
    targets = adapter.resolve_lora_targets(model)

    language_count = 0
    vision_count = 0
    projector_count = 0
    connector_count = 0
    other_count = 0

    for name in targets:
        if "language_model" in name or "text_model" in name:
            language_count += 1
        elif "visual" in name or "vision" in name:
            vision_count += 1
        elif "projector" in name or "multi_modal_projector" in name:
            projector_count += 1
        elif "connector" in name:
            connector_count += 1
        else:
            other_count += 1

    total_params = sum(p.numel() for p in model.parameters())
    named = dict(model.named_modules())
    selected_params = 0
    for name in targets:
        mod = named.get(name)
        if mod is not None:
            selected_params += sum(p.numel() for p in mod.parameters())

    p = adapter.profile
    inventory = {
        "model_key": p.key,
        "model_id": p.model_id,
        "model_revision": p.revision,
        "adapter_family": p.adapter_name,
        "scope_regex": p.lora_scope_regex,
        "target_leaf_names": list(p.lora_target_leaf_names),
        "selected_module_count": len(targets),
        "selected_modules": sorted(targets),
        "language_module_count": language_count,
        "vision_module_count": vision_count,
        "projector_module_count": projector_count,
        "connector_module_count": connector_count,
        "other_module_count": other_count,
        "selected_parameter_count": selected_params,
        "total_parameter_count": total_params,
    }
    inv_sha = sha256_str(json_dumps_safe(inventory))
    inventory["inventory_sha256"] = inv_sha
    return inventory


def compute_r2mu_candidates(n_layers: int) -> list[int]:
    """Compute R²MU candidate layers at ~25%, ~50%, ~75%, ~90% depth."""
    fractions = [0.25, 0.50, 0.75, 0.90]
    return [round(n_layers * f) for f in fractions]


def build_model_summary(
    model, processor, adapter, profile, profile_sha: str,
    layer_inv: dict, lora_inv: dict, proc_inv: dict,
    mode: str,
) -> dict:
    """Build the discovery/verification summary."""
    model_class = type(model).__name__

    # Structural metadata from runtime
    lang_cfg = adapter.language_config(model)
    hidden_size = getattr(lang_cfg, "hidden_size", None)
    intermediate_size = getattr(lang_cfg, "intermediate_size", None)
    n_layers = layer_inv.get("language_layer_count")

    # Attention topology
    attn_types = {}
    full_attn_indices = []
    linear_attn_indices = []
    for l in layer_inv.get("layers", []):
        t = l["attention_type"]
        attn_types[t] = attn_types.get(t, 0) + 1
        if l.get("has_q_proj") and l.get("has_o_proj"):
            full_attn_indices.append(l["layer_index"])
        elif l.get("linear_attention_type"):
            linear_attn_indices.append(l["layer_index"])

    r2mu = compute_r2mu_candidates(n_layers) if n_layers else []

    summary = {
        "mode": mode,
        "model_key": profile.key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "processor_id": profile.processor_id,
        "processor_revision": profile.processor_revision,
        "adapter_family": profile.adapter_name,
        "profile_sha256": profile_sha,
        "model_class": model_class,
        "processor_class": proc_inv.get("processor_type", ""),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_language_layers": n_layers,
        "language_layer_path": "model.language_model.layers",
        "attention_type_distribution": attn_types,
        "full_attention_layer_indices": full_attn_indices,
        "linear_attention_layer_indices": linear_attn_indices,
        "full_attention_count": len(full_attn_indices),
        "linear_attention_count": len(linear_attn_indices),
        "r2mu_candidate_layers": r2mu,
        "pad_token_id": proc_inv.get("pad_token_id"),
        "lora_target_count": lora_inv.get("selected_module_count", 0),
        "lora_language_count": lora_inv.get("language_module_count", 0),
        "lora_vision_count": lora_inv.get("vision_module_count", 0),
        "lora_projector_count": lora_inv.get("projector_module_count", 0),
        "lora_connector_count": lora_inv.get("connector_module_count", 0),
        "lora_inventory_sha256": lora_inv.get("inventory_sha256", ""),
    }

    # Verification checks
    if mode == "verify":
        checks = []
        if profile.num_language_layers > 0 and n_layers != profile.num_language_layers:
            checks.append(
                f"num_language_layers: runtime={n_layers} "
                f"profile={profile.num_language_layers}"
            )
        if profile.language_hidden_size > 0 and hidden_size != profile.language_hidden_size:
            checks.append(
                f"hidden_size: runtime={hidden_size} "
                f"profile={profile.language_hidden_size}"
            )
        if profile.intermediate_size > 0 and intermediate_size != profile.intermediate_size:
            checks.append(
                f"intermediate_size: runtime={intermediate_size} "
                f"profile={profile.intermediate_size}"
            )
        summary["verification_errors"] = checks
        summary["verification_pass"] = len(checks) == 0

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generic trainable model inventory generator"
    )
    parser.add_argument(
        "--model-profile", required=True,
        help="Path to model profile YAML",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device to load model on",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--discovery", action="store_true",
        help="Discovery mode: observe without requiring frozen profile",
    )
    group.add_argument(
        "--verify", action="store_true",
        help="Verification mode: require exact match with frozen profile",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    mode = "discovery" if args.discovery else "verify"
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_path = str(Path(args.model_profile).resolve())
    model_key = Path(args.model_profile).stem

    logger.info(f"Mode: {mode}")
    logger.info(f"Profile: {profile_path}")

    # Load profile and adapter
    profile, profile_sha = load_profile(profile_path)
    adapter = create_adapter(profile)
    logger.info(f"Profile SHA: {profile_sha}")
    logger.info(f"Model key: {profile.key}")
    logger.info(f"Adapter family: {profile.adapter_name}")

    # Load model
    logger.info(f"Loading model on {args.device} ...")
    t0 = time.time()
    model, processor = load_model_and_processor(adapter, args.device)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s: {type(model).__name__}")

    # Module inventory
    logger.info("\n--- Module Inventory ---")
    mod_inv = inventory_modules(model)
    logger.info(f"Total modules: {mod_inv['total_module_count']}")

    # Language layer inventory
    logger.info("\n--- Language Layer Inventory ---")
    layer_inv = inventory_language_layers(model, adapter)
    logger.info(f"Language layers: {layer_inv.get('language_layer_count')}")
    logger.info(f"Hidden size: {layer_inv.get('hidden_size')}")
    logger.info(f"Intermediate size: {layer_inv.get('intermediate_size')}")

    # Processor inventory
    logger.info("\n--- Processor Inventory ---")
    proc_inv = inventory_processor(processor, adapter)
    logger.info(f"Pad token: {proc_inv['pad_token_id']}")
    logger.info(f"Yes token IDs: {proc_inv['yes_token_ids']}")
    logger.info(f"No token IDs: {proc_inv['no_token_ids']}")

    # LoRA candidate inventory
    logger.info("\n--- LoRA Candidate Inventory ---")
    lora_inv = inventory_lora_candidates(model, adapter)
    logger.info(f"Selected: {lora_inv['selected_module_count']}")
    logger.info(f"  language: {lora_inv['language_module_count']}")
    logger.info(f"  vision: {lora_inv['vision_module_count']}")
    logger.info(f"  projector: {lora_inv['projector_module_count']}")
    logger.info(f"  connector: {lora_inv['connector_module_count']}")
    logger.info(f"Inventory SHA: {lora_inv['inventory_sha256']}")

    # Summary
    logger.info("\n--- Summary ---")
    summary = build_model_summary(
        model, processor, adapter, profile, profile_sha,
        layer_inv, lora_inv, proc_inv, mode,
    )
    for k in (
        "model_class", "hidden_size", "intermediate_size",
        "num_language_layers", "pad_token_id",
        "full_attention_count", "linear_attention_count",
        "lora_target_count", "lora_vision_count",
    ):
        logger.info(f"  {k}: {summary[k]}")

    if mode == "verify":
        if summary.get("verification_pass"):
            logger.info("Verification: PASSED")
        else:
            logger.error("Verification: FAILED")
            for e in summary.get("verification_errors", []):
                logger.error(f"  - {e}")

    # Write outputs
    prefix = model_key

    mod_path = out_dir / f"{prefix}_module_inventory.json"
    with open(mod_path, "w") as f:
        json.dump(mod_inv, f, indent=2)
    logger.info(f"\nWritten: {mod_path}")

    layer_path = out_dir / f"{prefix}_layer_inventory.json"
    with open(layer_path, "w") as f:
        json.dump(layer_inv, f, indent=2)
    logger.info(f"Written: {layer_path}")

    proc_path = out_dir / f"{prefix}_processor_inventory.json"
    with open(proc_path, "w") as f:
        json.dump(proc_inv, f, indent=2)
    logger.info(f"Written: {proc_path}")

    lora_path = out_dir / f"{prefix}_lora_candidate_inventory.json"
    with open(lora_path, "w") as f:
        json.dump(lora_inv, f, indent=2)
    logger.info(f"Written: {lora_path}")

    summary_path = out_dir / f"{prefix}_discovery_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Written: {summary_path}")

    # Fail-closed exit
    if mode == "verify" and not summary.get("verification_pass"):
        logger.error("Discovery summary: FAIL")
        sys.exit(1)

    logger.info(f"\n{model_key} inventory generation complete ({mode} mode)")


if __name__ == "__main__":
    main()
