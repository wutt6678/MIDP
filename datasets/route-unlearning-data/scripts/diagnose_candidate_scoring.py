#!/usr/bin/env python3
"""Diagnose candidate scoring for a model adapter.

Prints prefix length, Yes/No token IDs, sequence scores, finiteness,
and a brute-force/manual forward equivalence result.

Usage::

    python scripts/diagnose_candidate_scoring.py --model-profile configs/models/unlearning/qwen35_9b.yaml

Requires GPU access and the model weights to be available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose candidate scoring for a model adapter",
    )
    parser.add_argument(
        "--model-profile", required=True,
        help="Path to model profile YAML.",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for model loading (default: cuda:0).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    profile_path = Path(args.model_profile)
    if not profile_path.is_absolute():
        profile_path = project_root / profile_path

    from route_data.models.trainable.registry import (
        create_adapter,
        load_profile_from_yaml,
    )

    profile = load_profile_from_yaml(profile_path)
    adapter = create_adapter(profile.key)

    print(f"Model: {profile.key} ({profile.model_id})")
    print(f"Revision: {profile.revision[:12]}...")
    print()

    # Load model and processor
    print("Loading model and processor...")
    model, processor = adapter.load_model_processor(
        model_id=profile.model_id,
        revision=profile.revision,
        processor_revision=profile.processor_revision,
        dtype=profile.dtype,
        device=args.device,
        training=False,
    )
    print("Model loaded.")
    print()

    # Create a dummy image for testing
    from PIL import Image
    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    test_prompt = "Is this an image of a cat?"

    # Build prefix
    print("Building prefix...")
    prefix = adapter.build_prefix(
        processor, image=dummy_image, prompt=test_prompt,
    )
    prefix_len = prefix["_prefix_len"]
    print(f"  prefix_len = {prefix_len}")
    print(f"  input_ids shape = {prefix['input_ids'].shape}")
    print(f"  keys = {sorted(k for k in prefix if not k.startswith('_'))}")
    print()

    # Resolve candidate token IDs
    yes_ids = adapter.candidate_token_ids(processor, profile.candidate_positive)
    no_ids = adapter.candidate_token_ids(processor, profile.candidate_negative)
    print(f"  Yes token IDs: {yes_ids}")
    print(f"  No token IDs:  {no_ids}")
    print(f"  Yes and No differ: {yes_ids != no_ids}")
    print()

    # Build full supervised example
    print("Building supervised example...")
    supervised = adapter.build_supervised_example(
        processor, image=dummy_image, prompt=test_prompt,
        answer_text="Yes",
    )
    sup_prefix_len = supervised["_prefix_len"]
    print(f"  supervised prefix_len = {sup_prefix_len}")
    print(f"  input_ids shape = {supervised['input_ids'].shape}")
    print(f"  labels shape = {supervised['labels'].shape}")

    # P0 gate: prefix alignment
    prefix_ids = prefix["input_ids"]
    if prefix_ids.dim() == 2:
        prefix_ids = prefix_ids[0]
    full_ids = supervised["input_ids"][:sup_prefix_len]
    alignment_ok = torch.equal(prefix_ids.to(full_ids.device), full_ids)
    print(f"  P0 prefix alignment: {'PASS' if alignment_ok else 'FAIL'}")
    print()

    # Score candidates
    from route_data.models.scoring import score_candidate_sequence_tensor

    print("Scoring candidates...")
    with torch.no_grad():
        log_p_yes = score_candidate_sequence_tensor(
            model, prefix, yes_ids, adapter=adapter,
        )
        log_p_no = score_candidate_sequence_tensor(
            model, prefix, no_ids, adapter=adapter,
        )

    print(f"  log P(Yes) = {log_p_yes.item():.6f}")
    print(f"  log P(No)  = {log_p_no.item():.6f}")
    print(f"  margin     = {(log_p_yes - log_p_no).item():.6f}")
    print(f"  log P(Yes) finite: {torch.isfinite(log_p_yes).item()}")
    print(f"  log P(No) finite:  {torch.isfinite(log_p_no).item()}")
    print()

    # Verify prefix alignment
    print("Verifying prefix alignment via adapter method...")
    verified = adapter.verify_prefix_alignment(
        processor, image=dummy_image, prompt=test_prompt,
    )
    print(f"  verify_prefix_alignment: {'PASS' if verified else 'FAIL'}")
    print()

    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
