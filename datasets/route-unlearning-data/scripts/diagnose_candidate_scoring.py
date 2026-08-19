#!/usr/bin/env python3
"""Diagnose candidate scoring for a model adapter.

Implements both the shared scorer and an **independent** teacher-forced
reference scorer that does NOT call ``score_candidate_sequence_tensor``.
The two must agree within a frozen tolerance.

Usage::

    python scripts/diagnose_candidate_scoring.py \\
        --model-profile configs/models/unlearning/qwen35_9b.yaml

Requires GPU access and the model weights to be available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _batch_prefix(prefix: dict, device: str = "cpu") -> dict:
    """Ensure prefix tensors are batched (1, ...) and on the correct device."""
    batched = {}
    for key, val in prefix.items():
        if isinstance(val, torch.Tensor):
            if val.dim() == 1:
                val = val.unsqueeze(0)
            val = val.to(device)
        batched[key] = val
    return batched


def independent_reference_score(
    model: torch.nn.Module,
    prefix: dict,
    candidate_token_ids: list[int],
    *,
    adapter,
) -> torch.Tensor:
    """Independent teacher-forced reference scorer.

    This scorer does NOT call ``score_candidate_sequence_tensor``.
    It builds the forward kwargs manually and computes the sum of
    log-probabilities for the candidate tokens.
    """
    prefix_input_ids = prefix["input_ids"]
    prefix_len = prefix_input_ids.shape[1]
    device = prefix_input_ids.device
    dtype = prefix_input_ids.dtype

    cand_ids = torch.tensor(
        [candidate_token_ids], dtype=dtype, device=device,
    )

    # Build forward kwargs manually (not through adapter.append_candidate)
    full_input_ids = torch.cat([prefix_input_ids, cand_ids], dim=1)
    full_attention_mask = torch.cat(
        [prefix["attention_mask"], torch.ones_like(cand_ids)], dim=1,
    )

    forward_kwargs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
    }

    # Extend mm_token_type_ids for candidate tokens (text-only, type 0)
    if "mm_token_type_ids" in prefix:
        prefix_mm = prefix["mm_token_type_ids"]
        cand_mm = torch.zeros_like(cand_ids, dtype=prefix_mm.dtype)
        forward_kwargs["mm_token_type_ids"] = torch.cat([prefix_mm, cand_mm], dim=1)

    # Forward visual tensors from prefix
    for key in ("pixel_values", "image_sizes", "image_grid_thw"):
        if key in prefix:
            forward_kwargs[key] = prefix[key]

    # Run model directly
    outputs = model(**forward_kwargs)
    logits = outputs.logits  # [1, full_len, vocab]

    m = len(candidate_token_ids)

    # Extract prediction rows for candidate tokens
    pred_rows = logits[0, prefix_len - 1: prefix_len - 1 + m, :]
    target_ids = full_input_ids[0, prefix_len: prefix_len + m]

    # Compute log probabilities
    log_probs = torch.log_softmax(pred_rows.float(), dim=-1)
    gathered = log_probs.gather(
        -1, target_ids.to(log_probs.device).unsqueeze(-1),
    ).squeeze(-1)
    log_prob = gathered.sum()

    return log_prob


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
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path for diagnostic results.",
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
    adapter = create_adapter(profile.key, profile=profile)

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
    # Ensure prefix tensors are batched (1, ...) for the scorer
    prefix = _batch_prefix(prefix, device=args.device)
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

    # Score candidates using shared scorer
    from route_data.models.scoring import score_candidate_sequence_tensor

    print("Scoring candidates (shared scorer)...")
    with torch.no_grad():
        shared_yes = score_candidate_sequence_tensor(
            model, prefix, yes_ids, adapter=adapter,
        )
        shared_no = score_candidate_sequence_tensor(
            model, prefix, no_ids, adapter=adapter,
        )

    print(f"  Shared log P(Yes) = {shared_yes.item():.6f}")
    print(f"  Shared log P(No)  = {shared_no.item():.6f}")
    print(f"  Shared margin     = {(shared_yes - shared_no).item():.6f}")
    print(f"  Shared Yes finite: {torch.isfinite(shared_yes).item()}")
    print(f"  Shared No finite:  {torch.isfinite(shared_no).item()}")
    print()

    # Independent reference scorer
    print("Scoring candidates (independent reference)...")
    with torch.no_grad():
        ref_yes = independent_reference_score(
            model, prefix, yes_ids, adapter=adapter,
        )
        ref_no = independent_reference_score(
            model, prefix, no_ids, adapter=adapter,
        )

    print(f"  Reference log P(Yes) = {ref_yes.item():.6f}")
    print(f"  Reference log P(No)  = {ref_no.item():.6f}")
    print(f"  Reference margin     = {(ref_yes - ref_no).item():.6f}")
    print(f"  Reference Yes finite: {torch.isfinite(ref_yes).item()}")
    print(f"  Reference No finite:  {torch.isfinite(ref_no).item()}")
    print()

    # Compare shared vs reference
    atol, rtol = 1e-4, 1e-4
    yes_match = torch.allclose(shared_yes, ref_yes, atol=atol, rtol=rtol)
    no_match = torch.allclose(shared_no, ref_no, atol=atol, rtol=rtol)
    yes_diff = (shared_yes - ref_yes).abs().item()
    no_diff = (shared_no - ref_no).abs().item()

    print(f"  Yes match (atol={atol}, rtol={rtol}): {'PASS' if yes_match else 'FAIL'}")
    print(f"    diff = {yes_diff:.2e}")
    print(f"  No match (atol={atol}, rtol={rtol}): {'PASS' if no_match else 'FAIL'}")
    print(f"    diff = {no_diff:.2e}")
    print()

    # Verify prefix alignment via adapter method
    print("Verifying prefix alignment via adapter method...")
    verified = adapter.verify_prefix_alignment(
        processor, image=dummy_image, prompt=test_prompt,
    )
    print(f"  verify_prefix_alignment: {'PASS' if verified else 'FAIL'}")
    print()

    # Build diagnostic report
    report = {
        "model_key": profile.key,
        "model_id": profile.model_id,
        "model_revision": profile.revision,
        "prefix_len": prefix_len,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "yes_token_ids_non_empty": len(yes_ids) > 0,
        "no_token_ids_non_empty": len(no_ids) > 0,
        "yes_no_differ": yes_ids != no_ids,
        "shared_scorer": {
            "log_p_yes": shared_yes.item(),
            "log_p_no": shared_no.item(),
            "margin": (shared_yes - shared_no).item(),
            "yes_finite": torch.isfinite(shared_yes).item(),
            "no_finite": torch.isfinite(shared_no).item(),
        },
        "independent_reference": {
            "log_p_yes": ref_yes.item(),
            "log_p_no": ref_no.item(),
            "margin": (ref_yes - ref_no).item(),
            "yes_finite": torch.isfinite(ref_yes).item(),
            "no_finite": torch.isfinite(ref_no).item(),
        },
        "equivalence": {
            "atol": atol,
            "rtol": rtol,
            "yes_match": yes_match,
            "no_match": no_match,
            "yes_diff": yes_diff,
            "no_diff": no_diff,
        },
        "prefix_alignment_ok": alignment_ok,
        "verify_prefix_alignment": verified,
        "overall_pass": all([
            len(yes_ids) > 0,
            len(no_ids) > 0,
            yes_ids != no_ids,
            torch.isfinite(shared_yes).item(),
            torch.isfinite(shared_no).item(),
            torch.isfinite(ref_yes).item(),
            torch.isfinite(ref_no).item(),
            yes_match,
            no_match,
            alignment_ok,
            verified,
        ]),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written: {out_path}")

    status = "PASS" if report["overall_pass"] else "FAIL"
    print(f"\nDiagnostic complete: {status}")

    if not report["overall_pass"]:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
