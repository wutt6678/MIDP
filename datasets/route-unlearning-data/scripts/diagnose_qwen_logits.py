#!/usr/bin/env python3
"""Diagnostic: inspect next-token logits at the assistant prefix for Yes/No variants.

Constructs the official multimodal prefix via ``processor.apply_chat_template``
and prints token IDs, log-probabilities, and decoded forms for six candidate
strings.  This reveals whether the current ``score_candidates()`` is broken
(token-index misalignment) or whether all variants are genuinely collapsed.

Usage:
    PYTHONPATH=src python scripts/diagnose_qwen_logits.py \
        --image /path/to/face.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a real FIUBench face image",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-9B",
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Examine only the current image.\n"
            "Is this person smiling?\n"
            "Answer exactly one word: Yes or No."
        ),
        help="Binary question prompt",
    )
    args = parser.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    image = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path} ({image.size})")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading {args.model_id} ...")
    local_dir = snapshot_download(args.model_id)
    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        local_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    tokenizer = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    print(f"Model on device: {device}")

    # ── Construct official prefix ───────────────────────────────────────
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.prompt},
        ],
    }]

    try:
        prefix = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        # Fallback for older transformers without enable_thinking
        prefix = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    prefix = {k: v.to(device) for k, v in prefix.items()}
    prefix_len = prefix["input_ids"].shape[1]
    print(f"\nPrefix length: {prefix_len} tokens")

    # Show last few prefix tokens for context
    last_tokens = prefix["input_ids"][0, -10:]
    print(f"Last 10 prefix tokens: {tokenizer.decode(last_tokens)!r}")

    # ── Forward pass on prefix only ─────────────────────────────────────
    with torch.inference_mode():
        outputs = model(**prefix)
    next_logits = outputs.logits[0, -1]  # [vocab_size]
    next_log_probs = torch.log_softmax(next_logits.float(), dim=-1)

    # ── Candidate variants ──────────────────────────────────────────────
    candidates = [
        "yes",
        "no",
        "Yes",
        "No",
        " yes",
        " no",
    ]

    print(f"\n{'candidate':<12} {'token_ids':<20} {'logP':>10} {'decoded':<20}")
    print("-" * 65)
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 0:
            print(f"{cand:<12} {'(empty)':<20} {'N/A':>10} {'':<20}")
            continue
        # For multi-token candidates, we'd need sequence scoring.
        # For single-token candidates, we can read directly from next_logits.
        if len(ids) == 1:
            tid = ids[0]
            lp = next_log_probs[tid].item()
            decoded = tokenizer.decode(ids)
            print(f"{cand:<12} {ids!s:<20} {lp:>10.4f} {decoded!r:<20}")
        else:
            # Multi-token: score as sequence from the prefix
            from route_data.models.scoring import gather_sequence_log_probs
            m = len(ids)
            # We need logits at positions prefix_len-1, prefix_len, ..., prefix_len+m-2
            # But we only have the prefix forward pass, which gives logits up to prefix_len-1
            # For multi-token we'd need to extend. Just show tokenization for now.
            decoded = tokenizer.decode(ids)
            # Score first token from next_logits
            first_tid = ids[0]
            first_lp = next_log_probs[first_tid].item()
            print(f"{cand:<12} {ids!s:<20} {first_lp:>10.4f} {decoded!r:<20}  (multi-token, first token logP shown)")

    # ── Also show top-10 next tokens ────────────────────────────────────
    print("\nTop-10 next tokens by log-probability:")
    top_vals, top_idxs = torch.topk(next_log_probs, k=10)
    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        tok_str = tokenizer.decode([idx])
        print(f"  token_id={idx:<8} logP={val:>8.4f}  decoded={tok_str!r}")

    # ── Compare with old score_candidates approach ──────────────────────
    print("\n--- Comparison: old score_candidates approach ---")
    for cand in [" yes", " no"]:
        rendered_messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }]
        try:
            rendered = processor.apply_chat_template(
                rendered_messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = processor.apply_chat_template(
                rendered_messages,
                add_generation_prompt=True,
            )
        batch = processor(
            images=[image],
            text=[rendered + cand],
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        m = len(cand_ids)
        target_ids = input_ids[0, -m:]
        with torch.inference_mode():
            old_out = model(**batch)
        pred_rows = old_out.logits[0, -m - 1:-1, :]
        from route_data.models.scoring import gather_sequence_log_probs
        log_prob = gather_sequence_log_probs(pred_rows, target_ids.to(pred_rows.device))
        print(f"  old score({cand!r}) = {log_prob:.6f}  (target_ids={target_ids.tolist()}, cand_ids={cand_ids})")

    # ── Free generation check ───────────────────────────────────────────
    print("\n--- Free generation ---")
    with torch.inference_mode():
        gen_out = model.generate(
            **prefix,
            do_sample=False,
            temperature=None,
            max_new_tokens=4,
        )
    gen_tokens = gen_out[0, prefix_len:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"  Generated: {gen_text!r}")
    print(f"  Gen token IDs: {gen_tokens.tolist()}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
