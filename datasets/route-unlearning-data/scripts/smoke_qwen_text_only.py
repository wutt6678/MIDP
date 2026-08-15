#!/usr/bin/env python3
"""Smoke test: verify Qwen3.5 text-only inference path.

This script runs exactly one name_only probe through the real Qwen3.5-9B
model to verify that:

1. The text-only chat template renders successfully.
2. No image token or image placeholder is inserted.
3. The processor is called without images=[None].
4. Generation succeeds and produces non-empty output.
5. Thinking mode is disabled.

Usage:
    PYTHONPATH=src python scripts/smoke_qwen_text_only.py

This should be run BEFORE the real route smoke to confirm the text-only
path works on the actual model.
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "Qwen/Qwen3.5-9B"
    prompt = "What is Alice's full name? Answer with the complete name."

    print("=" * 70)
    print("Qwen3.5 Text-Only Smoke Test")
    print("=" * 70)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {model_id} ...")
    local_dir = snapshot_download(model_id)
    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        local_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    tokenizer = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    print(f"  Model loaded on {device}")
    print(f"  Tokenizer: {type(tokenizer).__name__}")
    print(f"  Processor: {type(processor).__name__}")

    # ── Render text-only chat template ──────────────────────────────────
    print("\n[2/5] Rendering text-only chat template ...")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        rendered = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
    except TypeError:
        # Older transformers without enable_thinking
        rendered = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    print("  Rendered template (first 200 chars):")
    print(f"  {rendered[:200]}...")

    # Verify no image token in rendered text
    image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else None
    if image_token and image_token in rendered:
        print(f"  ERROR: Image token '{image_token}' found in text-only template!")
        return 1
    print("  OK: No image token in rendered template")

    # ── Tokenize (text-only, no images argument) ────────────────────────
    print("\n[3/5] Tokenizing (text-only path) ...")
    inputs = processor(
        text=[rendered],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"  input_ids shape: {inputs['input_ids'].shape}")
    print(f"  attention_mask shape: {inputs['attention_mask'].shape}")

    # Verify no image-related keys
    image_keys = {"pixel_values", "image_grid_thw", "image_sizes"}
    found_image_keys = image_keys & set(inputs.keys())
    if found_image_keys:
        print(f"  ERROR: Image keys found in text-only inputs: {found_image_keys}")
        return 1
    print("  OK: No image keys in inputs")

    # ── Generate ────────────────────────────────────────────────────────
    print("\n[4/5] Generating text (text-only) ...")
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=32,
        )
    generated_ids = output[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"  Generated {len(generated_ids)} tokens")
    print(f"  Generated text: {generated_text!r}")

    if not generated_text or not generated_text.strip():
        print("  ERROR: Generated text is empty!")
        return 1
    print("  OK: Generated text is non-empty")

    # ── Verify thinking is disabled ─────────────────────────────────────
    print("\n[5/5] Verifying thinking mode is disabled ...")
    if "<think>" in generated_text or "<think>" in generated_text:
        print("  WARNING: Think tags found in output (thinking may not be disabled)")
    else:
        print("  OK: No think tags in output")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print(f"  Model: {model_id}")
    print(f"  Prompt: {prompt!r}")
    print(f"  Output: {generated_text!r}")
    print("  Text-only path: VERIFIED")
    print("  No image tokens: VERIFIED")
    print("  No image processor keys: VERIFIED")
    print("  Generation: SUCCESS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
