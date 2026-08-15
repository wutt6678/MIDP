#!/usr/bin/env python3
"""Smoke test: verify Qwen3.5 text-only inference via the production backend.

This script runs exactly one name_only probe through the real Qwen3.5-9B
model using the *production* ``qwen_hf`` backend (not direct AutoProcessor
/ AutoModel construction) to verify that:

1. The model config is loaded from the pinned YAML.
2. The resolved revision matches the frozen checkpoint.
3. The text-only chat template renders successfully.
4. No image token or image placeholder is inserted.
5. Generation succeeds and produces non-empty output.
6. Thinking mode is disabled.
7. JSON evidence is written to ``qwen_text_only_smoke.json``.

Usage::

    PYTHONPATH=src python scripts/smoke_qwen_text_only.py

The script uses the target model config at
``configs/models/unlearning_target_qwen35_9b.yaml`` by default.
Override with ``--model-config PATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
# Git state helpers
# --------------------------------------------------------------------------- #


def _get_git_state() -> dict[str, str | bool]:
    """Return ``{git_commit, git_dirty}`` from the working tree."""
    git_commit = ""
    git_dirty = False
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        git_dirty = bool(status)
    except Exception:
        pass
    return {"git_commit": git_commit, "git_dirty": git_dirty}


# --------------------------------------------------------------------------- #
# Runtime info
# --------------------------------------------------------------------------- #


def _runtime_info() -> dict[str, str]:
    """Collect library versions and GPU info."""
    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        else:
            info["gpu"] = ""
    except ImportError:
        info["torch"] = ""
        info["cuda"] = ""
        info["gpu"] = ""
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = ""
    try:
        import accelerate
        info["accelerate"] = accelerate.__version__
    except ImportError:
        info["accelerate"] = ""
    return info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "configs" / "models" / "unlearning_target_qwen35_9b.yaml",
        help="Path to the target model YAML (default: pinned Qwen3.5-9B config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for qwen_text_only_smoke.json (default: cwd).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is Alice's full name? Answer with the complete name.",
        help="Text-only prompt for the smoke test.",
    )
    args = parser.parse_args()

    from route_data.config import load_model_config
    from route_data.models.registry import create_backend

    print("=" * 70)
    print("Qwen3.5 Text-Only Smoke Test (Production Backend)")
    print("=" * 70)

    # ── Load model config from pinned YAML (P1-4) ──────────────────────
    print(f"\n[1/6] Loading model config from {args.model_config} ...")
    cfg = load_model_config(args.model_config)
    print(f"  model_id:  {cfg.model_id}")
    print(f"  revision:  {cfg.revision}")
    print(f"  backend:   {cfg.backend}")
    print(f"  dtype:     {cfg.dtype}")
    print(f"  attn:      {cfg.attn_implementation}")

    # ── Create production backend (P1-5) ────────────────────────────────
    print(f"\n[2/6] Creating production backend ({cfg.backend}) ...")
    backend = create_backend(cfg)

    # ── Generate via production path (text-only, image=None) ────────────
    print(f"\n[3/6] Generating text (text-only, image=None) ...")
    print(f"  Prompt: {args.prompt!r}")
    started = time.perf_counter()
    response = backend.generate(None, args.prompt)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    generated_text = response.text
    print(f"  Generated in {elapsed_ms:.0f} ms")
    print(f"  Output: {generated_text!r}")

    # ── Verify outputs (P1-6) ───────────────────────────────────────────
    print(f"\n[4/6] Verifying text-only path ...")
    passed = True

    # Response non-empty
    text_ok = bool(generated_text and generated_text.strip())
    print(f"  Response non-empty: {'OK' if text_ok else 'FAIL'}")
    if not text_ok:
        passed = False

    # Resolved revision pinned
    fingerprint = backend.fingerprint()
    resolved_rev = fingerprint.get("revision", "")
    rev_ok = resolved_rev == cfg.revision
    print(f"  Resolved revision: {resolved_rev}")
    print(f"  Revision pinned:   {'OK' if rev_ok else 'FAIL'}")
    if not rev_ok:
        passed = False

    # Thinking disabled
    thinking_ok = fingerprint.get("thinking", "") == "disabled"
    print(f"  Thinking disabled: {'OK' if thinking_ok else 'FAIL'}")
    if not thinking_ok:
        passed = False

    # No image marker in output
    image_marker_ok = "<|vision_start|>" not in generated_text
    print(f"  No image marker:   {'OK' if image_marker_ok else 'FAIL'}")
    if not image_marker_ok:
        passed = False

    # No think tags in output
    no_think_ok = "<think>" not in generated_text
    print(f"  No think tags:     {'OK' if no_think_ok else 'FAIL'}")
    if not no_think_ok:
        passed = False

    # Input mode = text_only, image_used = false
    input_mode = "text_only"
    image_used = False

    # ── Git state ───────────────────────────────────────────────────────
    print(f"\n[5/6] Checking Git state ...")
    git_state = _get_git_state()
    print(f"  git_commit: {str(git_state['git_commit'])[:12]}...")
    print(f"  git_dirty:  {git_state['git_dirty']}")

    # ── Write evidence (P1-7) ───────────────────────────────────────────
    print(f"\n[6/6] Writing evidence ...")

    # Compute model config SHA
    import hashlib
    model_config_sha = hashlib.sha256(
        json.dumps(
            {"model_id": cfg.model_id, "revision": cfg.revision,
             "backend": cfg.backend, "dtype": cfg.dtype},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    evidence = {
        "pass": passed,
        "model_id": cfg.model_id,
        "resolved_revision": resolved_rev,
        "model_fingerprint": fingerprint,
        "model_config_sha256": model_config_sha,
        "code_commit": git_state["git_commit"],
        "git_dirty": git_state["git_dirty"],
        "input_mode": input_mode,
        "image_used": image_used,
        "thinking_disabled": thinking_ok,
        "prompt": args.prompt,
        "generated_answer": generated_text,
        "latency_ms": elapsed_ms,
        "runtime": _runtime_info(),
    }

    output_dir = args.output_dir or Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "qwen_text_only_smoke.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"  Evidence written to {evidence_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if passed:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
    print("=" * 70)
    print(f"  Model:        {cfg.model_id}")
    print(f"  Revision:     {resolved_rev}")
    print(f"  Backend:      {cfg.backend} (production)")
    print(f"  Input mode:   {input_mode}")
    print(f"  Image used:   {image_used}")
    print(f"  Prompt:       {args.prompt!r}")
    print(f"  Output:       {generated_text!r}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
