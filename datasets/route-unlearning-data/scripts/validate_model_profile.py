#!/usr/bin/env python3
"""Validate a model profile YAML for research-mode use.

Checks
------
- Schema valid (all required fields present, correct types)
- Revisions are exact 40-hex SHA (not placeholders, not branch names)
- No unresolved ``<...>`` placeholders for enabled capabilities
- Candidate strings non-empty and different
- LoRA config sane (rank > 0, alpha > 0, scope_regex non-empty)
- R²MU layer list typed correctly (list[int])
- Support flags explicit (bool)
- Environment version compatible
- Adapter family can be resolved

Does NOT require full model weights.

Usage::

    python scripts/validate_model_profile.py \\
        --model-profile configs/models/unlearning/qwen35_9b.yaml

    # Validate all profiles
    python scripts/validate_model_profile.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def validate_profile(path: Path, *, verbose: bool = True) -> bool:
    """Validate a single profile YAML.  Returns True if valid."""
    from route_data.models.trainable.registry import (
        _ADAPTER_FAMILIES,
        _MODEL_KEY_TO_FAMILY,
        _ensure_builtin_adapters_loaded,
        compute_profile_sha256,
        load_profile_from_yaml,
        validate_research_profile,
    )

    if verbose:
        print(f"Validating: {path}")

    # --- Load ---
    try:
        profile = load_profile_from_yaml(path)
    except Exception as exc:
        print(f"  FAIL: schema invalid — {exc}")
        return False

    sha = compute_profile_sha256(path)
    if verbose:
        print(f"  model_key      = {profile.key}")
        print(f"  model_id       = {profile.model_id}")
        print(f"  profile_sha256 = {sha}")

    # --- Research validation ---
    errors = validate_research_profile(profile)
    if errors:
        for err in errors:
            print(f"  FAIL: {err}")
        return False

    # --- Adapter family resolution ---
    _ensure_builtin_adapters_loaded()
    if profile.key not in _MODEL_KEY_TO_FAMILY:
        print(f"  FAIL: model key {profile.key!r} not registered")
        return False

    family = _MODEL_KEY_TO_FAMILY[profile.key]
    if family not in _ADAPTER_FAMILIES:
        print(f"  FAIL: adapter family {family!r} not registered")
        return False

    if verbose:
        print(f"  adapter_family = {family}")

    # --- LoRA sanity ---
    if profile.lora_rank <= 0:
        print(f"  FAIL: lora_rank must be > 0, got {profile.lora_rank}")
        return False
    if profile.lora_alpha <= 0:
        print(f"  FAIL: lora_alpha must be > 0, got {profile.lora_alpha}")
        return False
    if not profile.lora_scope_regex:
        print("  WARN: lora_scope_regex is empty")

    # --- Environment compatibility ---
    if profile.min_transformers_version:
        try:
            import transformers
            from packaging.version import Version
            installed = Version(transformers.__version__)
            required = Version(profile.min_transformers_version)
            if installed < required:
                print(
                    f"  FAIL: transformers {installed} < "
                    f"required {required}"
                )
                return False
            if verbose:
                print(f"  transformers   = {installed} (>= {required})")
        except ImportError:
            print("  WARN: packaging not available; skipping version check")

    if verbose:
        print("  PASS")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a model profile YAML for research use",
    )
    parser.add_argument(
        "--model-profile",
        help="Path to model profile YAML",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all built-in profiles",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    verbose = not args.quiet

    if args.all:
        profiles_dir = project_root / "configs" / "models" / "unlearning"
        if not profiles_dir.is_dir():
            print(f"ERROR: profiles directory not found: {profiles_dir}")
            sys.exit(1)
        paths = sorted(profiles_dir.glob("*.yaml"))
        paths = [p for p in paths if p.name != "support_matrix.yaml"]
    elif args.model_profile:
        p = Path(args.model_profile)
        if not p.is_absolute():
            p = project_root / p
        paths = [p]
    else:
        parser.error("Provide --model-profile or --all")
        return

    all_pass = True
    for path in paths:
        if not path.is_file():
            print(f"ERROR: file not found: {path}")
            all_pass = False
            continue
        ok = validate_profile(path, verbose=verbose)
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\nAll {len(paths)} profile(s) valid.")
    else:
        print("\nSome profiles FAILED validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
