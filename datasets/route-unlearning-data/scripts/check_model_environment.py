#!/usr/bin/env python3
"""Check model environment compatibility.

Emits Python, PyTorch, Transformers, PEFT, Accelerate, CUDA versions,
model key, declared minimum version, and whether the current environment
satisfies the profile.

Usage::

    python scripts/check_model_environment.py configs/models/unlearning/glm46v_flash.yaml
    python scripts/check_model_environment.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _get_versions() -> dict[str, str]:
    """Collect installed package versions."""
    versions: dict[str, str] = {}

    versions["python"] = sys.version.split()[0]

    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["cuda_version"] = torch.version.cuda or "N/A"
            versions["gpu_count"] = str(torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                versions[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem
                versions[f"gpu_{i}_total_memory_gb"] = f"{mem / 1e9:.1f}"
    except ImportError:
        versions["torch"] = "NOT INSTALLED"

    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "NOT INSTALLED"

    try:
        import peft
        versions["peft"] = peft.__version__
    except ImportError:
        versions["peft"] = "NOT INSTALLED"

    try:
        import accelerate
        versions["accelerate"] = accelerate.__version__
    except ImportError:
        versions["accelerate"] = "NOT INSTALLED"

    return versions


def _check_compatibility(
    versions: dict[str, str],
    min_transformers: str | None,
) -> bool:
    """Check if the current environment satisfies the profile requirements."""
    if min_transformers is None:
        return True

    try:
        from packaging.version import Version
        current = Version(versions.get("transformers", "0.0.0"))
        minimum = Version(min_transformers)
        return current >= minimum
    except Exception:
        # Fallback: simple string comparison
        return versions.get("transformers", "0") >= min_transformers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check model environment compatibility",
    )
    parser.add_argument(
        "profiles", nargs="*",
        help="Model profile YAML files to check.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Check all profiles in configs/models/unlearning/.",
    )
    args = parser.parse_args()

    versions = _get_versions()

    print("=" * 60)
    print("Environment Versions")
    print("=" * 60)
    for key, val in sorted(versions.items()):
        print(f"  {key:30s} {val}")
    print()

    if args.all:
        project_root = Path(__file__).resolve().parent.parent
        config_dir = project_root / "configs" / "models" / "unlearning"
        args.profiles = sorted(config_dir.glob("*.yaml"))
        # Exclude support_matrix.yaml
        args.profiles = [p for p in args.profiles if p.name != "support_matrix.yaml"]

    if not args.profiles:
        print("No profiles specified. Use --all or pass profile YAML paths.")
        return

    from route_data.models.trainable.registry import load_profile_from_yaml

    for profile_path in args.profiles:
        profile_path = Path(profile_path)
        if not profile_path.exists():
            print(f"SKIP: {profile_path} (not found)")
            continue

        try:
            profile = load_profile_from_yaml(profile_path)
        except Exception as e:
            print(f"ERROR loading {profile_path.name}: {e}")
            continue

        compatible = _check_compatibility(
            versions, profile.min_transformers_version,
        )

        status = "OK" if compatible else "FAIL"
        print(f"[{status}] {profile.key}")
        print(f"  model_id:       {profile.model_id}")
        print(f"  adapter:        {profile.adapter_name}")
        print(f"  min_transformers: {profile.min_transformers_version or 'none'}")
        print(f"  trust_remote_code: {profile.trust_remote_code}")
        print(f"  requires_hf_auth:  {profile.requires_hf_auth}")
        print()


if __name__ == "__main__":
    main()
