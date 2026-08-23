"""E2C provenance capture.

Every generated condition must bind to:
    code SHA, git state, base model, revision, processor/tokenizer fingerprints,
    LoRA config, seed, optimizer config, all manifest SHAs, all dataset SHAs,
    all adapter SHAs.

Writes:
    e2c/outputs/<code_sha>/<condition>/provenance.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .synthetic_manifest import sha256_json

# --------------------------------------------------------------------------- #
# Git state
# --------------------------------------------------------------------------- #

def get_git_state(repo_dir: str | Path | None = None) -> dict[str, str]:
    """Capture git commit SHA and clean/dirty state."""
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": 10,
    }
    if repo_dir:
        kwargs["cwd"] = str(repo_dir)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], **kwargs, check=False,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], **kwargs, check=False,
        ).stdout.strip()
        dirty = len(status) > 0
    except Exception:
        dirty = True

    return {
        "code_sha": commit,
        "dirty": str(dirty),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------- #
# Model fingerprint
# --------------------------------------------------------------------------- #

def capture_model_fingerprint(
    *,
    model_id: str,
    revision: str,
    processor_id: str,
    processor_revision: str,
    dtype: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> dict[str, str]:
    """Capture model identification fingerprint."""
    return {
        "model_id": model_id,
        "model_revision": revision,
        "processor_id": processor_id,
        "processor_revision": processor_revision,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "trust_remote_code": str(trust_remote_code),
    }


def capture_processor_fingerprint(processor: Any) -> dict[str, str]:
    """Capture processor/tokenizer/chat-template fingerprints."""
    fp: dict[str, str] = {}
    try:
        fp["processor_class"] = type(processor).__name__
    except Exception:
        pass
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok:
            fp["tokenizer_class"] = type(tok).__name__
    except Exception:
        pass
    try:
        tpl = getattr(processor, "chat_template", None)
        if tpl:
            fp["chat_template_sha256"] = hashlib.sha256(
                tpl.encode()
            ).hexdigest()[:16]
    except Exception:
        pass
    return fp


# --------------------------------------------------------------------------- #
# LoRA config capture
# --------------------------------------------------------------------------- #

def capture_lora_config(
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    scope_regex: str,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": sorted(target_modules),
        "scope_regex": scope_regex,
    }


# --------------------------------------------------------------------------- #
# Training config capture
# --------------------------------------------------------------------------- #

def capture_training_config(
    *,
    learning_rate: float,
    optimizer_steps: int,
    warmup_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    weight_decay: float,
    seed: int,
    condition: str,
) -> dict[str, Any]:
    return {
        "learning_rate": learning_rate,
        "optimizer_steps": optimizer_steps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "weight_decay": weight_decay,
        "seed": seed,
        "condition": condition,
        "optimizer": "AdamW",
    }


# --------------------------------------------------------------------------- #
# Full provenance assembly
# --------------------------------------------------------------------------- #

def build_provenance(
    *,
    git_state: dict[str, str],
    model_fingerprint: dict[str, str],
    processor_fingerprint: dict[str, str],
    lora_config: dict[str, Any],
    training_config: dict[str, Any],
    manifest_shas: dict[str, str],
    dataset_shas: dict[str, str],
    adapter_shas: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the full provenance record for an E2C condition run."""
    provenance: dict[str, Any] = {
        "schema_version": "e2c_provenance_v1",
        "git": git_state,
        "model": model_fingerprint,
        "processor": processor_fingerprint,
        "lora": lora_config,
        "training": training_config,
        "manifests": manifest_shas,
        "datasets": dataset_shas,
    }
    if adapter_shas:
        provenance["adapter"] = adapter_shas

    # Self-hash
    provenance["provenance_sha256"] = sha256_json(provenance)
    return provenance


def write_provenance(
    provenance: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write provenance.json and return the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "provenance.json"

    if path.exists():
        raise FileExistsError(
            f"Provenance already exists at {path}. "
            f"Do not overwrite silently."
        )

    with open(path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    return path


def validate_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Validate that a provenance record has all mandatory fields.

    Returns report dict with pass/fail.
    """
    mandatory_top = [
        "schema_version", "git", "model", "processor",
        "lora", "training", "manifests", "datasets",
    ]
    missing = [k for k in mandatory_top if k not in provenance]

    mandatory_git = ["code_sha", "dirty"]
    missing_git = [
        k for k in mandatory_git
        if k not in provenance.get("git", {})
    ]

    errors = []
    if missing:
        errors.append(f"Missing top-level keys: {missing}")
    if missing_git:
        errors.append(f"Missing git keys: {missing_git}")

    return {
        "pass": len(errors) == 0,
        "errors": errors,
    }
