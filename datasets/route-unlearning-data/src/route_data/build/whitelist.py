"""Model-specific attribute whitelist loading and validation.

A whitelist JSON file declares which CelebA attributes are eligible for
automatic weak-label acceptance for a given annotator model.  All 40 raw
scores are always preserved in ``*_model_scores.jsonl``; only the whitelisted
attributes may receive ``source="source_model"`` labels in the processed
dataset.

Whitelist schema
----------------
{
    "model_id": "Qwen/Qwen3.5-9B",
    "source_commit": "<git sha>",
    "source_result": "<diagnostic result path>",
    "policy": "<policy label>",
    "notes": ["..."],
    "attributes": ["Smiling", "Eyeglasses", ...]
}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..constants.celeba_attributes import CELEBA_ATTRIBUTE_SET
from ..config import ConfigError


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class WhitelistError(ConfigError):
    """Raised when a whitelist file is invalid or incompatible."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttributeWhitelist:
    """Validated, immutable whitelist for a specific model."""

    model_id: str
    attributes: frozenset[str]
    source_commit: str | None = None
    source_result: str | None = None
    policy: str | None = None
    notes: list[str] = field(default_factory=list)
    path: Path | None = None
    sha256: str | None = None

    def contains(self, attribute: str) -> bool:
        return attribute in self.attributes


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_attribute_whitelist(
    path: str | Path,
    *,
    expected_model_id: str | None = None,
) -> AttributeWhitelist:
    """Load and validate a whitelist JSON file.

    Parameters
    ----------
    path:
        Path to the whitelist JSON file.
    expected_model_id:
        If provided, the whitelist ``model_id`` must match exactly.

    Raises
    ------
    WhitelistError
        If the file is missing, malformed, or incompatible.
    """
    path = Path(path)
    if not path.is_file():
        raise WhitelistError(f"Whitelist file not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise WhitelistError(f"Failed to read whitelist JSON: {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise WhitelistError(f"Whitelist root must be a mapping: {path}")

    # --- model_id ---
    model_id = raw.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise WhitelistError(f"Whitelist missing or invalid 'model_id': {path}")
    model_id = model_id.strip()

    if expected_model_id is not None and model_id != expected_model_id:
        raise WhitelistError(
            f"Whitelist model_id mismatch: expected '{expected_model_id}', "
            f"got '{model_id}' in {path}"
        )

    # --- attributes ---
    attrs_raw = raw.get("attributes")
    if not isinstance(attrs_raw, list) or len(attrs_raw) == 0:
        raise WhitelistError(
            f"Whitelist 'attributes' must be a non-empty list: {path}"
        )

    attributes: list[str] = []
    for i, a in enumerate(attrs_raw):
        if not isinstance(a, str) or not a.strip():
            raise WhitelistError(
                f"Whitelist attribute at index {i} is not a valid string: {a!r}"
            )
        attributes.append(a.strip())

    # Check for duplicates.
    if len(attributes) != len(set(attributes)):
        from collections import Counter
        dupes = [a for a, c in Counter(attributes).items() if c > 1]
        raise WhitelistError(
            f"Whitelist contains duplicate attributes: {sorted(dupes)}"
        )

    # Check all are valid CelebA attributes.
    unknown = set(attributes) - CELEBA_ATTRIBUTE_SET
    if unknown:
        raise WhitelistError(
            f"Whitelist contains unknown CelebA attributes: {sorted(unknown)}"
        )

    attr_set = frozenset(attributes)

    # --- optional metadata ---
    source_commit = raw.get("source_commit")
    if source_commit is not None and not isinstance(source_commit, str):
        raise WhitelistError(f"Whitelist 'source_commit' must be a string: {path}")

    source_result = raw.get("source_result")
    if source_result is not None and not isinstance(source_result, str):
        raise WhitelistError(f"Whitelist 'source_result' must be a string: {path}")

    policy = raw.get("policy")
    if policy is not None and not isinstance(policy, str):
        raise WhitelistError(f"Whitelist 'policy' must be a string: {path}")

    notes_raw = raw.get("notes", [])
    if not isinstance(notes_raw, list):
        raise WhitelistError(f"Whitelist 'notes' must be a list: {path}")
    notes = [str(n) for n in notes_raw]

    sha = _compute_sha256(path)

    return AttributeWhitelist(
        model_id=model_id,
        attributes=attr_set,
        source_commit=source_commit,
        source_result=source_result,
        policy=policy,
        notes=notes,
        path=path,
        sha256=sha,
    )
