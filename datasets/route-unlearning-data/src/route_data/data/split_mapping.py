"""Centralized source-split mapping (P1-10).

Maps benchmark-specific split names to the three internal buckets used
by the unlearning pipeline: ``train``, ``eval``, ``exclude``.  Records
with no official partition assignment map to ``hash`` (identity-hashing
fallback).

Every build stage, verifier, and audit script MUST use these helpers
instead of duplicating the mapping locally.
"""

from __future__ import annotations

from typing import Any

# Default source mapping covering common benchmark partition vocabularies.
# Benchmarks can override via data.extras["source_mapping"] in their
# data config YAML.
DEFAULT_SOURCE_MAPPING: dict[str, str] = {
    "train": "train",
    "retain_train": "train",
    "retain": "train",
    "validation": "eval",
    "val": "eval",
    "eval": "eval",
    "retain_eval": "eval",
    "evaluation": "eval",
    "test": "eval",
    "forget": "exclude",
    "exclude": "exclude",
    "unassigned": "hash",
}


def load_source_mapping(data_cfg: Any) -> dict[str, str]:
    """Build the effective source mapping for a benchmark.

    Starts from ``DEFAULT_SOURCE_MAPPING`` and applies any benchmark-specific
    overrides from ``data_cfg.extras["source_mapping"]``.

    Parameters
    ----------
    data_cfg:
        A ``DataConfig`` instance (or raw dict with ``extras`` key).

    Returns
    -------
    dict[str, str]
        Mapping from raw split name to internal bucket name.
    """
    mapping = DEFAULT_SOURCE_MAPPING.copy()
    # Support both DataConfig objects and raw dicts.
    if hasattr(data_cfg, "extras"):
        extras = data_cfg.extras
    elif isinstance(data_cfg, dict):
        extras = data_cfg.get("extras", {})
    else:
        extras = {}
    extra_mapping = extras.get("source_mapping") if extras else None
    if extra_mapping and isinstance(extra_mapping, dict):
        mapping.update(extra_mapping)
    return mapping


def resolve_effective_split(
    sample: dict[str, Any],
    data_cfg: Any = None,
    *,
    source_mapping: dict[str, str] | None = None,
) -> str | None:
    """Resolve the effective split bucket for a canonical sample.

    Checks ``source_split``, ``source_metadata.source_split``, and
    ``split`` fields, then maps through the canonical source mapping.

    Parameters
    ----------
    sample:
        A canonical sample dict (or dict-like).
    data_cfg:
        Optional data config for benchmark-specific overrides.
    source_mapping:
        Optional pre-built mapping (avoids re-building from data_cfg).

    Returns
    -------
    str | None
        The mapped bucket name (``train``, ``eval``, ``exclude``, ``hash``)
        or the raw value if no mapping exists.  Returns ``None`` if no
        split field is found.
    """
    if source_mapping is None:
        source_mapping = load_source_mapping(data_cfg) if data_cfg else DEFAULT_SOURCE_MAPPING
    raw = (
        sample.get("source_split")
        or sample.get("source_metadata", {}).get("source_split")
        or sample.get("split")
    )
    if raw is None:
        return None
    return source_mapping.get(raw, raw)
