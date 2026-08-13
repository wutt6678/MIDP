"""Centralized source-split mapping (P1-10).

Maps benchmark-specific split names to the three internal buckets used
by the unlearning pipeline: ``train``, ``eval``, ``exclude``.  Records
with no official partition assignment map to ``hash`` (identity-hashing
fallback).

Every build stage, verifier, and audit script MUST use these helpers
instead of duplicating the mapping locally.
"""

from __future__ import annotations

import hashlib
import json
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
    "out_of_protocol": "out_of_protocol",
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


# --------------------------------------------------------------------------- #
# Protocol-exclusive resolution (P0-1 / P0-2)
# --------------------------------------------------------------------------- #


def resolve_protocol_role(
    memberships: list[str],
    protocol: dict[str, Any],
    source_subject_id: str | None = None,
) -> str:
    """Resolve official bucket memberships to an experiment role via protocol.

    When a ``fiubench_protocol`` is active it is the SOLE authority for
    experiment roles.  Identities with official memberships that do not
    match any configured protocol bucket receive ``"out_of_protocol"``
    rather than falling through to a generic source mapping.

    Parameters
    ----------
    memberships:
        List of official FIUBench bucket names the identity belongs to
        (e.g. ``["forget1", "forget10"]``).
    protocol:
        The ``fiubench_protocol`` config dict with keys
        ``forget_bucket``, ``train_bucket``, ``eval_bucket``,
        ``eval_fraction``, ``eval_seed``.
    source_subject_id:
        Released subject identifier used for deterministic holdout.
        Required when the identity matches the train bucket.

    Returns
    -------
    str
        One of ``"exclude"``, ``"eval"``, ``"train"``, or
        ``"out_of_protocol"``.
    """
    forget_bucket = protocol.get("forget_bucket")
    train_bucket = protocol.get("train_bucket")
    eval_bucket = protocol.get("eval_bucket")

    # Forget takes priority (P0-1: exclusive forget_bucket).
    if forget_bucket and forget_bucket in memberships:
        return "exclude"

    # Explicit eval bucket (when configured separately from holdout).
    if eval_bucket and eval_bucket in memberships:
        return "eval"

    # Train bucket with deterministic holdout (P0-4).
    if train_bucket and train_bucket in memberships:
        eval_fraction = protocol.get("eval_fraction", 0.0)
        eval_seed = protocol.get("eval_seed", 0)
        if eval_fraction > 0 and source_subject_id is not None:
            return compute_holdout_role(source_subject_id, eval_fraction, eval_seed)
        return "train"

    # Official identity not selected by the configured experiment.
    return "out_of_protocol"


def compute_holdout_role(
    source_subject_id: str,
    eval_fraction: float,
    eval_seed: int = 0,
) -> str:
    """Deterministic train/eval assignment via stable subject ID hashing.

    Uses ``sha256(f"{eval_seed}|{source_subject_id}")`` so the assignment
    is independent of source row order and reproducible across runs.

    Parameters
    ----------
    source_subject_id:
        The released FIUBench / SFHQ subject identifier (e.g. ``"00044363"``).
    eval_fraction:
        Fraction of the retain pool reserved for eval (e.g. ``0.20``).
    eval_seed:
        Seed for the holdout hash (configurable, default ``17``).

    Returns
    -------
    str
        ``"eval"`` or ``"train"``.
    """
    h = hashlib.sha256(f"{eval_seed}|{source_subject_id}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / (2**64)
    return "eval" if x < eval_fraction else "train"


# --------------------------------------------------------------------------- #
# Protocol SHA / fingerprint (P1-2)
# --------------------------------------------------------------------------- #


def compute_protocol_sha256(
    protocol: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Compute a canonical SHA-256 fingerprint for a protocol configuration.

    The fingerprint covers every parameter that affects role assignment so
    that any protocol change is detectable in downstream manifests.

    Returns
    -------
    tuple[str, dict]
        ``(protocol_sha256, canonical_dict)`` — the hex digest and the
        canonical representation that was hashed.
    """
    canonical = {
        "algorithm_version": 1,
        "eval_fraction": protocol.get("eval_fraction"),
        "eval_seed": protocol.get("eval_seed"),
        "eval_bucket": protocol.get("eval_bucket"),
        "forget_bucket": protocol.get("forget_bucket"),
        "name": protocol.get("name"),
        "source_population": protocol.get("source_population"),
        "train_bucket": protocol.get("train_bucket"),
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(text.encode()).hexdigest()
    return sha, canonical
