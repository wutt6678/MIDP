"""Unlearning split construction (plan section 17).

Builds *explicit split families* rather than one generic forget flag. Four
forget scopes are supported (plan 17.1-17.4):

- ``identity``: all fine-tuning examples for selected identities are forgotten;
- ``identity_fact``: only selected profile facts are forgotten;
- ``visual_identity_link``: face-to-name/biography mappings are forgotten while
  name-only facts are retained;
- ``global_attribute``: every example for one visual attribute is forgotten
  across identities (reported separately from identity unlearning).

:class:`validate_split_invariants` enforces the automated checks from plan 17.5.
Assignment is deterministic (a SHA-256 of the sample id drives the retain
train/eval partition) so a split is reproducible without storing large state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from ..data.schemas import CanonicalSample

FORGET_SCOPES: frozenset[str] = frozenset(
    {"identity", "identity_fact", "visual_identity_link", "global_attribute"}
)

# Deterministic retain train/eval split.
_DEFAULT_EVAL_FRACTION = 0.2


class SplitError(ValueError):
    """Raised when a split spec is invalid or invariants are violated."""


# --------------------------------------------------------------------------- #
# Spec / result containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitSpec:
    name: str
    forget_scope: str
    forget_identity_ids: tuple[str, ...] = ()
    forget_fact_ids: tuple[str, ...] = ()
    attribute: str | None = None
    eval_fraction: float = _DEFAULT_EVAL_FRACTION

    def validate(self) -> "SplitSpec":
        if not self.name:
            raise SplitError("SplitSpec.name must be non-empty")
        if self.forget_scope not in FORGET_SCOPES:
            raise SplitError(
                f"forget_scope must be one of {sorted(FORGET_SCOPES)}, got {self.forget_scope!r}"
            )
        if self.forget_scope == "identity" and not self.forget_identity_ids:
            raise SplitError("identity-level deletion requires forget_identity_ids")
        if self.forget_scope == "identity_fact" and not self.forget_fact_ids:
            raise SplitError("association-level deletion requires forget_fact_ids")
        if self.forget_scope == "global_attribute" and not self.attribute:
            raise SplitError("global-attribute deletion requires an attribute")
        if not (0.0 < self.eval_fraction < 1.0):
            raise SplitError("eval_fraction must be in (0, 1)")
        return self


@dataclass
class SplitResult:
    spec: SplitSpec
    forget: list[CanonicalSample] = field(default_factory=list)
    retain_train: list[CanonicalSample] = field(default_factory=list)
    retain_eval: list[CanonicalSample] = field(default_factory=list)
    unassigned: list[CanonicalSample] = field(default_factory=list)

    def manifest(self) -> dict:
        return {
            "name": self.spec.name,
            "forget_scope": self.spec.forget_scope,
            "forget_identity_ids": list(self.spec.forget_identity_ids),
            "forget_fact_ids": list(self.spec.forget_fact_ids),
            "attribute": self.spec.attribute,
            "counts": {
                "forget": len(self.forget),
                "retain_train": len(self.retain_train),
                "retain_eval": len(self.retain_eval),
                "unassigned": len(self.unassigned),
            },
            "forget_retain_ratio": (
                len(self.forget) / max(1, len(self.retain_train) + len(self.retain_eval))
            ),
        }


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #


def _unit_hash(sample_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _is_eval(sample: CanonicalSample, spec: SplitSpec) -> bool:
    return _unit_hash(sample.source_sample_id, spec.name) < spec.eval_fraction


# --------------------------------------------------------------------------- #
# Per-scope assignment
# --------------------------------------------------------------------------- #


def _assign(sample: CanonicalSample, spec: SplitSpec) -> str | None:
    """Return 'forget', 'retain', or None for one sample under ``spec``."""
    scope = spec.forget_scope

    if scope == "identity":
        return "forget" if sample.identity_id in spec.forget_identity_ids else "retain"

    if scope == "identity_fact":
        fact_ids = {fact.fact_id for fact in sample.profile_facts}
        if fact_ids & set(spec.forget_fact_ids) and sample.task_type == "identity_fact":
            return "forget"
        # Identity recognition, other facts, and all visual attributes are kept.
        return "retain"

    if scope == "visual_identity_link":
        is_visual_link = (
            sample.modality == "image_text" and bool(sample.identity_name)
        )
        if is_visual_link:
            return "forget"
        # Name-only (text) facts and image-only perception tasks are retained.
        return "retain"

    if scope == "global_attribute":
        target = f"extended_attributes.celeba40.{spec.attribute}"
        obs = sample.visual_attributes.get(target)
        # Only accepted observations (explicit boolean label) drive
        # attribute-level forgetting; uncertain labels never force a split.
        if obs is not None and obs.label is not None:
            return "forget"
        return "retain"

    raise SplitError(f"Unhandled forget_scope {scope!r}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


class SplitBuilder:
    def __init__(self, samples: Iterable[CanonicalSample]):
        self.samples: list[CanonicalSample] = list(samples)

    def build(self, spec: SplitSpec) -> SplitResult:
        spec.validate()
        result = SplitResult(spec=spec)
        for sample in self.samples:
            assignment = _assign(sample, spec)
            if assignment is None:
                result.unassigned.append(sample)
                continue
            if assignment == "forget":
                result.forget.append(replace(sample, forget_scope=spec.forget_scope))
            elif _is_eval(sample, spec):
                result.retain_eval.append(sample)
            else:
                result.retain_train.append(sample)
        return result


# --------------------------------------------------------------------------- #
# Invariants (plan 17.5)
# --------------------------------------------------------------------------- #


def _visual_label_counts(samples: Sequence[CanonicalSample]) -> dict[str, int]:
    prefix = "extended_attributes.celeba40."
    pos = neg = 0
    for sample in samples:
        for key, obs in sample.visual_attributes.items():
            if key.startswith(prefix) and obs.label is not None:
                if obs.label:
                    pos += 1
                else:
                    neg += 1
    return {"positive": pos, "negative": neg}


def validate_split_invariants(result: SplitResult, *, strict: bool = True) -> list[str]:
    """Check the plan-17.5 invariants; return issues, raising when ``strict``."""
    issues: list[str] = []
    spec = result.spec

    # 1) No sample may be both forget and retain.
    forget_ids = {s.source_sample_id for s in result.forget}
    retain_ids = {s.source_sample_id for s in result.retain_train + result.retain_eval}
    overlap = forget_ids & retain_ids
    if overlap:
        issues.append(f"samples in both forget and retain: {sorted(overlap)[:10]}")

    # 2) Held-out images of forget identities must not be used for training.
    if spec.forget_scope == "identity":
        forget_identity_set = set(spec.forget_identity_ids)
        leaked = [
            s.source_sample_id
            for s in result.retain_train
            if s.identity_id in forget_identity_set
        ]
        if leaked:
            issues.append(
                f"forget-identity samples leaked into retain_train: {leaked[:10]}"
            )

    # 3) Every forget sample carries the expected scope.
    bad_scope = [s.source_sample_id for s in result.forget if s.forget_scope != spec.forget_scope]
    if bad_scope:
        issues.append(f"forget samples with wrong forget_scope: {bad_scope[:10]}")

    # 4) Each split should include positive and negative visual-attribute cases.
    for name, bucket in (
        ("forget", result.forget),
        ("retain_train", result.retain_train),
        ("retain_eval", result.retain_eval),
    ):
        counts = _visual_label_counts(bucket)
        if bucket and (counts["positive"] == 0 or counts["negative"] == 0):
            issues.append(
                f"split '{name}' is missing positive or negative visual cases: {counts}"
            )

    if issues and strict:
        raise SplitError(f"Split '{spec.name}' violates invariants: {'; '.join(issues)}")
    return issues
