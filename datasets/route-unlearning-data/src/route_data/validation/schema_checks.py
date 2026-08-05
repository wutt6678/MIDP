"""Schema tests for canonical records (plan section 19.1).

Checks every exported/built record for:

- structural validity (``CanonicalSample.validate`` / dataclass round-trip);
- non-empty required identifiers;
- recognized enum values (sources, confidence bands, modalities, scopes);
- binary labels where expected (``bool`` or ``None``, never a proxy value);
- confidence scores within ``[0, 1]``;
- image hashes that are valid SHA-256 hex strings.

The plan mentions Pydantic; this repository uses equivalent dataclass
validators (see ``data/schemas.py``) so the checks below play the same role.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..constants.celeba_attributes import CELEBA_ATTRIBUTES
from ..data.schemas import (
    CONFIDENCE_BANDS,
    MODALITIES,
    CanonicalSample,
    SchemaError,
)
from .common import ValidationError, ValidationIssue

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
CELEBA40_PREFIX = "extended_attributes.celeba40."


class SchemaValidationError(ValidationError):
    """Raised in strict mode when schema checks fail."""


def is_valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


# --------------------------------------------------------------------------- #
# Per-record checks
# --------------------------------------------------------------------------- #


def check_observation(sample_id: str, key: str, obs) -> list[ValidationIssue]:
    """Validate one attribute observation (plan 19.1 rules 3-5)."""
    issues: list[ValidationIssue] = []
    where = f"{sample_id}:{key}"
    if obs.label is not None and not isinstance(obs.label, bool):
        issues.append(
            ValidationIssue("label_not_binary", where, f"label={obs.label!r}")
        )
    if obs.score is not None:
        try:
            score = float(obs.score)
        except (TypeError, ValueError):
            issues.append(ValidationIssue("score_not_numeric", where, repr(obs.score)))
        else:
            if not (0.0 <= score <= 1.0):
                issues.append(ValidationIssue("score_out_of_range", where, f"score={score}"))
    if obs.confidence_band not in CONFIDENCE_BANDS:
        issues.append(
            ValidationIssue("unknown_confidence_band", where, obs.confidence_band)
        )
    if key.startswith(CELEBA40_PREFIX):
        attribute = key[len(CELEBA40_PREFIX):]
        if attribute not in CELEBA_ATTRIBUTES:
            issues.append(
                ValidationIssue("unknown_celeba40_attribute", where, attribute)
            )
    return issues


def check_sample(sample: CanonicalSample) -> list[ValidationIssue]:
    """Run all plan-19.1 checks against one canonical sample."""
    issues: list[ValidationIssue] = []
    sid = sample.source_sample_id or "<missing-id>"

    # 1) full structural validation (raises SchemaError on failure).
    try:
        sample.validate()
    except SchemaError as exc:
        issues.append(ValidationIssue("schema", sid, str(exc)))
        return issues

    # 2) required identifiers are non-empty.
    for name, value in (
        ("benchmark", sample.benchmark),
        ("source_sample_id", sample.source_sample_id),
        ("identity_id", sample.identity_id),
    ):
        if not (isinstance(value, str) and value.strip()):
            issues.append(ValidationIssue("empty_id", sid, f"{name}={value!r}"))

    # 3) enum fields carry recognized values.
    if sample.modality not in MODALITIES:
        issues.append(ValidationIssue("unknown_modality", sid, sample.modality))
    if sample.forget_scope is not None and sample.forget_scope not in {
        "identity",
        "identity_fact",
        "visual_identity_link",
        "global_attribute",
    }:
        issues.append(ValidationIssue("unknown_forget_scope", sid, sample.forget_scope))

    # 6) image hashes are valid SHA-256 strings whenever present.
    if sample.image_sha256 is not None and not is_valid_sha256(sample.image_sha256):
        issues.append(
            ValidationIssue(
                "bad_image_sha256", sid, f"image_sha256={sample.image_sha256!r}"
            )
        )

    # 4) + 5) per-observation checks (binary labels, [0,1] scores, bands).
    for key, obs in sample.visual_attributes.items():
        issues.extend(check_observation(sid, key, obs))

    return issues


def check_qa_row(row: Mapping[str, Any]) -> list[ValidationIssue]:
    """Checks specific to exported visual-QA rows (plan 16.2 + 19.1)."""
    issues: list[ValidationIssue] = []
    qid = str(row.get("qa_id") or "<missing-qa-id>")
    for name in ("qa_id", "sample_id", "question", "template_id", "registry_hash"):
        if not row.get(name):
            issues.append(ValidationIssue("missing_field", qid, name))
    if row.get("answer_label") is not None and not isinstance(row["answer_label"], bool):
        issues.append(
            ValidationIssue("label_not_binary", qid, repr(row.get("answer_label")))
        )
    score = row.get("answer_score")
    if score is not None:
        try:
            if not (0.0 <= float(score) <= 1.0):
                issues.append(ValidationIssue("score_out_of_range", qid, f"score={score}"))
        except (TypeError, ValueError):
            issues.append(ValidationIssue("score_not_numeric", qid, repr(score)))
    digest = row.get("registry_hash")
    if digest is not None and not re.match(r"^[0-9a-f]{16}$", str(digest)):
        issues.append(ValidationIssue("bad_registry_hash", qid, str(digest)))
    sha = row.get("image_sha256")
    if sha is not None and not is_valid_sha256(sha):
        issues.append(ValidationIssue("bad_image_sha256", qid, str(sha)))
    return issues


# --------------------------------------------------------------------------- #
# Batch entry point
# --------------------------------------------------------------------------- #


@dataclass
class SchemaReport:
    checked_samples: int = 0
    checked_qa_rows: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_samples": self.checked_samples,
            "checked_qa_rows": self.checked_qa_rows,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues[:100]],
        }


def validate_schema(
    samples: Sequence[CanonicalSample],
    *,
    qa_rows: Iterable[Mapping[str, Any]] = (),
    strict: bool = False,
) -> SchemaReport:
    """Validate many records; raise :class:`SchemaValidationError` when strict."""
    report = SchemaReport()
    for sample in samples:
        report.checked_samples += 1
        report.issues.extend(check_sample(sample))
    for row in qa_rows:
        report.checked_qa_rows += 1
        report.issues.extend(check_qa_row(row))
    if report.issues and strict:
        first = "; ".join(str(i) for i in report.issues[:5])
        raise SchemaValidationError(
            f"{len(report.issues)} schema issue(s); first: {first}"
        )
    return report
