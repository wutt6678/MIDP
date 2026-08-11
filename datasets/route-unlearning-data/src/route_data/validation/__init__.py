"""Validation suite (plan section 19).

Combines schema, source-integrity, distribution, and leakage checks into a
single auditable report for one benchmark extension.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..build.split_generation import SplitResult, validate_split_invariants
from ..data.schemas import CanonicalSample
from .common import ValidationError, ValidationIssue, summarize_issues
from .distribution_checks import (
    distribution_report,
    render_distribution_md,
    summarize_samples,
    summarize_split,
)
from .image_checks import (
    ImageCheckError,
    check_image_references,
    check_source_reconciliation,
    is_remote_uri,
    resolve_image,
    validate_source_integrity,
)
from .leakage_checks import (
    LeakageError,
    average_hash,
    check_fact_value_overlap,
    check_identity_names_in_uris,
    check_identity_overlap,
    check_image_checksum_overlap,
    check_perceptual_duplicates,
    check_question_overlap,
    hamming_distance,
    normalize_question,
    validate_split_leakage,
)
from .schema_checks import (
    SchemaReport,
    SchemaValidationError,
    check_qa_row,
    check_sample,
    is_valid_sha256,
    validate_schema,
)

__all__ = [
    "ImageCheckError",
    "LeakageError",
    "SchemaReport",
    "SchemaValidationError",
    "ValidationError",
    "ValidationIssue",
    "average_hash",
    "check_fact_value_overlap",
    "check_identity_names_in_uris",
    "check_identity_overlap",
    "check_image_checksum_overlap",
    "check_image_references",
    "check_perceptual_duplicates",
    "check_qa_row",
    "check_question_overlap",
    "check_sample",
    "check_source_reconciliation",
    "distribution_report",
    "hamming_distance",
    "is_remote_uri",
    "is_valid_sha256",
    "normalize_question",
    "render_distribution_md",
    "resolve_image",
    "summarize_issues",
    "summarize_samples",
    "summarize_split",
    "validate_dataset",
    "validate_schema",
    "validate_source_integrity",
    "validate_split_leakage",
]


def validate_dataset(
    samples: Sequence[CanonicalSample],
    *,
    qa_rows: Iterable[Mapping[str, Any]] = (),
    split_results: Iterable[SplitResult] = (),
    image_base_dirs: Sequence[str | Path] = (),
    strict: bool = False,
) -> dict[str, Any]:
    """Run every automated plan-19 check against one benchmark extension.

    Returns a machine-readable report; raises :class:`ValidationError`
    subclasses when ``strict`` and any check family fails.
    """
    schema_report = validate_schema(samples, qa_rows=qa_rows, strict=False)

    issues: list[ValidationIssue] = []
    for result in split_results:
        # Plan 17.5 structural invariants.
        split_issues = validate_split_invariants(result, strict=False)
        issues += [
            ValidationIssue("split_invariant", result.spec.name, text)
            for text in split_issues
        ]
        identity_disjoint = result.spec.forget_scope == "identity"
        issues += validate_split_leakage(
            result.forget,
            result.retain_train,
            result.retain_eval,
            identity_disjoint=identity_disjoint,
            strict=False,
        )

    image_issues, image_statuses = check_image_references(
        samples, base_dirs=image_base_dirs
    )
    issues += list(image_issues)

    report: dict[str, Any] = {
        "schema": schema_report.to_dict(),
        "image_statuses": dict(image_statuses),
        "distribution": distribution_report(
            samples, split_results=list(split_results)
        ),
        "leakage_and_integrity": summarize_issues(issues),
    }
    failed = bool(schema_report.issues) or bool(issues)
    report["ok"] = not failed
    if failed and strict:
        if schema_report.issues:
            raise SchemaValidationError(
                f"{len(schema_report.issues)} schema issue(s); "
                f"first: {schema_report.issues[0]}"
            )
        raise ValidationError(
            f"{len(issues)} validation issue(s); first: {issues[0]}"
        )
    return report
