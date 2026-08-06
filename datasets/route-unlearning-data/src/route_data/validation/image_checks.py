"""Source-integrity checks for image references (plan sections 3.2, 19.2).

Images are never copied into this repository; samples only carry a URI and a
SHA-256. These checks confirm that every reference is either:

- ``resolved``: a local file exists (and optionally matches its checksum);
- ``remote``: explicitly marked as a remote/URI-only reference; or
- ``unavailable``: flagged so downstream steps fail loudly instead of
  silently skipping data.

Also implements the plan-19.2 exclusion audit: no source id may disappear
without a logged exclusion reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..data.checksums import verify_checksum
from ..data.schemas import CanonicalSample
from .common import ValidationError, ValidationIssue

_REMOTE_PREFIXES = ("http://", "https://", "hf://", "s3://", "gs://")


class ImageCheckError(ValidationError):
    """Raised in strict mode when image references cannot be accounted for."""


def is_remote_uri(uri: str) -> bool:
    return any(uri.startswith(prefix) for prefix in _REMOTE_PREFIXES)


def resolve_image(sample: CanonicalSample, base_dirs: Sequence[str | Path] = ()) -> str:
    """Classify one sample's image reference as resolved/remote/unavailable."""
    uri = sample.image_uri
    if not uri:
        # Text-only records legitimately carry no image reference.
        return "resolved" if sample.modality == "text_only" else "unavailable"
    if is_remote_uri(uri):
        return "remote"
    for base in base_dirs:
        candidate = Path(base) / uri
        if candidate.is_file():
            return "resolved"
    direct = Path(uri)
    return "resolved" if direct.is_file() else "unavailable"


def check_image_references(
    samples: Iterable[CanonicalSample],
    *,
    base_dirs: Sequence[str | Path] = (),
    verify_existing_checksums: bool = False,
) -> tuple[list[ValidationIssue], Mapping[str, int]]:
    """Resolve every image reference; return issues and a status histogram."""
    issues: list[ValidationIssue] = []
    statuses: dict[str, int] = {"resolved": 0, "remote": 0, "unavailable": 0}
    for sample in samples:
        status = resolve_image(sample, base_dirs)
        statuses[status] += 1
        if status == "unavailable":
            issues.append(
                ValidationIssue(
                    "image_unavailable",
                    sample.source_sample_id,
                    f"image_uri={sample.image_uri!r}",
                )
            )
        elif (
            status == "resolved"
            and verify_existing_checksums
            and sample.image_sha256
        ):
            path = _resolved_path(sample, base_dirs)
            if path is not None and not verify_checksum(path, sample.image_sha256):
                issues.append(
                    ValidationIssue(
                        "checksum_mismatch",
                        sample.source_sample_id,
                        f"expected {sample.image_sha256[:12]}… for {path}",
                    )
                )
    return issues, statuses


def _resolved_path(sample: CanonicalSample, base_dirs: Sequence[str | Path]) -> Path | None:
    uri = sample.image_uri or ""
    for base in base_dirs:
        candidate = Path(base) / uri
        if candidate.is_file():
            return candidate
    direct = Path(uri)
    return direct if direct.is_file() else None


# --------------------------------------------------------------------------- #
# Exclusion audit (plan 19.2: no silent drops)
# --------------------------------------------------------------------------- #


def check_source_reconciliation(
    source_ids: Iterable[str],
    kept_samples: Sequence[CanonicalSample],
    exclusions: Mapping[str, str],
    *,
    source_name: str = "source",
) -> list[ValidationIssue]:
    """Every source id must be kept or excluded with a logged reason."""
    kept = {s.source_sample_id for s in kept_samples}
    issues: list[ValidationIssue] = []
    for source_id in source_ids:
        if source_id in kept:
            continue
        reason = exclusions.get(source_id)
        if not reason:
            issues.append(
                ValidationIssue(
                    "unlogged_drop", source_name, f"source_id={source_id} disappeared"
                )
            )
    return issues


def validate_source_integrity(
    samples: Sequence[CanonicalSample],
    *,
    base_dirs: Sequence[str | Path] = (),
    source_ids: Iterable[str] = (),
    exclusions: Mapping[str, str] | None = None,
    verify_existing_checksums: bool = False,
    strict: bool = False,
) -> dict:
    """Combined plan-19.2 source integrity audit for one benchmark."""
    issues, statuses = check_image_references(
        samples,
        base_dirs=base_dirs,
        verify_existing_checksums=verify_existing_checksums,
    )
    if source_ids:
        issues += check_source_reconciliation(
            source_ids, samples, exclusions or {}, source_name="source"
        )
    if issues and strict:
        raise ImageCheckError(
            f"{len(issues)} source-integrity issue(s); first: {issues[0]}"
        )
    return {
        "image_statuses": dict(statuses),
        "issue_count": len(issues),
        "issues": [issue.to_dict() for issue in issues[:100]],
    }
