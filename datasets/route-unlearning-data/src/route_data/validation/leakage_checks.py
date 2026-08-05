"""Leakage tests (plan section 19.4).

Six leak families are checked:

1. exact and normalized question overlap between splits;
2. image checksum overlap between splits that must be image-disjoint;
3. perceptual-hash overlap for near-duplicate images;
4. identity overlap where a split must be identity-disjoint;
5. fact-value overlap for synthetic unique identifiers;
6. accidental identity names embedded in image metadata or filenames.

All checks return :class:`ValidationIssue` lists and never raise unless
``strict=True``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..data.schemas import CanonicalSample
from .common import ValidationError, ValidationIssue


class LeakageError(ValidationError):
    """Raised in strict mode when any leakage is detected."""


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_question(text: str) -> str:
    """Lowercase, strip punctuation/whitespace so paraphrase-free leaks match."""
    return " ".join(_WORD_RE.findall(text.lower()))


# --------------------------------------------------------------------------- #
# 1. Question overlap
# --------------------------------------------------------------------------- #


def check_question_overlap(
    source_questions: Iterable[str],
    target_questions: Iterable[str],
    *,
    source_name: str = "train",
    target_name: str = "test",
) -> list[ValidationIssue]:
    """Flag questions in ``target`` that also appear in ``source``."""
    source_exact = {q.strip() for q in source_questions if q}
    source_norm = {normalize_question(q) for q in source_questions if q}
    issues: list[ValidationIssue] = []
    for question in target_questions:
        if not question:
            continue
        if question.strip() in source_exact:
            issues.append(
                ValidationIssue(
                    "exact_question_leak",
                    f"{target_name}",
                    question[:120],
                )
            )
        elif normalize_question(question) in source_norm:
            issues.append(
                ValidationIssue(
                    "normalized_question_leak",
                    f"{target_name}",
                    question[:120],
                )
            )
    return issues


# --------------------------------------------------------------------------- #
# 2. + 4. Checksum and identity overlap
# --------------------------------------------------------------------------- #


def _field_set(samples: Iterable[CanonicalSample], attr: str) -> set[str]:
    return {
        str(getattr(s, attr))
        for s in samples
        if getattr(s, attr) not in (None, "")
    }


def check_image_checksum_overlap(
    left: Sequence[CanonicalSample],
    right: Sequence[CanonicalSample],
    *,
    left_name: str = "forget",
    right_name: str = "retain",
) -> list[ValidationIssue]:
    left_shas = _field_set(left, "image_sha256")
    shared = left_shas & _field_set(right, "image_sha256")
    return [
        ValidationIssue("image_checksum_leak", right_name, sha)
        for sha in sorted(shared)[:50]
    ]


def check_identity_overlap(
    left: Sequence[CanonicalSample],
    right: Sequence[CanonicalSample],
    *,
    left_name: str = "forget",
    right_name: str = "retain",
) -> list[ValidationIssue]:
    """Identity-disjointness check (plan 17.5 / 19.4)."""
    left_ids = _field_set(left, "identity_id")
    shared = left_ids & _field_set(right, "identity_id")
    return [
        ValidationIssue("identity_leak", right_name, identity)
        for identity in sorted(shared)[:50]
    ]


# --------------------------------------------------------------------------- #
# 3. Perceptual hashing (near-duplicate detection)
# --------------------------------------------------------------------------- #


def average_hash(path: str | Path, hash_size: int = 8) -> int:
    """Tiny dependency-light average hash (aHash) for near-duplicate checks."""
    from PIL import Image

    with Image.open(path) as img:
        gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | (1 if pixel >= mean else 0)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def check_perceptual_duplicates(
    hashes: Mapping[str, int],
    *,
    threshold: int = 5,
    group_name: str = "split",
) -> list[ValidationIssue]:
    """Pairwise near-duplicate scan over precomputed aHash values."""
    issues: list[ValidationIssue] = []
    keys = sorted(hashes)
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            distance = hamming_distance(hashes[key_a], hashes[key_b])
            if distance <= threshold:
                issues.append(
                    ValidationIssue(
                        "perceptual_duplicate",
                        group_name,
                        f"{key_a} ~ {key_b} (distance={distance})",
                    )
                )
    return issues


# --------------------------------------------------------------------------- #
# 5. Synthetic unique-identifier fact leaks
# --------------------------------------------------------------------------- #


def check_fact_value_overlap(
    forget_samples: Sequence[CanonicalSample],
    retain_samples: Sequence[CanonicalSample],
    *,
    relations: Iterable[str] = ("unique_identifier", "ssn", "passport", "email", "phone"),
    retain_name: str = "retain",
) -> list[ValidationIssue]:
    """Synthetic unique identifiers must never survive in retained data."""
    wanted = set(relations)
    forget_values: dict[str, str] = {}
    for sample in forget_samples:
        for fact in sample.profile_facts:
            if fact.relation in wanted and fact.value:
                forget_values[fact.value] = sample.source_sample_id
    issues: list[ValidationIssue] = []
    for sample in retain_samples:
        for fact in sample.profile_facts:
            if fact.relation in wanted and fact.value in forget_values:
                issues.append(
                    ValidationIssue(
                        "fact_value_leak",
                        f"{retain_name}:{sample.source_sample_id}",
                        f"{fact.relation}={fact.value}",
                    )
                )
    return issues


# --------------------------------------------------------------------------- #
# 6. Accidental names in image metadata / filenames
# --------------------------------------------------------------------------- #


def check_identity_names_in_uris(samples: Iterable[CanonicalSample]) -> list[ValidationIssue]:
    """Flag images whose URI/filename embeds the identity's name."""
    issues: list[ValidationIssue] = []
    for sample in samples:
        if not sample.identity_name:
            continue
        name_parts = [p for p in re.split(r"\s+", sample.identity_name.lower()) if p]
        for field_name in ("image_uri", "image_id"):
            value = getattr(sample, field_name)
            if not value:
                continue
            lowered = str(value).lower()
            hits = [part for part in name_parts if part and part in lowered]
            if hits:
                issues.append(
                    ValidationIssue(
                        "name_in_image_reference",
                        sample.source_sample_id,
                        f"{field_name}={value} matches {hits}",
                    )
                )
    return issues


# --------------------------------------------------------------------------- #
# Combined split leakage check
# --------------------------------------------------------------------------- #


def validate_split_leakage(
    forget: Sequence[CanonicalSample],
    retain_train: Sequence[CanonicalSample],
    retain_eval: Sequence[CanonicalSample],
    *,
    identity_disjoint: bool = True,
    strict: bool = False,
) -> list[ValidationIssue]:
    """All leakage checks relevant to one unlearning split family."""
    issues: list[ValidationIssue] = []
    issues += check_image_checksum_overlap(
        forget, retain_train, left_name="forget", right_name="retain_train"
    )
    issues += check_image_checksum_overlap(
        forget, retain_eval, left_name="forget", right_name="retain_eval"
    )
    if identity_disjoint:
        issues += check_identity_overlap(
            forget, retain_train, left_name="forget", right_name="retain_train"
        )
        issues += check_identity_overlap(
            forget, retain_eval, left_name="forget", right_name="retain_eval"
        )
    issues += check_fact_value_overlap(forget, retain_train + retain_eval)
    issues += check_identity_names_in_uris(retain_train + retain_eval)
    if issues and strict:
        raise LeakageError(
            f"{len(issues)} leakage issue(s); first: {issues[0]}"
        )
    return issues
