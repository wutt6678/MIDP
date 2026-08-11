"""CelebA raw-root validation and immutable manifest construction.

Milestone 1 (plan sections 7.1-7.4). CelebA is license-restricted, so this
module never downloads or redistributes images; it validates a user-supplied
local root and derives manifests from the researcher's own copy.

Expected root layout::

    img_align_celeba/
    list_attr_celeba.txt
    list_eval_partition.txt
    identity_CelebA.txt              # optional for the attribute test
    list_landmarks_align_celeba.txt  # optional
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..constants.celeba_attributes import (
    EXPECTED_IMAGE_COUNT,
    PARTITION_TO_SPLIT,
    verify_attribute_header,
)
from .checksums import sha256_file


class CelebaValidationError(ValueError):
    """Raised when the CelebA root fails structural validation."""


@dataclass
class CelebaRoot:
    root: Path
    image_dir: Path
    attr_file: Path
    partition_file: Path
    identity_file: Path | None = None
    landmarks_file: Path | None = None

    @classmethod
    def discover(cls, root: str | Path) -> CelebaRoot:
        root = Path(root)
        image_dir = root / "img_align_celeba"
        attr_file = root / "list_attr_celeba.txt"
        partition_file = root / "list_eval_partition.txt"
        identity_file = root / "identity_CelebA.txt"
        landmarks_file = root / "list_landmarks_align_celeba.txt"
        return cls(
            root=root,
            image_dir=image_dir,
            attr_file=attr_file,
            partition_file=partition_file,
            identity_file=identity_file if identity_file.exists() else None,
            landmarks_file=landmarks_file if landmarks_file.exists() else None,
        )


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise CelebaValidationError("\n".join(self.errors))


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def parse_attribute_file(path: str | Path) -> tuple[list[str], dict[str, list[int]]]:
    """Return ``(attribute_names, filename -> 40 raw values in {-1,1})``."""
    path = Path(path)
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        raise CelebaValidationError(f"Attribute file too short: {path}")
    try:
        declared_count = int(lines[0])
    except ValueError as exc:
        raise CelebaValidationError(f"First line of {path} is not a count") from exc
    header = lines[1].split()
    verify_attribute_header(header)
    rows: dict[str, list[int]] = {}
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) != 41:
            raise CelebaValidationError(
                f"Attribute row must have 1 filename + 40 values, got {len(parts)}: {ln!r}"
            )
        filename, values = parts[0], parts[1:]
        try:
            ivals = [int(v) for v in values]
        except ValueError as exc:
            raise CelebaValidationError(f"Non-integer attribute value in row {ln!r}") from exc
        if any(v not in (-1, 1) for v in ivals):
            raise CelebaValidationError(f"Attribute values must be -1 or 1 in row {ln!r}")
        if filename in rows:
            raise CelebaValidationError(f"Duplicate attribute row for {filename}")
        rows[filename] = ivals
    if len(rows) != declared_count:
        raise CelebaValidationError(
            f"Declared {declared_count} images but parsed {len(rows)} attribute rows"
        )
    return header, rows


def parse_partition_file(path: str | Path) -> dict[str, int]:
    """Return ``filename -> partition`` with partition in {0,1,2}."""
    partitions: dict[str, int] = {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for i, ln in enumerate(lines):
        parts = ln.split()
        # Some releases prefix a count line; skip a bare integer first line.
        if i == 0 and len(parts) == 1 and parts[0].lstrip("-").isdigit():
            continue
        if len(parts) != 2:
            raise CelebaValidationError(f"Malformed partition row: {ln!r}")
        filename, raw = parts
        try:
            p = int(raw)
        except ValueError as exc:
            raise CelebaValidationError(f"Non-integer partition for {filename}") from exc
        if p not in (0, 1, 2):
            raise CelebaValidationError(f"Partition must be in {{0,1,2}} for {filename}: {p}")
        if filename in partitions:
            raise CelebaValidationError(f"Duplicate partition row for {filename}")
        partitions[filename] = p
    return partitions


def parse_identity_file(path: str | Path) -> dict[str, int]:
    """Return ``filename -> identity_id`` (positive integers)."""
    identities: dict[str, int] = {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for i, ln in enumerate(lines):
        parts = ln.split()
        if i == 0 and len(parts) == 1 and parts[0].lstrip("-").isdigit():
            continue
        if len(parts) != 2:
            raise CelebaValidationError(f"Malformed identity row: {ln!r}")
        filename, raw = parts
        try:
            ident = int(raw)
        except ValueError as exc:
            raise CelebaValidationError(f"Non-integer identity for {filename}") from exc
        if ident <= 0:
            raise CelebaValidationError(f"Identity ID must be positive for {filename}: {ident}")
        identities[filename] = ident
    return identities


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_raw(
    root: str | Path,
    *,
    require_identity: bool = False,
    sample_open: int = 8,
    seed: int = 0,
) -> ValidationReport:
    """Validate a user-supplied CelebA root (plan section 7.1)."""
    report = ValidationReport()
    celeba = CelebaRoot.discover(root)

    if not celeba.image_dir.is_dir():
        report.errors.append(f"Missing image directory: {celeba.image_dir}")
    if not celeba.attr_file.is_file():
        report.errors.append(f"Missing attribute file: {celeba.attr_file}")
    if not celeba.partition_file.is_file():
        report.errors.append(f"Missing partition file: {celeba.partition_file}")
    if require_identity and celeba.identity_file is None:
        report.errors.append("identity_CelebA.txt is required but missing")
    if report.errors:
        return report

    _header, attrs = parse_attribute_file(celeba.attr_file)
    partitions = parse_partition_file(celeba.partition_file)
    identities = (
        parse_identity_file(celeba.identity_file) if celeba.identity_file else None
    )

    report.counts["attr_rows"] = len(attrs)
    report.counts["partition_rows"] = len(partitions)
    report.counts["expected_images"] = EXPECTED_IMAGE_COUNT

    if len(attrs) != EXPECTED_IMAGE_COUNT:
        report.warnings.append(
            f"Attribute rows {len(attrs)} != expected {EXPECTED_IMAGE_COUNT}"
        )

    # Every annotation filename exists in the image directory.
    image_names = {p.name for p in celeba.image_dir.iterdir() if p.is_file()}
    report.counts["image_files"] = len(image_names)
    missing_images = [fn for fn in attrs if fn not in image_names]
    if missing_images:
        report.errors.append(
            f"{len(missing_images)} annotated images missing from image dir "
            f"(e.g. {missing_images[:5]})"
        )

    # Image count agrees with annotation count.
    if len(image_names) != len(attrs):
        report.warnings.append(
            f"Image file count {len(image_names)} != annotation count {len(attrs)}"
        )

    # Partition coverage and (filename, partition) uniqueness.
    no_partition = [fn for fn in attrs if fn not in partitions]
    if no_partition:
        report.errors.append(
            f"{len(no_partition)} images lack a partition entry (e.g. {no_partition[:5]})"
        )

    # Identity coverage when provided.
    if identities is not None:
        no_identity = [fn for fn in attrs if fn not in identities]
        if no_identity:
            report.warnings.append(
                f"{len(no_identity)} images lack an identity ID (e.g. {no_identity[:5]})"
            )
    else:
        report.warnings.append("identity_CelebA.txt not provided; identity analyses disabled")

    # Random images can be opened and converted to RGB.
    if sample_open > 0 and image_names:
        rng = random.Random(seed)
        sample = rng.sample(sorted(image_names), min(sample_open, len(image_names)))
        try:
            from PIL import Image
        except ImportError:
            report.warnings.append("Pillow not installed; skipped RGB open check")
            sample = []
        for name in sample:
            try:
                with Image.open(celeba.image_dir / name) as im:
                    im.convert("RGB")
            except Exception as exc:
                report.errors.append(f"Cannot open/convert {name}: {exc}")

    return report


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #


def build_wide_manifest(
    root: str | Path,
    *,
    compute_sha256: bool = False,
    source_version: str = "celeba-raw",
) -> pd.DataFrame:
    """Build the wide (one-row-per-image) manifest DataFrame."""
    celeba = CelebaRoot.discover(root)
    header, attrs = parse_attribute_file(celeba.attr_file)
    partitions = parse_partition_file(celeba.partition_file)
    identities = (
        parse_identity_file(celeba.identity_file) if celeba.identity_file else None
    )

    records: list[dict[str, Any]] = []
    for filename, values in attrs.items():
        partition_raw = partitions.get(filename)
        split = PARTITION_TO_SPLIT.get(partition_raw, "unknown")
        rec: dict[str, Any] = {
            "image_filename": filename,
            "image_path": str(celeba.image_dir / filename),
            "partition_raw": partition_raw,
            "split": split,
            "identity_id": identities.get(filename) if identities else None,
        }
        if compute_sha256:
            rec["image_sha256"] = sha256_file(celeba.image_dir / filename)
        else:
            rec["image_sha256"] = None
        for name, raw in zip(header, values):
            rec[f"attr_raw::{name}"] = raw
            rec[f"attr::{name}"] = 1 if raw == 1 else 0
        records.append(rec)

    df = pd.DataFrame(records)
    df["source_dataset"] = "celeba"
    df["source_version"] = source_version
    return df


def melt_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Melt a wide manifest into the long format required by section 7.3."""
    attr_cols = [c for c in wide.columns if c.startswith("attr::")]
    raw_cols = [c for c in wide.columns if c.startswith("attr_raw::")]
    id_cols = [
        "image_filename",
        "image_path",
        "image_sha256",
        "partition_raw",
        "split",
        "identity_id",
        "source_dataset",
        "source_version",
    ]
    labels = wide.melt(
        id_vars=id_cols,
        value_vars=attr_cols,
        var_name="_attr_key",
        value_name="label",
    )
    labels["attribute_name"] = labels["_attr_key"].str[len("attr::"):]

    raws = wide.melt(
        id_vars=["image_filename"],
        value_vars=raw_cols,
        var_name="_raw_key",
        value_name="label_raw",
    )
    raws["attribute_name"] = raws["_raw_key"].str[len("attr_raw::"):]

    melted = labels.merge(
        raws[["image_filename", "attribute_name", "label_raw"]],
        on=["image_filename", "attribute_name"],
        how="left",
    )
    melted["sample_id"] = melted["image_filename"] + "::" + melted["attribute_name"]
    out = melted[
        [
            "sample_id",
            "image_filename",
            "image_path",
            "image_sha256",
            "partition_raw",
            "split",
            "identity_id",
            "attribute_name",
            "label_raw",
            "label",
            "source_dataset",
            "source_version",
        ]
    ]
    return out.sort_values(["image_filename", "attribute_name"]).reset_index(drop=True)


def build_long_manifest(
    root: str | Path, *, compute_sha256: bool = False, source_version: str = "celeba-raw"
) -> pd.DataFrame:
    wide = build_wide_manifest(
        root, compute_sha256=compute_sha256, source_version=source_version
    )
    return melt_long(wide)


# --------------------------------------------------------------------------- #
# Deterministic pilot sampling (plan section 7.4)
# --------------------------------------------------------------------------- #


def select_split(wide: pd.DataFrame, split: str) -> pd.DataFrame:
    return wide[wide["split"] == split].reset_index(drop=True)


def sample_pilot_coverage(
    wide: pd.DataFrame,
    *,
    target_images: int,
    per_attribute_per_class: int = 10,
    split: str | None = None,
    seed: int = 17,
) -> pd.DataFrame:
    """Deterministically choose images that maximize attribute coverage.

    For each attribute we pull up to ``per_attribute_per_class`` positive and
    negative examples (seeded), unioning them until ``target_images`` distinct
    images are collected. The result is sorted by filename for determinism.
    """
    pool = wide if split is None else select_split(wide, split)
    rng = random.Random(seed)
    chosen: set[str] = set()
    attr_cols = [c for c in pool.columns if c.startswith("attr::")]
    for col in attr_cols:
        if len(chosen) >= target_images:
            break
        attr = col.split("::")[1]
        for target_label in (1, 0):
            if len(chosen) >= target_images:
                break
            candidates = pool.loc[
                (pool[col] == target_label) & (~pool["image_filename"].isin(chosen)),
                "image_filename",
            ].tolist()
            if not candidates:
                continue
            need = min(per_attribute_per_class, target_images - len(chosen))
            picked = rng.sample(candidates, min(need, len(candidates)))
            chosen.update(picked)
        _ = attr  # retained for readability of the loop intent
    return pool[pool["image_filename"].isin(sorted(chosen))].reset_index(drop=True)


def sample_pilot_prevalence(
    wide: pd.DataFrame,
    *,
    target_images: int,
    split: str = "validation",
    seed: int = 17,
) -> pd.DataFrame:
    """Prevalence-aware pilot: stratified random sample from ``split``."""
    pool = select_split(wide, split)
    rng = random.Random(seed)
    files = pool["image_filename"].unique().tolist()
    if len(files) <= target_images:
        return pool.reset_index(drop=True)
    picked = sorted(rng.sample(files, target_images))
    return pool[pool["image_filename"].isin(picked)].reset_index(drop=True)
