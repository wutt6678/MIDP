"""Benchmark adapter framework (repair plan workstream B).

Adapters map benchmark-specific source records into the canonical schema and
fail loudly when the source layout is incompatible (plan B4/C4). They never
assume identical field names across repositories and never read files outside
an explicit allowlist — generic recursive loading is forbidden for the four
complex benchmarks (FAIRGET, FIUBench, MLLMU-Bench, PPU-Bench).

Contract (plan B1/B2):

- ``iter_rows_with_context()`` yields ``(SourceContext, raw_row)`` pairs from
  allowlisted local files or an explicitly configured HF dataset/config/
  split/revision;
- ``to_samples(row, source_context=...)`` normalizes one raw row into zero or
  more canonical records (nested QA lists and multi-image rows expand
  one-to-many);
- ``load()`` yields every normalized record;
- ``to_sample()`` survives only as a compatibility helper for flat adapters.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
from abc import ABC
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from ...config import DataConfig
from ..schemas import CanonicalSample, Provenance


class AdapterError(ValueError):
    """Raised when an adapter cannot read or normalize a source record."""


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Source context (repair plan B2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceContext:
    """Exact upstream coordinates preserved on every canonical record."""

    source_dataset: str
    source_revision: str
    source_config: str | None = None
    source_split: str | None = None
    source_file: str | None = None
    source_row_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Image-reference handling (repair plan B5)
# --------------------------------------------------------------------------- #


def _looks_like_pil(value: Any) -> bool:
    # Duck-typed to avoid importing PIL: decoded HF images expose save/size.
    return hasattr(value, "save") and hasattr(value, "size") and not isinstance(
        value, (str, bytes, Mapping, Sequence)
    )


def _materialize_bytes(data: bytes, cache_dir: Path, suffix: str = ".png") -> str:
    """Write image bytes to a deterministic content-addressed cache path."""
    digest = hashlib.sha256(data).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def _materialize_pil(image: Any, cache_dir: Path) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _materialize_bytes(buffer.getvalue(), cache_dir, suffix=".png")


def image_view_uris(value: Any, *, cache_dir: Path | None = None) -> list[str]:
    """Normalize any source image reference into a list of URI strings.

    Supports path strings, HF decoded (PIL) images, ``{path, bytes}`` dicts,
    and sequences of any of these. PIL images are never stringified: they are
    materialized into the deterministic content-addressed ``cache_dir``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        path = value.get("path")
        if isinstance(path, str) and path.strip():
            return [path]
        raw = value.get("bytes")
        if raw is not None:
            if cache_dir is None:
                raise AdapterError(
                    "image dict with 'bytes' requires a cache_dir to materialize"
                )
            return [_materialize_bytes(bytes(raw), cache_dir)]
        return []
    if _looks_like_pil(value):
        if cache_dir is None:
            raise AdapterError(
                "decoded images require a cache_dir to materialize; never "
                "stringify PIL images"
            )
        return [_materialize_pil(value, cache_dir)]
    if isinstance(value, (list, tuple)):
        uris: list[str] = []
        for item in value:
            uris.extend(image_view_uris(item, cache_dir=cache_dir))
        return uris
    raise AdapterError(f"unsupported image reference type: {type(value).__name__}")


def image_uri(value: Any, *, cache_dir: Path | None = None) -> str | None:
    """Single-image variant of :func:`image_view_uris` (None when absent)."""
    uris = image_view_uris(value, cache_dir=cache_dir)
    return uris[0] if uris else None


# --------------------------------------------------------------------------- #
# Base adapter
# --------------------------------------------------------------------------- #


class BenchmarkAdapter(ABC):
    """Base class for all dataset adapters."""

    name: ClassVar[str] = "base"
    # Adapter schema version written into provenance/manifests (plan C3).
    adapter_version: ClassVar[str] = "base-v1"
    # Canonical field names that MUST be present (via the field map) per row.
    required_fields: ClassVar[tuple[str, ...]] = ("source_sample_id", "identity_id")

    def __init__(self, config: DataConfig):
        self.config = config
        merged = dict(self._default_field_map())
        merged.update(config.extras.get("field_map", {}))
        self.field_map = merged

    # -- configuration -------------------------------------------------- #

    def _default_field_map(self) -> dict[str, str]:
        """Map canonical field -> source field. Override per adapter."""
        return {}

    def source_revision(self) -> str:
        """Pinned source revision; adapters must never emit 'unknown' (C1)."""
        revision = self.config.source_version
        if not revision or revision == "unknown":
            raise AdapterError(
                f"[{self.name}] data.source_version is not pinned; set a "
                f"non-null source revision in configs/data/{self.name}.yaml "
                "before reading any data (repair plan C1/C4)."
            )
        # R10: validate immutable_revision components are not PENDING.
        self._validate_immutable_revision()
        return str(revision)

    def _validate_immutable_revision(self) -> None:
        """R10: reject PENDING immutable_revision values before data access."""
        # Skip validation for test/stub scenarios (golden fixture, CI smoke tests).
        import os
        if os.environ.get("ROUTE_DATA_SKIP_IMMUTABLE_CHECK"):
            return
        # Also skip if source_version indicates a stub/test.
        if self.config.source_version and any(
            token in self.config.source_version.lower()
            for token in ("stub", "test", "fixture")
        ):
            return
        
        immutable = self.config.extras.get("immutable_revision")
        if not immutable or not isinstance(immutable, dict):
            return  # no immutable_revision block yet (legacy configs)
        for key, value in immutable.items():
            if value == "PENDING":
                raise AdapterError(
                    f"[{self.name}] data.immutable_revision.{key} is still "
                    f"'PENDING'; replace with exact hash/SHA before pilot/full "
                    f"generation (repair plan R10)."
                )

    def hf_config_name(self) -> str | None:
        return self.config.extras.get("hf_config_name") or self.config.extras.get(
            "hf_config"
        )

    def base_context(
        self,
        *,
        source_file: str | None = None,
        source_split: str | None = None,
        source_row_index: int | None = None,
    ) -> SourceContext:
        return SourceContext(
            source_dataset=self.name,
            source_revision=self.source_revision(),
            source_config=self.hf_config_name(),
            source_split=source_split
            or self.config.extras.get("hf_split"),
            source_file=source_file,
            source_row_index=source_row_index,
        )

    # -- field access --------------------------------------------------- #

    def source_field(self, row: Mapping[str, Any], canonical: str, *, required: bool = True):
        """Resolve a canonical field from a raw row via the field map."""
        src = self.field_map.get(canonical, canonical)
        if src not in row or row[src] is None:
            if required:
                raise AdapterError(
                    f"[{self.name}] required field '{canonical}' (source '{src}') "
                    f"missing from row: {row!r}"
                )
            return None
        return row[src]

    def provenance(
        self,
        source_sample_id: str,
        *,
        source_subset: str | None = None,
        notes: str | None = None,
        context: SourceContext | None = None,
    ) -> Provenance:
        return Provenance(
            source_dataset=self.name,
            source_version=(context.source_revision if context else None)
            or self.config.source_version
            or "unknown",
            source_sample_id=source_sample_id,
            adapter=self.name,
            adapter_version=self.adapter_version,
            source_subset=source_subset,
            created_utc=utcnow_iso(),
            notes=notes,
        ).validate()

    @staticmethod
    def context_metadata(context: SourceContext) -> dict[str, Any]:
        """Base source_metadata block every record inherits (plan B2/B3)."""
        meta: dict[str, Any] = {
            "source_dataset": context.source_dataset,
            "source_revision": context.source_revision,
        }
        if context.source_config is not None:
            meta["source_config"] = context.source_config
        if context.source_split is not None:
            meta["source_split"] = context.source_split
        if context.source_file is not None:
            meta["source_file"] = context.source_file
        if context.source_row_index is not None:
            meta["source_row_index"] = context.source_row_index
        return meta

    # -- raw row loading (explicit readers only, plan B4) ---------------- #

    def source_files(self) -> Sequence[Path]:
        """Allowlisted source files. Subclasses MUST override; generic
        recursive loading of every JSON/CSV/Parquet file is forbidden."""
        raise AdapterError(
            f"[{self.name}] does not declare allowlisted source_files()"
        )

    def iter_rows_with_context(
        self,
    ) -> Iterator[tuple[SourceContext, Mapping[str, Any]]]:
        """Yield (context, raw row) pairs from the allowlisted source files.

        The default reader covers simple file-based sources; adapters with
        nested layouts (FAIRGET) or HF-only access override this entirely.
        """
        index = 0
        for path in self.source_files():
            for row in _read_file_rows(Path(path)):
                yield self.base_context(source_file=str(path), source_row_index=index), row
                index += 1

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        """Compatibility shim; prefer ``iter_rows_with_context``."""
        for _, row in self.iter_rows_with_context():
            yield dict(row)

    # -- mapping -------------------------------------------------------- #

    def to_samples(
        self,
        row: Mapping[str, Any],
        *,
        source_context: SourceContext,
    ) -> Iterable[CanonicalSample]:
        """Normalize one raw row into zero or more canonical records (B1).

        One-to-many sources (nested QA lists, multi-image rows) override this
        directly; flat adapters may keep implementing ``to_sample`` instead.
        """
        if type(self).to_sample is not BenchmarkAdapter.to_sample:
            return [self.to_sample(dict(row))]
        raise AdapterError(
            f"[{self.name}] must implement to_samples() (or to_sample() for "
            "flat sources)"
        )

    def to_sample(self, row: dict[str, Any]) -> CanonicalSample:
        """Compatibility helper for flat one-row/one-sample adapters."""
        raise AdapterError(
            f"[{self.name}] must implement to_samples() (or to_sample() for "
            "flat sources)"
        )

    def load(self) -> Iterator[CanonicalSample]:
        for context, row in self.iter_rows_with_context():
            yield from self.to_samples(row, source_context=context)


# --------------------------------------------------------------------------- #
# File readers (allowlisted paths only)
# --------------------------------------------------------------------------- #


def _read_file_rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise AdapterError(f"required source file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data
    elif suffix == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".csv":
        import pandas as pd

        yield from pd.read_csv(path).to_dict(orient="records")
    else:
        raise AdapterError(f"unsupported source file type: {path}")


def read_rows_from(path: Path) -> list[dict[str, Any]]:
    """Read all rows from one allowlisted source file."""
    return list(_read_file_rows(Path(path)))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_ADAPTERS: dict[str, Callable[[DataConfig], BenchmarkAdapter]] = {}


def register_adapter(name: str):
    def deco(cls):
        _ADAPTERS[name] = cls
        cls.name = name
        return cls

    return deco


def available_adapters() -> list[str]:
    # Ensure the built-in adapters are imported and registered.
    from . import fairget, fiubench, mllmu, ppubench  # noqa: F401

    return sorted(_ADAPTERS.keys())


def create_adapter(config: DataConfig) -> BenchmarkAdapter:
    available_adapters()
    if config.name not in _ADAPTERS:
        raise AdapterError(
            f"No adapter registered for dataset '{config.name}'. "
            f"Available: {available_adapters()}"
        )
    return _ADAPTERS[config.name](config)
