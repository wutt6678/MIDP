"""Benchmark adapter framework (coding plan sections 2.5, 12-15).

Adapters map benchmark-specific source fields into the canonical schema and
fail loudly when a required field is absent (plan section 2.5). They never
assume identical field names across repositories.

Each adapter reads raw rows from either a local directory of JSONL/Parquet/CSV
files (``data.root``) or, when configured, a Hugging Face dataset. Rows are
yielded lazily so large benchmarks are never fully materialized in memory.
"""

from __future__ import annotations

import datetime as _dt
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Iterator

from ...config import DataConfig
from ..schemas import CanonicalSample, Provenance


class AdapterError(ValueError):
    """Raised when an adapter cannot map a source record to the schema."""


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BenchmarkAdapter(ABC):
    """Base class for all dataset adapters."""

    name: ClassVar[str] = "base"
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

    # -- field access --------------------------------------------------- #

    def source_field(self, row: dict[str, Any], canonical: str, *, required: bool = True):
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
        self, source_sample_id: str, *, source_subset: str | None = None, notes: str | None = None
    ) -> Provenance:
        return Provenance(
            source_dataset=self.name,
            source_version=self.config.source_version or "unknown",
            source_sample_id=source_sample_id,
            adapter=self.name,
            source_subset=source_subset,
            created_utc=utcnow_iso(),
            notes=notes,
        ).validate()

    # -- raw row loading ------------------------------------------------- #

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        """Yield raw source rows from the configured local root or HF dataset."""
        hf_name = self.config.extras.get("hf_dataset_id") or self.config.extras.get(
            "hf_dataset"
        )
        if hf_name:
            yield from self._iter_hf_rows(hf_name)
            return
        root = self.config.require_root()
        yield from self._iter_local_rows(root)

    def _iter_local_rows(self, root: Path) -> Iterator[dict[str, Any]]:
        root = Path(root)
        if not root.exists():
            raise AdapterError(f"[{self.name}] data.root does not exist: {root}")
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jsonl", ".json", ".parquet", ".csv"}
        )
        if not files:
            raise AdapterError(
                f"[{self.name}] no .jsonl/.json/.parquet/.csv files under {root}"
            )
        for path in files:
            yield from _read_file_rows(path)

    def _iter_hf_rows(self, hf_name: str) -> Iterator[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise AdapterError(
                f"[{self.name}] loading '{hf_name}' requires the `datasets` package"
            ) from exc
        hf_config = self.config.extras.get("hf_config_name") or self.config.extras.get(
            "hf_config"
        )
        split = self.config.extras.get("hf_split")
        ds = load_dataset(hf_name, hf_config, split=split) if split else load_dataset(
            hf_name, hf_config
        )
        iterable = ds if isinstance(ds, dict) is False else _chain_splits(ds)
        for row in iterable:
            yield dict(row)

    # -- mapping -------------------------------------------------------- #

    @abstractmethod
    def to_sample(self, row: dict[str, Any]) -> CanonicalSample:
        """Map one raw source row to a validated canonical sample."""

    def load(self) -> Iterator[CanonicalSample]:
        for row in self.iter_rows():
            yield self.to_sample(row)


def _chain_splits(ds_dict) -> Iterator[dict[str, Any]]:
    for split_name, split in ds_dict.items():
        for row in split:
            rec = dict(row)
            rec.setdefault("_source_split", split_name)
            yield rec


def _read_file_rows(path: Path) -> Iterator[dict[str, Any]]:
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
