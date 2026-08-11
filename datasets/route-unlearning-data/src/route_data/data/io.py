"""Parquet / JSONL / JSON I/O with atomic writes and resumable shards.

Serialization policy (plan section 5.4):
- Parquet for normalized tabular records (manifests, predictions, annotations);
- JSONL for conversational training/evaluation instances;
- YAML/JSON for run manifests and split definitions.

All writers write to a temp file then atomically rename, so an interrupted
run never leaves a truncated file at the destination path. Prediction shards
are appended as ``part-NNNNN.parquet`` so a long evaluation can resume after
interruption without holding the full prediction set in memory (plan section
8.5).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from typing_extensions import Self


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_text(path: str | Path, write_fn) -> None:
    ensure_parent_dir(path)
    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write_fn(f)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- #
# JSON / JSONL
# --------------------------------------------------------------------------- #


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    _atomic_write_text(path, lambda f: json.dump(data, f, indent=indent, default=str))


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    def _write(f):
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")

    _atomic_write_text(path, _write)


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Append rows to a JSONL file; returns the number of rows written."""
    ensure_parent_dir(path)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Parquet
# --------------------------------------------------------------------------- #


def write_parquet(rows: Iterable[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    """Write a list of flat records to Parquet atomically."""
    df = pd.DataFrame(list(rows))
    ensure_parent_dir(path)
    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp.parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return df


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_parquet(path).to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Incremental shard writer (resumable evaluation)
# --------------------------------------------------------------------------- #


class ParquetShardWriter:
    """Append prediction rows into fixed-size Parquet shards.

    A directory of ``part-*.parquet`` shards can be resumed after a crash:
    callers scan existing shards for already-completed keys and skip them.
    """

    def __init__(self, output_dir: str | Path, prefix: str = "part", shard_size: int = 1000):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.shard_size = max(1, shard_size)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._shard_index = self._next_shard_index()
        self._written = 0

    def _next_shard_index(self) -> int:
        existing = list(self.output_dir.glob(f"{self.prefix}-*.parquet"))
        return len(existing)

    def _shard_path(self, index: int) -> Path:
        return self.output_dir / f"{self.prefix}-{index:05d}.parquet"

    def add(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def add_many(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.add(row)

    def flush(self) -> None:
        if not self._buffer:
            return
        write_parquet(self._buffer, self._shard_path(self._shard_index))
        self._written += len(self._buffer)
        self._buffer = []
        self._shard_index += 1

    def close(self) -> None:
        self.flush()

    @property
    def rows_written(self) -> int:
        return self._written

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_shards(output_dir: str | Path, prefix: str = "part") -> pd.DataFrame:
    """Concatenate all ``prefix-*.parquet`` shards in a directory."""
    paths = sorted(Path(output_dir).glob(f"{prefix}-*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
