"""SHA-256 checksum helpers (coding plan sections 3.2, 7.3, 16.2).

Every image in a manifest carries ``image_sha256`` so that a corrupted or
silently swapped file is detectable. Checksums are also used to key the
prediction cache: one record per ``(image_sha256, model_fingerprint,
prompt_id, scoring_mode)`` (plan section 8.5).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    """Stream-hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_digest(value: str, length: int = 16) -> str:
    """Truncate a hex digest for compact identifiers (e.g. registry hashes)."""
    return value[:length]


def verify_checksum(path: str | Path, expected_sha256: str) -> bool:
    """Return True when the file's sha256 matches ``expected_sha256``."""
    actual = sha256_file(path)
    return actual.lower() == expected_sha256.lower()


def assert_checksum(path: str | Path, expected_sha256: str) -> None:
    """Raise ``ValueError`` when a file's checksum does not match."""
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
