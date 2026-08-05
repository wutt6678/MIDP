"""Shared pytest configuration.

The package is not pip-installed in CI; insert ``src/`` (and the tests
directory, for the redistributable golden fixture) on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS_DIR = Path(__file__).resolve().parent

for _path in (SRC, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def golden_root(tmp_path: Path) -> Path:
    """Materialize the redistributable golden fixture and return its root."""
    from fixtures.golden_fixture import build_golden_fixture

    root = tmp_path / "golden_root"
    build_golden_fixture(root)
    return root
