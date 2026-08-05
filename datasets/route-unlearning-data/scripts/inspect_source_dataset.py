#!/usr/bin/env python
"""Inspect a source benchmark through its fail-loud adapter.

Thin wrapper around ``route-data source inspect`` (plan section 20). All
command-line flags are passed through (--dataset, --config, --limit, ...).
Exits non-zero if the adapter rejects the source schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from route_data.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["source", "inspect", *sys.argv[1:]]))
