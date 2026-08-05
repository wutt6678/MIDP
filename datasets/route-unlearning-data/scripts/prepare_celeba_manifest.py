#!/usr/bin/env python
"""Validate the local CelebA root and build wide/long manifests.

Thin wrapper around ``route-data celeba prepare`` (plan section 20). All
command-line flags are passed through (--config, --dry-run, --limit, ...).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from route_data.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["celeba", "prepare", *sys.argv[1:]]))
