#!/usr/bin/env python
"""Run a CelebA-40 evaluation with the configured backend.

Thin wrapper around ``route-data celeba evaluate`` (plan section 20). All
command-line flags are passed through (--config, --dry-run, --limit, ...).
Follow up with ``route-data celeba report --run-id <id>`` and
``route-data celeba freeze-protocol --run-id <id>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from route_data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["celeba", "evaluate", *sys.argv[1:]]))
