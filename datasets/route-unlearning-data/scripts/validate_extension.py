#!/usr/bin/env python
"""Run every plan-19 validation check on a built extension.

Thin wrapper around ``route-data validate dataset`` (plan sections 19, 20).
All command-line flags are passed through (--dataset, --config, --strict).
Exit code 1 means a validation check failed; 2 means a config/IO error.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from route_data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", "dataset", *sys.argv[1:]]))
