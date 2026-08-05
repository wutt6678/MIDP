#!/usr/bin/env python
"""Render the dataset card markdown for a built extension.

Thin wrapper around ``route-data card render`` (plan sections 14, 20). All
command-line flags are passed through (--dataset, --config, --output-dir).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from route_data.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["card", "render", *sys.argv[1:]]))
