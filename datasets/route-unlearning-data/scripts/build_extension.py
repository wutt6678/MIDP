#!/usr/bin/env python
"""Run the full extension construction pipeline for one benchmark.

Runs the plan-20 build chain in order, stopping at the first failure:

    build annotate -> build qa -> build route-probes -> build splits
    -> build export

All command-line flags are passed through to every stage
(--dataset, --config, --dry-run, --limit, --resume, --output-dir).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from route_data.cli import main  # noqa: E402

_STAGES = ("annotate", "qa", "route-probes", "splits", "export")

if __name__ == "__main__":
    for stage in _STAGES:
        code = main(["build", stage, *sys.argv[1:]])
        if code != 0:
            raise SystemExit(code)
    raise SystemExit(0)
