"""Final-verification driver: end-state commands from the repair plan.

Runs on the bundled golden fixture with the stub backend so the checks are
self-contained (no live model / restricted data required).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from fixtures.golden_fixture import build_golden_fixture  # noqa: E402
from route_data.cli import main  # noqa: E402

STUB_CFG = "configs/runs/golden_stub.yaml"
failures: list[str] = []


def run(label: str, argv: list[str], expect: int = 0) -> None:
    print(f"\n=== {label}: route-data {' '.join(argv)}")
    rc = main(argv)
    status = "OK" if rc == expect else "FAIL"
    if rc != expect:
        failures.append(label)
    print(f"--- {label}: rc={rc} (expected {expect}) [{status}]")


def main_check() -> int:
    work = REPO / "data" / "tmp_final_verify"
    golden_root = work / "golden_root"
    out = work / "out"
    if golden_root.exists():
        import shutil

        shutil.rmtree(golden_root)
    build_golden_fixture(golden_root)
    os.environ["FAIRGET_ROOT"] = str(golden_root)

    fairget_data_cfg = "configs/data/fairget.yaml"

    # -- source inspect -------------------------------------------------- #
    run(
        "source inspect fairget --limit 5",
        ["source", "inspect", "--dataset", "fairget", "--config", fairget_data_cfg, "--limit", "5"],
    )
    # Released sources for the other three benchmarks are not mirrored into
    # CI; source inspect must degrade to a clean rc=2 diagnostic (no crash).
    for name in ("fiubench", "mllmu", "ppubench"):
        run(
            f"source inspect {name} (no live source)",
            ["source", "inspect", "--dataset", name, "--config", f"configs/data/{name}.yaml", "--limit", "5"],
            expect=2,
        )

    # -- small end-to-end build with the stub backend -------------------- #
    for stage in ("annotate", "qa", "route-probes", "splits", "export"):
        run(
            f"build {stage} --limit 3 (stub)",
            [
                "build", stage, "--dataset", "fairget",
                "--config", STUB_CFG, "--output-dir", str(out), "--limit", "3",
            ],
        )

    # -- dry-run every build stage against the production run config ----- #
    # Annotate dry-run needs no prior artifacts; downstream stages load the
    # annotated samples built above, so they dry-run against ``out``.
    run(
        "build annotate dry-run (build_all_extensions)",
        [
            "build", "annotate", "--dataset", "fairget",
            "--config", "configs/runs/build_all_extensions.yaml",
            "--output-dir", str(work / "prod_out"), "--limit", "3", "--dry-run",
        ],
    )
    for stage in ("qa", "route-probes", "splits", "export"):
        run(
            f"build {stage} dry-run (build_all_extensions)",
            [
                "build", stage, "--dataset", "fairget",
                "--config", "configs/runs/build_all_extensions.yaml",
                "--output-dir", str(out), "--limit", "3", "--dry-run",
            ],
        )

    # -- strict validation on the built fixture --------------------------- #
    run(
        "validate dataset --strict",
        [
            "validate", "dataset", "--dataset", "fairget_celeba40",
            "--config", STUB_CFG, "--output-dir", str(out), "--strict",
        ],
    )

    print("\n=== SUMMARY:", "ALL CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main_check())
