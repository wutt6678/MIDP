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

    # -- P2-22: post-build artifact existence checks ---------------------- #
    # Verify that every expected artifact was actually produced by the
    # export stage.  This catches regressions in the export pipeline
    # (e.g. missing checksums, manifest, or coverage report).
    import json as _json

    # Resolve the export directory: out / <model_dir> / fairget/
    # The stub config uses local/stub-vlm-v1 as model_id.
    model_dir_name = "local--stub-vlm-v1"
    export_dir = out / model_dir_name / "fairget"
    if not export_dir.exists():
        # Fallback: try the benchmark name directly under out/.
        export_dir = out / "fairget"

    required_artifacts = [
        "fairget_celeba40_image_annotations.parquet",
        "fairget_celeba40_visual_qa_train.jsonl",
        "fairget_celeba40_visual_qa_eval.jsonl",
        "fairget_route_conflict_eval.jsonl",
        "fairget_extension_card.md",
        # P1-15: checksums and export manifest.
        "fairget_checksums.json",
        "fairget_export_manifest.json",
    ]
    for artifact in required_artifacts:
        path = export_dir / artifact
        label = f"artifact exists: {artifact}"
        if not path.exists():
            failures.append(label)
            print(f"--- {label}: MISSING [{path}]")
        else:
            print(f"--- {label}: OK")

    # P1-15: verify checksums.json does not contain itself.
    checksums_path = export_dir / "fairget_checksums.json"
    if checksums_path.exists():
        label = "checksums.json excludes self-reference"
        try:
            ckdata = _json.loads(checksums_path.read_text())
            self_ref = "fairget_checksums.json" in ckdata
            if self_ref:
                failures.append(label)
                print(f"--- {label}: FAIL (self-reference found)")
            else:
                print(f"--- {label}: OK")
        except Exception as exc:
            failures.append(f"checksums.json parse: {exc}")

    # P1-15: verify manifest contains checksums path.
    manifest_path = export_dir / "fairget_export_manifest.json"
    if manifest_path.exists():
        label = "export manifest references checksums"
        try:
            mdata = _json.loads(manifest_path.read_text())
            paths = mdata.get("paths", {})
            if "checksums" in paths:
                print(f"--- {label}: OK")
            else:
                failures.append(label)
                print(f"--- {label}: FAIL (no checksums key in paths)")
        except Exception as exc:
            failures.append(f"manifest parse: {exc}")

    # P1-18: route-probe coverage report must exist and have required keys.
    intermediate_dir = out / model_dir_name if (out / model_dir_name).exists() else out
    # The coverage report is in the intermediate (dataset) directory.
    for candidate in [export_dir, intermediate_dir, out]:
        report_path = candidate / "fairget_route_probe_report.json"
        if report_path.exists():
            break
    # Also check the dataset_dir (before export) — the route-probes stage
    # writes it alongside the other intermediate artifacts.
    if not report_path.exists():
        # Search recursively under out/.
        for p in sorted(out.rglob("fairget_route_probe_report.json")):
            report_path = p
            break

    label = "route-probe coverage report"
    if report_path.exists():
        try:
            rdata = _json.loads(report_path.read_text())
            required_keys = {
                "identities_total",
                "identities_with_visual_anchors",
                "identities_with_profile_facts",
                "probe_families",
            }
            missing = required_keys - set(rdata)
            if missing:
                failures.append(f"{label}: missing keys {missing}")
                print(f"--- {label}: FAIL (missing {missing})")
            else:
                print(f"--- {label}: OK")
        except Exception as exc:
            failures.append(f"{label} parse: {exc}")
    else:
        failures.append(f"{label}: file not found")
        print(f"--- {label}: MISSING")

    print("\n=== SUMMARY:", "ALL CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main_check())
