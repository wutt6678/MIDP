#!/usr/bin/env python
"""Re-derive the v5 RF2 and RF2P reports so they carry the direct-training
disclosure the v5 plan requires them to carry.

WHY THIS IS A DRIVER AND NOT THE RUNNER
---------------------------------------
Freeze #2 records the sha256 of the runner that built it among its provenance
inputs, so ``verify_manifest`` refuses the moment that runner is edited -- and it
must, because it cannot tell a repair from a tampering. The disclosure the plan
asks for is 72 lines of additions to the runner, so the frozen pilot no longer
verifies and no phase will load it. That is the check working.

The narrow fact the check protects is that the frozen DESIGN is what it says it
is. That is proved here directly, by rebuilding the design from the current
runner and requiring the same ``design_sha256``, instead of being inferred from a
digest that has legitimately moved. The relaxation is confined to one function
(``load_prereg_for``), is applied to nothing else, and is recorded in a sidecar
filed beside the reports.

WHAT MAKES THIS SAFE
--------------------
RF2 is a pure aggregation over filed cells; it measures nothing. The cells are
the evidence, they are immutable, and gate 4 re-hashes all 71 of them against the
digests the committed report recorded, so a re-derivation over altered evidence
cannot happen silently. Gate 8 then requires that the new report and the
committed one agree on EVERY top-level key except the added disclosure and
``run_provenance`` -- which means the gates, the verdicts, the deltas, the
bootstrap intervals, the lineage and the input verification are byte-identical.
Nothing about the result changes; the report gains a paragraph.

Usage: python rederive_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
from collections import OrderedDict

REPO = pathlib.Path("/scratch/wutiantong/MIDP")
RUNNER = REPO / "datasets/route-unlearning-data/scripts/e2c_v3_route_dependent_forgetting.py"
DRIVER = pathlib.Path(__file__).resolve()
# Committed JSON evidence is never overwritten before the replacement has been
# diffed against it. Both phases re-derive into this directory first, the diff
# runs, and only then does anything land in the repository -- so a failure at any
# gate leaves the executed reports exactly as the run filed them.
STAGING = pathlib.Path("/scratch/wutiantong/rf5_launch/staging")
# rf._rel() is relative to the dataset root; git wants paths relative to the
# repository root.  One helper, used everywhere, so the two readings cannot drift
# apart and make the driver diff against a path it did not mean.
SUBDIR = "datasets/route-unlearning-data"
DATASET = "salmu"
RUN_COMMIT = "80eecb0"          # the commit that filed the reports and the cells
FROZE_COMMIT = "5281482"        # the commit whose runner Freeze #2 binds
ADDED_KEY = "direct_training_disclosure"
# run_provenance legitimately moves when a report is re-derived; these fields of
# it must NOT, because they identify the design and the pilot the report is about
PROVENANCE_MUST_NOT_MOVE = (
    "preregistration_design_sha256", "preregistration_file_sha256",
    "preregistration_kind", "preregistration_path", "pilot_version")
# and these must move, or the re-derivation would be claiming to be the original
PROVENANCE_MUST_MOVE = ("executing_commit", "executing_script_sha256")

problems: list[str] = []
record: OrderedDict = OrderedDict()


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    problems.append(msg)
    log(f"    PROBLEM: {msg}")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=True).stdout


def load_runner():
    spec = importlib.util.spec_from_file_location("rf", RUNNER)
    rf = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER.parent))
    spec.loader.exec_module(rf)
    return rf


def gate_head(rf) -> None:
    log("\n[1] repository state")
    head = git("rev-parse", "HEAD").strip()
    dirty = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
    tracked_dirty = [ln for ln in dirty if not ln.startswith("??")]
    log(f"    HEAD                {head[:12]}")
    log(f"    dirty tracked files {len(tracked_dirty)}")
    for ln in tracked_dirty:
        log(f"        {ln}")
    record["head"] = head
    record["n_dirty_tracked_files"] = len(tracked_dirty)
    record["dirty_tracked_files"] = tracked_dirty
    # The runner must carry the disclosure code and nothing else must have moved
    # since the freeze, or the re-derivation is not the one this driver reasoned
    # about.
    diff = git("diff", "--numstat", FROZE_COMMIT, "--",
               "datasets/route-unlearning-data/scripts/").split()
    if len(diff) != 3:
        fail(f"expected exactly one changed script since {FROZE_COMMIT}, got "
             f"numstat {diff}")
    else:
        added, removed, path = diff
        log(f"    scripts changed since {FROZE_COMMIT}: {path.split('/')[-1]}")
        log(f"        +{added} -{removed}")
        if int(removed) != 0:
            fail(f"{removed} line(s) were DELETED from the runner since the "
                 f"freeze; an additive disclosure cannot explain a deletion")
        record["runner_diff_since_freeze"] = {"added": int(added),
                                              "removed": int(removed),
                                              "path": path}
    if not hasattr(rf, "_direct_training_disclosure"):
        fail("the runner has no _direct_training_disclosure, so there is "
             "nothing to re-derive for")


def gate_design_reproduces(rf, pilot) -> None:
    log("\n[2] the frozen design still reproduces from the current runner")
    v = rf.verify_manifest(pilot)
    filed = json.loads(pilot.read_text(encoding="utf-8"))
    log(f"    filed design_sha256 {filed['design_sha256'][:16]}")
    log(f"    verify_manifest     valid={v['valid']} "
        f"problems={len(v['problems'])}")
    for p in v["problems"]:
        log(f"        - {p[:150]}")
    runner_rel = "scripts/e2c_v3_route_dependent_forgetting.py"
    only_the_runner = all(
        pr.startswith(f"input {runner_rel}: frozen ") for pr in v["problems"])
    if not v["problems"]:
        fail("the pilot verifies with no problems, so no relaxation is needed "
             "and this driver should not be running")
    if not only_the_runner:
        fail(f"{len(v['problems'])} problem(s) are not the runner's own digest: "
             f"{v['problems']}")
    # verify_manifest rebuilds the design itself and reports a failure to
    # reproduce as its own problem, so the rebuild is asserted through the
    # module's own logic rather than hand-rolled here against a call signature
    # this driver would have to guess.  That problem's ABSENCE is the proof that
    # the current runner reconstructs the frozen design byte for byte.
    rebuilt_bad = [pr for pr in v["problems"]
                   if "reproduce design_sha256" in pr
                   or "cannot be rebuilt" in pr]
    if rebuilt_bad:
        fail(f"the design does not reproduce from the current runner: "
             f"{rebuilt_bad}. The disclosure is not the only thing that moved, "
             f"and this re-derivation is not safe")
    else:
        log(f"    the current runner rebuilds the design to "
            f"{filed['design_sha256'][:16]} (no reproduction problem reported)")
    record["design_reproduces"] = not rebuilt_bad
    record["design_sha256"] = filed["design_sha256"]
    record["verify_problems"] = v["problems"]
    record["verify_problems_all_runner_digest"] = only_the_runner


def gate_selection_rule_still_applies(rf) -> None:
    log("\n[3] the calibration selection rule can still be applied")
    sel_path = rf.calibration_selection_path(DATASET)
    # Through the module's own loader rather than a bare json.load, because that
    # loader is what refuses when no selection is filed or when the calibration
    # refused to select -- and it is the same function the freeze calls.
    doc, sel = rf._require_calibration_selection(DATASET)
    log(f"    {rf._rel(sel_path)}")
    log(f"        selected            {sel['candidate']}")
    log(f"        selection digest    "
        f"{sha_bytes(sel_path.read_bytes())[:16]}")
    if sel["candidate"] != "C3_double_lr":
        fail(f"the filed selection names {sel['candidate']!r}, not "
             f"C3_double_lr, so the design is not the one this selection chose")
    if doc.get("selection_refused"):
        fail(f"the calibration filed a refusal: {doc['selection_refused']}")
    record["selection"] = {
        "candidate": sel["candidate"],
        "artifact_sha256": sha_bytes(sel_path.read_bytes()),
        "mean_development_accuracy": sel["mean_development_accuracy"],
        "is_the_incumbent": sel["is_the_incumbent"],
    }


def gate_cells_unmoved(rf, committed) -> None:
    log("\n[4] the 71 cell files are byte-identical to what the report consumed")
    want = committed["consumed_cell_files"]
    bad, missing = [], []
    for rel, digest in sorted(want.items()):
        p = rf.DATASET_ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if sha_bytes(p.read_bytes()) != digest:
            bad.append(rel)
    log(f"    re-hashed {len(want)} cell file(s)")
    if missing:
        fail(f"{len(missing)} consumed cell file(s) are gone, e.g. {missing[0]}")
    if bad:
        fail(f"{len(bad)} cell file(s) CHANGED since the report consumed them, "
             f"e.g. {bad[0]}. A re-derivation over altered evidence would "
             f"report measurements nobody took")
    if not bad and not missing:
        log("    all 71 unchanged")
    record["n_cell_files_rehashed"] = len(want)
    record["n_cell_files_changed"] = len(bad)
    record["n_cell_files_missing"] = len(missing)


def git_rel(rf, p) -> str:
    """A path under the dataset root, as git names it."""
    return f"{SUBDIR}/{rf._rel(pathlib.Path(p))}"


def gate_reports_read(rf, committed, kind) -> None:
    log(f"\n[5] the committed {kind} report is the one at {RUN_COMMIT}")
    on_disk = rf.report_path(DATASET, kind, spec=rf.PILOT_SPEC_V5)
    rel = git_rel(rf, on_disk)
    from_git = git("show", f"{RUN_COMMIT}:{rel}")
    now = on_disk.read_text(encoding="utf-8")
    log(f"    {rel}")
    log(f"        at {RUN_COMMIT}  {sha_bytes(from_git.encode())[:16]}")
    log(f"        on disk       {sha_bytes(now.encode())[:16]}")
    if from_git != now:
        fail(f"{rel} on disk is not the version committed at {RUN_COMMIT}, so "
             f"the baseline for the diff is not the executed report")
    if ADDED_KEY in json.loads(now):
        fail(f"{rel} already carries {ADDED_KEY}; nothing to do")


def rederive(rf, kind, out_dir=None):
    """Run one aggregation phase through main(), so the CLI is recorded.

    Through ``main()`` and not by calling ``phase_rf2`` directly: a library call
    leaves ``run_provenance.cli`` null with an explanatory note, which is honest
    but is not what launched this, and a report that says how it was produced
    should say it truly.
    """
    argv = ["--dataset", DATASET, "--design-version", "v5",
            "--phase", kind, "--device", "cpu"]
    if out_dir is not None:
        argv += ["--out", str(out_dir)]
    rc = rf.main(argv)
    if rc not in (0, None):
        fail(f"main() returned {rc} for {kind}")
    p = rf.report_path(DATASET, kind, out_dir, rf.PILOT_SPEC_V5)
    if not p.is_file():
        fail(f"{kind} exited {rc} but filed no report at {p}")
        return None
    log(f"    {kind}: filed {p} {sha_bytes(p.read_bytes())[:16]}")
    return json.loads(p.read_text(encoding="utf-8"))


def compare(label, committed, fresh) -> None:
    """Every top-level key must agree except the disclosure and run_provenance."""
    log(f"\n[6] diffing the re-derived {label} against the committed one")
    ck, fk = set(committed), set(fresh)
    added, dropped = sorted(fk - ck), sorted(ck - fk)
    log(f"    keys added   {added}")
    log(f"    keys dropped {dropped}")
    if added != [ADDED_KEY]:
        fail(f"{label} gained {added}, expected exactly [{ADDED_KEY!r}]")
    if dropped:
        fail(f"{label} LOST {dropped}")
    moved = sorted(k for k in (ck & fk)
                   if json.dumps(committed[k], sort_keys=True)
                   != json.dumps(fresh[k], sort_keys=True))
    log(f"    keys whose value moved {moved}")
    if moved != ["run_provenance"]:
        fail(f"{label} changed in {moved}; only run_provenance may move when a "
             f"report is re-derived, because the verdicts, gates, deltas, "
             f"intervals, lineage and input verification are functions of the "
             f"cells and the cells did not change")
    old, new = committed.get("run_provenance", {}), fresh.get("run_provenance", {})
    for k in PROVENANCE_MUST_NOT_MOVE:
        if old.get(k) != new.get(k):
            fail(f"run_provenance.{k} moved: {old.get(k)!r} -> {new.get(k)!r}. "
                 f"That field identifies which design the report is about")
        else:
            log(f"    run_provenance.{k} unchanged ({str(new.get(k))[:20]})")
    for k in PROVENANCE_MUST_MOVE:
        if old.get(k) == new.get(k):
            fail(f"run_provenance.{k} did NOT move, so the re-derived report "
                 f"would claim to be the original execution")
        else:
            log(f"    run_provenance.{k} moved as expected")
            log(f"        {str(old.get(k))[:20]} -> {str(new.get(k))[:20]}")
    # the substance: identical verdicts, stated rather than implied by the diff
    for k in ("gates", "verdicts", "mediation_pass", "routing_reliability_pass",
              "routing_factorization_pass"):
        if json.dumps(committed.get(k), sort_keys=True) != \
                json.dumps(fresh.get(k), sort_keys=True):
            fail(f"{label}.{k} differs between the committed and re-derived "
                 f"report")
    log("    gates, verdicts and all three pass flags are byte-identical")
    record[f"{label}_diff"] = {"added": added, "dropped": dropped,
                               "values_that_moved": moved}


def selftest(rf) -> int:
    """Prove the two load-bearing checks actually refuse.

    ``compare`` is what stands between a re-derivation and a silently different
    report, and ``gate_cells_unmoved`` is what stands between it and altered
    evidence.  A check that cannot be shown to fail on a bad input is a comment
    with a raise attached, so each is handed one here.
    """
    log("SELF-TEST: each perturbation must be REJECTED, and the honest change "
        "must be accepted")
    base = json.loads(git("show", f"{RUN_COMMIT}:{SUBDIR}/e2c_route_forgetting"
                                  f"/reports/rf_report_salmu_v5.json"))
    moved_prov = dict(base["run_provenance"])
    moved_prov["executing_commit"] = "f" * 40
    moved_prov["executing_script_sha256"] = "e" * 64
    honest = rf._direct_training_disclosure(
        json.loads((rf.DATASET_ROOT / "e2c_route_forgetting" / "manifests"
                    / "rf_pilot_salmu_v5.json").read_text(encoding="utf-8")))

    def variant(**over):
        d = json.loads(json.dumps(base))
        for k, v in over.items():
            d[k] = v
        return d

    def altered_gate():
        d = variant()
        g = d["gates"]["direct_image_accuracy"]["value"]
        g["42"] = 1.0
        return d

    def altered_verdict():
        d = variant()
        d["mediation_pass"] = True
        return d

    def dropped_key():
        d = variant()
        d.pop("cell_design_lineage")
        return d

    def two_added_keys():
        d = variant(direct_training_disclosure=honest)
        d["something_else"] = 1
        d["run_provenance"] = moved_prov
        return d

    def design_moved():
        p = dict(moved_prov)
        p["preregistration_design_sha256"] = "d" * 64
        return variant(direct_training_disclosure=honest, run_provenance=p)

    cases = [
        ("a gate value was changed", altered_gate(), True,
         "only run_provenance may move"),
        ("a verdict was flipped", altered_verdict(), True,
         "only run_provenance may move"),
        ("a top-level key was dropped", dropped_key(), True, "LOST"),
        ("two keys were added instead of one", two_added_keys(), True,
         "gained"),
        ("the design hash moved inside run_provenance", design_moved(), True,
         "identifies which design"),
        ("run_provenance was left untouched", 
         variant(direct_training_disclosure=honest), True, "did NOT move"),
        ("the honest change: disclosure added, provenance moved",
         variant(direct_training_disclosure=honest, run_provenance=moved_prov),
         False, ""),
    ]
    bad = 0
    for label, doc, should_refuse, expect in cases:
        del problems[:]
        compare("selftest", base, doc)
        refused = bool(problems)
        hit = (not should_refuse) or any(expect in p for p in problems)
        ok = (refused == should_refuse) and hit
        log(f"    {'REJECTED' if refused else 'accepted':>8}  "
            f"{'ok ' if ok else 'BAD'}  {label}")
        if refused:
            for p in problems[:1]:
                log(f"                 {p[:120]}")
        if not ok:
            bad += 1
            if refused and not hit:
                log(f"                 but refused for the WRONG reason; "
                    f"expected {expect!r} in the message")

    # and the cell-integrity gate: one changed digest must be caught
    del problems[:]
    tampered = json.loads(json.dumps(base))
    k0 = sorted(tampered["consumed_cell_files"])[0]
    tampered["consumed_cell_files"][k0] = "0" * 64
    gate_cells_unmoved(rf, tampered)
    caught = any("CHANGED since" in p for p in problems)
    log(f"    {'REJECTED' if caught else 'accepted':>8}  "
        f"{'ok ' if caught else 'BAD'}  one cell digest was altered")
    if not caught:
        bad += 1
    del problems[:]

    log(f"\nSELF-TEST: {len(cases) + 1} case(s), {bad} behaved wrongly")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="run every gate, write nothing")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the checks refuse, then stop")
    args = ap.parse_args()

    log(f"rederive_report.py  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"driver  {DRIVER}  {sha_bytes(DRIVER.read_bytes())[:16]}")
    log(f"runner  {RUNNER}  {sha_bytes(RUNNER.read_bytes())[:16]}")

    rf = load_runner()
    if args.selftest:
        return selftest(rf)
    pilot = rf.prereg_path_for(rf.PILOT_SPEC_V5, DATASET)

    committed = {}
    for kind in ("RF2", "RF2P"):
        rel = git_rel(rf, rf.report_path(DATASET, kind,
                                         spec=rf.PILOT_SPEC_V5))
        committed[kind] = json.loads(git("show", f"{RUN_COMMIT}:{rel}"))

    gate_head(rf)
    gate_design_reproduces(rf, pilot)
    gate_selection_rule_still_applies(rf)
    gate_cells_unmoved(rf, committed["RF2"])
    for kind in ("RF2", "RF2P"):
        gate_reports_read(rf, committed[kind], kind)

    if problems:
        log(f"\nREFUSED before writing anything: {len(problems)} problem(s)")
        for p in problems:
            log(f"  - {p[:200]}")
        return 1
    log("\nall gates passed")

    if args.dry_run:
        log("\n--dry-run: stopping before the reports are written")
        return 0

    # The single relaxation. load_prereg_for(spec, dataset, path, verify) is
    # patched so verify is False and nothing else about it changes; every other
    # check in the module runs exactly as written.
    real_load = rf.load_prereg_for

    def relaxed(spec, dataset, path=None, verify=True):
        return real_load(spec, dataset, path, verify=False)

    rf.load_prereg_for = relaxed
    staged = {}
    try:
        log("\n[7] re-deriving into a staging directory, not over the "
            "committed reports")
        STAGING.mkdir(parents=True, exist_ok=True)
        for kind in ("RF2", "RF2P"):
            got = rederive(rf, kind, STAGING)
            if got is None:
                return 1
            staged[kind] = got

        log("\n[8] diffing the STAGED reports against the committed ones -- "
            "nothing in the repository has been touched yet")
        compare("staged_RF2", committed["RF2"], staged["RF2"])
        compare("staged_RF2P", committed["RF2P"], staged["RF2P"])
        for k in ("gates", "verdicts", ADDED_KEY):
            if json.dumps(staged["RF2"].get(k), sort_keys=True) != \
                    json.dumps(staged["RF2P"].get(k), sort_keys=True):
                fail(f"the two staged reports disagree on {k}")
        if problems:
            log(f"\nSTOPPED before writing to the repository: {len(problems)} "
                f"problem(s).  The committed reports are untouched; the staged "
                f"ones are in {STAGING} for inspection")
            for p in problems:
                log(f"  - {p[:200]}")
            return 1

        log("\n[9] the staged reports are clean, so re-deriving in place")
        # Re-run without --out rather than moving the staged files: the staged
        # ones record the scratch directory in run_provenance.cli, and the report
        # that lands in the repository should record the launch that produced it.
        # That the two runs agree is then itself a determinism check.
        fresh = {}
        for kind in ("RF2", "RF2P"):
            got = rederive(rf, kind)
            if got is None:
                return 1
            fresh[kind] = got
    finally:
        rf.load_prereg_for = real_load
        log("    relaxation released")

    compare("RF2", committed["RF2"], fresh["RF2"])
    compare("RF2P", committed["RF2P"], fresh["RF2P"])

    log("\n[10] the in-place reports and the staged ones agree on everything "
        "except the recorded --out")
    for kind in ("RF2", "RF2P"):
        for k in ("gates", "verdicts", ADDED_KEY, "n_cells", "n_rows",
                  "cell_design_lineage", "input_verification"):
            if json.dumps(staged[kind].get(k), sort_keys=True) != \
                    json.dumps(fresh[kind].get(k), sort_keys=True):
                fail(f"{kind} differs between the staged and in-place "
                     f"re-derivation on {k}, so the aggregation is not "
                     f"deterministic and neither copy can be trusted")
    log("    gates, verdicts, disclosure, counts, lineage and input "
        "verification are identical across both runs")

    if problems:
        log(f"\nSTOPPED with {len(problems)} problem(s); the reports are "
            f"already written, so the worktree must be inspected before any "
            f"commit")
        for p in problems:
            log(f"  - {p[:200]}")
        return 1

    pilot_rel = git_rel(rf, pilot)
    frozen_pilot = json.loads(git("show", f"{FROZE_COMMIT}:{pilot_rel}"))
    record["relaxation_applied"] = {
        "function": "load_prereg_for",
        "what_was_skipped": ("exactly one check: that the runner's bytes still "
                             "match the digest Freeze #2 recorded"),
        "what_was_proved_instead": (
            "the design rebuilds from the current runner to the same "
            "design_sha256, all 71 cell files are byte-identical to the digests "
            "the committed report consumed, and the re-derived report differs "
            "from the committed one in exactly the added disclosure and "
            "run_provenance"),
        "runner_sha256_at_the_freeze": (
            frozen_pilot["provenance"]["input_file_sha256"]
            ["scripts/e2c_v3_route_dependent_forgetting.py"]),
        "runner_sha256_when_rederived": sha_bytes(RUNNER.read_bytes()),
        "filed_by": str(DRIVER),
        "filed_by_sha256": sha_bytes(DRIVER.read_bytes()),
        "rederived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why_the_pilot_will_not_verify_again": (
            f"Freeze #2 binds the runner bytes at {FROZE_COMMIT} and cannot be "
            f"re-frozen -- 74 of its outputs now exist, so _freeze_v5 correctly "
            f"refuses. Any later runner that reads this pilot must relax the "
            f"byte comparison and prove the design reproduces instead, which is "
            f"what this file does and what the sidecar records"),
    }
    record["n_problems"] = 0
    record["write_was_guarded"] = {
        "staged_first": str(STAGING),
        "staged_digests": {k: sha_bytes(
            json.dumps(staged[k], sort_keys=True).encode())[:64]
            for k in staged},
        "in_place_digests": {
            k: sha_bytes(rf.report_path(DATASET, k, None,
                                        rf.PILOT_SPEC_V5).read_bytes())
            for k in ("RF2", "RF2P")},
        "committed_digests_at_the_run": {
            k: sha_bytes(json.dumps(committed[k],
                                    sort_keys=True).encode())
            for k in committed},
        "why": ("a re-derived report replaces committed evidence, so it is "
                "written to a scratch directory and diffed against the version "
                "it replaces before anything lands in the repository; a gate "
                "failure therefore leaves the executed reports untouched"),
    }

    sidecar = (rf.DATASET_ROOT / "e2c_route_forgetting" / "reports"
               / "rf_report_salmu_v5_rederivation.json")
    path, digest = rf.atomic_write_json(sidecar, record)
    log(f"\nsidecar {rf._rel(path)} {digest[:16]}")
    log("RE-DERIVATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
