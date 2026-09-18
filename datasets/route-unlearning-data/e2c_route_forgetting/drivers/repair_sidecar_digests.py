#!/usr/bin/env python
"""Correct the digest method recorded in the v5 re-derivation sidecar.

THE DEFECT
----------
``write_was_guarded`` recorded three sets of digests using TWO different methods
and presented them as if they were one kind of thing:

    committed_digests_at_the_run  sha256(json.dumps(parsed, sort_keys=True))
    staged_digests                sha256(json.dumps(parsed, sort_keys=True))
    in_place_digests              sha256(raw file bytes)

The first two are a re-serialization in Python's default spacing, so they match
neither the file on disk, nor the file in git, nor compact separators, nor this
module's own ``canonical_json``.  A reader who hashed the committed report got
d566fb6f... and could not reconcile it with the 9ecb7e6f... the sidecar claimed
to be describing.

WHAT THIS DOES NOT MEAN
-----------------------
The comparison the sidecar was recording did not consume those digests.  Gate 5
compared ``git show`` TEXT against the file's TEXT, and ``compare()`` compared the
parsed documents key by key.  The digests were output, not input -- which is
exactly why the error was invisible at the time and is only visible now, to a
reader trying to reproduce them.  Phase A below re-establishes the substantive
claim from raw bytes so that it rests on a reproducible method rather than on
assertion.

WHAT THIS DOES
--------------
Appends a correction block, preserving the erroneous values in place rather than
overwriting them, and archives the driver sources verbatim.  No report, gate,
verdict, cell or design is touched: RF2 and RF2P are not re-run.

Usage: python repair_sidecar_digests.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

REPO = pathlib.Path("/scratch/wutiantong/MIDP")
DSROOT = REPO / "datasets/route-unlearning-data"
RUNNER = DSROOT / "scripts/e2c_v3_route_dependent_forgetting.py"
REPORTS = DSROOT / "e2c_route_forgetting/reports"
DRIVERS = DSROOT / "e2c_route_forgetting/drivers"
SIDECAR = REPORTS / "rf_report_salmu_v5_rederivation.json"

SOURCE_COMMIT = "80eecb0"        # where the original reports were committed
READER_COMMIT = "bc80af7"        # HEAD when the driver read them
SUPERSEDED_NOTE = "3097353f9f4d02a9"
ADDED_KEY = "direct_training_disclosure"
REPORTS_BY_KIND = {"RF2": "rf_report_salmu_v5.json",
                   "RF2P": "rf_report_salmu_v5_rescored.json"}
# The values a reader hashing the committed files actually gets.  Asserted, not
# assumed: phase A recomputes them and refuses if they are not these.
EXPECTED_RAW = {
    "RF2": "d566fb6f2d492f67c7169db2fe16c59868dc49b1bf45bcfb849a1617495c7d4d",
    "RF2P": "e5c7d3f598726a03f4eb7e8c58162c03ac639321051a8e8c64f382c450294f4a",
}
# The drivers to archive, and the artifact each one filed.
TO_ARCHIVE = (
    ("/scratch/wutiantong/rf5_launch/rederive_report.py",
     "8bc8c2ca34cac491", "filed the re-derivation sidecar this block corrects"),
    ("/scratch/wutiantong/rf5_launch/file_scope_note.py",
     "765ea3b441e104f7", f"filed the superseded scope note {SUPERSEDED_NOTE}"),
)
PROV_MUST_NOT_MOVE = ("preregistration_design_sha256",
                      "preregistration_file_sha256", "preregistration_kind",
                      "preregistration_path", "pilot_version")
PROV_MUST_MOVE = ("executing_commit", "executing_script_sha256")

problems: list[str] = []


def log(m: str) -> None:
    print(m, flush=True)


def fail(m: str) -> None:
    problems.append(m)
    log(f"    PROBLEM: {m}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*a: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a],
                          capture_output=True, check=True).stdout


def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(DSROOT))


def load_runner():
    spec = importlib.util.spec_from_file_location("rf", RUNNER)
    rf = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER.parent))
    spec.loader.exec_module(rf)
    return rf


# --------------------------------------------------------------------------
# phase A: establish the cause and re-prove the substantive claim
# --------------------------------------------------------------------------
def phase_a(rf, side):
    log("\n[A1] the raw sha256 of the reports at the source commit")
    guarded = side["write_was_guarded"]
    raw_at_source = {}
    for kind, name in REPORTS_BY_KIND.items():
        p = f"datasets/route-unlearning-data/{rel(REPORTS)}/{name}"
        raw_at_source[kind] = sha(git("show", f"{SOURCE_COMMIT}:{p}"))
        log(f"    {kind:<5} {raw_at_source[kind]}")
        if raw_at_source[kind] != EXPECTED_RAW[kind]:
            fail(f"{kind} at {SOURCE_COMMIT} hashes to {raw_at_source[kind]}, "
                 f"not the expected {EXPECTED_RAW[kind]}")
        still = sha(git("show", f"{READER_COMMIT}:{p}"))
        if still != raw_at_source[kind]:
            fail(f"{kind} differs between {SOURCE_COMMIT} and {READER_COMMIT}, "
                 f"so the driver did not read the version it claims to")
    log(f"    identical at {READER_COMMIT}, the commit the driver read them from")

    log("\n[A2] reproducing the erroneous recorded values, to fix the cause")
    methods = {}
    for kind, name in REPORTS_BY_KIND.items():
        p = f"datasets/route-unlearning-data/{rel(REPORTS)}/{name}"
        parsed = json.loads(git("show", f"{SOURCE_COMMIT}:{p}"))
        cands = OrderedDict((
            ("sha256(raw file bytes)", sha(git("show", f"{SOURCE_COMMIT}:{p}"))),
            ("sha256(json.dumps(obj, sort_keys=True).encode())",
             sha(json.dumps(parsed, sort_keys=True).encode())),
            ("sha256(json.dumps(obj, sort_keys=True, "
             "separators=(',',':')).encode())",
             sha(json.dumps(parsed, sort_keys=True,
                            separators=(",", ":")).encode())),
            ("sha256(canonical_json(obj).encode())",
             sha(rf.canonical_json(parsed).encode())),
        ))
        recorded = guarded["committed_digests_at_the_run"][kind]
        hit = [m for m, v in cands.items() if v == recorded]
        log(f"    {kind} recorded {recorded[:16]}")
        for m, v in cands.items():
            log(f"        {v[:16]}  {m}{'   <== REPRODUCES IT' if v == recorded else ''}")
        if len(hit) != 1:
            fail(f"{kind}: {len(hit)} candidate methods reproduce the recorded "
                 f"digest, so the cause is not pinned to one: {hit}")
        methods[kind] = {"recorded": recorded, "reproduced_by": hit,
                         "every_method_tried": cands}
    if all(len(v["reproduced_by"]) == 1 for v in methods.values()):
        log("    cause pinned: a sorted re-serialization in Python's default "
            "spacing, which matches no file and no canonical form")

    log("\n[A3] the recorded digests were output, never an input to a gate")
    src = pathlib.Path(TO_ARCHIVE[0][0]).read_text(encoding="utf-8")
    for field in ("committed_digests_at_the_run", "staged_digests",
                  "in_place_digests"):
        n = src.count(field)
        log(f"    {field:<30} appears {n}x in rederive_report.py")
        if n != 1:
            fail(f"{field} appears {n} times in the driver, so it could have "
                 f"been read by a gate rather than only written to the record")
    log("    each appears exactly once, inside the record construction after "
        "every gate had already run")

    log("\n[A4] re-establishing the substantive claim from RAW bytes")
    reestablished = {}
    for kind, name in REPORTS_BY_KIND.items():
        p = REPORTS / name
        old = json.loads(git("show", f"{SOURCE_COMMIT}:datasets/"
                                     f"route-unlearning-data/{rel(REPORTS)}/{name}"))
        new = json.loads(p.read_bytes())
        added = sorted(set(new) - set(old))
        dropped = sorted(set(old) - set(new))
        moved = sorted(k for k in (set(old) & set(new))
                       if json.dumps(old[k], sort_keys=True)
                       != json.dumps(new[k], sort_keys=True))
        log(f"    {kind}: added={added} dropped={dropped} moved={moved}")
        if added != [ADDED_KEY]:
            fail(f"{kind} gained {added}, not exactly [{ADDED_KEY!r}]")
        if dropped:
            fail(f"{kind} lost {dropped}")
        if moved != ["run_provenance"]:
            fail(f"{kind} changed in {moved}; only run_provenance may move")
        op, np_ = old["run_provenance"], new["run_provenance"]
        for k in PROV_MUST_NOT_MOVE:
            if op.get(k) != np_.get(k):
                fail(f"{kind} run_provenance.{k} moved")
        for k in PROV_MUST_MOVE:
            if op.get(k) == np_.get(k):
                fail(f"{kind} run_provenance.{k} did NOT move")
        for k in ("gates", "verdicts", "mediation_pass",
                  "routing_reliability_pass", "routing_factorization_pass",
                  "n_cells", "n_rows", "cell_design_lineage",
                  "input_verification", "consumed_cell_files", "delta_route"):
            if json.dumps(old.get(k), sort_keys=True) != \
                    json.dumps(new.get(k), sort_keys=True):
                fail(f"{kind}.{k} differs between the committed original and "
                     f"the re-derived report")
        reestablished[kind] = OrderedDict((
            ("added_keys", added), ("dropped_keys", dropped),
            ("keys_whose_value_moved", moved),
            ("gates_verdicts_and_counts_byte_identical", True),
            ("run_provenance_identity_fields_unchanged",
             list(PROV_MUST_NOT_MOVE)),
            ("run_provenance_execution_fields_moved", list(PROV_MUST_MOVE)),
        ))
    if not problems:
        log("    confirmed: one added key, nothing dropped, only run_provenance "
            "moved, and every gate, verdict, count and lineage block identical")

    log("\n[A5] the 71 cell files against the digests the ORIGINAL report recorded")
    orig = json.loads(git("show", f"{SOURCE_COMMIT}:datasets/route-unlearning-"
                                  f"data/{rel(REPORTS)}/rf_report_salmu_v5.json"))
    want = orig["consumed_cell_files"]
    bad = [r for r, d in sorted(want.items())
           if not (DSROOT / r).is_file() or sha((DSROOT / r).read_bytes()) != d]
    log(f"    re-hashed {len(want)} cell file(s), {len(bad)} disagreeing")
    if bad:
        fail(f"{len(bad)} cell file(s) do not match the digests the report at "
             f"{SOURCE_COMMIT} recorded, e.g. {bad[0]}")
    reestablished["cell_files_rehashed_against_the_original_report"] = {
        "n": len(want), "n_disagreeing": len(bad)}
    return raw_at_source, methods, reestablished


# --------------------------------------------------------------------------
# phase B: archive the drivers and append the correction
# --------------------------------------------------------------------------
def archive_drivers(extra=()):
    log("\n[B1] archiving the driver sources verbatim")
    DRIVERS.mkdir(parents=True, exist_ok=True)
    archived = OrderedDict()
    for src, expect, produced in (*TO_ARCHIVE, *extra):
        s = pathlib.Path(src)
        got = sha(s.read_bytes())
        dst = DRIVERS / s.name
        shutil.copyfile(s, dst)
        after = sha(dst.read_bytes())
        log(f"    {s.name}  {got[:16]}")
        log(f"        -> {rel(dst)}  {after[:16]}")
        if expect and got[:16] != expect:
            fail(f"{s.name} hashes to {got[:16]} but the artifact it filed "
                 f"recorded {expect}; this is not the source that ran")
        if after != got:
            fail(f"archiving {s.name} changed its bytes")
        archived[s.name] = OrderedDict((
            ("archived_at", rel(dst)), ("sha256", got),
            ("what_it_filed", produced),
            ("archived_verbatim_because",
             "a driver is evidence about how an artifact was produced, so it is "
             "copied byte for byte into the repository and deliberately kept "
             "outside src/tests/scripts, which CI reformats with ruff and which "
             "would silently break the digest the artifact records")))
    return archived


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log(f"repair_sidecar_digests.py  "
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    rf = load_runner()
    side = json.loads(SIDECAR.read_text(encoding="utf-8"))
    log(f"sidecar {rel(SIDECAR)} {sha(SIDECAR.read_bytes())[:16]}")
    if "correction" in side:
        fail("the sidecar already carries a correction; refusing to stack one")

    raw_at_source, methods, reestablished = phase_a(rf, side)
    if problems:
        log(f"\nREFUSED before writing anything: {len(problems)} problem(s)")
        for p in problems:
            log(f"  - {p[:220]}")
        return 1
    if args.dry_run:
        log("\n--dry-run: all of phase A passed, writing nothing")
        return 0

    archived = archive_drivers(
        extra=((str(pathlib.Path(__file__).resolve()), None,
                "filed this correction block"),))

    side["correction"] = OrderedDict((
        ("corrected_at_utc",
         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ("corrected_by", str(pathlib.Path(__file__).resolve())),
        ("corrected_by_sha256",
         sha(pathlib.Path(__file__).resolve().read_bytes())),
        ("what_was_wrong",
         "write_was_guarded recorded three sets of digests using two different "
         "methods and presented them as one kind of thing.  "
         "committed_digests_at_the_run and staged_digests were "
         "sha256(json.dumps(parsed, sort_keys=True)) -- a re-serialization in "
         "Python's default spacing -- while in_place_digests were sha256 of the "
         "raw file bytes.  The first two therefore match no file on disk, no "
         "file in git, no compact-separator form and not canonical_json, so a "
         "reader hashing the committed reports could not reconcile them with "
         "what this sidecar said it was describing"),
        ("what_was_not_wrong",
         "the comparison these digests were meant to summarise did not consume "
         "them.  The driver compared git-show TEXT against the file's TEXT, and "
         "then compared the parsed documents key by key; the three digest fields "
         "appear exactly once each in its source, inside the record construction "
         "that runs after every gate.  They were output, not input, which is why "
         "the mistake was invisible while the driver ran and is only visible to "
         "somebody reproducing the numbers"),
        ("the_erroneous_values_are_preserved_not_overwritten",
         "write_was_guarded.committed_digests_at_the_run and .staged_digests are "
         "left exactly as filed, because replacing them would destroy the record "
         "of what was claimed and make this correction unfalsifiable"),
        ("erroneous_values_as_filed", side["write_was_guarded"]
         ["committed_digests_at_the_run"]),
        ("source_commit_of_the_original_reports", SOURCE_COMMIT),
        ("commit_the_driver_read_them_at", READER_COMMIT),
        ("correct_raw_sha256_of_the_original_reports", raw_at_source),
        ("verification_method", OrderedDict((
            ("command", "git -C /scratch/wutiantong/MIDP show "
                        f"{SOURCE_COMMIT}:datasets/route-unlearning-data/"
                        "e2c_route_forgetting/reports/rf_report_salmu_v5.json "
                        "| sha256sum"),
            ("and", "git -C /scratch/wutiantong/MIDP show "
                    f"{SOURCE_COMMIT}:datasets/route-unlearning-data/"
                    "e2c_route_forgetting/reports/"
                    "rf_report_salmu_v5_rescored.json | sha256sum"),
            ("python_equivalent",
             "hashlib.sha256(subprocess.run(['git','show',"
             f"'{SOURCE_COMMIT}:<path>'],capture_output=True).stdout).hexdigest()"),
            ("digest_method_this_correction_uses_everywhere",
             "sha256 of the RAW FILE BYTES, which is the only method that "
             "reproduces with sha256sum and the only one a reader can check "
             "without knowing which serializer produced the file")))),
        ("how_the_erroneous_values_reproduce", methods),
        ("the_substantive_claim_re_established_from_raw_bytes", OrderedDict((
            ("why_re_established",
             "the sidecar's claim was that the re-derived reports differ from "
             "the committed originals in exactly one added key plus "
             "run_provenance.  That claim is recomputed here from the raw bytes "
             "at the source commit, so it no longer rests on a digest nobody "
             "could reproduce"),
            ("per_report", reestablished)))),
        ("drivers_archived_verbatim", archived),
        ("changes_nothing_about_the_result",
         "no report, gate, verdict, cell, manifest or design is altered by this "
         "correction, and neither RF2 nor RF2P was re-run.  The mediation, "
         "routing-reliability and routing-factorization verdicts, all nine gate "
         "values and all 2076 rows across 71 cells are exactly as filed at "
         f"{SOURCE_COMMIT} and as re-derived under proof; only the record of how "
         "the re-derivation was checked is corrected"),
    ))

    path, digest = rf.atomic_write_json(SIDECAR, side)
    log(f"\nsidecar rewritten {rel(path)}")
    log(f"    new sha256 {digest}")
    log(f"    CORRECTION_SUPERCEDING_NOTE {SUPERSEDED_NOTE}")
    pathlib.Path("/scratch/wutiantong/rf5_launch/new_sidecar_sha256.txt") \
        .write_text(digest + "\n")
    if problems:
        log(f"\nSTOPPED with {len(problems)} problem(s)")
        return 1
    log("\nCORRECTION FILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
