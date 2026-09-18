#!/usr/bin/env python
"""File the v5 scope note, superseding 3097353f9f4d02a9.

This is version 2 of the generator.  Version 1 is archived verbatim under
e2c_route_forgetting/drivers/ and is not edited, because the note it filed records
its digest; a generator whose bytes move after filing leaves an artifact pointing
at a file that no longer exists.

WHAT VERSION 2 ADDS
-------------------
* binds the CORRECTED re-derivation sidecar.  The sidecar's recorded digests for
  the committed original reports were sha256 of a sorted re-serialization of the
  parsed document rather than of its raw bytes, so they matched no file; it now
  carries a /correction block preserving the erroneous values and supplying the
  correct raw sha256 at the source commit.
* a supersedes_scope_note lineage block naming the digest and commit it replaces.
* the archived driver sources, each cross-checked against the sidecar's own
  record of archiving it.
* disclosure 5's single "carried" field split into carried /
  carried_in_frozen_artifact / carried_in_report / carried_by_scope_note, applied
  uniformly to all five, because "carried": false beside "carried_here_instead":
  true read as a contradiction.
* every claim below is validated BEFORE the note is written.  Version 1 wrote the
  note and then validated it, so a note contradicting its own artifacts could have
  been filed and only reported afterwards.

The five disclosures, the eight deviations and the derivations behind them are
otherwise unchanged.

WHY A SEPARATE FILE
-------------------
The plan lists five disclosures "the artifacts must carry".  Four are carried, but
the two natural homes for the fifth are closed: Freeze #1 and Freeze #2 are frozen
and re-freezing is impossible (74 of Freeze #2's outputs now exist, so _freeze_v5
correctly refuses), and the calibration selection cannot be edited either because
the v5 design binds its sha256.  A statement added after the fact therefore has to
be its own file.  It changes nothing it describes and binds each of them by digest,
so it is an index with a checksum rather than a claim.

Everything factual here is DERIVED by calling the module: the two PPUBench
refusals are captured as the exceptions the functions actually raise, and the
digests are computed from the files on disk.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
import time
from collections import OrderedDict

REPO = pathlib.Path("/scratch/wutiantong/MIDP")
DSROOT = REPO / "datasets/route-unlearning-data"
RUNNER = DSROOT / "scripts/e2c_v3_route_dependent_forgetting.py"
OUT = DSROOT / "e2c_route_forgetting/reports/rf_scope_note_salmu_v5.json"
DRIVERS = DSROOT / "e2c_route_forgetting/drivers"
SIDECAR = (DSROOT / "e2c_route_forgetting/reports"
           / "rf_report_salmu_v5_rederivation.json")
# The scope note this one replaces, filed at 9515035.  Superseded by digest and
# left in git history rather than deleted: it bound the re-derivation sidecar
# BEFORE that sidecar's digest correction, so it is part of the record of how this
# one came to exist.
SUPERSEDED_NOTE_SHA256 = (
    "3097353f9f4d02a9ff8b77a34d2935bb6128c89ccef2bdd8355637b543060d92")
SUPERSEDED_AT_COMMIT = "9515035"


def load():
    spec = importlib.util.spec_from_file_location("rf", RUNNER)
    rf = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER.parent))
    spec.loader.exec_module(rf)
    return rf


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def refusal(fn, *a, **k):
    """Call a guard and return what it actually said, or that it did not fire."""
    try:
        fn(*a, **k)
    except RuntimeError as exc:
        return {"refused": True, "message": str(exc)}
    return {"refused": False,
            "note": "it did not refuse, so this note would be claiming an "
                    "enforcement that does not exist"}


def main() -> int:
    rf = load()
    man_salmu = rf.load_manifest("salmu")
    man_ppu = rf.load_manifest("ppubench")

    bound = OrderedDict()
    for label, p in (
            ("calibration_preregistration_freeze_1",
             rf.prereg_path_for(rf.PILOT_SPEC_V5, "salmu").parent
             / "rf_calibration_salmu_v5.json"),
            ("confirmatory_design_freeze_2",
             rf.prereg_path_for(rf.PILOT_SPEC_V5, "salmu")),
            ("calibration_selection", rf.calibration_selection_path("salmu")),
            ("rf2_report", rf.report_path("salmu", "RF2", None, rf.PILOT_SPEC_V5)),
            ("rf2p_report",
             rf.report_path("salmu", "RF2P", None, rf.PILOT_SPEC_V5)),
            ("rederivation_sidecar",
             rf.report_path("salmu", "RF2", None, rf.PILOT_SPEC_V5).parent
             / "rf_report_salmu_v5_rederivation.json")):
        bound[label] = OrderedDict((("path", rf._rel(p)),
                                    ("sha256", sha(p) if p.is_file() else None),
                                    ("present", p.is_file())))

    ppu_policy = rf.pilot_seed_policy_v2("ppubench", man=man_ppu)
    salmu_split = rf.dev_split(man_salmu)

    # The two claims about the calibration grid are derived from its own frozen
    # artifacts rather than restated, because both were wrong the first time they
    # were written from memory: the grid varies the schedule and NOT the LoRA
    # shape, and "monotone in learning rate across all four candidates" is false
    # since C2 varies steps at the incumbent learning rate.
    cal = json.loads((rf.calibration_selection_path("salmu").parent.parent
                      / "manifests" / "rf_calibration_salmu_v5.json")
                     .read_text(encoding="utf-8"))
    sel = json.loads(rf.calibration_selection_path("salmu")
                     .read_text(encoding="utf-8"))
    # The manifest nests the grid: cal["grid"]["candidates"] is the per-candidate
    # schedule table, and cal["grid"] carries n_candidates, n_trainings,
    # overridable_keys and seeds.
    grid = cal["grid"]
    measured = sel["candidates"]
    incumbent = grid["candidates"]["C0_incumbent"]
    lr_only = sorted(
        ((c["lr"], name, measured[name]["mean_development_accuracy"])
         for name, c in grid["candidates"].items()
         if c["steps"] == incumbent["steps"]
         and c["warmup"] == incumbent["warmup"]
         and c["repeat"] == incumbent["repeat"]))
    monotone = all(lr_only[i][2] <= lr_only[i + 1][2]
                   for i in range(len(lr_only) - 1))
    all_lrs = sorted(c["lr"] for c in grid["candidates"].values())
    grid_evidence = OrderedDict((
        ("n_candidates", grid["n_candidates"]),
        ("n_trainings", grid["n_trainings"]),
        ("overridable_keys", grid["overridable_keys"]),
        ("lora_shape_is_overridable",
         any(k in grid["overridable_keys"]
             for k in ("lora_rank", "lora_alpha", "lora_dropout"))),
        ("candidates_varying_only_the_learning_rate", [
            OrderedDict((("lr", lr), ("candidate", name),
                         ("mean_development_accuracy", acc)))
            for lr, name, acc in lr_only]),
        ("mean_accuracy_is_monotone_in_lr_across_those", monotone),
        ("the_selected_lr", sel["selected"]["schedule"]["lr"]),
        ("the_selected_lr_is_the_maximum_in_the_grid",
         sel["selected"]["schedule"]["lr"] == all_lrs[-1]),
        ("all_learning_rates_in_the_grid", all_lrs),
        ("the_candidate_that_varies_steps_instead", OrderedDict((
            ("candidate", "C2_half_as_many_steps_again"),
            ("lr", grid["candidates"]["C2_half_as_many_steps_again"]["lr"]),
            ("steps",
             grid["candidates"]["C2_half_as_many_steps_again"]["steps"]),
            ("mean_development_accuracy",
             measured["C2_half_as_many_steps_again"]
             ["mean_development_accuracy"]),
            ("why_it_matters", "a claim of monotonicity in learning rate across "
                               "all four candidates would be false, because "
                               "this one moves steps at the incumbent learning "
                               "rate")))),
    ))

    # Archive THIS generator too, so the note's own provenance is a file in the
    # repository and not only a digest of something outside it.  Copied before the
    # listing is built, so the listing includes it.
    DRIVERS.mkdir(parents=True, exist_ok=True)
    me = pathlib.Path(__file__).resolve()
    shutil.copyfile(me, DRIVERS / me.name)

    # Cross-check the archive against what the sidecar's correction block says it
    # archived, rather than restating it: this note must not be able to describe an
    # archive that disagrees with the record of making it.  The sidecar was
    # corrected before this generator existed, so it cannot name this file; a
    # driver it does name must match exactly, and one it does not name is recorded
    # as archived by this note rather than silently treated as equivalent.
    side_doc = json.loads(SIDECAR.read_text(encoding="utf-8"))
    corr = side_doc.get("correction") or {}
    recorded_archive = corr.get("drivers_archived_verbatim") or {}
    archived = OrderedDict()
    for p in sorted(DRIVERS.glob("*.py")):
        got = sha(p)
        rec = recorded_archive.get(p.name)
        archived[p.name] = OrderedDict((
            ("path", rf._rel(p)),
            ("sha256", got),
            ("named_by_the_sidecar_correction", rec is not None),
            ("matches_the_sidecar_correction",
             (got == rec["sha256"]) if rec else None),
            ("what_it_filed",
             (rec or {}).get("what_it_filed")
             or "filed this scope note, superseding "
                f"{SUPERSEDED_NOTE_SHA256[:16]}"),
        ))

    note = OrderedDict((
        ("kind", "route_dependent_forgetting_scope_note_v5"),
        ("dataset", "salmu"),
        ("filed_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ("filed_by", str(pathlib.Path(__file__).resolve())),
        ("filed_by_sha256", sha(pathlib.Path(__file__).resolve())),
        ("runner_sha256", sha(RUNNER)),
        ("why_this_is_a_separate_file", (
            "the two natural homes are closed: the v5 pre-registrations are "
            "frozen and cannot be re-frozen because 74 of Freeze #2's outputs "
            "now exist, and the calibration selection cannot be edited because "
            "the v5 design binds its sha256.  A scope statement made after the "
            "fact has to be its own file, so it binds everything it describes "
            "instead of being part of it")),
        ("changes_nothing", (
            "no threshold, gate, verdict, design or cell is altered by this "
            "file; it records obligations and the refusals that enforce them")),
        ("bound_artifacts", bound),
        ("supersedes_scope_note", OrderedDict((
            ("sha256", SUPERSEDED_NOTE_SHA256),
            ("filed_at_commit", SUPERSEDED_AT_COMMIT),
            ("why_it_was_superseded",
             "that note bound the re-derivation sidecar at d7f4f8bc6cdf1035, "
             "before the sidecar's recorded digests were found to have been "
             "computed by re-serializing the parsed reports rather than by "
             "hashing their raw bytes. The sidecar now carries a correction "
             "block that preserves the erroneous values and supplies the correct "
             "raw sha256 of both reports at their source commit, so this note is "
             "regenerated to bind the corrected sidecar and to record the "
             "lineage. The superseded note is preserved in git history at the "
             "commit above and is not deleted"),
            ("what_changed_in_this_note", [
                "binds the corrected sidecar digest rather than d7f4f8bc6cdf1035",
                "adds this supersedes_scope_note lineage block",
                "lists the archived driver sources by digest and cross-checks "
                "them against the sidecar's correction block",
                "splits disclosure 5's single 'carried' field into carried / "
                "carried_in_frozen_artifact / carried_by_scope_note, because "
                "'carried': false beside 'carried_here_instead': true read as a "
                "contradiction"])))),
        ("sidecar_digest_correction", OrderedDict((
            ("see", f"{rf._rel(SIDECAR)} at /correction"),
            ("corrected_at_utc", corr.get("corrected_at_utc")),
            ("corrected_by", corr.get("corrected_by")),
            ("source_commit_of_the_original_reports",
             corr.get("source_commit_of_the_original_reports")),
            ("correct_raw_sha256_of_the_original_reports",
             corr.get("correct_raw_sha256_of_the_original_reports")),
            ("erroneous_values_preserved_in_the_sidecar",
             corr.get("the_erroneous_values_are_preserved_not_overwritten")
             is not None),
            ("changes_no_report_gate_or_verdict",
             corr.get("changes_nothing_about_the_result")),
            ("digest_method_used_everywhere_in_this_note",
             "sha256 of the RAW FILE BYTES, the only method that reproduces with "
             "sha256sum and the only one a reader can check without knowing "
             "which serializer produced the file")))),
        ("drivers_archived_verbatim", archived),
        ("calibration_grid_as_frozen_and_as_measured", grid_evidence),
        ("disclosures_the_plan_required", [
            OrderedDict((
                ("n", 1),
                ("statement", "the router is not calibrated"),
                ("where_it_is_carried",
                 "rf_calibration_salmu_v5.json at "
                 "/what_is_being_calibrated/what_is_NOT_being_calibrated"),
                ("carried", True),
                ("carried_in_frozen_artifact", True),
                ("carried_in_report", False),
                ("carried_by_scope_note", False))),
            OrderedDict((
                ("n", 2),
                ("statement", "D_s trains on 72 images while the frozen g "
                              "trained on all 96, and the direction of that "
                              "asymmetry favours the route"),
                ("where_it_is_carried",
                 "Freeze #2 at /direct_training/training_data_asymmetry, and "
                 "the RF2 and RF2P reports at "
                 "/direct_training_disclosure/training_data_asymmetry"),
                ("carried", True),
                ("carried_in_frozen_artifact", True),
                ("carried_in_report", True),
                ("carried_by_scope_note", False),
                ("n_fit_images", salmu_split["n_fit_images"]),
                ("n_development_images", salmu_split["n_development_images"]),
                ("n_test_images", salmu_split["n_test_images"]),
                ("n_train_images", salmu_split["n_train_images"]))),
            OrderedDict((
                ("n", 3),
                ("statement", "D_s's recipe may differ from the route's once a "
                              "non-incumbent candidate is selected"),
                ("where_it_is_carried",
                 "Freeze #2 at /direct_training/selected_candidate and "
                 "/direct_training/schedule, and the RF2 and RF2P reports at "
                 "/direct_training_disclosure, where the divergence is COMPUTED "
                 "against _incumbent_schedule() rather than asserted"),
                ("carried", True),
                ("carried_in_frozen_artifact", True),
                ("carried_in_report", True),
                ("carried_by_scope_note", False),
                ("the_live_case", "a non-incumbent WAS selected, so this is not "
                                  "hypothetical"))),
            OrderedDict((
                ("n", 4),
                ("statement", "every v5 cell is produced_under_this_design, so "
                              "the 28 v3 cells take no part in any v5 verdict"),
                ("where_it_is_carried",
                 "the RF2 report at /cell_design_lineage, with an empty "
                 "ancestor and 71 of 71 cells produced_under_this_design"),
                ("carried", True),
                # Not in a frozen artifact: this is a property of the RUN, and a
                # design frozen before the run could not have known it.  The
                # report is where a fact about execution belongs.
                ("carried_in_frozen_artifact", False),
                ("carried_in_report", True),
                ("carried_by_scope_note", False))),
            OrderedDict((
                ("n", 5),
                ("statement", "PPUBench is not run as a v5 pilot, and a future "
                              "arm there is restricted to new forget sets at "
                              "router seed 17, reported as forget-set "
                              "generalisation rather than a router-seed cross"),
                ("where_it_is_carried",
                 "nowhere in a frozen artifact, which is why this file exists"),
                # Three fields rather than one overloaded "carried": "carried":
                # false sitting beside "carried_here_instead": true reads as a
                # contradiction, and what a reader actually needs is WHICH CLASS
                # of artifact holds it, because a frozen one can never be edited
                # and this one can.
                ("carried", True),
                ("carried_in_frozen_artifact", False),
                ("carried_in_report", False),
                ("carried_by_scope_note", True),
                ("why_not_a_frozen_artifact",
                 "Freeze #1 and Freeze #2 cannot be re-frozen because 74 of "
                 "Freeze #2's outputs now exist and _freeze_v5 correctly refuses, "
                 "and the calibration selection cannot be edited because the v5 "
                 "design binds its sha256"),
                ("not_a_preference",
                 "two guards in the runner refuse it, reproduced verbatim below"))),
        ]),
        ("ppubench_cannot_host_a_v5_pilot", OrderedDict((
            ("held_out_image_content_exists",
             ppu_policy["held_out_image_content_exists"]),
            ("frozen_router_seeds", ppu_policy["router_seeds"]),
            ("frozen_direct_seeds", ppu_policy["direct_seeds"]),
            ("seed_budget_refusal", refusal(
                rf.check_seed_budget, "ppubench", [17, 42, 123], [17, 42, 123],
                man=man_ppu)),
            ("development_split_refusal", refusal(rf.dev_split, man_ppu)),
            ("salmu_by_contrast", OrderedDict((
                ("held_out_image_content_exists",
                 rf.pilot_seed_policy_v2("salmu", man=man_salmu)
                 ["held_out_image_content_exists"]),
                ("development_split_is_disjoint_in_content", True),
                ("split_sizes", OrderedDict(
                    (k, salmu_split[k]) for k in
                     ("n_train_images", "n_fit_images",
                      "n_development_images", "n_test_images")))))),
            ("what_this_means",
             "PPUBench is not merely deferred: its held-out filenames are bytes "
             "that also appear in its train split, so no development split of it "
             "can be content-disjoint from its test images, and a configuration "
             "selected on one would be selected partly on the measurement it is "
             "supposed to predict.  v5 is therefore SALMU-only, and the "
             "route-seed cross it exists to run is only possible on a dataset "
             "with held-out image content")))),
        ("deviations_from_the_approved_plan", [
            OrderedDict((
                ("n", 1),
                ("deviation", "the calibration grid varies the training SCHEDULE "
                              "and not the LoRA shape"),
                ("authority", "approved by the user when the grid was scoped"),
                ("reason", "lora_rank 8, lora_alpha 16 and lora_dropout 0.05 are "
                           "hardcoded at lines 440 and 464 of "
                           "scripts/e2c_v3_research_validity.py, so varying them "
                           "would have meant editing the module the frozen route "
                           "was trained under"),
                ("evidence", OrderedDict((
                    ("overridable_keys", grid["overridable_keys"]),
                    ("no_lora_key_is_overridable",
                     not grid_evidence["lora_shape_is_overridable"])))))),
            OrderedDict((
                ("n", 2),
                ("deviation", "DECLARED_HYGIENE_SCOPE_V5 inherits v4's "
                              "scope-defining keys but not "
                              "post_outcome_scope_amendment and its two "
                              "explanatory keys"),
                ("reason", "those record that v4's scope was amended after its "
                           "results were seen; v5's scope is declared before "
                           "any result exists, so inheriting the amendment "
                           "would assert a history v5 does not have"))),
            OrderedDict((
                ("n", 3),
                ("deviation", "the development split is a COUNT PER IDENTITY "
                              "rather than the index list (6, 7) the plan named"),
                ("reason", "a count works on both datasets' index layouts and is "
                           "what dev_split checks against image BYTES; the "
                           "resulting SALMU split is 24 development images, "
                           "2 per identity, as the plan required"))),
            OrderedDict((
                ("n", 4),
                ("deviation", "v5 is SALMU-only"),
                ("reason", "see ppubench_cannot_host_a_v5_pilot: dev_split "
                           "refuses PPUBench outright"))),
            OrderedDict((
                ("n", 5),
                ("deviation", "direct and hybrid cell ids are namespaced "
                              "(direct__v5__d17) and they are NOT qualified by "
                              "forget set, while baseline, intervention and "
                              "natural ids are"),
                ("reason", "the plan did not specify ids; namespacing is what "
                           "stops v5 overwriting the v3 cells whose digests "
                           "committed v4 reports still cite, and the direct and "
                           "hybrid cells depend on no forget set so qualifying "
                           "them would have multiplied 6 cells into 30 "
                           "identical ones"))),
            OrderedDict((
                ("n", 6),
                ("deviation", "the selection rule floors the MEAN of two "
                              "calibration seeds while the confirmatory gate "
                              "requires EVERY one of three seeds to clear 0.9"),
                ("reason", "left as frozen rather than repaired, because "
                           "changing the rule after the measurements existed "
                           "would have been selecting against them"),
                ("consequence", "C3_double_lr was selected at a mean of 0.9375 "
                                "from 24/24 and 21/24, and then scored 34/36, "
                                "32/36 and 32/36 on the confirmatory images, so "
                                "two seeds fell one image short of the 33/36 the "
                                "floor requires and the mediation verdict is a "
                                "fail.  This risk was reported before Freeze #2 "
                                "and the rule was applied as frozen anyway, "
                                "which is the correct order even though the "
                                "outcome is the one it predicted"))),
            OrderedDict((
                ("n", 7),
                ("deviation", "the selected schedule sits at the TOP of the "
                              "pre-registered learning-rate range, and mean "
                              "development accuracy is monotone increasing in "
                              "learning rate across the three candidates that "
                              "vary it"),
                ("reason", "reported rather than acted on: extending the grid "
                           "upward after seeing that trend would be tuning "
                           "against measurements that already exist, which is "
                           "the thing the calibration was built to prevent"),
                ("consequence", "the selection is a boundary solution, so this "
                                "pilot cannot distinguish 'double the learning "
                                "rate is the right schedule' from 'more is "
                                "still better'"),
                ("evidence", grid_evidence))),
            OrderedDict((
                ("n", 8),
                ("deviation", "the RF2 and RF2P reports were re-derived after "
                              "the run, under a labelled driver, to add the "
                              "disclosure the plan required them to carry"),
                ("reason", "the reports were filed before that code existed"),
                ("authority", "chosen by the user over filing a sidecar beside "
                              "the executed reports"),
                ("proof", "rf_report_salmu_v5_rederivation.json: all 71 cell "
                          "files byte-identical to the digests the committed "
                          "report consumed, the design still reproducing "
                          "dbddf8b78a7af66b, and the report differing in "
                          "exactly one added key plus run_provenance"),
                ("defect_found_afterwards_and_corrected",
                 "the sidecar recorded the digests of the committed originals, "
                 "the staged copies and the in-place copies in one block using "
                 "two different methods: the first two were sha256 of a sorted "
                 "re-serialization of the parsed document and only the third was "
                 "sha256 of raw file bytes. The committed originals therefore "
                 "did not hash to what the sidecar said, and no common "
                 "normalization reproduced it. The comparison itself never "
                 "consumed those digests -- it compared git-show text against "
                 "file text, then the parsed documents key by key -- so no gate "
                 "was weakened, but a record that cannot be reproduced does not "
                 "support the claim it was filed to support. Corrected in the "
                 "sidecar's /correction block, which preserves the erroneous "
                 "values, names the source commit, gives both correct raw "
                 "sha256, states the verifying command, and re-establishes the "
                 "substantive claim from raw bytes"))),
        ]),
    ))

    # Checked BEFORE the write.  The version this supersedes wrote the note and
    # then validated it, which means a note contradicting its own artifacts could
    # have been filed and only reported afterwards.
    bad = []
    dscl = note["disclosures_the_plan_required"]
    not_carried = [d["n"] for d in dscl if not d["carried"]]
    if not_carried:
        bad.append(f"disclosure(s) {not_carried} are not carried anywhere")
    # Every disclosure says which CLASS of artifact holds it, so a reader can tell
    # a statement that can never be edited from one that can.
    for d in dscl:
        for f in ("carried_in_frozen_artifact", "carried_in_report",
                  "carried_by_scope_note"):
            if f not in d:
                bad.append(f"disclosure {d['n']} does not record {f}")
        if not (d["carried_in_frozen_artifact"] or d["carried_in_report"]
                or d["carried_by_scope_note"]):
            bad.append(f"disclosure {d['n']} says it is carried but names no "
                       f"artifact that carries it")
    d5 = next(d for d in dscl if d["n"] == 5)
    if d5["carried_in_frozen_artifact"] is not False:
        bad.append("disclosure 5 claims a frozen artifact carries it; none does, "
                   "and saying so would hide why this note exists")
    if d5["carried_by_scope_note"] is not True:
        bad.append("disclosure 5 is not marked as carried by this note")
    mismatched = [k for k, v in archived.items()
                  if v["named_by_the_sidecar_correction"]
                  and not v["matches_the_sidecar_correction"]]
    if mismatched:
        bad.append(f"archived driver(s) {mismatched} do not match the digest the "
                   f"sidecar's correction block recorded for them")
    unnamed_by_sidecar = sorted(set(recorded_archive) - set(archived))
    if unnamed_by_sidecar:
        bad.append(f"the sidecar's correction names {unnamed_by_sidecar} but "
                   f"they are not in the archive")
    if me.name not in archived:
        bad.append("this generator did not archive itself")
    elif archived[me.name]["sha256"] != sha(me):
        bad.append("the archived copy of this generator is not its own bytes")
    bound_side = bound["rederivation_sidecar"]["sha256"]
    if bound_side != sha(SIDECAR):
        bad.append("the note does not bind the sidecar's actual raw bytes")
    if "correction" not in side_doc:
        bad.append("the sidecar carries no correction block, so this note would "
                   "be binding an uncorrected one")
    elif side_doc["correction"].get("source_commit_of_the_original_reports") \
            != "80eecb0":
        bad.append("the correction does not name 80eecb0 as the source commit")
    missing = [k for k, v in bound.items() if not v["present"]]
    if missing:
        bad.append(f"bound artifacts absent: {missing}")
    ge = note["calibration_grid_as_frozen_and_as_measured"]
    for label, ok in (("no LoRA key overridable",
                       not ge["lora_shape_is_overridable"]),
                      ("monotone in lr",
                       ge["mean_accuracy_is_monotone_in_lr_across_those"]),
                      ("selected lr is the maximum",
                       ge["the_selected_lr_is_the_maximum_in_the_grid"])):
        if not ok:
            bad.append(f"the note asserts {label} but the artifacts do not")
    ppu = note["ppubench_cannot_host_a_v5_pilot"]
    for label, key in (("seed budget refusal", "seed_budget_refusal"),
                       ("development split refusal", "development_split_refusal")):
        if not ppu[key]["refused"]:
            bad.append(f"the note claims a {label} but the guard did not refuse")

    for b in bad:
        print(f"  PROBLEM: {b}")
    if bad:
        print(f"\nREFUSED: {len(bad)} problem(s); nothing was filed")
        return 1

    path, digest = rf.atomic_write_json(OUT, note)
    print(f"filed {rf._rel(path)} {digest[:16]}")
    print(f"  supersedes                         : "
          f"{note['supersedes_scope_note']['sha256'][:16]} "
          f"at {note['supersedes_scope_note']['filed_at_commit']}")
    print(f"  disclosures carried                : {len(dscl)} of {len(dscl)}")
    print(f"    in a frozen artifact             : "
          f"{sum(1 for d in dscl if d['carried_in_frozen_artifact'])}")
    print(f"    in a report                      : "
          f"{sum(1 for d in dscl if d['carried_in_report'])}")
    print(f"    by this scope note               : "
          f"{sum(1 for d in dscl if d['carried_by_scope_note'])}")
    print(f"  deviations recorded                : "
          f"{len(note['deviations_from_the_approved_plan'])}")
    print(f"  drivers archived verbatim          : {len(archived)} "
          f"({sum(1 for v in archived.values() if v['named_by_the_sidecar_correction'])} "
          f"cross-checked against the sidecar's correction)")
    print(f"  corrected sidecar bound at         : {bound_side[:16]}")
    print(f"  ppubench seed budget refused       : "
          f"{ppu['seed_budget_refusal']['refused']}")
    print(f"  ppubench dev split refused         : "
          f"{ppu['development_split_refusal']['refused']}")
    print(f"  bound artifacts                    : {len(bound)}, all present")
    print(f"  grid                               : {ge['n_candidates']} candidates "
          f"/ {ge['n_trainings']} trainings, no LoRA key overridable")
    print(f"  monotone in lr across the "
          f"{len(ge['candidates_varying_only_the_learning_rate'])} lr-varying "
          f"candidates: {ge['mean_accuracy_is_monotone_in_lr_across_those']}")
    print(f"  selected lr is the grid maximum    : "
          f"{ge['the_selected_lr_is_the_maximum_in_the_grid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
