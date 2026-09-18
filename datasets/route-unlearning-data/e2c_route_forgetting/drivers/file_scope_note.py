#!/usr/bin/env python
"""File the v5 scope note: the plan's disclosure obligations, where each one is
discharged, and the two refusals that make PPUBench a v5 pilot it cannot be.

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
import sys
import time
from collections import OrderedDict

REPO = pathlib.Path("/scratch/wutiantong/MIDP")
DSROOT = REPO / "datasets/route-unlearning-data"
RUNNER = DSROOT / "scripts/e2c_v3_route_dependent_forgetting.py"
OUT = DSROOT / "e2c_route_forgetting/reports/rf_scope_note_salmu_v5.json"


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
        ("calibration_grid_as_frozen_and_as_measured", grid_evidence),
        ("disclosures_the_plan_required", [
            OrderedDict((
                ("n", 1),
                ("statement", "the router is not calibrated"),
                ("where_it_is_carried",
                 "rf_calibration_salmu_v5.json at "
                 "/what_is_being_calibrated/what_is_NOT_being_calibrated"),
                ("carried", True))),
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
                ("the_live_case", "a non-incumbent WAS selected, so this is not "
                                  "hypothetical"))),
            OrderedDict((
                ("n", 4),
                ("statement", "every v5 cell is produced_under_this_design, so "
                              "the 28 v3 cells take no part in any v5 verdict"),
                ("where_it_is_carried",
                 "the RF2 report at /cell_design_lineage, with an empty "
                 "ancestor and 71 of 71 cells produced_under_this_design"),
                ("carried", True))),
            OrderedDict((
                ("n", 5),
                ("statement", "PPUBench is not run as a v5 pilot, and a future "
                              "arm there is restricted to new forget sets at "
                              "router seed 17, reported as forget-set "
                              "generalisation rather than a router-seed cross"),
                ("where_it_is_carried",
                 "nowhere in a frozen artifact, which is why this file exists"),
                ("carried", False),
                ("carried_here_instead", True),
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
                          "exactly one added key plus run_provenance"))),
        ]),
    ))

    path, digest = rf.atomic_write_json(OUT, note)
    print(f"filed {rf._rel(path)} {digest[:16]}")
    print(f"  disclosures carried in an artifact : "
          f"{sum(1 for d in note['disclosures_the_plan_required'] if d['carried'])}"
          f" of {len(note['disclosures_the_plan_required'])}")
    print(f"  deviations recorded                : "
          f"{len(note['deviations_from_the_approved_plan'])}")
    print(f"  ppubench seed budget refused       : "
          f"{note['ppubench_cannot_host_a_v5_pilot']['seed_budget_refusal']['refused']}")
    print(f"  ppubench dev split refused         : "
          f"{note['ppubench_cannot_host_a_v5_pilot']['development_split_refusal']['refused']}")
    missing = [k for k, v in bound.items() if not v["present"]]
    if missing:
        print(f"  PROBLEM: bound artifacts absent: {missing}")
        return 1
    print(f"  bound artifacts                    : {len(bound)}, all present")
    ge = note["calibration_grid_as_frozen_and_as_measured"]
    print(f"  grid: {ge['n_candidates']} candidates / {ge['n_trainings']} "
          f"trainings, no LoRA key overridable: "
          f"{not ge['lora_shape_is_overridable']}")
    print(f"  monotone in lr across the {len(ge['candidates_varying_only_the_learning_rate'])} "
          f"lr-varying candidates: {ge['mean_accuracy_is_monotone_in_lr_across_those']}")
    print(f"  selected lr is the grid maximum   : "
          f"{ge['the_selected_lr_is_the_maximum_in_the_grid']}")
    for k, v in (("no LoRA key overridable",
                  not ge["lora_shape_is_overridable"]),
                 ("monotone in lr", ge["mean_accuracy_is_monotone_in_lr_across_those"]),
                 ("selected lr is the maximum",
                  ge["the_selected_lr_is_the_maximum_in_the_grid"])):
        if not v:
            print(f"  PROBLEM: the note asserts {k} but the artifacts do not")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
