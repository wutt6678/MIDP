#!/usr/bin/env python
"""The canonical status record for the E2C-v3 MLLMU G6.1 coarsening pilot.

One file answers, for anyone arriving at ``e2c_granularity/outputs/mllmu/``:

  * which artifact is AUTHORITATIVE, and what its bytes hash to;
  * what the pilot concluded;
  * what was never launched and will not be;
  * what role G6 plays in MIDP's claims -- auxiliary, not the headline;
  * and, per sealed file, EXACTLY which statements in it are stale and why.

The sealed reports are kept, not rewritten: a report is evidence of what was
concluded when it was written, and deleting it would erase the fact that the
pilot was once read differently.  What was missing is a record of which reading
is current, which is what this file is.

Every number and verdict here is READ from the artifact it describes, never
retyped.  A status record that restates its own summary's conclusions from
prose is the same failure mode as the stale report it supersedes: it keeps
asserting the old value after the evidence moves underneath it.  Where a
declaration is a decision rather than a measurement -- G6's role, and the
decision not to tune further -- it is labelled as such and carries the derived
facts that motivated it beside it.

Torch-free by construction: ``SCORING_DIGEST_HISTORY`` is read out of
``e2c_v3_research_validity.py`` with ``ast`` rather than by importing it,
because importing that module pulls in torch and this record has to be
derivable on a machine with no CUDA.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
OUT_BASE = DATASET_ROOT / "e2c_granularity" / "outputs" / "mllmu"

AUTHORITATIVE_NAME = "g6_pilot_gates_reevaluated.json"
STATUS_NAME = "g6_canonical_status.json"
STUDY_NAME = "study_protoB_summary.json"
KIND = "g6_canonical_status_v1"

#: The three reports written by the pilot run itself.  Retained as historical
#: artifacts; this tool reads them and never writes them.
HISTORICAL_SEALED = ("g6_pilot_gates.json", "granularity_summary.json",
                     "run_manifest.json")

#: The oracle families whose fit the repaired matched-oracle gate reads, and
#: whose PRE-GX2H fit is what makes the sealed summary's reading stale.
GATE_ORACLE_FAMILIES = ("matched_retrain", "loo_retrain")
PRE_REEVAL_SUFFIX = ".pre_hard_reeval.json"

RESEARCH_VALIDITY_PATH = SCRIPT_DIR / "e2c_v3_research_validity.py"

#: A decision, not a measurement.  Stated plainly so that no later reader
#: promotes a five-set pilot that failed its gates into a MIDP headline.
G6_ROLE = {
    "is": ("auxiliary granularity evidence: one controlled study of whether an "
           "edit can be confined to a coarser taxonomic level than the "
           "identity it was trained on"),
    "is_not": ("MIDP's main claim.  Nothing in G6.1 supports or is required by "
               "the route-establishment or route-dependent-forgetting claims, "
               "and no G6 result is load-bearing for them"),
    "why": ("the pilot FAILED its frozen gates, so it produced no positive "
            "granularity finding to headline; what it produced is a measured "
            "negative plus two repaired instruments -- a strict per-target "
            "oracle gate and a candidate-score-sum reading that no longer "
            "pretends to be a probability"),
    "status_of_the_frozen_route": (
        "unchanged by this pilot.  g, h, the prompts and the LoRA recipe are "
        "frozen, and no G6 result is a reason to modify them"),
}

#: A decision, not a measurement.  The sets it names are injected from the
#: frozen matrix's own ``target_label`` rather than typed here: the sibling
#: label of one of these sets is "Software Developer" while its TARGET is
#: "Software Engineer", and prose that names a set by hand is exactly where
#: that pair gets swapped.
NO_FURTHER_TUNING = {
    "planned": False,
    "why": ("the two knobs that could be tuned were already crossed -- retain "
            "repetition and edit steps, over three recipes -- and the study "
            "filed the result: the score-sum shortfall on the set that fails "
            "it is not repaired by either.  Continuing to search would be "
            "tuning a measurement threshold until the desired verdict appears, "
            "which is the practice the frozen criteria exist to prevent"),
    "what_would_reopen_it": (
        "a change to the candidate-set construction or to the score-sum "
        "semantics themselves -- i.e. a decision that the floor measures the "
        "wrong quantity -- and not another training recipe"),
    "decision_not_measurement": True,
}


def sha256_file(path):
    """Streamed SHA-256, so a multi-megabyte adapter never lands in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_sibling(name, filename):
    """Load a sibling script by path (these are not an installed package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path, what):
    if not path.is_file():
        raise RuntimeError(f"{what} is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def accepted_scoring_digests_without_torch(path=None):
    """The digests ``rv`` accepts, read out of its source by AST.

    Re-derived here rather than imported: ``e2c_v3_research_validity`` pulls in
    torch, and this record has to be derivable with no CUDA present.  The
    current digest is recomputed from the file's bytes exactly as
    ``rv.script_sha256()`` does, and the history entries are taken from the
    module's own literal, so a change to either moves this result.

    Only ``reason == "provenance_only"`` entries are accepted.  A history entry
    recording an actual scoring change would excuse a digest whose measurement
    content differs, which is the blanket amnesty the filter exists to prevent.
    """
    path = Path(path) if path is not None else RESEARCH_VALIDITY_PATH
    tree = ast.parse(path.read_text(encoding="utf-8"))
    history = None
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "SCORING_DIGEST_HISTORY"
                        for t in node.targets)):
            try:
                history = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                # literal_eval raises its own errors on a computed value.  The
                # contract here is "refuse, do not guess", so a leaked ValueError
                # would still stop the run but would not say why acceptance
                # could not be determined.
                raise RuntimeError(
                    f"SCORING_DIGEST_HISTORY in {path.name} is not a literal "
                    f"({type(exc).__name__}: {exc}), so digest acceptance "
                    "cannot be re-derived without importing torch; update this "
                    "reader rather than hardcoding an accepted set") from exc
    if history is None:
        raise RuntimeError(
            "SCORING_DIGEST_HISTORY is no longer a literal assignment in "
            f"{path.name}, so digest acceptance cannot be re-derived without "
            "importing torch; update this reader rather than hardcoding an "
            "accepted set")
    if not isinstance(history, (tuple, list)):
        # RuntimeError, not TypeError: every refusal in this reader is a
        # RuntimeError because every one means "do not file the record", and a
        # caller catching one kind would miss the others.  Iterating a dict
        # literal here would otherwise fail later with a message about string
        # indices, which would not say that acceptance could not be determined.
        raise RuntimeError(  # noqa: TRY004
            f"SCORING_DIGEST_HISTORY in {path.name} is a literal "
            f"{type(history).__name__}, not a sequence of entries")
    return ({sha256_file(path)}
            | {e["previous_script_sha256"] for e in history
               if e.get("reason") == "provenance_only"}), history


def staleness_explanation(supersession, g31_coverage, matrix):
    """Check the claim that the sealed summary's failures were an ORACLE-fit
    artifact, and derive the mechanism that produced it.

    Two checks, both raising rather than annotating:

      every row the sealed summary called established-and-failing must sit in a
      set whose oracle ``fit_ok`` flipped false -> true at GX2H -- a row failing
      anywhere else means the explanation is incomplete and the current reading
      is not entitled to disagree with the sealed one;

      the sets repaired at GX2H must be exactly the sets whose coarsened label
      contains a comma, since GX2H repaired a punctuation-asymmetric matcher.
      Without this the causal story is unfalsifiable prose.
    """
    flipped = {p["set_id"] for p in supersession["pairs"]
               if p["fit_ok_flipped_false_to_true"]}
    rows = list((g31_coverage or {}).get("established_and_failing") or [])
    failing_sets = {r["cell_id"].split("__")[0] for r in rows}
    unexplained = sorted(failing_sets - flipped)
    if unexplained:
        raise RuntimeError(
            f"the sealed summary reports {len(rows)} established-and-failing "
            f"rows in sets {sorted(failing_sets)}, but only {sorted(flipped)} "
            f"had an oracle fit repaired at GX2H.  Unexplained by the "
            f"oracle-fit story: {unexplained}.  The staleness explanation "
            "would be incomplete, so refusing to file it")

    coarser = {e["set_id"]: (e.get("coarser_label") or "")
               for e in matrix["sets"]}
    comma_sets = {sid for sid, lab in coarser.items() if "," in lab}
    if comma_sets != flipped:
        raise RuntimeError(
            f"the sets whose oracle fit flipped at GX2H ({sorted(flipped)}) "
            f"are not the sets whose coarsened label contains a comma "
            f"({sorted(comma_sets)}), so the punctuation-matcher explanation "
            "does not account for the repair; refusing to file it as the cause")

    return {
        "claim": ("every row the sealed summary called established-and-failing "
                  "belongs to a set whose ORACLE fit -- not whose edit -- was "
                  "repaired at GX2H"),
        "checked_not_asserted": True,
        "n_rows_called_failing": len(rows),
        "sets_those_rows_are_in": sorted(failing_sets),
        "sets_whose_oracle_fit_flipped": sorted(flipped),
        "rows_unexplained_by_the_oracle_fit": [],
        "agrees": not unexplained,
        "mechanism_also_derived": {
            "claim": ("the sets needing an oracle-fit repair are exactly the "
                      "sets whose coarsened label contains a comma, which is "
                      "the punctuation-asymmetric matcher defect GX2H "
                      "repaired"),
            "sets_whose_coarsened_label_has_a_comma": sorted(comma_sets),
            "coarser_labels": {sid: coarser[sid] for sid in sorted(coarser)},
            "matches_the_flipped_sets": comma_sets == flipped,
        },
        "note": ("both checks raise rather than file: if a failing row sat "
                 "outside the repaired sets, or the repaired sets were not the "
                 "comma-containing ones, this record's explanation of the "
                 "staleness would be incomplete and the authoritative reading "
                 "would not be entitled to disagree"),
    }


def oracle_fit_supersession(out_base, set_ids):
    """Derived proof of WHY the sealed summary's oracle reading is stale.

    For each gate family and set, compares the oracle's own fit as recorded
    before the GX2H hard re-evaluation against the same file now.  The sealed
    ``granularity_summary.json`` was written between the two, so it read the
    left-hand column; the authoritative report reads the right-hand one.
    """
    rows, n_flipped = [], 0
    for sid in sorted(set_ids):
        for fam in GATE_ORACLE_FAMILIES:
            d = out_base / "oracles" / f"{fam}_{sid}"
            cur = d / "oracle_results.json"
            pre = d / f"oracle_results{PRE_REEVAL_SUFFIX}"
            if not cur.is_file():
                raise RuntimeError(
                    f"oracle results absent for {fam}/{sid}: {cur}")
            c = json.loads(cur.read_text(encoding="utf-8"))
            p = (json.loads(pre.read_text(encoding="utf-8"))
                 if pre.is_file() else None)
            flipped = bool(p) and (not p.get("fit_ok")) and bool(c.get("fit_ok"))
            n_flipped += int(flipped)
            rows.append({
                "family": fam,
                "set_id": sid,
                "pre_gx2h": (None if p is None else {
                    "present": True,
                    "fit_ok": p.get("fit_ok"),
                    "strict_all_expected": p.get("strict_all_expected"),
                    "min_candidate_mass": p.get("min_candidate_mass")}),
                "current": {
                    "fit_ok": c.get("fit_ok"),
                    "strict_all_expected": c.get("strict_all_expected"),
                    "min_candidate_mass": c.get("min_candidate_mass")},
                "fit_ok_flipped_false_to_true": flipped,
            })
    return {
        "comparison": (f"oracle_results{PRE_REEVAL_SUFFIX} (what the sealed "
                       "summary read) against oracle_results.json (what the "
                       "authoritative report reads)"),
        "n_pairs": len(rows),
        "n_fit_ok_flipped_false_to_true": n_flipped,
        "pairs": rows,
    }


def build(out_base, authoritative_path=None):
    """Build the record.

    ``authoritative_path`` allows summarising a freshly derived report that is
    not yet in place.  Without it, regenerating both artifacts in one commit is
    impossible at a clean worktree: the report has to be written into the tree
    before this record can hash it, and writing it makes the tree dirty, so the
    record would always report the commit it summarises as uncommitted.  Reading
    the report from outside the tree lets both be derived clean and committed
    together.  The recorded NAME and relative path are always the destination in
    ``out_base``, never the staging path, so the record describes where the
    artifact lives rather than where it happened to be generated.
    """
    g6m = _load_sibling("e2c_v3_mllmu_matrix", "e2c_v3_mllmu_matrix.py")
    manifest, matrix = g6m.load_frozen_g6()

    auth_path = (Path(authoritative_path) if authoritative_path
                 else out_base / AUTHORITATIVE_NAME)
    auth = load_json(auth_path, "the authoritative re-derived gate report")
    gates = auth["gates"]
    per_gate = gates.get("gates") or {}
    if not per_gate:
        raise RuntimeError(
            f"{AUTHORITATIVE_NAME} carries no per-gate results, so its verdict "
            "cannot be summarised")

    study_path = out_base / STUDY_NAME
    study = (json.loads(study_path.read_text(encoding="utf-8"))
             if study_path.is_file() else None)

    set_ids = [e["set_id"] for e in matrix["sets"]]
    target_label = {e["set_id"]: e.get("target_label") for e in matrix["sets"]}
    # Cells are globbed independently of the design so "not launched" is a fact
    # about the tree, not a restatement of the frozen manifest.  Only the pilot
    # layout counts: a study arm lives one level deeper and is not pilot design.
    disk_set_ids = {
        p.parent.parent.name
        for p in (out_base / "cells").glob("*/seed_*/cell_results.json")}

    sealed = {}
    for name in HISTORICAL_SEALED:
        p = out_base / name
        sealed[name] = {
            "path": str(p.relative_to(out_base)),
            "sha256": sha256_file(p) if p.is_file() else None,
            "present": p.is_file(),
            "status": "historical sealed artifact; read-only, superseded where noted",
            "loaded": (json.loads(p.read_text(encoding="utf-8"))
                       if p.is_file() else None),
        }
    gs = sealed["granularity_summary.json"]["loaded"] or {}
    pg = sealed["g6_pilot_gates.json"]["loaded"] or {}
    rm = sealed["run_manifest.json"]["loaded"] or {}

    accepted, history = accepted_scoring_digests_without_torch()
    pinned = (rm.get("provenance") or {}).get("shared_scoring_script_sha256")
    supersession = oracle_fit_supersession(out_base, set_ids)

    g31 = gs.get("g3_1_gate") or {}
    g31_cov = g31.get("coverage") or {}
    sealed_mo = ((pg.get("gates") or {}).get("gates") or {}).get("matched_oracle") or {}
    auth_mo = per_gate.get("matched_oracle") or {}

    # The causal claim "the sealed summary's failures were an oracle-fit
    # artifact" is CHECKED, not asserted -- see staleness_explanation, which
    # raises if any failing row falls outside the sets GX2H repaired.
    stale_g31 = staleness_explanation(
        supersession, g31.get("coverage") or {}, matrix)

    cm = ((auth.get("metric_semantics") or {}).get("candidate_score_sum_reading")
          or {})

    study_attr = ((study or {}).get("attribution") or {}).get(
        "candidate_mass_254012") or {}

    return {
        "kind": KIND,
        "dataset": "mllmu",
        "generated_by": f"scripts/{Path(__file__).resolve().name}",
        "purpose": (
            "the single place that says which G6.1 artifact is authoritative, "
            "what the pilot concluded, what was never launched, what role G6 "
            "plays, and exactly which statements in the sealed reports are "
            "stale and why"),

        "authoritative_result": {
            "file": AUTHORITATIVE_NAME,
            "path": AUTHORITATIVE_NAME,
            "sha256": sha256_file(auth_path),
            "read_from": (str(auth_path)
                          if auth_path.parent.resolve() != out_base.resolve()
                          else AUTHORITATIVE_NAME),
            "kind": auth.get("kind"),
            "why_this_one": (
                "it is the only report that applies the repaired criteria of "
                "992efa9 -- a strict per-target fresh-retrain oracle gate with "
                "no averaging and an exact (set_id, seed) coverage audit -- to "
                "the post-GX2H oracle fit, and it binds the bytes it read: "
                "every cell and gate oracle is digest-verified against the "
                "sealed run_manifest"),
            "content_binding": (auth.get("inputs") or {}).get(
                "cell_identity_and_content_cross_check", {}).get("content"),
            "read_any_of_these_first": [
                "pilot_verdict below for the decision",
                f"{AUTHORITATIVE_NAME} -> gates for the per-gate detail",
                (f"{AUTHORITATIVE_NAME} -> metric_semantics for what "
                 "candidate_mass is and is not"),
            ],
        },

        "pilot_verdict": {
            "passed": gates["passed"],
            "failed_gates": list(gates["failed_gates"]),
            "proceed_to_full_matrix": gates.get("proceed_to_full_matrix"),
            "coverage_exact": (gates.get("coverage") or {}).get("exact"),
            "coverage_defects": list(gates.get("coverage_defects") or []),
            "per_gate_passed": {k: bool(v.get("passed"))
                                for k, v in sorted(per_gate.items())},
            "n_cells_evaluated": gates.get("n_cells_evaluated"),
            "n_cells_expected_when_complete": (
                (gates.get("coverage") or {}).get(
                    "n_cells_expected_when_complete")),
            "statement": (
                f"the G6.1 five-set pilot FAILED its frozen gates "
                f"({', '.join(gates['failed_gates'])}); "
                f"proceed_to_full_matrix is "
                f"{gates.get('proceed_to_full_matrix')}"),
            "source": f"read from {AUTHORITATIVE_NAME}, not retyped",
        },

        "full_matrix": {
            "launched": False,
            "will_not_proceed": True,
            "design_of_record": (
                "the frozen G6.1 pilot design covers "
                f"{len(set_ids)} sets x {len(matrix['sets'][0]['seeds'])} edit "
                f"seeds = {matrix.get('n_cells')} cells, all of which were "
                "evaluated; the larger matrix the gates decide about was never "
                "frozen as a design artifact and no cells exist for it"),
            "sets_in_frozen_pilot_design": sorted(set_ids),
            "set_ids_with_cells_on_disk": sorted(disk_set_ids),
            "cells_outside_the_pilot_design": sorted(disk_set_ids - set(set_ids)),
            "pilot_study_cells_excluded_from_this_count": (
                "cells under a study_<tag>/ subdirectory are the protocol study "
                "arms, are not part of the pilot design, and are not counted "
                "here"),
            "gate_that_would_have_authorised_it": gates.get("proceed_note"),
            "statement": (
                "the full 12-target matrix was NOT LAUNCHED and WILL NOT "
                "PROCEED: the gate that authorises it requires all four gates "
                "to pass with exact coverage, and two gates fail"),
        },

        "g6_role": {**G6_ROLE, "decision_not_measurement": True},

        "metric_terminology": {
            "candidate_mass_is_not_a_probability": (
                (auth.get("metric_semantics") or {}).get("candidate_mass")),
            "set_failing_only_the_score_sum": {
                "set_id": cm.get("set_id"),
                "target_label": target_label.get(cm.get("set_id")),
                "candidate_mass_floor": cm.get("candidate_mass_floor"),
                "min_over_seeds": cm.get("min_candidate_mass_over_seeds"),
                "max_over_seeds": cm.get("max_candidate_mass_over_seeds"),
                "floor_cleared_on_any_seed": cm.get(
                    "candidate_mass_floor_cleared_on_any_seed"),
                "target_rows_correct": cm.get("target_rows_correct"),
                "target_rows": cm.get("target_rows"),
                "only_failed_criterion_is_the_score_sum": cm.get(
                    "only_failed_criterion_is_the_score_sum"),
                "statement": cm.get("verdict"),
                "interpretation": cm.get("interpretation"),
            },
            "source": (f"recomputed from the pilot cells by "
                       f"{AUTHORITATIVE_NAME}; quoted here, never restated "
                       f"from prose"),
        },

        "protocol_study": {
            "file": STUDY_NAME if study is not None else None,
            "sha256": (sha256_file(study_path)
                       if study_path.is_file() else None),
            "narrowed_conclusion": study_attr.get("detail"),
            "scope_of_that_claim": study_attr.get("scope_of_that_claim"),
            "recipes_evaluated": study_attr.get("recipes_evaluated"),
            "cells_with_all_target_outputs_strictly_correct": study_attr.get(
                "cells_with_all_target_outputs_strictly_correct"),
            "no_further_tuning": {
                **NO_FURTHER_TUNING,
                "sets_it_covers": [
                    {"set_id": sid, "target_label": target_label.get(sid),
                     "target_soc": next(
                         (e.get("target_soc") for e in matrix["sets"]
                          if e["set_id"] == sid), None)}
                    for sid in sorted(
                        (study or {}).get("arms") or [])],
                "scope": (
                    "no further protocol tuning of the studied cells is "
                    "planned -- no fourth recipe, no re-run of the pilot, and "
                    "no relaxation of the frozen candidate-score-sum floor to "
                    "make a set pass.  The sets this covers are named in "
                    "sets_it_covers, taken from the frozen matrix's own "
                    "target_label"),
            },
            "superseded_wording": (
                "an earlier revision of this study called the shortfall 'a "
                "property of that set under this route rather than a recipe "
                "defect to tune away'.  That claimed more than three recipes "
                "over two knobs can support and was narrowed to what was "
                "tested"),
        },

        "historical_sealed_artifacts": {
            "policy": (
                "retained ONLY as historical sealed artifacts.  They are "
                "evidence of what the pilot concluded when it was written and "
                "are never rewritten; where a statement in one is stale, the "
                "staleness is recorded here rather than edited out of the "
                "file.  Do not cite them for a current verdict"),
            "files": {
                name: {
                    "path": sealed[name]["path"],
                    "sha256": sealed[name]["sha256"],
                    "present": sealed[name]["present"],
                }
                for name in HISTORICAL_SEALED},
            "stale_claims": {
                "granularity_summary.json": {
                    "run_stage": {
                        "value_as_written": gs.get("run_stage"),
                        "why_stale": (
                            "'full_matrix' describes the completeness of the "
                            "five-set PILOT design (its own "
                            f"full_matrix_cells_expected is "
                            f"{gs.get('full_matrix_cells_expected')}), not the "
                            "larger 12-target matrix, which was never "
                            "launched.  Read beside the authoritative report "
                            "it implies a stage that does not exist"),
                        "current_value": "pilot_only_full_matrix_not_launched",
                    },
                    "g3_1_gate": {
                        "value_as_written": {
                            "status": g31.get("status"),
                            "passed": g31.get("passed"),
                            "n_transformation_targets": g31.get(
                                "n_transformation_targets"),
                            "n_separation_established": g31_cov.get(
                                "n_separation_established"),
                            "n_established_and_failing": g31_cov.get(
                                "n_established_and_failing"),
                        },
                        "why_stale": (
                            "computed against the PRE-GX2H oracle fit.  The "
                            "retrain oracles' own strict parsing was broken by "
                            "a punctuation-asymmetric matcher, so several "
                            "reported fit_ok=false, and rows failed on the "
                            "ORACLE's fit rather than on the edit's behaviour. "
                            "GX2H re-parsed the stored raw generations and "
                            "repaired that fit; the authoritative report reads "
                            "the repaired one"),
                        "evidence": supersession,
                        "causal_check": stale_g31,
                        "current_reading": {
                            "matched_oracle_passed": auth_mo.get("passed"),
                            "n_target_rows": sum(
                                v.get("n_target_rows", 0)
                                for v in (auth_mo.get("per_set") or {}).values()),
                            "n_rows_failing": sum(
                                v.get("n_rows_failing", 0)
                                for v in (auth_mo.get("per_set") or {}).values()),
                        },
                    },
                },
                "g6_pilot_gates.json": {
                    "matched_oracle_criteria": {
                        "value_as_written": sealed_mo.get("criteria"),
                        "passed_as_written": sealed_mo.get("passed"),
                        "why_stale": (
                            "the sealed gate PASSED, but on criteria that "
                            "cannot support the claim: a mean delta over edit "
                            "seeds against a threshold of zero, over the "
                            "*_finetune families, which continue from the "
                            "trained baseline h and so are not fresh-retrain "
                            "references at all.  Averaging also lets one "
                            "inverted seed cancel another"),
                        "replaced_by": auth_mo.get("criteria"),
                        "verdict_changed": (
                            bool(sealed_mo.get("passed"))
                            != bool(auth_mo.get("passed"))),
                        "note": (
                            "the pass/fail COINCIDES -- both True -- so nothing "
                            "here changes the pilot's outcome.  What changes is "
                            "that the pass now rests on per-target "
                            "fresh-retrain evidence instead of an average over "
                            "a threshold of zero"),
                    },
                    "overall_verdict": {
                        "value_as_written": {
                            "passed": (pg.get("gates") or {}).get("passed"),
                            "failed_gates": (pg.get("gates") or {}).get(
                                "failed_gates")},
                        "why_stale": ("not stale: the sealed overall verdict "
                                      "agrees with the authoritative one"),
                        "agrees_with_authoritative": (
                            (pg.get("gates") or {}).get("passed")
                            == gates["passed"]
                            and list((pg.get("gates") or {}).get(
                                "failed_gates") or [])
                            == list(gates["failed_gates"])),
                    },
                },
                "run_manifest.json": {
                    "shared_scoring_script_sha256": {
                        "value_as_written": pinned,
                        "why_stale": (
                            "it pins the digest of e2c_v3_research_validity.py "
                            "from before 992efa9 extracted the three strict "
                            "parser functions into e2c_v3_label_parser.  That "
                            "move was provenance-only -- the functions are "
                            "byte-identical -- so the pinned digest is accepted "
                            "rather than invalidated"),
                        "accepted_by_rv": pinned in accepted,
                        "accepted_digests": sorted(accepted),
                        "supersession_history": [
                            {"previous_script_sha256": h["previous_script_sha256"],
                             "reason": h.get("reason"),
                             "superseded_by_commit": h.get(
                                 "superseded_by_commit")}
                            for h in history],
                        "note": (
                            "acceptance is re-derived from rv's own "
                            "SCORING_DIGEST_HISTORY by AST here, not retyped, "
                            "and only reason='provenance_only' entries are "
                            "accepted"),
                    },
                },
            },
        },

        "frozen_design_unchanged": {
            "g6_manifest_sha256": sha256_file(g6m.G6_MANIFEST_PATH),
            "g6_matrix_sha256": sha256_file(g6m.G6_MATRIX_PATH),
            "statement": (
                "closing G6 changes no frozen design: the hierarchy, the "
                "target selection, the matrix and the route architecture (g, "
                "h, prompts, LoRA recipe) are all untouched.  A failed "
                "auxiliary pilot is not a reason to modify the route"),
        },

        "provenance": {
            **g6m.g6_provenance(out_base),
            "frozen_design": {
                "n_identities": len(manifest.get("identity_ids") or []),
                "n_professions": len(manifest.get("professions") or []),
                "n_sets": len(set_ids),
                "n_cells_in_pilot_design": matrix.get("n_cells"),
                "held_out_note": manifest.get("held_out_note"),
            },
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-base", default=str(OUT_BASE))
    p.add_argument("--dest", default=None,
                   help=f"where to write; defaults to <out-base>/{STATUS_NAME}")
    p.add_argument("--authoritative", default=None,
                   help=f"read the authoritative report from this path instead "
                        f"of <out-base>/{AUTHORITATIVE_NAME}, so a freshly "
                        "derived report can be summarised before it is placed; "
                        "the record names the destination, not the staging path")
    p.add_argument("--dry-run", action="store_true",
                   help="print the record without writing it")
    args = p.parse_args(argv)

    out_base = Path(args.out_base).resolve()
    dest = (Path(args.dest).resolve() if args.dest
            else out_base / STATUS_NAME)
    sealed_paths = {(out_base / n).resolve() for n in HISTORICAL_SEALED}
    sealed_paths.add((out_base / AUTHORITATIVE_NAME).resolve())
    if dest in sealed_paths:
        raise RuntimeError(
            f"--dest resolves to an artifact this tool only reads: {dest}")

    block = build(out_base, authoritative_path=args.authoritative)
    text = json.dumps(block, indent=2)
    if args.dry_run:
        print(text)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")

    v = block["pilot_verdict"]
    print(f"authoritative          : {block['authoritative_result']['file']}"
          f"  {block['authoritative_result']['sha256'][:16]}", file=sys.stderr)
    staged = block["authoritative_result"]["read_from"]
    if staged != AUTHORITATIVE_NAME:
        print(f"  read from staging path : {staged}", file=sys.stderr)
        print("  (the record names the destination; place the bytes there "
              "unchanged or the hash will not match)", file=sys.stderr)
    print(f"pilot verdict          : passed={v['passed']} "
          f"failed={v['failed_gates']}", file=sys.stderr)
    print(f"proceed_to_full_matrix : {v['proceed_to_full_matrix']}",
          file=sys.stderr)
    print(f"full 12-target matrix  : launched="
          f"{block['full_matrix']['launched']} "
          f"will_not_proceed={block['full_matrix']['will_not_proceed']}",
          file=sys.stderr)
    print("G6 role                : auxiliary granularity evidence, not "
          "MIDP's main claim", file=sys.stderr)
    print(f"further tuning planned : "
          f"{block['protocol_study']['no_further_tuning']['planned']}",
          file=sys.stderr)
    fit = block["historical_sealed_artifacts"]["stale_claims"][
        "granularity_summary.json"]["g3_1_gate"]["evidence"]
    print(f"oracle fit superseded  : {fit['n_fit_ok_flipped_false_to_true']}"
          f" of {fit['n_pairs']} family/set pairs flipped false -> true at "
          f"GX2H", file=sys.stderr)
    if not args.dry_run:
        print(f"written                : {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
