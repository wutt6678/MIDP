#!/usr/bin/env python3
"""Re-derive the frozen G6.1 pilot's four gates on CPU and FILE the result
beside the sealed reports, without rewriting them.

WHY THIS EXISTS AS A SEPARATE TOOL

Commit 992efa9 repaired the gate logic itself: ``matched_oracle_gate`` had
measured the ``*_finetune`` oracle families and averaged delta over edit seeds
per set, so it never checked whether the oracle FIT -- which is exactly why a
``fit_ok=False`` on two sets went unnoticed until GX2H re-scored them.  The
coverage audit had been set-level, so a missing seed produced four green gates.
The report wording attributed an image-router rationale to a design that has no
image router.

The three committed reports -- ``g6_pilot_gates.json``,
``granularity_summary.json`` and ``run_manifest.json`` -- still carry the OLD
verdict, because they were sealed by the run that produced them.  Re-running
GX7 would overwrite all three.  They are derived outputs rather than frozen
inputs (``g6_pilot_queue.FROZEN`` covers the hierarchy, selection, G6 manifest
and matrix, not these), so overwriting is permitted -- but a report is also
evidence of what was concluded at the time, and the correction is more useful
next to it than on top of it.  So this tool writes ONE new file and refuses to
touch the sealed ones.

WHAT IT DOES NOT DO

No GPU, no model, no training and no torch: the frozen matrix comes from
``g6m.load_frozen_g6()``, the cells are read from the JSON the runner already
committed, and the oracle fit comes from ``oracles_summary.json`` as written by
GX2H.  Every number here is a re-derivation from stored evidence, which is why
it can be checked against the sealed report rather than merely asserted.

The cells glob is the runner's own ``load_all_cells`` contract -- two levels,
``cells/*/seed_*/cell_results.json`` -- so a protocol-repair study cell written
three levels down under ``study_<tag>/`` can never be picked up as a committed
pilot cell.  The loaded count is cross-checked against the sealed
``run_manifest.json`` rather than trusted, because silently evaluating a
different set of cells than the pilot ran is the one failure that would make
this comparison meaningless.

Usage::

    python scripts/e2c_v3_g6_gate_reevaluation.py            # write the file
    python scripts/e2c_v3_g6_gate_reevaluation.py --dry-run  # print only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_BASE = SCRIPT_DIR.parent / "e2c_granularity" / "outputs" / "mllmu"

#: Written by this tool.  Deliberately a NEW name: the sealed reports stay
#: byte-identical so the verdict the pilot was filed under is still readable.
OUTPUT_NAME = "g6_pilot_gates_reevaluated.json"

#: The sealed reports this tool must never write.  Enforced, not documented
#: only: a guard that lives in a comment does not survive a rename.
SEALED_REPORTS = ("g6_pilot_gates.json", "granularity_summary.json",
                  "run_manifest.json")

KIND = "g6_pilot_gate_reevaluation_v1"

#: Hashed directly rather than imported, so this tool never pulls in torch.
RESEARCH_VALIDITY_PATH = SCRIPT_DIR / "e2c_v3_research_validity.py"
LABEL_PARSER_PATH = SCRIPT_DIR / "e2c_v3_label_parser.py"

#: A cell's ``checkpoint_sha256`` is the digest of this file beneath the cell
#: directory, and an oracle's is this file beneath its family directory.
#: Verified by recomputation rather than assumed from the field name.
ADAPTER_RELPATH = ("edited_h", "adapter_final", "adapter_model.safetensors")
ORACLE_ADAPTER_RELPATH = ("adapter_final", "adapter_model.safetensors")

#: The only two oracle families whose bytes the repaired matched-oracle gate
#: reads.  The ``*_finetune`` families stay on disk for the G3 reports written
#: against them, but they are not this gate's evidence, so binding them here
#: would claim a dependency the re-derivation does not have.
GATE_ORACLE_FAMILIES = ("matched_retrain", "loo_retrain")

#: ``candidate_mass`` is not a probability and must not be described as one.
#: It is the sum of teacher-forced FULL-SEQUENCE scores over the recognized
#: candidate labels, and no termination event makes those scores a normalized
#: distribution: each is a product of per-token conditionals over a different
#: label string, so the terms overlap and the sum has no reason to reach 1.
#: ``other_mass = 1 - candidate_mass`` inherits that, and reading it as "the
#: probability the model put elsewhere" is the error this note exists to stop.
#:
#: Deliberately carries NO numbers.  Every figure beside it is recomputed from
#: the cells by ``candidate_score_sum_reading`` -- a number quoted in prose
#: about evidence has to be produced by the code that files the prose, or it
#: silently survives the evidence moving underneath it.
METRIC_SEMANTICS = {
    "candidate_mass": {
        "is_not": "a probability, and not a normalized distribution",
        "is": ("the sum of teacher-forced full-sequence scores over the "
               "recognized candidate labels, each a product of per-token "
               "conditionals over a different label string"),
        "why_it_need_not_reach_one": (
            "there is no termination event and no normalization across "
            "candidates, so the per-label terms overlap and their sum is not "
            "constrained to 1"),
        "consequence_for_other_mass": (
            "other_mass = 1 - candidate_mass is a residual against an "
            "arbitrary ceiling, not the probability of everything else; it is "
            "structurally nonzero even for a perfectly concentrated edit"),
        "gate_reading": ("the frozen floor is therefore a candidate-SCORE-SUM "
                         "threshold, and failing it is a protocol failure "
                         "rather than a statement that the model put weight "
                         "outside the candidate set"),
    },
}

#: The set whose only failing criterion is the candidate score sum, and whose
#: targets are all strictly correct -- the case where calling candidate_mass a
#: probability would turn a protocol threshold into a false claim that the
#: transformation failed.
SCORE_SUM_SET = "gx_mll_254012"
SCORE_SUM_SET_NAME = "Museum Curator"


def candidate_score_sum_reading(cells, mass_floor):
    """Recompute the score-sum-vs-strict-correctness contrast from the cells.

    Returns the block that says, with figures taken from the evidence rather
    than from prose: this set fails the candidate-score-sum gate on every seed
    while every one of its strict target outputs is correct -- a valid protocol
    failure, and not evidence that the transformation itself failed.
    """
    rows = sorted((c for c in cells if c["set_id"] == SCORE_SUM_SET),
                  key=lambda c: c["seed"])
    if not rows:
        raise RuntimeError(
            f"no cells for {SCORE_SUM_SET}, so the candidate-score-sum reading "
            f"cannot be recomputed rather than quoted")
    per_seed, targets_ok, targets_total = {}, 0, 0
    only_mass = True
    for c in rows:
        cr = c["criteria"]
        tgt = [p for p in c["hard_preds"] if p["group"] == "target"]
        ok = sum(1 for p in tgt if p["correct_post_edit"])
        targets_ok += ok
        targets_total += len(tgt)
        per_seed[str(c["seed"])] = {
            "min_candidate_mass": cr["min_candidate_mass"],
            "clears_floor": cr["min_candidate_mass"] >= mass_floor,
            "strict_expected_accuracy": cr["strict_expected_accuracy"],
            "target_rows_correct": ok,
            "target_rows": len(tgt),
            "cell_pass": bool(cr["cell_pass"]),
            "failed_criteria": list(cr["failed_criteria"]),
        }
        if list(cr["failed_criteria"]) != ["min_candidate_mass>=0.99"]:
            only_mass = False
    return {
        "set_id": SCORE_SUM_SET,
        "set_name": SCORE_SUM_SET_NAME,
        "candidate_mass_floor": mass_floor,
        "per_edit_seed": per_seed,
        "target_rows_correct": targets_ok,
        "target_rows": targets_total,
        "all_target_rows_strictly_correct": targets_ok == targets_total,
        "min_candidate_mass_over_seeds": min(
            v["min_candidate_mass"] for v in per_seed.values()),
        "max_candidate_mass_over_seeds": max(
            v["min_candidate_mass"] for v in per_seed.values()),
        "candidate_mass_floor_cleared_on_any_seed": any(
            v["clears_floor"] for v in per_seed.values()),
        "only_failed_criterion_is_the_score_sum": only_mass,
        "verdict": (
            f"{SCORE_SUM_SET_NAME} ({SCORE_SUM_SET}) fails the frozen "
            f"{mass_floor} candidate-SCORE-SUM gate on all "
            f"{len(rows)} edit seeds while "
            f"{targets_ok} of {targets_total} strict target outputs are "
            "correct; the only criterion it fails is the score sum"),
        "interpretation": (
            "a valid protocol failure, and the pilot is correctly recorded as "
            "not passing -- but it is NOT evidence that the transformation "
            "itself failed, because the edit produced the intended label on "
            "every target row it was scored on.  Reading the score sum as a "
            "probability is what would turn this threshold into that claim."),
    }


def _load_sibling(name, filename):
    """Load a sibling script by path.

    These scripts are exec'd through importlib by their callers and are not an
    installed package, so a bare ``import e2c_v3_mllmu_matrix`` would depend on
    whichever sys.path the caller happened to have.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pilot_cells(out_base):
    """The committed pilot cells, by the runner's own two-level glob.

    Loaded unmodified: no field is added or rewritten, so what the gates see is
    byte-for-byte what the runner committed.
    """
    cells = []
    for p in sorted((out_base / "cells").glob("*/seed_*/cell_results.json")):
        with open(p, encoding="utf-8") as f:
            cells.append(json.load(f))
    return cells


def cell_content_index(out_base, sha):
    """Per-cell bytes: the JSON read, the adapter it describes, and the digest
    the cell itself records for that adapter."""
    index = {}
    for p in sorted((out_base / "cells").glob("*/seed_*/cell_results.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        adapter = p.parent.joinpath(*ADAPTER_RELPATH)
        index[rec["cell_id"]] = {
            "cell_results_path": str(p.relative_to(out_base)),
            "cell_results_sha256": sha(p),
            "adapter_path": (str(adapter.relative_to(out_base))
                             if adapter.is_file() else None),
            "adapter_present": adapter.is_file(),
            "adapter_sha256_recomputed": (sha(adapter)
                                          if adapter.is_file() else None),
            "checkpoint_sha256_recorded_by_cell": rec.get("checkpoint_sha256"),
        }
    return index


def oracle_content_index(out_base, sha, oracle_fit):
    """Per-oracle bytes for the two families the matched-oracle gate reads."""
    index = {}
    for sid in sorted(oracle_fit):
        for fam in GATE_ORACLE_FAMILIES:
            fr = (oracle_fit[sid] or {}).get(fam) or {}
            d = out_base / "oracles" / f"{fam}_{sid}"
            adapter = d.joinpath(*ORACLE_ADAPTER_RELPATH)
            results = d / "oracle_results.json"
            index[f"oracle_{fam}_{sid}"] = {
                "family": fam,
                "set_id": sid,
                "dir": str(d.relative_to(out_base)) if d.is_dir() else None,
                "oracle_results_present": results.is_file(),
                "oracle_results_sha256": (sha(results)
                                          if results.is_file() else None),
                "adapter_present": adapter.is_file(),
                "adapter_sha256_recomputed": (sha(adapter)
                                              if adapter.is_file() else None),
                "sha256_recorded_by_summary": fr.get("sha256"),
                "fit_ok": fr.get("fit_ok"),
                "strict_all_expected": fr.get("strict_all_expected"),
                "min_candidate_mass": fr.get("min_candidate_mass"),
            }
    return index


def verify_content_bindings(cells_index, oracles_index, manifest):
    """Three-way comparison of every digest this re-derivation rests on.

    For each cell and each gate oracle, three values from three different
    writers must agree:

      recomputed now      the adapter bytes on disk, hashed by this run
      recorded in place   the digest the cell / oracle summary carries
      sealed manifest     run_manifest.json's own ``checkpoints_sha256``

    The third is what makes this enforce rather than merely describe.  A hash
    stored inside the file it describes is a cross-check only -- whoever wrote
    the data wrote that too -- but the sealed manifest was written by the pilot
    run at e6fd1ce, before this tool existed and in an artifact this tool never
    writes.  A cell swapped, retrained or edited after the fact therefore cannot
    agree with it, whereas comparing the first two alone would pass on any
    self-consistent forgery.

    Raises rather than reporting a partial pass: a binding that could not be
    checked is not a weaker check, it is no check, and reading "verified 12 of
    15" later is indistinguishable from having verified the evidence.
    """
    ckpts = manifest.get("checkpoints_sha256") or {}
    problems = []

    def check(label, entry, sealed_key, recorded_key, adapter_key):
        sealed = ckpts.get(sealed_key)
        recorded = entry.get(recorded_key)
        recomputed = entry.get(adapter_key)
        if sealed is None:
            problems.append(f"{label}: no {sealed_key!r} in the sealed "
                            f"run_manifest.json")
        if recorded != sealed:
            problems.append(f"{label}: recorded {recorded} != sealed manifest "
                            f"{sealed}")
        if not entry.get("adapter_present"):
            problems.append(f"{label}: adapter_model.safetensors absent, so "
                            f"its digest cannot be recomputed and the record "
                            f"is unverified against bytes")
        elif recomputed != recorded:
            problems.append(f"{label}: recomputed adapter digest {recomputed} "
                            f"!= recorded {recorded}")

    for cid, entry in sorted(cells_index.items()):
        check(cid, entry, f"cell_{cid}",
              "checkpoint_sha256_recorded_by_cell",
              "adapter_sha256_recomputed")
    for name, entry in sorted(oracles_index.items()):
        check(name, entry, name,
              "sha256_recorded_by_summary", "adapter_sha256_recomputed")

    if problems:
        raise RuntimeError(
            "content binding failed for "
            f"{len(problems)} artifact(s); refusing to re-derive gates over "
            "evidence whose bytes do not match the sealed run: "
            + "; ".join(problems[:6])
            + ("..." if len(problems) > 6 else ""))
    return {
        "verified": True,
        "method": ("three-way digest agreement: adapter bytes recomputed now, "
                   "the digest recorded in the cell / oracle summary, and the "
                   "sealed run_manifest.json checkpoints_sha256"),
        "why_the_sealed_manifest_is_the_enforcing_side": (
            "it was written by the pilot run itself and is never written by "
            "this tool, so its value was fixed before this re-derivation "
            "existed; the other two could both be produced by whoever wrote "
            "the data"),
        "n_cells_bound": len(cells_index),
        "n_oracles_bound": len(oracles_index),
        "oracle_families_bound": list(GATE_ORACLE_FAMILIES),
    }


def cross_check_cells(cells, out_base, sha, oracle_fit):
    """Refuse to re-derive gates over anything but the pilot's own evidence.

    Two independent questions, both answered before any gate is evaluated.

    IDENTITY.  The sealed ``run_manifest.json`` hashes one ``cell_<cell_id>``
    entry per cell the pilot aggregated, so the set of IDs must match exactly.
    A count alone would pass on a run that silently swapped one cell for
    another.

    CONTENT.  Matching IDs say nothing about bytes: a cell could keep its name
    and carry re-scored, re-trained or hand-edited results.  So every
    ``cell_results.json`` and every gate-relevant ``oracle_results.json`` is
    hashed, and each cell's and oracle's adapter digest is recomputed from disk
    and compared against both its own record and the sealed manifest.

    Raises rather than degrading.  A check that reports "could not verify" and
    continues is indistinguishable from no check at all once its output is read
    later, so the only honest outcomes here are "verified" and "stop".
    """
    globbed = {c["cell_id"] for c in cells}
    manifest_path = out_base / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"cannot cross-check the cells: {manifest_path} absent, so there "
            f"is no sealed record of which cells the pilot aggregated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashed = {k[len("cell_"):] for k in manifest.get("checkpoints_sha256") or {}
              if k.startswith("cell_")}
    if not hashed:
        raise RuntimeError(
            "run_manifest.json hashes no cell_ checkpoints, so the globbed "
            "cells cannot be tied to the sealed run")
    if globbed != hashed:
        raise RuntimeError(
            f"cell identity disagrees with the sealed run_manifest.json: only "
            f"on disk {sorted(globbed - hashed)}, only in the manifest "
            f"{sorted(hashed - globbed)}.  Refusing to re-derive gates over a "
            f"different set of cells.")

    sealed_gates = out_base / "g6_pilot_gates.json"
    sealed_n = None
    if sealed_gates.is_file():
        sealed_n = json.loads(
            sealed_gates.read_text(encoding="utf-8")).get("n_cells_adapted")
        if sealed_n is not None and sealed_n != len(cells):
            raise RuntimeError(
                f"the sealed g6_pilot_gates.json adapted {sealed_n} cells, "
                f"the glob finds {len(cells)}")

    cells_index = cell_content_index(out_base, sha)
    oracles_index = oracle_content_index(out_base, sha, oracle_fit)
    if len(cells_index) != len(cells):
        raise RuntimeError(
            f"indexed {len(cells_index)} cell directories but loaded "
            f"{len(cells)} cells")
    bindings = verify_content_bindings(cells_index, oracles_index, manifest)

    return {
        "checked": True,
        "identity": {
            "method": "cell_id set equality",
            "sealed_source": "run_manifest.json checkpoints_sha256 cell_*",
            "n_cells": len(cells),
            "n_cells_hashed_in_sealed_manifest": len(hashed),
            "sealed_n_cells_adapted": sealed_n,
        },
        "content": bindings,
        "cell_files": {cid: cells_index[cid] for cid in sorted(cells_index)},
        "oracle_files": {n: oracles_index[n] for n in sorted(oracles_index)},
    }


def _per_gate(evaluation):
    """The per-gate dict inside an ``evaluate_gates`` result.

    ``evaluate_gates`` nests them under a key ALSO called ``gates``: the outer
    result is the whole verdict, the inner dict maps a gate name to its result.
    The sealed ``g6_pilot_gates.json`` wraps that result once more under its own
    ``gates`` key, so the FILE must be unwrapped before calling this -- one
    level per wrapper, not one more.
    """
    return evaluation.get("gates") or {}


def delta_vs_sealed(gates, sealed_path):
    """What the repaired gate logic concludes differently, stated explicitly.

    A reader should not have to diff two large JSON trees to learn whether the
    verdict moved.  And the verdict is not the whole story: ``matched_oracle``
    PASSED under the sealed criteria and passes under the repaired ones too, so
    comparing pass flags alone would report "no change" across a gate whose
    evidence was replaced wholesale -- averaged ``*_finetune`` deltas per set
    becoming every-target-per-seed ``*_retrain`` conditions that also check the
    oracle FIT.  The criteria are therefore compared beside the verdicts,
    because a gate that passes for a different reason is a different gate.
    """
    if not sealed_path.is_file():
        return {"available": False, "reason": f"{sealed_path.name} absent"}
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    old = sealed.get("gates") or {}
    new = gates
    fields = ("passed", "failed_gates", "proceed_to_full_matrix",
              "coverage_defects")
    old_pg, new_pg = _per_gate(old), _per_gate(new)

    # An empty read here would report "nothing differs" across four gates,
    # which is indistinguishable from the truth it is standing in for.  The
    # nesting is one level per wrapper and has already been got wrong once, so
    # the comparison refuses to run on a path that resolved to nothing.
    if not new_pg:
        raise RuntimeError(
            "the re-derived gates expose no per-gate results; "
            "evaluate_gates nests them under its own 'gates' key")
    if old and not old_pg:
        raise RuntimeError(
            f"{sealed_path.name} records a gates block but no per-gate "
            f"results; its structure is not the one this comparison assumes")

    per_gate = {}
    for name in sorted({*old_pg, *new_pg}):
        o, n = old_pg.get(name) or {}, new_pg.get(name) or {}
        entry = {}
        if o.get("passed") != n.get("passed"):
            entry["passed"] = {"sealed": o.get("passed"),
                               "reevaluated": n.get("passed")}
        if len(o.get("failures") or []) != len(n.get("failures") or []):
            entry["n_failures"] = {"sealed": len(o.get("failures") or []),
                                   "reevaluated": len(n.get("failures") or [])}
        if o.get("criteria") != n.get("criteria"):
            entry["criteria"] = {
                "sealed": o.get("criteria"), "reevaluated": n.get("criteria"),
                "keys_added": sorted(set(n.get("criteria") or {})
                                     - set(o.get("criteria") or {})),
                "keys_removed": sorted(set(o.get("criteria") or {})
                                       - set(n.get("criteria") or {}))}
        if entry:
            per_gate[name] = entry

    def matched(pg):
        m = pg.get("matched_oracle") or {}
        crit = m.get("criteria") or {}
        per_set = m.get("per_set") or {}
        return {
            "passed": m.get("passed"),
            "n_failures": len(m.get("failures") or []),
            "oracle_families": crit.get("oracle_families"),
            "aggregation": crit.get("aggregation")
            or crit.get("min_mean_delta_oracle"),
            "n_target_rows": sum(v.get("n_target_rows") or 0
                                 for v in per_set.values()),
            "worst_case_delta_retrain": {
                sid: v.get("worst_case_delta_retrain")
                for sid, v in sorted(per_set.items())
                if v.get("worst_case_delta_retrain") is not None},
        }

    return {
        "available": True,
        "sealed_report": sealed_path.name,
        "sealed_n_cells_adapted": sealed.get("n_cells_adapted"),
        "verdict_fields": {
            f: {"sealed": old.get(f), "reevaluated": new.get(f)}
            for f in fields if old.get(f) != new.get(f)},
        "gates_that_differ": per_gate,
        "matched_oracle": {"sealed": matched(old_pg),
                           "reevaluated": matched(new_pg)},
        "note": ("the sealed report is left byte-identical; this records what "
                 "the repaired criteria conclude about the SAME stored cells, "
                 "verified to be the same cells by cell_id"),
    }


#: Recorded in the artifact because it is a property of regenerating it, and
#: mistaking it for drift would send someone looking for a bug that is not
#: there.  ``g6m.worktree_state`` deliberately does NOT apply its
#: ``exclude_prefixes`` to the tracked-only determination -- exclusions scope
#: the untracked accounting, while what provenance RECORDS stays unscoped -- so
#: writing this file necessarily makes the next run see a modified tracked file.
REGENERATION_NOTE = (
    "provenance.clean_worktree is recorded unscoped, as g6m.worktree_state "
    "intends: its exclude_prefixes apply only to the untracked accounting, and "
    "the tracked-only determination is what provenance records.  Provenance is "
    "read BEFORE the write, so to derive this artifact at a genuinely clean "
    "worktree pass --dest a path OUTSIDE the tracked output tree, then commit "
    "the file from there; writing straight into the tree leaves the artifact "
    "dirty for whatever runs next and records dirty_tracked_only=True.  The "
    "reproducible part is everything else: inputs, metric_semantics, gates, "
    "sets and delta_vs_sealed are byte-identical across regenerations at an "
    "unchanged commit, and are pinned by a test that compares them with "
    "provenance.clean_worktree excluded.")


def build(out_base):
    g6m = _load_sibling("e2c_v3_mllmu_matrix", "e2c_v3_mllmu_matrix.py")
    sha = g6m.sha256_file

    _manifest, matrix = g6m.load_frozen_g6()
    cells = load_pilot_cells(out_base)
    if not cells:
        raise RuntimeError(f"no pilot cells under {out_base / 'cells'}")

    summary_path = out_base / "oracles" / "oracles_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"oracle summary absent: {summary_path}")
    oracle_fit = json.loads(summary_path.read_text(encoding="utf-8"))

    gate_cells = g6m.cells_for_gates(cells, oracle_fit=oracle_fit)
    gates = g6m.evaluate_gates(gate_cells, matrix)

    # Read from the frozen criteria rather than retyped: the floor this whole
    # terminology correction is about belongs to gx.PASS_CRITERIA, and a second
    # copy of it here is how a report comes to describe a gate the runner does
    # not actually apply.
    mass_floor = g6m.gx.PASS_CRITERIA["min_candidate_mass"]

    return {
        "kind": KIND,
        "dataset": "mllmu",
        "purpose": (
            "the four pilot gates re-derived with the criteria repaired in "
            "992efa9, on the SAME stored cells and the SAME post-GX2H oracle "
            "fit the sealed reports were computed from; filed beside them "
            "rather than over them"),
        "metric_semantics": {
            **METRIC_SEMANTICS,
            "candidate_score_sum_reading": candidate_score_sum_reading(
                cells, mass_floor),
            "criteria_source": ("gx.PASS_CRITERIA, read at run time; the floor "
                                f"quoted throughout is {mass_floor}"),
        },
        "relationship_to_sealed_reports": {
            "rewrites_nothing": list(SEALED_REPORTS),
            "why": ("a report is evidence of what was concluded at the time; "
                    "the repaired criteria are recorded next to it so both "
                    "verdicts remain readable and the difference is explicit"),
        },
        "inputs": {
            "matrix": {"path": str(g6m.G6_MATRIX_PATH),
                       "sha256": sha(g6m.G6_MATRIX_PATH)},
            "oracles_summary": {"path": str(summary_path),
                                "sha256": sha(summary_path)},
            "cells_glob": "cells/*/seed_*/cell_results.json",
            "cells_glob_depth_note": (
                "two levels, matching the runner's load_all_cells, so a study "
                "cell under study_<tag>/ is never picked up as a pilot cell"),
            "n_cells": len(cells),
            "cell_ids": sorted(c["cell_id"] for c in cells),
            "cell_identity_and_content_cross_check": cross_check_cells(
                cells, out_base, sha, oracle_fit),
        },
        "gates": gates,
        "sets": g6m.per_set_provenance(matrix, gate_cells),
        "n_cells_adapted": len(gate_cells),
        "delta_vs_sealed": delta_vs_sealed(
            gates, out_base / "g6_pilot_gates.json"),
        "provenance": {
            **g6m.g6_provenance(out_base),
            "regeneration_note": REGENERATION_NOTE,
            "gate_module_sha256": sha(Path(g6m.__file__).resolve()),
            # Hashed from the files directly rather than by calling
            # rv.script_sha256() / rv.label_parser_sha256(), so this tool stays
            # torch-free: e2c_v3_research_validity imports torch at module
            # scope, and a sha256 is a function of the bytes, so reading them
            # here gives the identical digest without the import.  Pinned by a
            # test asserting the two routes agree.
            "shared_scoring_script_sha256": sha(RESEARCH_VALIDITY_PATH),
            "label_parser_script_sha256": sha(LABEL_PARSER_PATH),
            "reevaluated_by": Path(__file__).name,
            "reevaluation_script_sha256": sha(Path(__file__).resolve()),
            "oracle_fit_source": (
                "oracles_summary.json (matched_retrain, post-GX2H hard "
                "re-evaluation)"),
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-base", default=str(OUT_BASE))
    p.add_argument("--dest", default=None,
                   help="where to write the artifact; defaults to "
                        f"<out-base>/{OUTPUT_NAME}.  Pass a path OUTSIDE the "
                        "tracked output tree to derive the report at a clean "
                        "worktree -- provenance is read before the write, but "
                        "writing into the tree leaves the file dirty for "
                        "whatever runs next")
    p.add_argument("--dry-run", action="store_true",
                   help="print the re-derivation without writing it")
    args = p.parse_args(argv)

    out_base = Path(args.out_base).resolve()
    dest = (Path(args.dest).resolve() if args.dest
            else out_base / OUTPUT_NAME)

    # Compare resolved paths, not names: a caller can point --dest at a sealed
    # report by absolute path, and a name check alone would wave that through.
    sealed = {n: (out_base / n).resolve() for n in SEALED_REPORTS}
    for name, path in sealed.items():
        if dest == path:
            raise RuntimeError(
                f"--dest resolves to the sealed report {name}; this tool files "
                f"a re-derivation beside the sealed reports and never over "
                f"one")

    block = build(out_base)
    text = json.dumps(block, indent=2)

    if args.dry_run:
        print(text)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")

    g = block["gates"]
    content = block["inputs"]["cell_identity_and_content_cross_check"]["content"]
    print(f"cells re-derived       : {block['n_cells_adapted']}", file=sys.stderr)
    print(f"content bound          : {content['n_cells_bound']} cells + "
          f"{content['n_oracles_bound']} oracles "
          f"({'verified' if content['verified'] else 'UNVERIFIED'})",
          file=sys.stderr)
    print(f"gates passed           : {g['passed']}", file=sys.stderr)
    print(f"failed gates           : {g['failed_gates']}", file=sys.stderr)
    print(f"coverage exact         : {g['coverage']['exact']}", file=sys.stderr)
    print(f"coverage defects       : {g['coverage_defects']}", file=sys.stderr)
    print(f"proceed_to_full_matrix : {g['proceed_to_full_matrix']}",
          file=sys.stderr)
    cw = block["provenance"]["clean_worktree"]
    print(f"executing commit       : "
          f"{block['provenance']['executing_commit'][:12]}", file=sys.stderr)
    print(f"dirty_tracked_only     : {cw['dirty_tracked_only']}",
          file=sys.stderr)
    if not args.dry_run:
        print(f"written                : {dest}", file=sys.stderr)
        print(f"inside tracked output  : "
              f"{dest == (out_base / OUTPUT_NAME).resolve()}", file=sys.stderr)
        for name in SEALED_REPORTS:
            print(f"untouched (sealed)     : {out_base / name}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
