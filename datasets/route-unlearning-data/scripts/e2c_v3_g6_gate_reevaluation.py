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


def cross_check_cells(cells, out_base):
    """Refuse to compare against a different set of cells than the pilot ran.

    Checked by IDENTITY, not by count: the sealed ``run_manifest.json`` hashes
    one ``cell_<cell_id>`` checkpoint entry per cell it aggregated, so the set
    of IDs must match exactly.  A count alone would pass on a run that silently
    swapped one cell for another -- and silently evaluating a different pilot
    than the one that was filed is the single failure that would make the
    comparison below meaningless.

    Raises rather than degrading.  A check that reports "could not verify" and
    continues is indistinguishable from no check at all once its output is
    read later, so the only honest outcomes here are "verified" and "stop".
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
    return {"checked": True, "method": "cell_id set equality",
            "sealed_source": "run_manifest.json checkpoints_sha256 cell_*",
            "n_cells": len(cells),
            "n_cells_hashed_in_sealed_manifest": len(hashed),
            "sealed_n_cells_adapted": sealed_n}


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

    return {
        "kind": KIND,
        "dataset": "mllmu",
        "purpose": (
            "the four pilot gates re-derived with the criteria repaired in "
            "992efa9, on the SAME stored cells and the SAME post-GX2H oracle "
            "fit the sealed reports were computed from; filed beside them "
            "rather than over them"),
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
            "cell_identity_cross_check": cross_check_cells(cells, out_base),
        },
        "gates": gates,
        "sets": g6m.per_set_provenance(matrix, gate_cells),
        "n_cells_adapted": len(gate_cells),
        "delta_vs_sealed": delta_vs_sealed(
            gates, out_base / "g6_pilot_gates.json"),
        "provenance": {
            **g6m.g6_provenance(out_base),
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
    p.add_argument("--dry-run", action="store_true",
                   help="print the re-derivation without writing it")
    args = p.parse_args(argv)

    out_base = Path(args.out_base).resolve()
    dest = out_base / OUTPUT_NAME
    if dest.name in SEALED_REPORTS:  # guard against a rename making this true
        raise RuntimeError(f"{OUTPUT_NAME} must not be a sealed report name")

    block = build(out_base)
    text = json.dumps(block, indent=2)

    if args.dry_run:
        print(text)
    else:
        dest.write_text(text + "\n", encoding="utf-8")

    g = block["gates"]
    print(f"cells re-derived       : {block['n_cells_adapted']}", file=sys.stderr)
    print(f"gates passed           : {g['passed']}", file=sys.stderr)
    print(f"failed gates           : {g['failed_gates']}", file=sys.stderr)
    print(f"coverage exact         : {g['coverage']['exact']}", file=sys.stderr)
    print(f"coverage defects       : {g['coverage_defects']}", file=sys.stderr)
    print(f"proceed_to_full_matrix : {g['proceed_to_full_matrix']}",
          file=sys.stderr)
    if not args.dry_run:
        print(f"written                : {dest}", file=sys.stderr)
        for name in SEALED_REPORTS:
            print(f"untouched (sealed)     : {out_base / name}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
