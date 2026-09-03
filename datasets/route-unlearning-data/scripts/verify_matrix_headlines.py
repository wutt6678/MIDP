#!/usr/bin/env python3
"""CPU-only repair + verification of the E2C-v3 matrix headline values.

Reconstructs EVERY headline number directly from the 75 committed cell
files (e2c_matrix/outputs/<ds>/cells/**/cell_results.json) -- no model
execution, no GPU -- and cross-checks the committed per-set summaries.

Two modes
=========
--repair-salmu
    The executed runner counted the one unrouted SALMU image (where h was
    never queried) as an h-unparseable output.  This deterministically
    recomputes, for every SALMU cell, from the committed e2e_rows:
        g_routing_failures          (unchanged, authoritative)
        h_unparseable_outputs       = routed rows with pred_alias None
        h_multi_label_ambiguous     = routed rows with >1 recognized label
    rewrites the cell files in place (all other fields untouched), and
    regenerates e2c_matrix/outputs/salmu/matrix_summary.json from the
    repaired cells via the runner's own aggregate().

--verify (default)
    Rebuilds the canonical headline values for all three datasets and
    cross-checks them against the committed matrix_summary.json files:
    cells/pass/failure counts, retain accuracy, per-cell max target
    p(original alias) WITH attribution (set, seed, target), conditional
    and unconditional E2E, unparseable/multi-label totals (recomputed
    from rows, authoritative), oracle distances, MLLMU QA probes.
    Writes e2c_matrix/reports/headline_verification.json (canonical
    values; paper tables must be generated from this file or from the
    committed matrix_summary.json files it validates).

Exact commands
==============
    python scripts/verify_matrix_headlines.py --repair-salmu
    python scripts/verify_matrix_headlines.py --verify
"""
import argparse
import importlib.util
import json
import logging
import statistics
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("verify_matrix")

SCRIPT_DIR = Path(__file__).resolve().parents[1]
OUT_ROOT = SCRIPT_DIR / "e2c_matrix" / "outputs"
REPORTS = SCRIPT_DIR / "e2c_matrix" / "reports"
DATASETS = ["ppubench", "mllmu", "salmu"]


def _load_mx():
    spec = importlib.util.spec_from_file_location(
        "e2c_mx_for_verify", SCRIPT_DIR / "scripts" / "e2c_v3_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_paths(ds):
    return sorted((OUT_ROOT / ds / "cells").glob("*/seed_*/cell_results.json"))


def _load_cell(p):
    with open(p) as f:
        return json.load(f)


def recount_e2e(rows):
    """Authoritative h-side output-health counts from committed rows.

    h is only queried where g routed (pred_code present); unrouted rows are
    g failures and must NOT inflate h_unparseable_outputs.
    """
    routed = [r for r in rows if r.get("g_routed") and r.get("pred_code")]
    return {
        "h_unparseable_outputs": sum(
            1 for r in routed if r.get("pred_alias") is None),
        "h_multi_label_ambiguous_outputs": sum(
            1 for r in routed if r.get("multi_label_ambiguous")),
        "g_routing_failures": len(rows) - len(routed),
        "n_rows": len(rows),
    }


def repair_salmu():
    changed = 0
    for p in _cell_paths("salmu"):
        cell = _load_cell(p)
        fixed = recount_e2e(cell["e2e_rows"])
        e = cell["e2e"]
        diff = {k: (e[k], fixed[k]) for k in
                ("h_unparseable_outputs", "h_multi_label_ambiguous_outputs",
                 "g_routing_failures") if e[k] != fixed[k]}
        if diff:
            e.update({k: fixed[k] for k in fixed if k in e})
            with open(p, "w") as f:
                json.dump(cell, f, indent=2)
            changed += 1
            logger.info(f"repaired {p.parent.parent.name}/{p.parent.name}: "
                        f"{ {k: f'{a}->{b}' for k, (a, b) in diff.items()} }")
    logger.info(f"repair: {changed} SALMU cell files updated")
    # regenerate the SALMU summary from repaired cells with the runner's
    # own aggregation (CPU-only)
    mx = _load_mx()
    args = argparse.Namespace(
        dataset="salmu", seeds=[17, 42, 123], smoke=False, device="cpu",
        ul_steps=500, ul_warmup=50, ul_lr=2e-5, ul_repeat=50,
        route_steps=3000, route_warmup=200, route_lr=2e-5, route_repeat=50,
        max_gen_tokens=12, archive_pilots=False, phase="MX3")
    matrix = mx.load_matrix(args)
    all_cells = {f"{c['set_id']}__seed{c['seed']}": c
                 for c in (_load_cell(p) for p in _cell_paths("salmu"))}
    mx.aggregate(args, OUT_ROOT / "salmu", matrix, all_cells)
    logger.info("repair: salmu matrix_summary.json regenerated")
    return changed


def rebuild_dataset(ds):
    """Canonical headline values rebuilt from the cell files alone."""
    cells = [_load_cell(p) for p in _cell_paths(ds)]
    if not cells:
        raise RuntimeError(f"no cells found for {ds}")
    worst = {"p_original": -1.0}
    unparse_total = multi_total = routing_fail_total = 0
    cond, uncond, retain, tgt_l2, ret_l2_max = [], [], [], [], 0.0
    fails = {"leak": 0, "soft_leak": 0, "suppress_incomplete": 0,
             "retain_broken": 0, "oov_garbage": 0, "catastrophic": 0,
             "cell_pass": 0}
    qa = []
    for c in cells:
        ff = c["failure_flags"]
        fails["leak"] += bool(ff["hard_leaks"])
        fails["soft_leak"] += bool(ff["soft_leaks"])
        fails["suppress_incomplete"] += bool(ff["suppress_incomplete"])
        fails["retain_broken"] += bool(ff["retain_broken"])
        fails["oov_garbage"] += bool(ff["oov_garbage_ids"])
        fails["catastrophic"] += bool(ff["catastrophic"])
        fails["cell_pass"] += bool(ff["cell_pass"])
        retain.append(ff["retain_acc"])
        for t in c["targets"]:
            p = c["soft"][t]["p_original_alias"]
            if p > worst["p_original"]:
                worst = {"p_original": p, "set_id": c["set_id"],
                         "seed": c["seed"], "target": t}
        rc = recount_e2e(c["e2e_rows"])
        unparse_total += rc["h_unparseable_outputs"]
        multi_total += rc["h_multi_label_ambiguous_outputs"]
        routing_fail_total += rc["g_routing_failures"]
        cond.append(c["e2e"]["e2e_conditional_acc"])
        uncond.append(c["e2e"]["e2e_unconditional_acc"])
        td = [c["gated_distance_to_oracle"][t]["distance"]["l2"]
              for t in c["targets"]
              if c["gated_distance_to_oracle"][t]["reliable"]]
        if td and len(td) == len(c["targets"]):
            tgt_l2.append(statistics.fmean(td))
        for iid, v in c["gated_distance_to_oracle"].items():
            if iid not in c["targets"] and v["reliable"]:
                ret_l2_max = max(ret_l2_max, v["distance"]["l2"])
        if "qa_probes_post" in c:
            qa.append(c["qa_probes_post"]["accuracy"])
    out = {
        "dataset": ds,
        "n_cells": len(cells),
        "failure_counts": fails,
        "cell_pass_rate": fails["cell_pass"] / len(cells),
        "retain_acc": {"mean": statistics.fmean(retain),
                       "min": min(retain)},
        "max_target_p_original": worst,
        "e2e_conditional_acc": {"mean": statistics.fmean(cond),
                                "min": min(cond), "max": max(cond)},
        "e2e_unconditional_acc": {"mean": statistics.fmean(uncond),
                                  "min": min(uncond), "max": max(uncond)},
        "h_unparseable_outputs_total": unparse_total,
        "h_multi_label_outputs_total": multi_total,
        "g_routing_failures_total": routing_fail_total,
        "oracle_target_l2_mean": (statistics.fmean(tgt_l2)
                                  if tgt_l2 else None),
        "oracle_retained_l2_max": ret_l2_max,
    }
    if qa:
        out["qa_probes_post_accuracy"] = {
            "mean": statistics.fmean(qa), "min": min(qa), "max": max(qa)}
    return out


def cross_check_summaries(rebuilt):
    """Rebuilt values vs committed per-set matrix_summary.json aggregates."""
    problems = []
    for ds, rb in rebuilt.items():
        with open(OUT_ROOT / ds / "matrix_summary.json") as f:
            summary = json.load(f)
        cells_total = sum(a["failure_counts"]["cells"]
                          for a in summary["per_set"].values())
        pass_total = sum(a["failure_counts"]["cell_pass"]
                         for a in summary["per_set"].values())
        if cells_total != rb["n_cells"]:
            problems.append(f"{ds}: cells {cells_total} != {rb['n_cells']}")
        if pass_total != rb["failure_counts"]["cell_pass"]:
            problems.append(f"{ds}: pass {pass_total} != "
                            f"{rb['failure_counts']['cell_pass']}")
        unparse = sum(a["unparseable_outputs_total"]["mean"] * a["n_seeds"]
                      for a in summary["per_set"].values())
        if round(unparse) != rb["h_unparseable_outputs_total"]:
            problems.append(f"{ds}: summary unparseable {round(unparse)} != "
                            f"rebuilt {rb['h_unparseable_outputs_total']}")
        max_p = max(a["max_target_p_original"]["max"]
                    for a in summary["per_set"].values())
        if abs(max_p - rb["max_target_p_original"]["p_original"]) > 1e-15:
            problems.append(f"{ds}: summary max p_orig {max_p!r} != rebuilt "
                            f"{rb['max_target_p_original']['p_original']!r}")
    return problems


def verify():
    rebuilt = {ds: rebuild_dataset(ds) for ds in DATASETS}
    problems = cross_check_summaries(rebuilt)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = {
        "note": "canonical headline values rebuilt from the 75 committed "
                "cell files (CPU-only); paper tables must be generated "
                "from this file or the matrix_summary.json files it "
                "validates",
        "recount_rule": "h_unparseable_outputs / h_multi_label counted "
                        "ONLY over routed rows (h actually queried); "
                        "unrouted rows are g_routing_failures",
        "per_dataset": rebuilt,
        "cross_check_vs_committed_summaries": problems or "OK",
    }
    path = REPORTS / "headline_verification.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    for ds, rb in rebuilt.items():
        w = rb["max_target_p_original"]
        logger.info(
            f"{ds}: cells={rb['n_cells']} pass={rb['failure_counts']['cell_pass']}"
            f" retain_min={rb['retain_acc']['min']:.4f}"
            f" max_p_orig={w['p_original']:.6e}"
            f" ({w['set_id']} seed{w['seed']} target {w['target']})"
            f" e2e_cond={rb['e2e_conditional_acc']['mean']:.4f}"
            f" e2e_uncond=[{rb['e2e_unconditional_acc']['min']:.4f},"
            f"{rb['e2e_unconditional_acc']['max']:.4f}]"
            f" unparse={rb['h_unparseable_outputs_total']}"
            f" multi={rb['h_multi_label_outputs_total']}"
            f" tgtL2={rb['oracle_target_l2_mean']}"
            f" retL2max={rb['oracle_retained_l2_max']:.2e}")
    if problems:
        for p in problems:
            logger.error(f"MISMATCH: {p}")
        logger.error(f"verification written to {path} WITH MISMATCHES")
        return 1
    logger.info(f"verification OK, written to {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repair-salmu", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.repair_salmu:
        repair_salmu()
    if a.verify or not a.repair_salmu:
        return verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
