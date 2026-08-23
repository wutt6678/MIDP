#!/usr/bin/env python3
"""E2C analyze routes — compute route metrics, bootstrap, and R1–R7 gates.

Usage:
    python scripts/e2c_analyze_routes.py \
        --eval-dir-m e2c/outputs/<sha>/M/eval \
        --eval-dir-d e2c/outputs/<sha>/D/eval \
        --eval-dir-ms e2c/outputs/<sha>/M_shuffled/eval \
        --manifest-dir e2c/manifests \
        --output-dir e2c/reports

Generates:
    e2c_route_comparison.json
    e2c_route_validation.json
    e2c_route_establishment_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.route_metrics import (
    compute_accuracy_from_probes,
    compute_i2n_accuracy,
    compute_route_effects,
    compute_shuffled_analysis,
    identity_clustered_bootstrap,
)
from route_data.e2c.route_validation import (
    aggregate_route_decision,
    classify_failure,
    evaluate_r1,
    evaluate_r2,
    evaluate_r3,
    evaluate_r4,
    evaluate_r5,
    evaluate_r6,
    evaluate_r7,
)
from route_data.e2c.synthetic_manifest import (
    load_json_manifest,
    write_json_manifest,
)


def load_eval_results(eval_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all eval JSONL files from an eval directory."""
    results: dict[str, list[dict[str, Any]]] = {}
    families = ["I2N", "NAME", "DV_syn", "IPN_syn", "WN", "VTC", "VISUAL_CONTROL"]

    for family in families:
        path = eval_dir / f"{family}.jsonl"
        if path.exists():
            family_results = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        family_results.append(json.loads(line))
            results[family] = family_results
        else:
            results[family] = []

    return results


def main():
    parser = argparse.ArgumentParser(description="E2C route analysis")
    parser.add_argument("--eval-dir-m", required=True)
    parser.add_argument("--eval-dir-d", required=True)
    parser.add_argument("--eval-dir-ms", required=True)
    parser.add_argument("--manifest-dir", default="e2c/manifests")
    parser.add_argument("--output-dir", default="e2c/reports")
    parser.add_argument("--base-eval-dir", default=None,
                        help="Base model eval dir for R7 visual controls")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir)

    # ------------------------------------------------------------------ #
    # Load evaluation results
    # ------------------------------------------------------------------ #
    print("[E2C] Loading evaluation results...")
    m_results = load_eval_results(Path(args.eval_dir_m))
    d_results = load_eval_results(Path(args.eval_dir_d))
    ms_results = load_eval_results(Path(args.eval_dir_ms))

    # Load manifests
    true_mapping = load_json_manifest(
        manifest_dir / "synthetic_attribute_mapping.json"
    )
    shuffled_mapping = load_json_manifest(
        manifest_dir / "synthetic_attribute_mapping_shuffled.json"
    )

    from route_data.e2c.synthetic_manifest import generate_identity_ids
    exp_ids, _ = generate_identity_ids()

    # ------------------------------------------------------------------ #
    # Compute accuracies
    # ------------------------------------------------------------------ #
    print("[E2C] Computing accuracies...")

    i2n_acc_m = compute_i2n_accuracy(m_results.get("I2N", []))
    name_acc_m = compute_accuracy_from_probes(m_results.get("NAME", []))
    name_acc_d = compute_accuracy_from_probes(d_results.get("NAME", []))
    dv_acc_m = compute_accuracy_from_probes(m_results.get("DV_syn", []))
    dv_acc_d = compute_accuracy_from_probes(d_results.get("DV_syn", []))

    print(f"  I2N_M:       {i2n_acc_m:.4f}")
    print(f"  NAME_M:      {name_acc_m:.4f}")
    print(f"  NAME_D:      {name_acc_d:.4f}")
    print(f"  DV_M:        {dv_acc_m:.4f}")
    print(f"  DV_D:        {dv_acc_d:.4f}")

    # ------------------------------------------------------------------ #
    # Compute route effects
    # ------------------------------------------------------------------ #
    print("[E2C] Computing route effects...")

    m_effects = compute_route_effects(
        dv_results=m_results.get("DV_syn", []),
        ipn_results=m_results.get("IPN_syn", []),
        wn_results=m_results.get("WN", []),
        vtc_results=m_results.get("VTC", []),
    )
    d_effects = compute_route_effects(
        dv_results=d_results.get("DV_syn", []),
        ipn_results=d_results.get("IPN_syn", []),
        wn_results=d_results.get("WN", []),
        vtc_results=d_results.get("VTC", []),
    )

    print(f"  M NameEffect:       {m_effects['NameEffect']:.4f}")
    print(f"  M WrongNameEffect:  {m_effects['WrongNameEffect']:.4f}")
    print(f"  M ConflictEffect:   {m_effects['ConflictEffect']:.4f}")
    print(f"  D WrongNameEffect:  {d_effects['WrongNameEffect']:.4f}")
    print(f"  D ConflictEffect:   {d_effects['ConflictEffect']:.4f}")

    # ------------------------------------------------------------------ #
    # M-shuffled analysis
    # ------------------------------------------------------------------ #
    print("[E2C] M-shuffled analysis...")
    ms_analysis = compute_shuffled_analysis(
        ms_results.get("NAME", []) + ms_results.get("DV_syn", []) + ms_results.get("IPN_syn", []),
        true_mapping,
        shuffled_mapping,
    )
    for family, stats in ms_analysis.items():
        print(
            f"  {family}: true={stats['agreement_with_true_mapping']:.4f} "
            f"shuffled={stats['agreement_with_shuffled_mapping']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Identity-clustered bootstrap
    # ------------------------------------------------------------------ #
    print(f"[E2C] Bootstrap ({args.bootstrap_resamples} resamples)...")
    all_m_results = (
        m_results.get("NAME", [])
        + m_results.get("DV_syn", [])
        + m_results.get("IPN_syn", [])
        + m_results.get("WN", [])
        + m_results.get("VTC", [])
    )
    all_d_results = (
        d_results.get("NAME", [])
        + d_results.get("DV_syn", [])
        + d_results.get("IPN_syn", [])
        + d_results.get("WN", [])
        + d_results.get("VTC", [])
    )

    bootstrap = identity_clustered_bootstrap(
        probe_results_m=all_m_results,
        probe_results_d=all_d_results,
        experimental_ids=exp_ids,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    for key, stats in bootstrap["bootstrap"].items():
        print(
            f"  {key}: mean={stats['mean']:.4f} "
            f"CI=[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] "
            f"excl0={stats['ci_excludes_zero']}"
        )

    # ------------------------------------------------------------------ #
    # R1–R7 gates
    # ------------------------------------------------------------------ #
    print("[E2C] Evaluating R1–R7 gates...")

    wn_ci = bootstrap["bootstrap"].get("abs_WrongNameEffect_M_minus_D", {})
    vtc_ci = bootstrap["bootstrap"].get("abs_ConflictEffect_M_minus_D", {})

    gates = {}
    gates["R1"] = evaluate_r1(i2n_acc_m)
    gates["R2"] = evaluate_r2(name_acc_m, name_acc_d)
    gates["R3"] = evaluate_r3(dv_acc_m, dv_acc_d)
    gates["R4"] = evaluate_r4(
        m_effects["abs_WrongNameEffect"],
        d_effects["abs_WrongNameEffect"],
        wn_ci.get("ci_excludes_zero", False),
        bootstrap_ci=wn_ci,
    )
    gates["R5"] = evaluate_r5(
        m_effects["abs_ConflictEffect"],
        d_effects["abs_ConflictEffect"],
        vtc_ci.get("ci_excludes_zero", False),
        bootstrap_ci=vtc_ci,
    )

    # R6: M-shuffled
    # Use NAME family as primary signal
    ms_name = ms_analysis.get("NAME", {})
    gates["R6"] = evaluate_r6(
        ms_name.get("agreement_with_shuffled_mapping", 0.0),
        ms_name.get("agreement_with_true_mapping", 0.0),
    )

    # R7: Visual controls
    visual_results: dict[str, dict[str, float]] = {}
    vc_m = m_results.get("VISUAL_CONTROL", [])
    if args.base_eval_dir:
        base_results = load_eval_results(Path(args.base_eval_dir))
        vc_base = base_results.get("VISUAL_CONTROL", [])
    else:
        vc_base = []

    # Group by attribute
    for attr in ("smiling", "eyeglasses", "hat"):
        m_attr = [r for r in vc_m if r.get("visual_attribute") == attr]
        base_attr = [r for r in vc_base if r.get("visual_attribute") == attr]
        visual_results[attr] = {
            "trained_accuracy": compute_accuracy_from_probes(m_attr) if m_attr else 0.0,
            "base_accuracy": compute_accuracy_from_probes(base_attr) if base_attr else 1.0,
        }

    gates["R7"] = evaluate_r7(visual_results)

    # Print gate results
    for gate_id in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
        g = gates[gate_id]
        print(f"  {gate_id}: {g['status']}")

    # ------------------------------------------------------------------ #
    # Aggregate decision
    # ------------------------------------------------------------------ #
    decision = aggregate_route_decision(gates)
    print(f"\n[E2C] ROUTE_ESTABLISHED = {decision['route_established']}")
    print(f"[E2C] CONTROLLED_UNLEARNING_ALLOWED = {decision['controlled_unlearning_allowed']}")

    if not decision["route_established"]:
        failures = classify_failure(gates)
        for f in failures:
            print(f"  Failure {f['code']}: {f['pattern']} — {f['action']}")

    # ------------------------------------------------------------------ #
    # Write artifacts
    # ------------------------------------------------------------------ #
    # Route comparison JSON
    comparison = {
        "M": {
            "I2N": i2n_acc_m,
            "NAME": name_acc_m,
            "DV_syn": dv_acc_m,
            "IPN_syn": compute_accuracy_from_probes(m_results.get("IPN_syn", [])),
            "WN": compute_accuracy_from_probes(m_results.get("WN", [])),
            "VTC": compute_accuracy_from_probes(m_results.get("VTC", [])),
            "NameEffect": m_effects["NameEffect"],
            "WrongNameEffect": m_effects["WrongNameEffect"],
            "ConflictEffect": m_effects["ConflictEffect"],
        },
        "D": {
            "NAME": name_acc_d,
            "DV_syn": dv_acc_d,
            "IPN_syn": compute_accuracy_from_probes(d_results.get("IPN_syn", [])),
            "WN": compute_accuracy_from_probes(d_results.get("WN", [])),
            "VTC": compute_accuracy_from_probes(d_results.get("VTC", [])),
            "WrongNameEffect": d_effects["WrongNameEffect"],
            "ConflictEffect": d_effects["ConflictEffect"],
        },
        "M_shuffled": ms_analysis,
        "bootstrap": bootstrap,
        "contrasts": {
            "NAME_M_minus_D": name_acc_m - name_acc_d,
            "abs_WrongNameEffect_M_minus_D": (
                m_effects["abs_WrongNameEffect"] - d_effects["abs_WrongNameEffect"]
            ),
            "abs_ConflictEffect_M_minus_D": (
                m_effects["abs_ConflictEffect"] - d_effects["abs_ConflictEffect"]
            ),
        },
    }
    write_json_manifest(comparison, output_dir / "e2c_route_comparison.json")

    # Route validation JSON
    validation = {
        "route_established": decision["route_established"],
        "controlled_unlearning_allowed": decision["controlled_unlearning_allowed"],
        "gates": gates,
    }
    write_json_manifest(validation, output_dir / "e2c_route_validation.json")

    # Markdown report
    report_path = output_dir / "e2c_route_establishment_report.md"
    _write_markdown_report(
        report_path, comparison, gates, decision, bootstrap, ms_analysis,
    )

    print(f"\n[E2C] Reports written to {output_dir}")


def _write_markdown_report(
    path: Path,
    comparison: dict,
    gates: dict,
    decision: dict,
    bootstrap: dict,
    ms_analysis: dict,
) -> None:
    """Generate the Markdown route establishment report."""
    lines = [
        "# E2C Route Establishment Report",
        "",
        "## Summary",
        "",
        f"**ROUTE_ESTABLISHED**: {decision['route_established']}",
        f"**CONTROLLED_UNLEARNING_ALLOWED**: {decision['controlled_unlearning_allowed']}",
        "",
        "## Probe Accuracy Table",
        "",
        "| Condition | I2N | NAME | DV-syn | IPN-syn | WN | VTC | NameEffect | WrongNameEffect | ConflictEffect |",
        "|-----------|-----|------|--------|---------|----|-----|------------|-----------------|----------------|",
    ]

    m = comparison["M"]
    d = comparison["D"]
    lines.append(
        f"| M | {m.get('I2N', 0):.3f} | {m.get('NAME', 0):.3f} | {m.get('DV_syn', 0):.3f} "
        f"| {m.get('IPN_syn', 0):.3f} | {m.get('WN', 0):.3f} | {m.get('VTC', 0):.3f} "
        f"| {m.get('NameEffect', 0):.4f} | {m.get('WrongNameEffect', 0):.4f} "
        f"| {m.get('ConflictEffect', 0):.4f} |"
    )
    lines.append(
        f"| D | - | {d.get('NAME', 0):.3f} | {d.get('DV_syn', 0):.3f} "
        f"| {d.get('IPN_syn', 0):.3f} | {d.get('WN', 0):.3f} | {d.get('VTC', 0):.3f} "
        f"| - | {d.get('WrongNameEffect', 0):.4f} "
        f"| {d.get('ConflictEffect', 0):.4f} |"
    )
    lines.append("| M-shuffled | - | - | - | - | - | - | - | - | - |")

    lines.extend([
        "",
        "## Between-Condition Contrasts",
        "",
    ])

    contrasts = comparison["contrasts"]
    for key, val in contrasts.items():
        ci_data = bootstrap.get("bootstrap", {}).get(key, {})
        ci_str = ""
        if ci_data:
            ci_str = f" (CI: [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}])"
        lines.append(f"- **{key}**: {val:.4f}{ci_str}")

    lines.extend([
        "",
        "## M-shuffled Analysis",
        "",
    ])
    for family, stats in ms_analysis.items():
        lines.append(
            f"- {family}: true_agreement={stats['agreement_with_true_mapping']:.4f}, "
            f"shuffled_agreement={stats['agreement_with_shuffled_mapping']:.4f}"
        )

    lines.extend([
        "",
        "## Gate Results (R1–R7)",
        "",
    ])
    for gate_id in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
        g = gates.get(gate_id, {})
        lines.append(f"- **{gate_id}**: {g.get('status', 'MISSING')}")

    lines.extend([
        "",
        "## Decision",
        "",
        f"**ROUTE_ESTABLISHED = {decision['route_established']}**",
        "",
    ])

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
