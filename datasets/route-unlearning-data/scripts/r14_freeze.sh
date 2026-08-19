#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# R14 — Final Comparison and Research Freeze
#
# Converts R13 outputs into immutable comparison tables, Case A/B/C
# decision, and final freeze manifest.
#
# NO training occurs in R14.
#
# Prerequisites:
#   - R13 full evidence complete for all comparison methods
#   - All eval_results.json pass strict validation
#
# Usage:
#   conda activate midp-qwen35
#   cd /scratch/wutiantong/MIDP
#   source datasets/route-unlearning-data/scripts/r12r14_env.sh
#   bash datasets/route-unlearning-data/scripts/r14_freeze.sh
# --------------------------------------------------------------------------- #
set -euo pipefail

# -- Load shared environment ----------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/r12r14_env.sh"

# -- Prerequisites --------------------------------------------------------- #
verify_code_sha || exit 1

FULL_ROOT="${RUN_ROOT}/r13_full"
FREEZE_ROOT="${RUN_ROOT}/r14_freeze"
mkdir -p "${FREEZE_ROOT}"

echo ""
echo "================================================================"
echo "  R14 FREEZE — starting"
echo "  R13 input:  ${FULL_ROOT}"
echo "  freeze out: ${FREEZE_ROOT}"
echo "================================================================"
echo ""

# ----------------------------------------------------------------------- #
# Step 1: Validate all R13 artifacts (Section 30)
# ----------------------------------------------------------------------- #
echo "=== Step 1: R14 input validation ==="

python3 - "${FULL_ROOT}" "${FREEZE_ROOT}" "${CODE_SHA}" "${MODEL_REVISION}" \
    "${PROCESSED_DATASET_SHA}" "${ROUTE_PROBE_SHA}" "${SELECTION_MANIFEST_SHA}" <<'PYEOF'
import json, sys, hashlib, os
from pathlib import Path

full_root = Path(sys.argv[1])
freeze_root = Path(sys.argv[2])
code_sha = sys.argv[3]
model_revision = sys.argv[4]
processed_sha = sys.argv[5]
route_probe_sha = sys.argv[6]
selection_sha = sys.argv[7]

COMPARISON_METHODS = [
    "prompting", "ga", "gd", "kl", "npo", "midp_cm",
    "mmunlearner", "manu", "r2mu_adapted",
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

validation_results = {}
errors = []

for method in COMPARISON_METHODS:
    eval_dir = full_root / method / "eval"
    eval_file = eval_dir / "eval_results.json"

    if not eval_file.is_file():
        errors.append(f"{method}: eval_results.json not found at {eval_file}")
        validation_results[method] = {"valid": False, "error": "missing eval_results.json"}
        continue

    with open(eval_file) as f:
        ev = json.load(f)

    checks = {}
    # Core contract
    checks["pair_count_500"] = ev.get("exact_pair_count", 0) == 500
    checks["inference_errors_0"] = ev.get("inference_errors", 1) == 0
    checks["strict_validation"] = ev.get("strict_validation_pass", False) is True
    checks["exact_pairing"] = ev.get("exact_pairing_pass", False) is True
    checks["model_revision"] = ev.get("model_revision", "") == model_revision
    checks["route_probe_sha"] = ev.get("route_probe_sha256", "") == route_probe_sha
    checks["selection_manifest_sha"] = ev.get("selection_manifest_sha256", "") == selection_sha

    # Identity counts
    groups = ev.get("group_probe_counts", {})
    for grp in ["target", "retain", "control"]:
        grp_counts = groups.get(grp, {})
        for fam in ["DV", "IPN", "WN", "VTC", "name_only"]:
            expected = 2
            actual = grp_counts.get(fam, -1)
            checks[f"{grp}_{fam}_count"] = actual == expected
    untargeted = groups.get("untargeted", {})
    for fam in ["DV", "IPN", "WN", "VTC", "name_only"]:
        actual = untargeted.get(fam, -1)
        checks[f"untargeted_{fam}_count"] = actual == 94

    # Method-specific (informational — not blocking if field missing)
    if method == "manu":
        val = ev.get("all_restores_verified")
        if val is not None:
            checks["all_restores_verified"] = val is True
    if method == "r2mu_adapted":
        val = ev.get("answer_tokens_excluded")
        if val is not None:
            checks["answer_tokens_excluded"] = val is True

    all_pass = all(checks.values())
    validation_results[method] = {"valid": all_pass, "checks": checks}
    if not all_pass:
        failed = [k for k, v in checks.items() if not v]
        errors.append(f"{method}: FAILED checks: {failed}")

# Report
print(f"\nValidated {len(COMPARISON_METHODS)} methods:")
for m, r in validation_results.items():
    status = "PASS" if r["valid"] else "FAIL"
    print(f"  {m:20s} {status}")

if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
    print("\nR14 CANNOT PROCEED — fix errors above.")
    sys.exit(1)

# Write validation report
report = {
    "stage": "R14",
    "code_sha": code_sha,
    "model_revision": model_revision,
    "methods_validated": len(COMPARISON_METHODS),
    "all_valid": True,
    "per_method": validation_results,
}
report_path = freeze_root / "r14_validation_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
print(f"\nValidation report: {report_path}")
print("All R13 artifacts PASSED input validation.")
PYEOF

echo "  R14 input validation PASSED"

# ----------------------------------------------------------------------- #
# Step 2: Generate comparison tables (Section 31)
# ----------------------------------------------------------------------- #
echo ""
echo "=== Step 2: Comparison tables ==="

python3 - "${FULL_ROOT}" "${FREEZE_ROOT}" <<'PYEOF'
import json, sys
from pathlib import Path

full_root = Path(sys.argv[1])
freeze_root = Path(sys.argv[2])

COMPARISON_METHODS = [
    "prompting", "ga", "gd", "kl", "npo", "midp_cm",
    "mmunlearner", "manu", "r2mu_adapted",
]
FAMILIES = ["DV", "IPN", "WN", "VTC"]
GROUPS = ["target", "retain", "control"]

tables = {
    "table_A_route_effects": {},
    "table_B_preservation": {},
    "table_C_selectivity": {},
    "table_D_identity_text": {},
    "table_E_efficiency": {},
}

for method in COMPARISON_METHODS:
    eval_file = full_root / method / "eval" / "eval_results.json"
    if not eval_file.is_file():
        continue
    with open(eval_file) as f:
        ev = json.load(f)

    metrics = ev.get("eval_metrics", {})

    # Table A: signed delta-M per family per group
    row_a = {}
    for grp in GROUPS:
        for fam in FAMILIES:
            key = f"{grp}_delta_{fam}"
            grp_fam = metrics.get(f"{grp}_group", {}).get(fam, {})
            row_a[key] = grp_fam.get("signed_answer_margin_delta", None)
    tables["table_A_route_effects"][method] = row_a

    # Table B: DV preservation
    dv = metrics.get("dv_preservation", {})
    tables["table_B_preservation"][method] = {
        "dv_accuracy_global": dv.get("global_accuracy"),
        "dv_accuracy_target": dv.get("target_accuracy"),
        "dv_accuracy_retain": dv.get("retain_accuracy"),
        "dv_accuracy_control": dv.get("control_accuracy"),
        "dv_accuracy_untargeted": dv.get("untargeted_accuracy"),
        "preservation_gate": dv.get("gate_pass"),
    }

    # Table C: selectivity
    sel = metrics.get("selectivity", {})
    tables["table_C_selectivity"][method] = {
        "mean_target_delta_m": sel.get("mean_target_delta"),
        "target_forgetting": sel.get("target_forgetting"),
        "retain_drift": sel.get("retain_drift"),
        "control_drift": sel.get("control_drift"),
        "selectivity_score": sel.get("selectivity_score"),
        "dv_preservation_pass": dv.get("gate_pass"),
    }

    # Table D: name-only + route contrasts
    no = metrics.get("name_only", {})
    tables["table_D_identity_text"][method] = {
        "name_only_target": no.get("target_change"),
        "name_only_retain": no.get("retain_change"),
    }

    # Table E: efficiency
    eff = ev.get("efficiency", {})
    tables["table_E_efficiency"][method] = {
        "trainable_parameters": eff.get("trainable_params"),
        "optimizer_steps": eff.get("optimizer_steps"),
        "wall_clock_seconds": eff.get("wall_clock_seconds"),
        "peak_gpu_memory_gb": eff.get("peak_gpu_memory_gb"),
        "adapter_size_bytes": eff.get("adapter_size_bytes"),
    }

# Write tables
tables_path = freeze_root / "comparison_tables.json"
with open(tables_path, "w") as f:
    json.dump(tables, f, indent=2)
    f.write("\n")
print(f"Comparison tables: {tables_path}")
print(f"  {len(tables)} tables × {len(COMPARISON_METHODS)} methods")
PYEOF

# ----------------------------------------------------------------------- #
# Step 3: Suite summary + efficiency report
# ----------------------------------------------------------------------- #
echo ""
echo "=== Step 3: Suite summary ==="

python3 - "${FULL_ROOT}" "${FREEZE_ROOT}" "${CODE_SHA}" <<'PYEOF'
import json, sys
from pathlib import Path

full_root = Path(sys.argv[1])
freeze_root = Path(sys.argv[2])
code_sha = sys.argv[3]

COMPARISON_METHODS = [
    "prompting", "ga", "gd", "kl", "npo", "midp_cm",
    "mmunlearner", "manu", "r2mu_adapted",
]

method_results = {}
for method in COMPARISON_METHODS:
    eval_file = full_root / method / "eval" / "eval_results.json"
    if eval_file.is_file():
        with open(eval_file) as f:
            method_results[method] = json.load(f)

valid_eval = {m for m, r in method_results.items()
              if r.get("strict_validation_pass") and r.get("exact_pair_count") == 500}
required = set(COMPARISON_METHODS)
missing = sorted(required - valid_eval)

summary = {
    "stage": "R13/R14",
    "code_sha": code_sha,
    "execution_scope": "full" if not missing else "partial",
    "missing_comparison_methods": missing,
    "eval_complete": len(missing) == 0,
    "research_suite_complete": len(missing) == 0,
    "methods": {m: {"status": "PASS" if m in valid_eval else "FAIL"}
                for m in COMPARISON_METHODS},
}

summary_path = freeze_root / "suite_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
print(f"Suite summary: {summary_path}")
print(f"  eval_complete: {summary['eval_complete']}")
print(f"  missing: {missing}")
PYEOF

# ----------------------------------------------------------------------- #
# Step 4: Case A/B/C decision
# ----------------------------------------------------------------------- #
echo ""
echo "=== Step 4: E2C decision template ==="

python3 - "${FREEZE_ROOT}" "${CODE_SHA}" <<'PYEOF'
import json, sys
from pathlib import Path

freeze_root = Path(sys.argv[1])
code_sha = sys.argv[2]

summary_path = freeze_root / "suite_summary.json"
with open(summary_path) as f:
    summary = json.load(f)

if not summary["eval_complete"]:
    decision = {
        "decision_status": "INCOMPLETE",
        "decision_case": None,
        "reason": f"Missing methods: {summary['missing_comparison_methods']}",
    }
else:
    # Decision logic requires human inspection of comparison_tables.json
    # This template records the structure; the actual Case A/B/C determination
    # depends on quantitative results.
    decision = {
        "decision_status": "PENDING_REVIEW",
        "decision_case": "TO_BE_DETERMINED",
        "reason": (
            "All methods present. Case A/B/C requires quantitative "
            "inspection of comparison_tables.json and route_selectivity "
            "analysis. See checklist Sections 33-34."
        ),
        "case_criteria": {
            "Case_A": "selective route forgetting + DV preservation",
            "Case_B": "stable execution, no selective route separation",
            "Case_C": "broad degradation or preservation failure",
        },
    }

decision_path = freeze_root / "e2c_decision.json"
with open(decision_path, "w") as f:
    json.dump(decision, f, indent=2)
    f.write("\n")
print(f"E2C decision: {decision_path}")
print(f"  status: {decision['decision_status']}")
PYEOF

# ----------------------------------------------------------------------- #
# Step 5: Final freeze manifest (Section 37)
# ----------------------------------------------------------------------- #
echo ""
echo "=== Step 5: Final freeze manifest ==="

python3 - "${FREEZE_ROOT}" "${FREEZE_ROOT}" "${CODE_SHA}" "${MODEL_REVISION}" \
    "${PROCESSED_DATASET_SHA}" "${ROUTE_PROBE_SHA}" <<'PYEOF'
import json, sys, hashlib
from pathlib import Path

freeze_root = Path(sys.argv[1])
code_sha = sys.argv[2]
model_revision = sys.argv[3]
processed_sha = sys.argv[4]
route_probe_sha = sys.argv[5]

COMPARISON_METHODS = [
    "prompting", "ga", "gd", "kl", "npo", "midp_cm",
    "mmunlearner", "manu", "r2mu_adapted",
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# Hash all final artifacts
artifact_hashes = {}
for json_file in sorted(freeze_root.glob("*.json")):
    artifact_hashes[json_file.name] = sha256_file(json_file)

# Also hash per-method eval artifacts
for method in COMPARISON_METHODS:
    eval_dir = freeze_root.parent.parent / "r13_full" / method / "eval"
    if eval_dir.is_dir():
        for json_file in sorted(eval_dir.glob("*.json")):
            key = f"{method}/eval/{json_file.name}"
            artifact_hashes[key] = sha256_file(json_file)

# Load decision
decision_path = freeze_root / "e2c_decision.json"
with open(decision_path) as f:
    decision = json.load(f)

manifest = {
    "stage": "R14",
    "status": "FROZEN",
    "code_sha": code_sha,
    "model_id": "Qwen/Qwen3.5-9B",
    "model_revision": model_revision,
    "seed": 17,
    "processed_dataset_sha256": processed_sha,
    "route_probe_sha256": route_probe_sha,
    "comparison_methods": COMPARISON_METHODS,
    "artifact_hashes": artifact_hashes,
    "decision_status": decision["decision_status"],
    "decision_case": decision.get("decision_case"),
    "ready_for_e2c": decision["decision_status"] != "INCOMPLETE",
}

manifest_path = freeze_root / "final_freeze_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

# Re-hash the manifest itself (after writing)
manifest_hashes = sha256_file(manifest_path)
manifest["self_sha256"] = manifest_hashes
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print(f"Freeze manifest: {manifest_path}")
print(f"  artifacts hashed: {len(artifact_hashes)}")
print(f"  decision: {decision['decision_status']}")
PYEOF

# ----------------------------------------------------------------------- #
# Done
# ----------------------------------------------------------------------- #
echo ""
echo "================================================================"
echo "  R14 FREEZE — complete"
echo "================================================================"
echo ""
echo "Artifacts in: ${FREEZE_ROOT}/"
echo "  suite_summary.json"
echo "  comparison_tables.json"
echo "  r14_validation_report.json"
echo "  e2c_decision.json"
echo "  final_freeze_manifest.json"
echo ""
echo "If Case A: replicate with >= 3 seeds before strong claims."
echo "If Case B: freeze results → move to E2C."
echo "If Case C: document failure mode → move to E2C."
echo "If INCOMPLETE: repair missing evidence before freezing."
