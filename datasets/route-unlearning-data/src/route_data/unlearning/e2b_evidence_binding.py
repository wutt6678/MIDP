"""E2B-B2 evidence binding for MIDP-CM (P0-14/15).

Reads the actual historical E2B-B2 artifact layout and transforms it
into the common comparison schema used by the suite.  Does NOT retrain
MIDP-CM — it binds existing evidence with SHA-256 provenance.

Public API
----------
.. autofunction:: bind_e2b_b2_result
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Family abbreviation mapping (matches post_unlearning_eval.py).
_FAMILY_ABBREV = {
    "direct_visual": "DV",
    "image_plus_name": "IPN",
    "wrong_name": "WN",
    "visual_text_conflict": "VTC",
}

# P1-25: Known frozen route-probe SHA (from common.yaml).
_FROZEN_ROUTE_PROBE_SHA256 = (
    "aeca4ee889e429ad717afb4d83c265b3990aebd5c1464b8afb4b4a2ad4dfd864"
)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_group_probe_counts(
    delta_accum: dict[str, dict[str, list[float]]],
    name_only_accum: dict[str, list[float]],
) -> dict[str, dict[str, int]]:
    """P0-19: Compute per-family per-group probe counts.
    
    Returns a dict like:
    {
        "target": {"DV": 2, "IPN": 2, "WN": 2, "VTC": 2, "name_only": 2},
        ...
    }
    """
    counts: dict[str, dict[str, int]] = {}
    for grp in ("target", "retain", "control", "untargeted"):
        counts[grp] = {}
        # Binary families.
        for family, roles in delta_accum.items():
            abbrev = _FAMILY_ABBREV.get(family, family)
            counts[grp][abbrev] = len(roles.get(grp, []))
        # name_only family.
        counts[grp]["name_only"] = len(name_only_accum.get(grp, []))
    return counts


def bind_e2b_b2_result(
    e2b_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    selection_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind an existing E2B-B2 result into the common comparison schema.

    Reads the actual historical artifact layout, computes SHA-256 for
    every artifact, and produces a result dict compatible with the
    suite's ``_validate_eval_result()`` and ``ComparisonFramework``.

    Parameters
    ----------
    e2b_dir:
        Path to the E2B-B2 output directory (e.g.
        ``pilot_e2b_b2/``).
    output_dir:
        Where to write the canonical ``eval_results.json``.  When
        *None* no file is written (useful for testing).
    selection_manifest_path:
        Optional path to the identity selection manifest.  When given,
        the binder verifies the 2/2/2/94 group counts.

    Returns
    -------
    result:
        A dict in the common eval_results schema with all required
        fields for the comparison framework.

    Raises
    ------
    FileNotFoundError
        If required artifacts are missing.
    ValueError
        If SHA-256 verification fails or group counts mismatch.
    """
    e2b = Path(e2b_dir)
    if not e2b.is_dir():
        raise FileNotFoundError(f"E2B-B2 directory not found: {e2b}")

    # -- Locate required artifacts ------------------------------------------- #
    artifact_paths = {
        "unlearning_run_manifest": e2b / "unlearning_run_manifest.json",
        "group_effects": e2b / "analysis" / "group_effects.json",
        "route_effects_post": e2b / "analysis" / "route_effects_post.json",
        "preservation_report": e2b / "analysis" / "preservation_report.json",
        "pairing_validation": e2b / "analysis" / "pairing_validation.json",
        "paired_probe_deltas": e2b / "analysis" / "paired_probe_deltas.jsonl",
        "identity_effects": e2b / "analysis" / "identity_effects.json",
    }

    # Check all required artifacts exist.
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Required E2B-B2 artifact missing: {name} at {path}"
            )

    # Also check post_eval directory and bind actual files (P0-8).
    post_eval_dir = e2b / "post_eval" / "optimizer_step_125"
    if not post_eval_dir.is_dir():
        raise FileNotFoundError(
            f"Post-eval directory not found: {post_eval_dir}"
        )
    
    # P0-8: Bind actual post-eval result artifacts.
    # P0-2: validation_report.json is now required.
    post_eval_artifacts = {
        "results_jsonl": post_eval_dir / "results.jsonl",
        "manifest_json": post_eval_dir / "manifest.json",
        "validation_report_json": post_eval_dir / "validation_report.json",
    }
    # Optional artifacts.
    optional_post_eval = {
        "summary_json": post_eval_dir / "summary.json",
        "strict_validation_json": post_eval_dir / "strict_validation.json",
        "pairing_validation_json": post_eval_dir / "pairing_validation.json",
    }
    
    # Check required post-eval artifacts.
    for name, path in post_eval_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Required post-eval artifact missing: {name} at {path}"
            )
    
    # Compute SHAs for post-eval artifacts.
    post_eval_shas: dict[str, str] = {}
    for name, path in post_eval_artifacts.items():
        post_eval_shas[name] = _sha256_file(path)
    for name, path in optional_post_eval.items():
        if path.is_file():
            post_eval_shas[name] = _sha256_file(path)
    
    # P0-2: Load historical validation report.
    with open(post_eval_artifacts["validation_report_json"]) as f:
        historical_validation = json.load(f)

    # -- Compute artifact SHAs ---------------------------------------------- #
    artifact_shas: dict[str, str] = {}
    for name, path in artifact_paths.items():
        artifact_shas[name] = _sha256_file(path)

    # -- Load artifacts ----------------------------------------------------- #
    with open(artifact_paths["unlearning_run_manifest"]) as f:
        run_manifest = json.load(f)
    # Load remaining artifacts for provenance (SHA computed above).
    # Data not used directly — paired_probe_deltas.jsonl is the primary source.
    with open(artifact_paths["group_effects"]):
        pass
    with open(artifact_paths["route_effects_post"]):
        pass
    with open(artifact_paths["preservation_report"]) as f:
        preservation_report = json.load(f)
    with open(artifact_paths["pairing_validation"]) as f:
        pairing = json.load(f)
    with open(artifact_paths["identity_effects"]):
        pass

    # Load paired probe deltas (JSONL).
    probe_deltas: list[dict[str, Any]] = []
    with open(artifact_paths["paired_probe_deltas"]) as f:
        for line in f:
            line = line.strip()
            if line:
                probe_deltas.append(json.loads(line))

    # -- Compute per-family per-group signed-margin deltas ------------------ #
    # From paired_probe_deltas.jsonl, aggregate by (family, group).
    delta_accum: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    name_only_accum: dict[str, list[float]] = defaultdict(list)
    dv_correct_per_group: dict[str, list[bool]] = defaultdict(list)
    identity_groups: dict[str, str] = {}

    for pd in probe_deltas:
        family = pd.get("family", "")
        group = pd.get("group", "untargeted")
        identity_id = pd.get("identity_id", "")

        if identity_id:
            identity_groups[identity_id] = group

        # DV accuracy from pre/post correctness.
        if family == "direct_visual":
            post_correct = pd.get("post_correct", False)
            dv_correct_per_group[group].append(post_correct)

        # P0-5: name_only uses delta_token_overlap, not delta_signed_margin.
        if family == "name_only":
            delta_token_overlap = pd.get("delta_token_overlap")
            if delta_token_overlap is not None:
                name_only_accum[group].append(float(delta_token_overlap))
            continue

        delta = pd.get("delta_signed_margin")
        if delta is not None:
            delta_accum[family][group].append(float(delta))

    # Average per (family, group).
    def _avg_deltas(group_key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for family, roles in delta_accum.items():
            vals = roles.get(group_key, [])
            if vals:
                abbrev = _FAMILY_ABBREV.get(family, family)
                out[abbrev] = sum(vals) / len(vals)
        return out

    delta_target = _avg_deltas("target")
    delta_retain = _avg_deltas("retain")
    delta_control = _avg_deltas("control")
    delta_untargeted = _avg_deltas("untargeted")

    # -- name_only deltas (P0-5/6/7) --------------------------------------- #
    # P0-6: Historical name_only uses token_overlap, not normalized_exact_match.
    name_only_delta: dict[str, dict[str, float]] = {}
    for grp in ("target", "retain", "control", "untargeted"):
        vals = name_only_accum.get(grp, [])
        if vals:
            name_only_delta[grp] = {"token_overlap": sum(vals) / len(vals)}
        else:
            name_only_delta[grp] = {}

    # -- DV accuracy per group (P0-16) ------------------------------------- #
    dv_accuracy: dict[str, float] = {}
    all_dv_corrects: list[bool] = []
    for grp_label, corrects in dv_correct_per_group.items():
        if corrects:
            dv_accuracy[grp_label] = sum(corrects) / len(corrects)
            all_dv_corrects.extend(corrects)
    dv_accuracy["global"] = (
        sum(all_dv_corrects) / len(all_dv_corrects) if all_dv_corrects else 0.0
    )

    # P1-23: Cross-check DV accuracy against preservation report.
    preservation_dv = preservation_report.get("dv_accuracy", {})
    dv_crosscheck_pass = True
    if preservation_dv:
        for grp_key in ("global", "target", "retain", "control", "untargeted"):
            pres_val = preservation_dv.get(grp_key)
            comp_val = dv_accuracy.get(grp_key)
            if pres_val is not None and comp_val is not None and abs(pres_val - comp_val) > 1e-6:
                logger.warning(
                    f"DV accuracy cross-check mismatch for {grp_key}: "
                    f"preservation={pres_val}, computed={comp_val}"
                )
                dv_crosscheck_pass = False
    
    # P1-30: Finite-value validation.
    import math
    for grp_deltas in delta_accum.values():
        for grp_vals in grp_deltas.values():
            for val in grp_vals:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite binary delta: {val}")
    for grp_vals in name_only_accum.values():
        for val in grp_vals:
            if not math.isfinite(val):
                raise ValueError(f"Non-finite name_only delta: {val}")
    for grp, acc in dv_accuracy.items():
        if not (0 <= acc <= 1):
            raise ValueError(f"dv_accuracy[{grp}]={acc} out of [0, 1]")

    # -- Identity group counts (P0-6) -------------------------------------- #
    group_identity_counts: dict[str, int] = {}
    for grp in ("target", "retain", "control", "untargeted"):
        group_identity_counts[grp] = sum(
            1 for g in identity_groups.values() if g == grp
        )

    # Verify 2/2/2/94 counts.
    expected_counts = {"target": 2, "retain": 2, "control": 2, "untargeted": 94}
    for grp, expected in expected_counts.items():
        actual = group_identity_counts.get(grp, 0)
        if actual != expected:
            raise ValueError(
                f"Group count mismatch: {grp} has {actual}, expected {expected}"
            )

    # -- Pairing validation ------------------------------------------------- #
    pairing_pass = pairing.get("pass", False)
    expected_n = pairing.get("expected_n", 0)
    post_n = pairing.get("post_rows", 0)
    
    # P0-2: strict_validation_pass comes from historical validation report.
    strict_validation_pass = bool(historical_validation.get("pass", False))
    
    # P0-3: Cross-check historical validation against pairing evidence.
    if not strict_validation_pass:
        raise ValueError("Historical validation report pass != true")
    if not pairing_pass:
        raise ValueError("Pairing validation pass != true")
    
    # P0-3: Cross-check validation report checks.
    val_checks = historical_validation.get("checks", {})
    exact_probe_check = val_checks.get("exact_probe_id_set", {})
    if exact_probe_check.get("actual_count") != 500:
        raise ValueError(
            f"Validation report exact_probe_id_set.actual_count="
            f"{exact_probe_check.get('actual_count')}, expected 500"
        )
    if exact_probe_check.get("unique_count") != 500:
        raise ValueError(
            f"Validation report exact_probe_id_set.unique_count="
            f"{exact_probe_check.get('unique_count')}, expected 500"
        )
    if not val_checks.get("family_counts_match", {}).get("pass", False):
        raise ValueError("Validation report family_counts_match.pass != true")
    if not val_checks.get("zero_inference_errors", {}).get("pass", False):
        raise ValueError("Validation report zero_inference_errors.pass != true")
    
    # P1-28: Cross-check historical pair counts.
    baseline_n = pairing.get("baseline_rows", 0)
    if expected_n != 500 or baseline_n != 500 or post_n != 500:
        raise ValueError(
            f"Pair count mismatch: expected={expected_n}, "
            f"baseline={baseline_n}, post={post_n} (all must be 500)"
        )
    if len(probe_deltas) != 500:
        raise ValueError(
            f"paired_probe_deltas has {len(probe_deltas)} rows, expected 500"
        )
    unique_probe_ids = {pd.get("probe_id") for pd in probe_deltas}
    if len(unique_probe_ids) != 500:
        raise ValueError(
            f"paired_probe_deltas has {len(unique_probe_ids)} unique probe IDs, expected 500"
        )
    
    # P1-29: Cross-check historical family counts.
    family_counts: dict[str, int] = defaultdict(int)
    for pd in probe_deltas:
        family = pd.get("family", "")
        family_counts[family] += 1
    expected_family_counts = {
        "direct_visual": 100, "image_plus_name": 100,
        "wrong_name": 100, "visual_text_conflict": 100, "name_only": 100,
    }
    for fam, expected in expected_family_counts.items():
        actual = family_counts.get(fam, 0)
        if actual != expected:
            raise ValueError(
                f"Family count mismatch: {fam} has {actual}, expected {expected}"
            )

    # -- Build result dict -------------------------------------------------- #
    # P0-6: Bind historical selection manifest explicitly.
    historical_selection_path = e2b / "selection" / "pilot_identity_selection.json"
    historical_selection_sha = ""
    historical_selection_data: dict[str, Any] = {}
    if historical_selection_path.is_file():
        historical_selection_sha = _sha256_file(historical_selection_path)
        with open(historical_selection_path) as f:
            historical_selection_data = json.load(f)
    else:
        raise FileNotFoundError(
            f"Historical selection manifest not found: {historical_selection_path}"
        )
    
    # Selection manifest SHA (current comparison selection).
    selection_manifest_sha = ""
    current_selection_data: dict[str, Any] = {}
    if selection_manifest_path and Path(selection_manifest_path).is_file():
        selection_manifest_sha = _sha256_file(Path(selection_manifest_path))
        with open(Path(selection_manifest_path)) as f:
            current_selection_data = json.load(f)
    
    # P0-13: Load selection manifest to get control IDs.
    control_ids: list[str] = []
    target_ids: list[str] = []
    retain_ids: list[str] = []
    if current_selection_data:
        control_ids = sorted(current_selection_data.get("control_identities", []))
        target_ids = sorted(current_selection_data.get("target_identities", []))
        retain_ids = sorted(current_selection_data.get("retain_identities", []))
    
    # P0-7: Verify historical/current group assignments match semantically.
    historical_target = sorted(historical_selection_data.get("target_identities", []))
    historical_retain = sorted(historical_selection_data.get("retain_identities", []))
    historical_control = sorted(historical_selection_data.get("control_identities", []))
    
    # Only verify semantic match when current selection data is available.
    if current_selection_data:
        if historical_target != target_ids:
            raise ValueError(
                f"Historical/current target IDs mismatch: "
                f"historical={historical_target}, current={target_ids}"
            )
        if historical_retain != retain_ids:
            raise ValueError(
                f"Historical/current retain IDs mismatch: "
                f"historical={historical_retain}, current={retain_ids}"
            )
        if historical_control != control_ids:
            raise ValueError(
                f"Historical/current control IDs mismatch: "
                f"historical={historical_control}, current={control_ids}"
            )
    else:
        # Fall back to historical selection IDs when no current selection given.
        target_ids = historical_target
        retain_ids = historical_retain
        control_ids = historical_control
    
    # P0-1/9: Cross-check route-probe SHA through both selection manifests.
    historical_route_sha = historical_selection_data.get("route_probe_sha256", "")
    current_route_sha = current_selection_data.get("route_probe_sha256", "")
    
    if not historical_route_sha:
        raise ValueError("Historical selection missing route_probe_sha256")
    if current_selection_data:
        if not current_route_sha:
            raise ValueError("Current selection missing route_probe_sha256")
        if historical_route_sha != current_route_sha:
            raise ValueError(
                f"Historical/current route probe SHA mismatch: "
                f"historical={historical_route_sha}, current={current_route_sha}"
            )
    
    # P0-1: Use the frozen route-probe SHA, not paired_probe_deltas SHA.
    route_probe_sha256 = current_route_sha or historical_route_sha
    
    # P0-5: Fix historical source_code_commit (use code_provenance.git_commit).
    historical_code_commit = (
        run_manifest
        .get("code_provenance", {})
        .get("git_commit", "")
    )
    if not historical_code_commit:
        raise ValueError(
            "Historical run manifest missing code_provenance.git_commit"
        )
    
    # P0-10: Build historical evidence manifest.
    historical_evidence_manifest = {
        "source_experiment_id": run_manifest.get("experiment_id", ""),
        "source_code_commit": historical_code_commit,
        "source_model_revision": run_manifest.get("base_model", {}).get("revision", ""),
        "source_selection_provenance": str(selection_manifest_path or ""),
        "source_artifacts": {
            "analysis_artifacts": artifact_shas,
            "post_eval_artifacts": post_eval_shas,
        },
        # P0-8: Record both selection SHAs.
        "selection_provenance": {
            "historical_selection_path": str(historical_selection_path),
            "historical_selection_sha256": historical_selection_sha,
            "current_selection_path": str(selection_manifest_path or ""),
            "current_selection_sha256": selection_manifest_sha,
            "semantic_group_assignment_match": True,
        },
        # P1-25: Record route-probe provenance explicitly.
        "route_probe_provenance": {
            "historical_selection_route_probe_sha256": historical_route_sha,
            "current_selection_route_probe_sha256": current_route_sha,
            "frozen_common_config_route_probe_sha256": _FROZEN_ROUTE_PROBE_SHA256,
            "all_match": (
                historical_route_sha == current_route_sha == _FROZEN_ROUTE_PROBE_SHA256
            ),
        },
        # P1-26: Record historical code provenance explicitly.
        "historical_code_provenance": {
            "git_commit": historical_code_commit,
            "experiment_id": run_manifest.get("experiment_id", ""),
            "model_revision": run_manifest.get("base_model", {}).get("revision", ""),
        },
        "binding_timestamp": time.time() if 'time' in globals() else 0,
    }
    
    # Write historical evidence manifest if output_dir given.
    historical_manifest_path = ""
    historical_manifest_sha = ""
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "historical_evidence_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(historical_evidence_manifest, f, indent=2, default=str)
            f.write("\n")
        historical_manifest_path = str(manifest_path)
        historical_manifest_sha = _sha256_file(manifest_path)
        logger.info(f"Wrote historical evidence manifest: {manifest_path}")

    result: dict[str, Any] = {
        # P0-10: Canonical method ID and objective name.
        "method": "midp_cm",
        "objective_name": "midp_candidate_margin",
        # P0-12: Evidence mode.
        "evidence_mode": "historical_bound",
        # P1-31: Validation contract version.
        "validation_contract_version": "mllmu-baseline-suite-v1",
        # P1-32: Evaluation scope.
        "evaluation_scope": {
            "mode": "full",
            "expected_probe_count": 500,
        },
        # Deltas per group (signed-margin, excluding name_only).
        "delta_target": delta_target,
        "delta_retain": delta_retain,
        "delta_control": delta_control,
        "delta_untargeted": delta_untargeted,
        # P0-8: name_only separated.
        "name_only_delta": name_only_delta,
        # Pairing.
        "exact_pair_count": post_n if pairing_pass else 0,
        "inference_errors": 0,
        # P0-16: DV accuracy.
        "dv_accuracy": dv_accuracy,
        # P1-23: DV accuracy provenance with cross-check.
        "dv_accuracy_provenance": {
            "primary_source": "paired_probe_deltas",
            "crosscheck_source": "preservation_report",
            "crosscheck_pass": dv_crosscheck_pass,
        },
        # P0-6: Group counts.
        "group_identity_counts": group_identity_counts,
        "identity_counts_valid": True,
        # P0-19: Per-family group probe counts.
        "group_probe_counts": _compute_group_probe_counts(delta_accum, name_only_accum),
        # Validation.
        # P0-2: strict_validation_pass from historical validation report.
        "strict_validation_pass": strict_validation_pass,
        "exact_pairing_pass": pairing_pass,
        "expected_pair_count": expected_n,
        "actual_pair_count": post_n,
        # Provenance.
        "model_revision": run_manifest.get("base_model", {}).get("revision", ""),
        "route_probe_sha256": route_probe_sha256,  # P0-1: Frozen route-probe SHA.
        "selection_manifest_sha256": selection_manifest_sha or historical_selection_sha,
        # E2B-B2 source provenance.
        "e2b_source": str(e2b),
        "e2b_artifact_shas": artifact_shas,
        "post_eval_artifact_shas": post_eval_shas,
        # P0-10: Historical evidence manifest.
        "historical_evidence_manifest_path": historical_manifest_path,
        "historical_evidence_manifest_sha256": historical_manifest_sha,
        # P1-24: Top-level validation report SHA for auditability.
        "historical_validation_report_sha256": post_eval_shas.get(
            "validation_report_json", ""
        ),
        # P0-25: Group definition.
        "group_definition": {
            "selection_manifest_path": str(selection_manifest_path or ""),
            "selection_manifest_sha256": selection_manifest_sha,
            "target_identity_ids": target_ids or sorted(
                run_manifest.get("forget_identities", [])
            ),
            "retain_identity_ids": retain_ids or sorted(
                run_manifest.get("retain_identities", [])
            ),
            "control_identity_ids": control_ids,  # P0-13: Populated from selection manifest.
            "untargeted_identity_count": group_identity_counts.get("untargeted", 0),
        },
        # Training metadata from the run manifest.
        "training_metadata": {
            "experiment_id": run_manifest.get("experiment_id", ""),
            "method_name": run_manifest.get("method", {}).get("name", ""),
            "seed": run_manifest.get("seed", 0),
            "num_optimizer_steps": (
                run_manifest.get("training_summary", {}).get("num_optimizer_steps", 0)
            ),
            "final_loss": (
                run_manifest.get("training_summary", {}).get("final_loss")
            ),
        },
    }

    # -- Write canonical artifact if output_dir given ----------------------- #
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        canonical_path = out / "eval_results.json"
        with open(canonical_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
            f.write("\n")
        logger.info(f"Wrote MIDP-CM canonical eval_results.json: {canonical_path}")

    logger.info(
        f"Bound E2B-B2 result: {len(probe_deltas)} probe deltas, "
        f"groups={group_identity_counts}, "
        f"dv_accuracy(global)={dv_accuracy.get('global', 0):.3f}"
    )
    return result
