"""Paired analysis for the Stage 3 unlearning pilot.

Joins pre-unlearning (baseline) and post-unlearning evaluation results
by ``probe_id`` to compute per-probe deltas, identity-level aggregates,
group-level aggregates, a preservation report, and a route-effects report.

Public API
----------
.. autoclass:: PairedAnalysisConfig
.. autoclass:: PairedAnalysis
.. autofunction:: load_results_jsonl
.. autofunction:: compute_probe_deltas
.. autofunction:: compute_identity_effects
.. autofunction:: compute_group_effects
.. autofunction:: compute_preservation_report
.. autofunction:: compute_route_effects
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class PairedAnalysisConfig:
    """Paths and metadata required for paired analysis."""

    baseline_results_path: str = ""
    post_results_path: str = ""
    selection_manifest_path: str = ""
    output_dir: str = ""

    # Provenance
    code_commit: str = ""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_results_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL results file (baseline or post-eval) into a list of dicts."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_by_probe_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a probe_id → row mapping."""
    return {r["probe_id"]: r for r in rows}


# --------------------------------------------------------------------------- #
# Probe-level deltas
# --------------------------------------------------------------------------- #

def compute_probe_deltas(
    baseline_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
    identity_groups: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-probe deltas between baseline and post-eval results.

    Parameters
    ----------
    baseline_rows, post_rows:
        Lists of result dicts (from ``load_results_jsonl``).
    identity_groups:
        Optional mapping ``identity_id → group`` (target/retain/control).
        Identities not in the mapping are labelled ``"untargeted"``.

    Returns
    -------
    list[dict]
        One dict per probe with all fields specified in Section 41.
    """
    bl_idx = _index_by_probe_id(baseline_rows)
    po_idx = _index_by_probe_id(post_rows)

    if identity_groups is None:
        identity_groups = {}

    deltas: list[dict[str, Any]] = []
    for probe_id in sorted(bl_idx.keys()):
        if probe_id not in po_idx:
            continue
        bl = bl_idx[probe_id]
        po = po_idx[probe_id]

        identity_id = bl.get("identity_id", "")
        group = identity_groups.get(identity_id, "untargeted")
        family = bl.get("probe_family", "")

        row: dict[str, Any] = {
            "probe_id": probe_id,
            "identity_id": identity_id,
            "group": group,
            "family": family,
            "protocol_role": bl.get("protocol_role", ""),
            "target_attribute": bl.get("target_attribute"),
            "answer_label": bl.get("answer_label"),
        }

        # -- signed margin deltas (binary families) --
        if family != "name_only":
            pre_m = bl.get("signed_answer_margin")
            post_m = po.get("signed_answer_margin")
            row["pre_signed_margin"] = pre_m
            row["post_signed_margin"] = post_m
            if pre_m is not None and post_m is not None:
                row["delta_signed_margin"] = post_m - pre_m
            else:
                row["delta_signed_margin"] = None

            # predictions
            pre_pred = bl.get("predicted_label")
            post_pred = po.get("predicted_label")
            row["pre_prediction"] = pre_pred
            row["post_prediction"] = post_pred
            row["prediction_changed"] = (
                pre_pred != post_pred
                if pre_pred is not None and post_pred is not None
                else None
            )

            # correctness
            row["pre_correct"] = bl.get("correct")
            row["post_correct"] = po.get("correct")
        else:
            # name_only: token_overlap
            pre_to = bl.get("token_overlap")
            post_to = po.get("token_overlap")
            row["pre_token_overlap"] = pre_to
            row["post_token_overlap"] = post_to
            if pre_to is not None and post_to is not None:
                row["delta_token_overlap"] = post_to - pre_to
            else:
                row["delta_token_overlap"] = None

            # also carry prediction/correctness for name_only
            row["pre_prediction"] = bl.get("predicted_label")
            row["post_prediction"] = po.get("predicted_label")
            row["prediction_changed"] = (
                bl.get("predicted_label") != po.get("predicted_label")
                if bl.get("predicted_label") is not None
                and po.get("predicted_label") is not None
                else None
            )
            row["pre_correct"] = bl.get("correct")
            row["post_correct"] = po.get("correct")

        deltas.append(row)

    return deltas


# --------------------------------------------------------------------------- #
# Identity-level aggregates
# --------------------------------------------------------------------------- #

def _safe_mean(values: list[float | None]) -> float | None:
    """Mean of non-None values, or None if empty."""
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def compute_identity_effects(
    probe_deltas: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate probe deltas into per-identity mean-ΔM by family.

    Returns a dict keyed by ``identity_id`` with::

        {
            "direct_visual_dM": float | None,
            "image_plus_name_dM": float | None,
            "wrong_name_dM": float | None,
            "conflict_dM": float | None,
            "overall_visual_dM": float | None,
            "probe_count": int,
        }
    """
    by_identity: dict[str, dict[str, list[float | None]]] = defaultdict(
        lambda: {
            "direct_visual": [],
            "image_plus_name": [],
            "wrong_name": [],
            "visual_text_conflict": [],
        },
    )

    for d in probe_deltas:
        iid = d["identity_id"]
        fam = d["family"]
        if fam == "name_only":
            continue
        dm = d.get("delta_signed_margin")
        if fam in by_identity[iid]:
            by_identity[iid][fam].append(dm)

    result: dict[str, dict[str, Any]] = {}
    for iid in sorted(by_identity):
        families = by_identity[iid]
        dv = _safe_mean(families["direct_visual"])
        ipn = _safe_mean(families["image_plus_name"])
        wn = _safe_mean(families["wrong_name"])
        cfl = _safe_mean(families["visual_text_conflict"])

        # overall visual = mean across all non-name_only families
        all_margins: list[float | None] = []
        for fam_margins in families.values():
            all_margins.extend(fam_margins)
        overall = _safe_mean(all_margins)

        total_probes = sum(len(v) for v in families.values())

        result[iid] = {
            "direct_visual_dM": dv,
            "image_plus_name_dM": ipn,
            "wrong_name_dM": wn,
            "conflict_dM": cfl,
            "overall_visual_dM": overall,
            "probe_count": total_probes,
        }

    return result


# --------------------------------------------------------------------------- #
# Group-level aggregates
# --------------------------------------------------------------------------- #

def _summary_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, median, std, min, max, count for a list of floats."""
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    n = len(clean)
    mn = statistics.mean(clean)
    md = statistics.median(clean)
    sd = statistics.pstdev(clean) if n > 1 else 0.0
    return {
        "mean": mn,
        "median": md,
        "std": sd,
        "min": min(clean),
        "max": max(clean),
        "count": n,
    }


def compute_group_effects(
    probe_deltas: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate probe deltas into group-level statistics.

    Groups: ``target``, ``retain``, ``control``, ``untargeted``.

    P0-9: Visual signed-margin metrics are reported separately from
    name-only token-overlap metrics.  The primary gating key is
    ``overall_visual`` which averages only ``delta_signed_margin``
    across the four visual families.  Name-only metrics are reported
    in a dedicated ``name_only`` sub-dict.
    """
    # Visual (signed-margin) deltas per group.
    group_visual_deltas: dict[str, list[float]] = defaultdict(list)
    # Name-only (token-overlap) deltas per group.
    group_nameonly_deltas: dict[str, list[float]] = defaultdict(list)
    # Per-family breakdowns.
    group_family: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )

    for d in probe_deltas:
        grp = d.get("group", "untargeted")
        fam = d["family"]

        if fam == "name_only":
            delta = d.get("delta_token_overlap")
            if delta is not None:
                group_nameonly_deltas[grp].append(delta)
                group_family[grp][fam].append(delta)
        else:
            delta = d.get("delta_signed_margin")
            if delta is not None:
                group_visual_deltas[grp].append(delta)
                group_family[grp][fam].append(delta)

    result: dict[str, dict[str, Any]] = {}
    for grp in ["target", "retain", "control", "untargeted"]:
        # P0-9: overall_visual is the primary gating metric.
        overall_visual = _summary_stats(group_visual_deltas.get(grp, []))
        name_only_stats = _summary_stats(group_nameonly_deltas.get(grp, []))

        per_family: dict[str, dict[str, float]] = {}
        for fam in [
            "direct_visual",
            "image_plus_name",
            "wrong_name",
            "visual_text_conflict",
            "name_only",
        ]:
            vals = group_family.get(grp, {}).get(fam, [])
            if vals:
                per_family[fam] = _summary_stats(vals)

        result[grp] = {
            "overall_visual": overall_visual,
            "name_only": name_only_stats,
            "per_family": per_family,
        }

    return result


# --------------------------------------------------------------------------- #
# Preservation report
# --------------------------------------------------------------------------- #

def compute_preservation_report(
    baseline_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
    identity_groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute the preservation report (Section 44).

    Includes:
    - global direct_visual accuracy/margin pre/post
    - per-attribute accuracy pre/post
    - positive-state / negative-state accuracy pre/post
    - retain/control group behavior
    """
    if identity_groups is None:
        identity_groups = {}

    bl_idx = _index_by_probe_id(baseline_rows)
    po_idx = _index_by_probe_id(post_rows)

    # -- global direct_visual --
    dv_bl_acc, dv_po_acc = [], []
    dv_bl_margin, dv_po_margin = [], []

    # -- per-attribute --
    attr_bl: dict[str, list[tuple[bool | None, float | None]]] = defaultdict(list)
    attr_po: dict[str, list[tuple[bool | None, float | None]]] = defaultdict(list)

    # -- positive / negative state --
    pos_bl_acc, neg_bl_acc = [], []
    pos_po_acc, neg_po_acc = [], []

    # -- retain / control --
    retain_bl_acc, retain_po_acc = [], []
    control_bl_acc, control_po_acc = [], []

    for probe_id in sorted(bl_idx.keys()):
        if probe_id not in po_idx:
            continue
        bl = bl_idx[probe_id]
        po = po_idx[probe_id]
        fam = bl.get("probe_family", "")
        attr = bl.get("target_attribute")
        label = bl.get("answer_label")
        iid = bl.get("identity_id", "")
        grp = identity_groups.get(iid, "untargeted")

        if fam == "direct_visual":
            bl_correct = bl.get("correct")
            po_correct = po.get("correct")
            if bl_correct is not None:
                dv_bl_acc.append(bl_correct)
                dv_bl_margin.append(bl.get("signed_answer_margin"))
            if po_correct is not None:
                dv_po_acc.append(po_correct)
                dv_po_margin.append(po.get("signed_answer_margin"))

        # per-attribute (binary families only)
        if fam != "name_only" and attr is not None:
            attr_bl[attr].append((bl.get("correct"), bl.get("signed_answer_margin")))
            attr_po[attr].append((po.get("correct"), po.get("signed_answer_margin")))

        # positive / negative state
        if fam != "name_only" and label is not None:
            if label:
                if bl.get("correct") is not None:
                    pos_bl_acc.append(bl["correct"])
                if po.get("correct") is not None:
                    pos_po_acc.append(po["correct"])
            else:
                if bl.get("correct") is not None:
                    neg_bl_acc.append(bl["correct"])
                if po.get("correct") is not None:
                    neg_po_acc.append(po["correct"])

        # retain / control
        if grp == "retain" and fam != "name_only":
            if bl.get("correct") is not None:
                retain_bl_acc.append(bl["correct"])
            if po.get("correct") is not None:
                retain_po_acc.append(po["correct"])
        elif grp == "control" and fam != "name_only":
            if bl.get("correct") is not None:
                control_bl_acc.append(bl["correct"])
            if po.get("correct") is not None:
                control_po_acc.append(po["correct"])

    # -- assemble report --
    def _acc(vals: list[bool]) -> float | None:
        return statistics.mean(vals) if vals else None

    def _mean_margin(vals: list[float | None]) -> float | None:
        clean = [v for v in vals if v is not None]
        return statistics.mean(clean) if clean else None

    # per-attribute accuracy
    per_attr: dict[str, dict[str, float | None]] = {}
    for attr in sorted(attr_bl.keys()):
        bl_corrects = [c for c, _ in attr_bl[attr] if c is not None]
        po_corrects = [c for c, _ in attr_po[attr] if c is not None]
        per_attr[attr] = {
            "pre_accuracy": _acc(bl_corrects),
            "post_accuracy": _acc(po_corrects),
            "pre_count": len(bl_corrects),
            "post_count": len(po_corrects),
        }

    report: dict[str, Any] = {
        "global_direct_visual": {
            "pre_accuracy": _acc(dv_bl_acc),
            "post_accuracy": _acc(dv_po_acc),
            "pre_mean_margin": _mean_margin(dv_bl_margin),
            "post_mean_margin": _mean_margin(dv_po_margin),
            "count": len(dv_bl_acc),
        },
        "per_attribute": per_attr,
        "positive_state": {
            "pre_accuracy": _acc(pos_bl_acc),
            "post_accuracy": _acc(pos_po_acc),
            "count": len(pos_bl_acc),
        },
        "negative_state": {
            "pre_accuracy": _acc(neg_bl_acc),
            "post_accuracy": _acc(neg_po_acc),
            "count": len(neg_bl_acc),
        },
        "retain_group": {
            "pre_accuracy": _acc(retain_bl_acc),
            "post_accuracy": _acc(retain_po_acc),
            "count": len(retain_bl_acc),
        },
        "control_group": {
            "pre_accuracy": _acc(control_bl_acc),
            "post_accuracy": _acc(control_po_acc),
            "count": len(control_bl_acc),
        },
    }
    return report


# --------------------------------------------------------------------------- #
# Route-effects report
# --------------------------------------------------------------------------- #

def compute_route_effects(
    baseline_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
    identity_groups: dict[str, str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Compute route effects (Δ_name, Δ_wrong, Δ_conflict) pre and post.

    For each group (target, retain, control):

    - ``Δ_name = mean(image_plus_name) - mean(direct_visual)``
    - ``Δ_wrong = mean(wrong_name) - mean(image_plus_name)``
    - ``Δ_conflict = mean(visual_text_conflict) - mean(direct_visual)``

    Returns a nested dict: ``{group: {effect_name: {pre, post, change}}}``.
    """
    if identity_groups is None:
        identity_groups = {}

    bl_idx = _index_by_probe_id(baseline_rows)
    po_idx = _index_by_probe_id(post_rows)

    # Collect margins per (group, family, phase)
    margins: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {
            "direct_visual": {"pre": [], "post": []},
            "image_plus_name": {"pre": [], "post": []},
            "wrong_name": {"pre": [], "post": []},
            "visual_text_conflict": {"pre": [], "post": []},
        },
    )

    for probe_id in sorted(bl_idx.keys()):
        if probe_id not in po_idx:
            continue
        bl = bl_idx[probe_id]
        po = po_idx[probe_id]
        fam = bl.get("probe_family", "")
        if fam == "name_only":
            continue
        iid = bl.get("identity_id", "")
        grp = identity_groups.get(iid, "untargeted")

        bl_m = bl.get("signed_answer_margin")
        po_m = po.get("signed_answer_margin")
        if bl_m is not None:
            margins[grp][fam]["pre"].append(bl_m)
        if po_m is not None:
            margins[grp][fam]["post"].append(po_m)

    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for grp in ["target", "retain", "control"]:
        fam_means: dict[str, dict[str, float | None]] = {}
        for fam in [
            "direct_visual",
            "image_plus_name",
            "wrong_name",
            "visual_text_conflict",
        ]:
            pre_vals = margins[grp][fam]["pre"]
            post_vals = margins[grp][fam]["post"]
            fam_means[fam] = {
                "pre": statistics.mean(pre_vals) if pre_vals else None,
                "post": statistics.mean(post_vals) if post_vals else None,
            }

        def _delta(
            a_fam: str, b_fam: str, phase: str, _fm: dict = fam_means,
        ) -> float | None:
            va = _fm[a_fam].get(phase)
            vb = _fm[b_fam].get(phase)
            if va is not None and vb is not None:
                return va - vb
            return None

        dv = "direct_visual"
        ipn = "image_plus_name"
        wn = "wrong_name"
        vtc = "visual_text_conflict"

        pre_dname = _delta(ipn, dv, "pre")
        post_dname = _delta(ipn, dv, "post")
        pre_dwrong = _delta(wn, ipn, "pre")
        post_dwrong = _delta(wn, ipn, "post")
        pre_dconflict = _delta(vtc, dv, "pre")
        post_dconflict = _delta(vtc, dv, "post")

        def _change(pre_v: float | None, post_v: float | None) -> float | None:
            if pre_v is not None and post_v is not None:
                return post_v - pre_v
            return None

        result[grp] = {
            "delta_name": {
                "pre": pre_dname,
                "post": post_dname,
                "change": _change(pre_dname, post_dname),
            },
            "delta_wrong": {
                "pre": pre_dwrong,
                "post": post_dwrong,
                "change": _change(pre_dwrong, post_dwrong),
            },
            "delta_conflict": {
                "pre": pre_dconflict,
                "post": post_dconflict,
                "change": _change(pre_dconflict, post_dconflict),
            },
        }

    return result


# --------------------------------------------------------------------------- #
# High-level orchestrator
# --------------------------------------------------------------------------- #

class PairedAnalysis:
    """Run the full paired analysis pipeline.

    Loads baseline and post-eval results, joins by probe_id, and writes
    all five analysis artifacts to ``output_dir``.

    P0-10: Before any delta computation the pairing is validated:
    both files must have exactly 500 rows, 500 unique IDs, and
    identical ID sets.  Any violation aborts analysis.
    """

    def __init__(self, config: PairedAnalysisConfig) -> None:
        self.config = config
        self._baseline_rows: list[dict[str, Any]] | None = None
        self._post_rows: list[dict[str, Any]] | None = None
        self._identity_groups: dict[str, str] = {}
        self._probe_deltas: list[dict[str, Any]] | None = None
        self._pairing_validation: dict[str, Any] | None = None

    # -- loading ---------------------------------------------------------- #

    def load_data(self) -> None:
        """Load baseline, post-eval, and selection manifest."""
        self._baseline_rows = load_results_jsonl(self.config.baseline_results_path)
        self._post_rows = load_results_jsonl(self.config.post_results_path)

        # Load selection manifest to build identity → group mapping
        manifest_path = Path(self.config.selection_manifest_path)
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            for iid in manifest.get("target_identities", []):
                self._identity_groups[iid] = "target"
            for iid in manifest.get("retain_identities", []):
                self._identity_groups[iid] = "retain"
            for iid in manifest.get("control_identities", []):
                self._identity_groups[iid] = "control"

    @property
    def baseline_rows(self) -> list[dict[str, Any]]:
        assert self._baseline_rows is not None, "Call load_data() first"
        return self._baseline_rows

    @property
    def post_rows(self) -> list[dict[str, Any]]:
        assert self._post_rows is not None, "Call load_data() first"
        return self._post_rows

    # -- P0-10: pairing validation ---------------------------------------- #

    def validate_pairing(self) -> dict[str, Any]:
        """Validate exact 500↔500 probe pairing (P0-10).

        Raises ``RuntimeError`` if any check fails.  Returns the
        validation report dict on success.
        """
        bl = self.baseline_rows
        po = self.post_rows

        bl_ids = [r["probe_id"] for r in bl]
        po_ids = [r["probe_id"] for r in po]
        bl_unique = set(bl_ids)
        po_unique = set(po_ids)

        bl_dupes = [pid for pid in bl_ids if bl_ids.count(pid) > 1]
        po_dupes = [pid for pid in po_ids if po_ids.count(pid) > 1]

        missing = sorted(bl_unique - po_unique)
        extra = sorted(po_unique - bl_unique)

        passed = (
            len(bl) == 500
            and len(po) == 500
            and len(bl_unique) == 500
            and len(po_unique) == 500
            and bl_unique == po_unique
        )

        report: dict[str, Any] = {
            "pass": passed,
            "baseline_rows": len(bl),
            "post_rows": len(po),
            "baseline_unique_ids": len(bl_unique),
            "post_unique_ids": len(po_unique),
            "missing": missing,
            "extra": extra,
            "duplicates_baseline": sorted(set(bl_dupes)),
            "duplicates_post": sorted(set(po_dupes)),
        }
        self._pairing_validation = report

        if not passed:
            reasons: list[str] = []
            if len(bl) != 500:
                reasons.append(f"baseline has {len(bl)} rows, expected 500")
            if len(po) != 500:
                reasons.append(f"post has {len(po)} rows, expected 500")
            if len(bl_unique) != 500:
                reasons.append(
                    f"baseline has {len(bl_unique)} unique IDs, expected 500"
                )
            if len(po_unique) != 500:
                reasons.append(
                    f"post has {len(po_unique)} unique IDs, expected 500"
                )
            if bl_unique != po_unique:
                if missing:
                    reasons.append(f"{len(missing)} missing probe IDs")
                if extra:
                    reasons.append(f"{len(extra)} extra probe IDs")
            raise RuntimeError(
                "P0-10: Paired analysis aborted — exact 500↔500 "
                "pairing failed: " + "; ".join(reasons)
            )

        return report

    # -- analysis --------------------------------------------------------- #

    def run_probe_deltas(self) -> list[dict[str, Any]]:
        """Compute and cache probe-level deltas.

        P0-10: Validates exact 500↔500 pairing before computing
        any deltas.  Raises ``RuntimeError`` on pairing failure.
        """
        # P0-10: Hard-fail unless exact pairing is confirmed.
        self.validate_pairing()
        self._probe_deltas = compute_probe_deltas(
            self.baseline_rows, self.post_rows, self._identity_groups,
        )
        return self._probe_deltas

    def run_identity_effects(self) -> dict[str, dict[str, Any]]:
        """Compute per-identity effects (requires probe deltas)."""
        if self._probe_deltas is None:
            self.run_probe_deltas()
        return compute_identity_effects(self._probe_deltas)

    def run_group_effects(self) -> dict[str, dict[str, Any]]:
        """Compute group-level effects (requires probe deltas)."""
        if self._probe_deltas is None:
            self.run_probe_deltas()
        return compute_group_effects(self._probe_deltas)

    def run_preservation_report(self) -> dict[str, Any]:
        """Compute the preservation report."""
        return compute_preservation_report(
            self.baseline_rows, self.post_rows, self._identity_groups,
        )

    def run_route_effects(self) -> dict[str, dict[str, Any]]:
        """Compute the route-effects report."""
        return compute_route_effects(
            self.baseline_rows, self.post_rows, self._identity_groups,
        )

    # -- full pipeline ---------------------------------------------------- #

    def run_all(self) -> dict[str, Any]:
        """Run the full analysis pipeline and return all results."""
        probe_deltas = self.run_probe_deltas()
        identity_effects = self.run_identity_effects()
        group_effects = self.run_group_effects()
        preservation = self.run_preservation_report()
        route_effects = self.run_route_effects()
        return {
            "probe_deltas": probe_deltas,
            "identity_effects": identity_effects,
            "group_effects": group_effects,
            "preservation_report": preservation,
            "route_effects": route_effects,
            # P0-10: Include pairing validation in results so
            # downstream consumers (e.g. validation report) can
            # access it.
            "pairing_validation": self._pairing_validation or {},
        }

    # -- writing ---------------------------------------------------------- #

    def write_artifacts(self, results: dict[str, Any] | None = None) -> dict[str, Path]:
        """Write all analysis artifacts to ``output_dir``.

        Returns a mapping from artifact name to file path.
        """
        if results is None:
            results = self.run_all()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        # 0) P0-10: pairing_validation.json
        if self._pairing_validation is not None:
            p = output_dir / "pairing_validation.json"
            with p.open("w", encoding="utf-8") as fh:
                json.dump(self._pairing_validation, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            paths["pairing_validation"] = p

        # 1) paired_probe_deltas.jsonl
        p = output_dir / "paired_probe_deltas.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for row in results["probe_deltas"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths["paired_probe_deltas"] = p

        # 2) identity_effects.json
        p = output_dir / "identity_effects.json"
        with p.open("w", encoding="utf-8") as fh:
            json.dump(results["identity_effects"], fh, indent=2, ensure_ascii=False)
        paths["identity_effects"] = p

        # 3) group_effects.json
        p = output_dir / "group_effects.json"
        with p.open("w", encoding="utf-8") as fh:
            json.dump(results["group_effects"], fh, indent=2, ensure_ascii=False)
        paths["group_effects"] = p

        # 4) preservation_report.json
        p = output_dir / "preservation_report.json"
        with p.open("w", encoding="utf-8") as fh:
            json.dump(results["preservation_report"], fh, indent=2, ensure_ascii=False)
        paths["preservation_report"] = p

        # 5) route_effects_post.json
        p = output_dir / "route_effects_post.json"
        with p.open("w", encoding="utf-8") as fh:
            json.dump(results["route_effects"], fh, indent=2, ensure_ascii=False)
        paths["route_effects_post"] = p

        return paths
