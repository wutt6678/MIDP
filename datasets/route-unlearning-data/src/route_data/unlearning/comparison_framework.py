"""Phase 4: Unified comparison framework for all baselines.

Generates comparison tables, efficiency reports, and trajectory analysis
across all MLLMU-Bench baselines (B0–B9). Produces the route-selectivity
conclusion and E2C decision.

Public API
----------
.. autoclass:: ComparisonFramework
.. autoclass:: MethodResult
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class MethodResult:
    """Result from a single baseline method."""

    method_id: str  # e.g., "ga", "gd", "kl", "npo", "mmunlearner", "manu", "r2mu_adapted"
    baseline_id: str  # e.g., "B1", "B2", ..., "B9"
    description: str

    # Outcome metrics (ΔM for each attribute)
    delta_target: dict[str, float] = field(default_factory=dict)  # DV/IPN/WN/VTC
    delta_retain: dict[str, float] = field(default_factory=dict)
    delta_control: dict[str, float] = field(default_factory=dict)
    delta_untargeted: dict[str, float] = field(default_factory=dict)

    # Contrasts
    target_retain_contrast: dict[str, float] = field(default_factory=dict)
    target_control_contrast: dict[str, float] = field(default_factory=dict)

    # Preservation
    global_preservation: dict[str, float] = field(default_factory=dict)
    target_preservation: dict[str, float] = field(default_factory=dict)
    retain_preservation: dict[str, float] = field(default_factory=dict)
    control_preservation: dict[str, float] = field(default_factory=dict)

    # Exact pairing
    exact_pair_count: int = 0
    inference_errors: int = 0

    # Efficiency
    gpu_hours: float = 0.0
    peak_memory_gb: float = 0.0
    forward_backward_passes: int = 0
    trainable_parameters: int = 0
    modified_parameters: int = 0
    pruned_neurons: int = 0
    mask_generation_cost_seconds: float = 0.0
    reference_model_cost_seconds: float = 0.0
    checkpoint_size_mb: float = 0.0

    # Training
    num_steps: int = 0
    final_loss: float | None = None

    # Table assignment
    table: str = "A"  # "A" = LoRA-controlled, "B" = Native structural


# --------------------------------------------------------------------------- #
# Comparison tables
# --------------------------------------------------------------------------- #

class ComparisonFramework:
    """Unified comparison framework for all baselines.

    Generates comparison tables, efficiency reports, and makes the
    E2C decision based on route-selectivity evidence.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[MethodResult] = []

    def add_result(self, result: MethodResult) -> None:
        """Add a method result."""
        self.results.append(result)
        logger.info(f"Added result for {result.method_id} ({result.baseline_id})")

    def generate_tables(self) -> dict[str, Any]:
        """Generate comparison tables.

        Returns
        -------
        tables:
            Dict with table_a, table_b, and combined tables.
        """
        table_a = [r for r in self.results if r.table == "A"]
        table_b = [r for r in self.results if r.table == "B"]

        tables = {
            "table_a": {
                "title": "LoRA-controlled baselines (B0–B6, B9)",
                "methods": [self._method_summary(r) for r in table_a],
            },
            "table_b": {
                "title": "Native structural baselines (B7–B8)",
                "methods": [self._method_summary(r) for r in table_b],
            },
            "combined": {
                "title": "All methods with common MIDP metrics",
                "methods": [self._method_summary(r) for r in self.results],
            },
        }

        # Save tables
        with open(self.output_dir / "comparison_tables.json", "w") as f:
            json.dump(tables, f, indent=2)
            f.write("\n")

        logger.info(f"Generated comparison tables: {len(self.results)} methods")
        return tables

    def _method_summary(self, result: MethodResult) -> dict[str, Any]:
        """Create a summary dict for a method."""
        return {
            "method_id": result.method_id,
            "baseline_id": result.baseline_id,
            "description": result.description,
            "table": result.table,
            "outcome": {
                "delta_target": result.delta_target,
                "delta_retain": result.delta_retain,
                "delta_control": result.delta_control,
                "delta_untargeted": result.delta_untargeted,
            },
            "contrasts": {
                "target_retain": result.target_retain_contrast,
                "target_control": result.target_control_contrast,
            },
            "preservation": {
                "global": result.global_preservation,
                "target": result.target_preservation,
                "retain": result.retain_preservation,
                "control": result.control_preservation,
            },
            "pairing": {
                "exact_pairs": result.exact_pair_count,
                "inference_errors": result.inference_errors,
            },
        }

    def generate_efficiency_report(self) -> dict[str, Any]:
        """Generate efficiency comparison report.

        Returns
        -------
        report:
            Efficiency metrics for all methods.
        """
        report = {
            "methods": [],
        }

        for r in self.results:
            method_eff = {
                "method_id": r.method_id,
                "baseline_id": r.baseline_id,
                "gpu_hours": r.gpu_hours,
                "peak_memory_gb": r.peak_memory_gb,
                "forward_backward_passes": r.forward_backward_passes,
                "trainable_parameters": r.trainable_parameters,
                "modified_parameters": r.modified_parameters,
                "pruned_neurons": r.pruned_neurons,
                "mask_generation_cost_seconds": r.mask_generation_cost_seconds,
                "reference_model_cost_seconds": r.reference_model_cost_seconds,
                "checkpoint_size_mb": r.checkpoint_size_mb,
                "num_steps": r.num_steps,
            }
            report["methods"].append(method_eff)

        # Save report
        with open(self.output_dir / "efficiency_report.json", "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

        logger.info(f"Generated efficiency report: {len(self.results)} methods")
        return report

    def make_e2c_decision(self) -> dict[str, Any]:
        """Make the E2C decision based on route-selectivity evidence.

        Decision rules (from plan Section 4.4):
        - Case A: At least one method achieves target-specific degradation
          while preserving retain/control → replicate ≥3 seeds before E2C
        - Case B: Method improves stability but not selectivity
          (target ≈ retain ≈ control) → record as evidence, proceed to E2C
        - Case C: All methods collapse non-selectively → freeze results,
          proceed directly to E2C

        Returns
        -------
        decision:
            E2C decision with rationale.
        """
        # Analyze each method for target-specific degradation
        case_a_methods = []
        case_b_methods = []
        case_c_methods = []

        missing_eval_methods = []

        for r in self.results:
            if not r.delta_target:
                continue

            # Check if target degradation is specific (not uniform)
            target_degradation = self._has_target_degradation(r)
            retain_status = self._check_preservation(r.delta_retain)
            control_status = self._check_preservation(r.delta_control)

            # Missing evidence: method cannot be Case A
            if retain_status == "MISSING" or control_status == "MISSING":
                missing_eval_methods.append(r.method_id)

            if target_degradation and retain_status == "PASS" and control_status == "PASS":
                case_a_methods.append(r.method_id)
            elif target_degradation and not (retain_status == "PASS" and control_status == "PASS"):
                # Target degrades but so do retain/control (or evidence missing)
                case_c_methods.append(r.method_id)
            else:
                # No specific target degradation
                case_b_methods.append(r.method_id)

        # Make decision
        if case_a_methods:
            decision = {
                "case": "A",
                "rationale": (
                    f"At least one method ({', '.join(case_a_methods)}) achieves "
                    f"target-specific degradation while preserving retain/control. "
                    f"Recommend: replicate with ≥3 seeds before E2C."
                ),
                "case_a_methods": case_a_methods,
                "action": "replicate_seeds",
            }
        elif case_b_methods and not case_c_methods:
            decision = {
                "case": "B",
                "rationale": (
                    "Methods improve stability but not selectivity "
                    "(target ≈ retain ≈ control). "
                    "Record as evidence that stability ≠ selectivity. "
                    "Proceed to E2C."
                ),
                "case_b_methods": case_b_methods,
                "action": "proceed_to_e2c",
            }
        else:
            decision = {
                "case": "C",
                "rationale": (
                    "All methods collapse non-selectively. "
                    "Strong support for claim that conventional objectives "
                    "don't isolate identity routes. "
                    "Freeze results and proceed directly to E2C."
                ),
                "case_c_methods": case_c_methods,
                "action": "freeze_and_proceed",
            }

        # Save decision
        with open(self.output_dir / "e2c_decision.json", "w") as f:
            json.dump(decision, f, indent=2)
            f.write("\n")

        if missing_eval_methods:
            logger.warning(
                f"Methods with MISSING evaluation data: {missing_eval_methods}. "
                f"These cannot be classified as Case A."
            )

        logger.info(f"E2C decision: Case {decision['case']}")
        return decision

    def _has_target_degradation(self, result: MethodResult) -> bool:
        """Check if target attributes show degradation."""
        if not result.delta_target:
            return False
        # Target degradation = negative delta (decrease in attribute accuracy)
        avg_delta = sum(result.delta_target.values()) / len(result.delta_target)
        return avg_delta < -0.1  # At least 10% degradation

    def _check_preservation(self, deltas: dict[str, float]) -> str:
        """Three-state preservation check: PASS, FAIL, or MISSING.

        Parameters
        ----------
        deltas:
            Per-attribute delta values. Empty dict means no data.

        Returns
        -------
        status:
            'PASS' if avg delta > -0.05, 'FAIL' if <= -0.05,
            'MISSING' if no data.
        """
        if not deltas:
            return "MISSING"
        avg_delta = sum(deltas.values()) / len(deltas)
        if avg_delta > -0.05:
            return "PASS"
        return "FAIL"

    def generate_trajectory_analysis(self) -> dict[str, Any]:
        """Generate trajectory analysis across checkpoints.

        Analyzes how each method's metrics evolve during training.

        Returns
        -------
        analysis:
            Trajectory analysis results.
        """
        analysis = {
            "methods": [],
        }

        for r in self.results:
            method_traj = {
                "method_id": r.method_id,
                "baseline_id": r.baseline_id,
                "num_steps": r.num_steps,
                "final_loss": r.final_loss,
                "outcome_summary": {
                    "target_avg": (
                        sum(r.delta_target.values()) / len(r.delta_target)
                        if r.delta_target else None
                    ),
                    "retain_avg": (
                        sum(r.delta_retain.values()) / len(r.delta_retain)
                        if r.delta_retain else None
                    ),
                    "control_avg": (
                        sum(r.delta_control.values()) / len(r.delta_control)
                        if r.delta_control else None
                    ),
                },
                "selectivity_score": self._compute_selectivity_score(r),
            }
            analysis["methods"].append(method_traj)

        # Save analysis
        with open(self.output_dir / "trajectory_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
            f.write("\n")

        logger.info(f"Generated trajectory analysis: {len(self.results)} methods")
        return analysis

    def _compute_selectivity_score(self, result: MethodResult) -> float | None:
        """Compute a selectivity score for a method.

        Selectivity = target degradation - max(retain, control) degradation.
        Higher = more selective (target-specific).
        """
        if not result.delta_target:
            return None

        target_deg = abs(sum(result.delta_target.values()) / len(result.delta_target))
        retain_deg = (
            abs(sum(result.delta_retain.values()) / len(result.delta_retain))
            if result.delta_retain else 0.0
        )
        control_deg = (
            abs(sum(result.delta_control.values()) / len(result.delta_control))
            if result.delta_control else 0.0
        )

        selectivity = target_deg - max(retain_deg, control_deg)
        return selectivity

    def write_route_selectivity_conclusion(self) -> str:
        """Write the route-selectivity conclusion.

        Returns
        -------
        conclusion:
            Text conclusion about route selectivity.
        """
        decision = self.make_e2c_decision()
        tables = self.generate_tables()
        efficiency = self.generate_efficiency_report()

        conclusion = self._format_conclusion(decision, tables, efficiency)

        # Save conclusion
        with open(self.output_dir / "route_selectivity_conclusion.txt", "w") as f:
            f.write(conclusion)

        logger.info("Wrote route-selectivity conclusion")
        return conclusion

    def _format_conclusion(
        self,
        decision: dict[str, Any],
        tables: dict[str, Any],
        efficiency: dict[str, Any],
    ) -> str:
        """Format the conclusion as text."""
        lines = [
            "=" * 80,
            "ROUTE-SELECTIVITY CONCLUSION",
            "=" * 80,
            "",
            f"E2C Decision: Case {decision['case']}",
            f"Rationale: {decision['rationale']}",
            "",
            "-" * 80,
            "SUMMARY",
            "-" * 80,
            "",
            f"Total methods evaluated: {len(self.results)}",
            f"  Table A (LoRA-controlled): {len([r for r in self.results if r.table == 'A'])}",
            f"  Table B (Native structural): {len([r for r in self.results if r.table == 'B'])}",
            "",
        ]

        # Add method summaries
        for r in self.results:
            lines.extend([
                f"{r.baseline_id} ({r.method_id}):",
                f"  {r.description}",
                f"  Target degradation: {self._format_dict(r.delta_target)}",
                f"  Retain preservation: {self._format_dict(r.delta_retain)}",
                f"  Control preservation: {self._format_dict(r.delta_control)}",
                f"  Selectivity score: {self._compute_selectivity_score(r)}",
                "",
            ])

        lines.extend([
            "-" * 80,
            "EFFICIENCY",
            "-" * 80,
            "",
        ])

        for method_eff in efficiency["methods"]:
            lines.extend([
                f"{method_eff['method_id']}:",
                f"  GPU-hours: {method_eff['gpu_hours']:.2f}",
                f"  Peak memory: {method_eff['peak_memory_gb']:.1f} GB",
                f"  Trainable params: {method_eff['trainable_parameters']:,}",
                f"  Modified params: {method_eff['modified_parameters']:,}",
                "",
            ])

        lines.extend([
            "=" * 80,
            "END OF CONCLUSION",
            "=" * 80,
        ])

        return "\n".join(lines)

    def _format_dict(self, d: dict[str, float]) -> str:
        """Format a dict for display."""
        if not d:
            return "N/A"
        parts = [f"{k}={v:+.3f}" for k, v in d.items()]
        return ", ".join(parts)
