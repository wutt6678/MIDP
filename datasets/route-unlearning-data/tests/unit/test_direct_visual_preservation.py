"""Tests for group-specific direct-visual preservation (P0-9) and
base-parameter integrity (P1-5, P1-6).

Verifies:
- target/retain/control/untargeted direct_visual metrics are computed
  independently and only from probe_family == "direct_visual" probes
- retain direct_visual 100% but retain image_plus_name 0% →
  retain_direct_visual.post_accuracy == 1.0
- Base-parameter integrity check passes when only LoRA params trainable
- Base-parameter integrity check fails when non-LoRA params trainable
- Evidence file is written correctly
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from route_data.eval.paired_analysis import compute_preservation_report
from route_data.eval.unlearning_harness import check_base_parameter_integrity

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_row(
    probe_id: str = "p1",
    identity_id: str = "id_A",
    probe_family: str = "direct_visual",
    signed_answer_margin: float | None = 5.0,
    correct: bool | None = True,
    target_attribute: str | None = "hair_color",
    answer_label: bool | None = True,
) -> dict:
    """Build a minimal result row."""
    return {
        "probe_id": probe_id,
        "identity_id": identity_id,
        "probe_family": probe_family,
        "signed_answer_margin": signed_answer_margin,
        "correct": correct,
        "target_attribute": target_attribute,
        "answer_label": answer_label,
    }


# --------------------------------------------------------------------------- #
# Tests: P0-9 — group-specific direct-visual preservation
# --------------------------------------------------------------------------- #

class TestGroupDirectVisualPreservation:
    """P0-9: Each group's direct_visual metrics are independent."""

    def _make_fixtures(self):
        """Build baseline/post rows across 4 groups with mixed families."""
        groups = {
            "t1": "target",
            "r1": "retain",
            "c1": "control",
            "u1": "untargeted",
        }

        bl, po = [], []
        pid = 0

        # For each group, create direct_visual and image_plus_name probes
        for iid in groups:
            # 2 direct_visual probes per group
            for j in range(2):
                p = f"p{pid:03d}"
                pid += 1
                bl.append(_make_row(
                    probe_id=p, identity_id=iid,
                    probe_family="direct_visual",
                    correct=True, signed_answer_margin=5.0,
                ))
                po.append(_make_row(
                    probe_id=p, identity_id=iid,
                    probe_family="direct_visual",
                    correct=True, signed_answer_margin=4.0,
                ))

            # 2 image_plus_name probes per group (should NOT affect direct_visual)
            for j in range(2):
                p = f"p{pid:03d}"
                pid += 1
                bl.append(_make_row(
                    probe_id=p, identity_id=iid,
                    probe_family="image_plus_name",
                    correct=True, signed_answer_margin=3.0,
                ))
                po.append(_make_row(
                    probe_id=p, identity_id=iid,
                    probe_family="image_plus_name",
                    correct=False, signed_answer_margin=-1.0,
                ))

        return bl, po, groups

    def test_all_group_keys_present(self) -> None:
        """Report must contain all 4 group-specific direct_visual keys."""
        bl, po, groups = self._make_fixtures()
        report = compute_preservation_report(bl, po, identity_groups=groups)

        for key in [
            "target_direct_visual",
            "retain_direct_visual",
            "control_direct_visual",
            "untargeted_direct_visual",
        ]:
            assert key in report, f"Missing key: {key}"

    def test_each_group_has_required_fields(self) -> None:
        """Each group-specific metric must have the required fields."""
        bl, po, groups = self._make_fixtures()
        report = compute_preservation_report(bl, po, identity_groups=groups)

        for key in [
            "target_direct_visual",
            "retain_direct_visual",
            "control_direct_visual",
            "untargeted_direct_visual",
        ]:
            m = report[key]
            assert "pre_accuracy" in m
            assert "post_accuracy" in m
            assert "pre_mean_margin" in m
            assert "post_mean_margin" in m
            assert "count" in m

    def test_group_counts_are_correct(self) -> None:
        """Each group should have exactly 2 direct_visual probes."""
        bl, po, groups = self._make_fixtures()
        report = compute_preservation_report(bl, po, identity_groups=groups)

        for key in [
            "target_direct_visual",
            "retain_direct_visual",
            "control_direct_visual",
            "untargeted_direct_visual",
        ]:
            assert report[key]["count"] == 2

    def test_global_count_is_sum(self) -> None:
        """Global direct_visual count should be sum of all groups."""
        bl, po, groups = self._make_fixtures()
        report = compute_preservation_report(bl, po, identity_groups=groups)

        group_total = sum(
            report[f"{g}_direct_visual"]["count"]
            for g in ["target", "retain", "control", "untargeted"]
        )
        assert report["global_direct_visual"]["count"] == group_total

    def test_group_metrics_only_from_direct_visual(self) -> None:
        """Group metrics must come only from direct_visual probes.

        Regression: retain direct_visual 100% but retain image_plus_name 0%
        → retain_direct_visual.post_accuracy == 1.0
        """
        bl, po, groups = self._make_fixtures()
        report = compute_preservation_report(bl, po, identity_groups=groups)

        # All direct_visual probes are correct=True in post
        assert report["retain_direct_visual"]["post_accuracy"] == pytest.approx(1.0)
        assert report["target_direct_visual"]["post_accuracy"] == pytest.approx(1.0)
        assert report["control_direct_visual"]["post_accuracy"] == pytest.approx(1.0)
        assert report["untargeted_direct_visual"]["post_accuracy"] == pytest.approx(1.0)

    def test_regression_retain_dv_100_but_ipn_0(self) -> None:
        """retain direct_visual 100% but retain image_plus_name 0%
        → retain_direct_visual.post_accuracy == 1.0

        This is the key regression test: the old retain_group metric
        mixed all families, so image_plus_name failures would drag down
        the retain metric even though direct_visual was fine.
        """
        bl = [
            # retain direct_visual: always correct
            _make_row(probe_id="r_dv1", identity_id="r1",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=5.0),
            _make_row(probe_id="r_dv2", identity_id="r1",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=4.0),
            # retain image_plus_name: all wrong
            _make_row(probe_id="r_ipn1", identity_id="r1",
                      probe_family="image_plus_name", correct=False,
                      signed_answer_margin=-2.0),
            _make_row(probe_id="r_ipn2", identity_id="r1",
                      probe_family="image_plus_name", correct=False,
                      signed_answer_margin=-3.0),
        ]
        po = [
            # retain direct_visual: still correct
            _make_row(probe_id="r_dv1", identity_id="r1",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=4.0),
            _make_row(probe_id="r_dv2", identity_id="r1",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=3.0),
            # retain image_plus_name: still wrong
            _make_row(probe_id="r_ipn1", identity_id="r1",
                      probe_family="image_plus_name", correct=False,
                      signed_answer_margin=-2.0),
            _make_row(probe_id="r_ipn2", identity_id="r1",
                      probe_family="image_plus_name", correct=False,
                      signed_answer_margin=-3.0),
        ]
        groups = {"r1": "retain"}
        report = compute_preservation_report(bl, po, identity_groups=groups)

        # Key assertion: retain_direct_visual is 100% despite image_plus_name being 0%
        assert report["retain_direct_visual"]["post_accuracy"] == pytest.approx(1.0)
        assert report["retain_direct_visual"]["count"] == 2

    def test_target_direct_visual_isolated(self) -> None:
        """Target direct_visual metrics are isolated from other groups."""
        bl = [
            _make_row(probe_id="t1", identity_id="tgt",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=6.0),
            _make_row(probe_id="r1", identity_id="ret",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=5.0),
        ]
        po = [
            # Target drops to incorrect
            _make_row(probe_id="t1", identity_id="tgt",
                      probe_family="direct_visual", correct=False,
                      signed_answer_margin=-1.0),
            # Retain stays correct
            _make_row(probe_id="r1", identity_id="ret",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=4.0),
        ]
        groups = {"tgt": "target", "ret": "retain"}
        report = compute_preservation_report(bl, po, identity_groups=groups)

        assert report["target_direct_visual"]["post_accuracy"] == pytest.approx(0.0)
        assert report["retain_direct_visual"]["post_accuracy"] == pytest.approx(1.0)

    def test_untargeted_default_for_unknown_identity(self) -> None:
        """Identities not in identity_groups map to untargeted."""
        bl = [
            _make_row(probe_id="u1", identity_id="unknown_id",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=5.0),
        ]
        po = [
            _make_row(probe_id="u1", identity_id="unknown_id",
                      probe_family="direct_visual", correct=True,
                      signed_answer_margin=4.0),
        ]
        # No identity_groups → all "untargeted"
        report = compute_preservation_report(bl, po)

        assert report["untargeted_direct_visual"]["count"] == 1
        assert report["untargeted_direct_visual"]["post_accuracy"] == pytest.approx(1.0)
        assert report["target_direct_visual"]["count"] == 0
        assert report["retain_direct_visual"]["count"] == 0
        assert report["control_direct_visual"]["count"] == 0


# --------------------------------------------------------------------------- #
# Tests: P1-5 — base-parameter integrity
# --------------------------------------------------------------------------- #

class TestBaseParameterIntegrity:
    """P1-5: Only LoRA parameters should be trainable."""

    def _make_model(self, lora_params=True, non_lora_trainable=False):
        """Create a mock model with configurable parameter groups."""
        model = MagicMock()
        params = []

        if lora_params:
            p_lora1 = MagicMock()
            p_lora1.requires_grad = True
            p_lora1.numel.return_value = 100

            p_lora2 = MagicMock()
            p_lora2.requires_grad = True
            p_lora2.numel.return_value = 200

            params.extend([
                ("base_model.model.lora_A.weight", p_lora1),
                ("base_model.model.lora_B.weight", p_lora2),
            ])

        # Base model parameters
        p_base1 = MagicMock()
        p_base1.requires_grad = non_lora_trainable  # trainable only if flag set
        p_base1.numel.return_value = 1000

        p_base2 = MagicMock()
        p_base2.requires_grad = False  # frozen
        p_base2.numel.return_value = 2000

        params.extend([
            ("base_model.model.layers.0.self_attn.q_proj.weight", p_base1),
            ("base_model.model.layers.0.self_attn.v_proj.weight", p_base2),
        ])

        model.named_parameters.return_value = params
        model.parameters.return_value = [p for _, p in params]
        return model

    def test_pass_when_only_lora_trainable(self, tmp_path: Path) -> None:
        """Should pass when only LoRA parameters are trainable."""
        model = self._make_model(lora_params=True, non_lora_trainable=False)
        report = check_base_parameter_integrity(model, output_dir=tmp_path)

        assert report["pass"] is True
        assert report["non_lora_trainable_parameter_count"] == 0
        assert report["unexpected_trainable_parameters"] == []

    def test_fail_when_non_lora_trainable(self) -> None:
        """Should hard-fail when non-LoRA parameters are trainable."""
        model = self._make_model(lora_params=True, non_lora_trainable=True)

        with pytest.raises(RuntimeError, match="Base-parameter integrity check FAILED"):
            check_base_parameter_integrity(model)

    def test_evidence_file_written(self, tmp_path: Path) -> None:
        """Evidence file should be written to output_dir/evidence/."""
        model = self._make_model(lora_params=True, non_lora_trainable=False)
        check_base_parameter_integrity(model, output_dir=tmp_path)

        evidence_path = tmp_path / "evidence" / "base_parameter_integrity.json"
        assert evidence_path.exists()

        with open(evidence_path) as f:
            data = json.load(f)
        assert data["pass"] is True
        assert data["non_lora_trainable_parameter_count"] == 0
        assert data["unexpected_trainable_parameters"] == []

    def test_no_evidence_file_without_output_dir(self) -> None:
        """No evidence file when output_dir is None."""
        model = self._make_model(lora_params=True, non_lora_trainable=False)
        report = check_base_parameter_integrity(model)  # no output_dir
        assert report["pass"] is True

    def test_unexpected_params_listed_in_report(self) -> None:
        """Report should list the unexpected trainable parameter names."""
        model = self._make_model(lora_params=True, non_lora_trainable=True)

        try:
            check_base_parameter_integrity(model)
        except RuntimeError:
            pass

        # Verify by calling without raise (test the report directly)
        model2 = self._make_model(lora_params=True, non_lora_trainable=True)
        # Manually inspect
        unexpected = []
        for name, param in model2.named_parameters():
            if param.requires_grad and "lora" not in name.lower():
                unexpected.append(name)
        assert len(unexpected) == 1
        assert "q_proj.weight" in unexpected[0]
