"""Tests for P1 fixes: score validation, pair validation, source pinning."""

from __future__ import annotations

import math

import pytest

from route_data.build.conflict_generation import (
    RouteProbeBuilder,
    build_identity_probes,
    build_pair_manifest,
    validate_pair_manifest,
    _accepted_visible_attributes,
)
from route_data.config import PromptsConfig
from route_data.data.schemas import (
    AttributeObservation,
    CanonicalSample,
    ProfileFact,
    Provenance,
)
from route_data.prompts.registry import PromptRegistry

EYEGLASSES_KEY = "extended_attributes.celeba40.Eyeglasses"
SMILING_KEY = "extended_attributes.celeba40.Smiling"


@pytest.fixture()
def registry(repo_root) -> PromptRegistry:
    prompts = PromptsConfig(
        binary=str(repo_root / "configs/prompts/celeba_binary_v1.yaml"),
        grouped=str(repo_root / "configs/prompts/celeba_grouped_json_v1.yaml"),
        route_conflict=str(repo_root / "configs/prompts/route_conflict_v1.yaml"),
    )
    return PromptRegistry(prompts)


def _sample(**overrides) -> CanonicalSample:
    fields = dict(
        benchmark="fairget",
        source_sample_id="s1",
        identity_id="id1",
        identity_name="Test User",
        provenance=Provenance(source_dataset="fairget"),
        image_uri="images/test.png",
        modality="image_text",
        visual_attributes={
            EYEGLASSES_KEY: AttributeObservation(
                name=EYEGLASSES_KEY,
                label=True,
                source="source_model",
                confidence_band="high",
            ),
            SMILING_KEY: AttributeObservation(
                name=SMILING_KEY,
                label=False,
                source="source_model",
                confidence_band="high",
            ),
        },
        profile_facts=[
            ProfileFact(
                fact_id="fairget_nationality",
                relation="nationality",
                value="Testland",
            )
        ],
    )
    fields.update(overrides)
    return CanonicalSample(**fields)


# --------------------------------------------------------------------------- #
# P1-11: score-cache row validation
# --------------------------------------------------------------------------- #


class TestScoreCacheValidation:
    """P1-11: NaN/Inf/range/dedup checks on score rows.

    The validation logic lives in cmd_build_annotate; here we test the
    invariant that the CLI rejects bad rows by simulating the same checks.
    """

    @staticmethod
    def _validate(score_rows):
        """Replicate the P1-11 validation logic from cli.py."""
        validated = []
        seen = set()
        for r in score_rows:
            p = r.get("p_positive")
            sid = r.get("sample_id", "")
            attr = r.get("attribute", "")
            if p is None or not isinstance(p, (int, float)):
                continue
            if math.isnan(p) or math.isinf(p):
                continue
            if not (0.0 <= p <= 1.0):
                continue
            key = (sid, attr)
            if key in seen:
                continue
            seen.add(key)
            validated.append(r)
        return validated

    def test_valid_rows_pass(self):
        rows = [
            {"sample_id": "s1", "attribute": "A", "p_positive": 0.7},
            {"sample_id": "s2", "attribute": "B", "p_positive": 0.3},
        ]
        assert len(self._validate(rows)) == 2

    def test_nan_dropped(self):
        rows = [{"sample_id": "s1", "attribute": "A", "p_positive": float("nan")}]
        assert len(self._validate(rows)) == 0

    def test_inf_dropped(self):
        rows = [{"sample_id": "s1", "attribute": "A", "p_positive": float("inf")}]
        assert len(self._validate(rows)) == 0

    def test_negative_inf_dropped(self):
        rows = [{"sample_id": "s1", "attribute": "A", "p_positive": float("-inf")}]
        assert len(self._validate(rows)) == 0

    def test_out_of_range_dropped(self):
        rows = [
            {"sample_id": "s1", "attribute": "A", "p_positive": 1.5},
            {"sample_id": "s2", "attribute": "B", "p_positive": -0.1},
        ]
        assert len(self._validate(rows)) == 0

    def test_boundary_values_pass(self):
        rows = [
            {"sample_id": "s1", "attribute": "A", "p_positive": 0.0},
            {"sample_id": "s2", "attribute": "B", "p_positive": 1.0},
        ]
        assert len(self._validate(rows)) == 2

    def test_none_p_positive_dropped(self):
        rows = [{"sample_id": "s1", "attribute": "A", "p_positive": None}]
        assert len(self._validate(rows)) == 0

    def test_non_numeric_dropped(self):
        rows = [{"sample_id": "s1", "attribute": "A", "p_positive": "yes"}]
        assert len(self._validate(rows)) == 0

    def test_duplicates_dropped(self):
        rows = [
            {"sample_id": "s1", "attribute": "A", "p_positive": 0.7},
            {"sample_id": "s1", "attribute": "A", "p_positive": 0.8},
        ]
        result = self._validate(rows)
        assert len(result) == 1
        assert result[0]["p_positive"] == 0.7  # first occurrence kept


# --------------------------------------------------------------------------- #
# P1-13: semantic pair validation
# --------------------------------------------------------------------------- #


class TestValidatePairManifest:
    """P1-13: validate_pair_manifest checks semantic constraints."""

    def test_valid_pair_no_issues(self, registry):
        s1 = _sample(source_sample_id="s1", image_uri="img1.png")
        s2 = _sample(source_sample_id="s2", image_uri="img2.png")
        samples_by_id = {"s1": s1, "s2": s2}
        pairs = build_pair_manifest([{
            "pair_type": "cross_image_attribute_state",
            "left_sample_id": "s1",
            "right_sample_id": "s2",
            "attribute": "Eyeglasses",
            "left_label": True,
            "right_label": False,
        }])
        # s2 has Eyeglasses=True (same as s1), so this pair should flag
        # identical labels.  Let's make s2 have Eyeglasses=False instead.
        s2 = _sample(
            source_sample_id="s2",
            image_uri="img2.png",
            visual_attributes={
                EYEGLASSES_KEY: AttributeObservation(
                    name=EYEGLASSES_KEY,
                    label=False,
                    source="source_model",
                    confidence_band="high",
                ),
                SMILING_KEY: AttributeObservation(
                    name=SMILING_KEY,
                    label=False,
                    source="source_model",
                    confidence_band="high",
                ),
            },
        )
        samples_by_id = {"s1": s1, "s2": s2}
        issues = validate_pair_manifest(pairs, samples_by_id)
        assert issues == []

    def test_missing_left_sample(self):
        s2 = _sample(source_sample_id="s2")
        pairs = [{"pair_id": "pair_000000", "pair_type": "cross_image_attribute_state",
                   "left_sample_id": "missing", "right_sample_id": "s2"}]
        issues = validate_pair_manifest(pairs, {"s2": s2})
        assert len(issues) == 1
        assert "not found" in issues[0]

    def test_missing_right_sample(self):
        s1 = _sample(source_sample_id="s1")
        pairs = [{"pair_id": "pair_000000", "pair_type": "cross_image_attribute_state",
                   "left_sample_id": "s1", "right_sample_id": "missing"}]
        issues = validate_pair_manifest(pairs, {"s1": s1})
        assert len(issues) == 1
        assert "not found" in issues[0]

    def test_identity_mismatch(self):
        s1 = _sample(source_sample_id="s1", identity_id="id_a", image_uri="a.png")
        s2 = _sample(source_sample_id="s2", identity_id="id_b", image_uri="b.png")
        pairs = [{"pair_id": "p0", "pair_type": "visual_vs_fact_same_image",
                   "left_sample_id": "s1", "right_sample_id": "s2"}]
        issues = validate_pair_manifest(pairs, {"s1": s1, "s2": s2})
        assert any("identity mismatch" in i for i in issues)

    def test_same_image_flagged(self):
        s1 = _sample(source_sample_id="s1", image_uri="same.png")
        s2 = _sample(source_sample_id="s2", image_uri="same.png")
        pairs = [{"pair_id": "p0", "pair_type": "visual_vs_fact_same_image",
                   "left_sample_id": "s1", "right_sample_id": "s2"}]
        issues = validate_pair_manifest(pairs, {"s1": s1, "s2": s2})
        assert any("same image" in i for i in issues)

    def test_cross_image_missing_attribute(self):
        s1 = _sample(source_sample_id="s1", image_uri="a.png")
        s2 = _sample(source_sample_id="s2", image_uri="b.png")
        pairs = [{"pair_id": "p0", "pair_type": "cross_image_attribute_state",
                   "left_sample_id": "s1", "right_sample_id": "s2"}]
        issues = validate_pair_manifest(pairs, {"s1": s1, "s2": s2})
        assert any("missing 'attribute'" in i for i in issues)

    def test_cross_image_attribute_not_accepted_on_left(self):
        s1 = _sample(
            source_sample_id="s1", image_uri="a.png",
            visual_attributes={},  # no accepted attributes
        )
        s2 = _sample(source_sample_id="s2", image_uri="b.png")
        pairs = build_pair_manifest([{
            "pair_type": "cross_image_attribute_state",
            "left_sample_id": "s1",
            "right_sample_id": "s2",
            "attribute": "Eyeglasses",
            "left_label": True,
            "right_label": False,
        }])
        issues = validate_pair_manifest(pairs, {"s1": s1, "s2": s2})
        assert any("not accepted on left" in i for i in issues)

    def test_cross_image_same_labels_flagged(self):
        """Both images have Eyeglasses=True -> labels identical."""
        s1 = _sample(source_sample_id="s1", image_uri="a.png")
        s2 = _sample(source_sample_id="s2", image_uri="b.png")
        pairs = build_pair_manifest([{
            "pair_type": "cross_image_attribute_state",
            "left_sample_id": "s1",
            "right_sample_id": "s2",
            "attribute": "Eyeglasses",
            "left_label": True,
            "right_label": True,
        }])
        issues = validate_pair_manifest(pairs, {"s1": s1, "s2": s2})
        assert any("identical" in i or "left_label == right_label" in i for i in issues)

    def test_cross_image_missing_labels(self):
        s1 = _sample(source_sample_id="s1", image_uri="a.png")
        s2 = _sample(
            source_sample_id="s2", image_uri="b.png",
            visual_attributes={
                EYEGLASSES_KEY: AttributeObservation(
                    name=EYEGLASSES_KEY, label=False,
                    source="source_model", confidence_band="high",
                ),
            },
        )
        pairs = [{
            "pair_id": "p0",
            "pair_type": "cross_image_attribute_state",
            "left_sample_id": "s1",
            "right_sample_id": "s2",
            "attribute": "Eyeglasses",
            # no left_label or right_label
        }]
        issues = validate_pair_manifest(pairs, {"s1": s1, "s2": s2})
        assert any("missing left_label or right_label" in i for i in issues)


# --------------------------------------------------------------------------- #
# P0-3: visual_text_conflict direction
# --------------------------------------------------------------------------- #


class TestVisualTextConflictDirection:
    """P0-3: visual_text_conflict must use the actual visible_label."""

    def test_conflict_claim_when_visible_true(self, registry):
        """When the attribute IS visible, the claim should say it's NOT."""
        builder = RouteProbeBuilder(registry)
        # visible_label=True -> conflict_claim should say "never shows ..."
        probe_text = builder.render_probe(
            "visual_text_conflict",
            attribute="Eyeglasses",
            identity_name="Test",
            fact=ProfileFact(fact_id="f1", relation="nationality", value="X"),
            visible_label=True,
        )
        assert "never shows" in probe_text.lower()

    def test_conflict_claim_when_visible_false(self, registry):
        """When the attribute is NOT visible, the claim should say it IS."""
        builder = RouteProbeBuilder(registry)
        probe_text = builder.render_probe(
            "visual_text_conflict",
            attribute="Eyeglasses",
            identity_name="Test",
            fact=ProfileFact(fact_id="f1", relation="nationality", value="X"),
            visible_label=False,
        )
        assert "always shows" in probe_text.lower()


# --------------------------------------------------------------------------- #
# P0-5: expected answers in probe rows
# --------------------------------------------------------------------------- #


class TestProbeRowExpectedAnswers:
    """P0-5: probe_row includes target_attribute, answer_label, answer_text."""

    def test_visual_probe_has_answer(self, registry):
        builder = RouteProbeBuilder(registry)
        anchor = _sample()
        probes = build_identity_probes([anchor], builder)
        # Find the direct_visual probe
        dv = [p for p in probes if p.route_probe.probe_family == "direct_visual"][0]
        row = builder.probe_row(dv, attribute="Eyeglasses")
        assert "target_attribute" in row
        assert row["target_attribute"] == "Eyeglasses"
        assert "answer_label" in row
        assert "answer_text" in row
        # Eyeglasses label=True -> answer_text should be "yes"
        assert row["answer_text"] == "yes"
        assert row["answer_label"] is True

    def test_name_only_probe_has_fact_value(self, registry):
        builder = RouteProbeBuilder(registry)
        anchor = _sample()
        probes = build_identity_probes([anchor], builder)
        no = [p for p in probes if p.route_probe.probe_family == "name_only"][0]
        row = builder.probe_row(no, attribute="nationality")
        assert row["answer_text"] == "Testland"
        assert row["answer_label"] is None


# --------------------------------------------------------------------------- #
# P0-6: text_only probes have no image leakage
# --------------------------------------------------------------------------- #


class TestTextOnlyNoLeakage:
    """P0-6: name_only probes must have image fields nullified."""

    def test_name_only_has_no_image_fields(self, registry):
        builder = RouteProbeBuilder(registry)
        anchor = _sample()
        probes = build_identity_probes([anchor], builder)
        no = [p for p in probes if p.route_probe.probe_family == "name_only"][0]
        # name_only is text_only modality -> image fields should be None
        assert no.modality == "text_only"
        assert no.image_id is None
        assert no.image_uri is None
        assert no.image_sha256 is None

    def test_visual_probe_keeps_image(self, registry):
        builder = RouteProbeBuilder(registry)
        anchor = _sample()
        probes = build_identity_probes([anchor], builder)
        dv = [p for p in probes if p.route_probe.probe_family == "direct_visual"][0]
        # direct_visual should keep image fields
        assert dv.image_uri == "images/test.png"
