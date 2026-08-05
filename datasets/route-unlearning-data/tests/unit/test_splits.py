"""Forget/retain split generation and invariant checks (plan 13)."""

from __future__ import annotations

import pytest

from route_data.build.split_generation import (
    SplitBuilder,
    SplitError,
    SplitResult,
    SplitSpec,
    validate_split_invariants,
)
from route_data.data.schemas import (
    AttributeObservation,
    CanonicalSample,
    ProfileFact,
    Provenance,
)

EYEGLASSES_KEY = "extended_attributes.celeba40.Eyeglasses"
SMILING_KEY = "extended_attributes.celeba40.Smiling"


def _sample(
    sid: str,
    identity_id: str,
    *,
    facts: tuple[ProfileFact, ...] = (),
    attrs: dict[str, bool] | None = None,
    task_type: str = "visual_attribute",
    modality: str = "image_only",
    name: str | None = None,
) -> CanonicalSample:
    visual = {
        f"extended_attributes.celeba40.{attr}": AttributeObservation(
            name=f"extended_attributes.celeba40.{attr}",
            label=value,
            confidence_band="high",
        )
        for attr, value in (attrs or {}).items()
    }
    return CanonicalSample(
        benchmark="fairget",
        source_sample_id=sid,
        identity_id=identity_id,
        provenance=Provenance(source_dataset="fairget"),
        identity_name=name,
        image_uri=f"images/{sid}.png",
        visual_attributes=visual,
        profile_facts=list(facts),
        modality=modality,
        task_type=task_type,
    )


def _balanced_population(n_each: int = 8) -> list[CanonicalSample]:
    """Two identities, each with pos+neg glasses and smiling labels."""
    samples = []
    for identity in ("alpha", "beta"):
        for i in range(n_each):
            positive = i % 2 == 0
            samples.append(
                _sample(
                    f"{identity}_s{i}",
                    identity,
                    attrs={"Eyeglasses": positive, "Smiling": not positive},
                )
            )
    return samples


class TestSplitSpecValidation:
    def test_empty_name_rejected(self):
        with pytest.raises(SplitError, match="name"):
            SplitSpec(
                name="", forget_scope="identity", forget_identity_ids=("a",)
            ).validate()

    def test_unknown_scope_rejected(self):
        with pytest.raises(SplitError, match="forget_scope"):
            SplitSpec(name="x", forget_scope="everything").validate()

    def test_identity_scope_requires_ids(self):
        with pytest.raises(SplitError, match="forget_identity_ids"):
            SplitSpec(name="x", forget_scope="identity").validate()

    def test_identity_fact_scope_requires_fact_ids(self):
        with pytest.raises(SplitError, match="forget_fact_ids"):
            SplitSpec(name="x", forget_scope="identity_fact").validate()

    def test_global_attribute_scope_requires_attribute(self):
        with pytest.raises(SplitError, match="attribute"):
            SplitSpec(name="x", forget_scope="global_attribute").validate()

    def test_eval_fraction_bounds(self):
        with pytest.raises(SplitError, match="eval_fraction"):
            SplitSpec(
                name="x",
                forget_scope="identity",
                forget_identity_ids=("a",),
                eval_fraction=1.5,
            ).validate()


class TestAssignment:
    def test_identity_scope_forgets_matching_identity_only(self):
        samples = [
            _sample("a1", "alpha"),
            _sample("a2", "alpha"),
            _sample("b1", "beta"),
        ]
        spec = SplitSpec(
            name="id", forget_scope="identity", forget_identity_ids=("alpha",)
        )
        result = SplitBuilder(samples).build(spec)
        assert sorted(s.source_sample_id for s in result.forget) == ["a1", "a2"]
        retained = {s.source_sample_id for s in result.retain_train}
        retained |= {s.source_sample_id for s in result.retain_eval}
        assert retained == {"b1"}
        assert all(s.forget_scope == "identity" for s in result.forget)

    def test_retain_split_is_deterministic(self):
        samples = [_sample(f"s{i}", f"id{i % 3}") for i in range(20)]
        spec = SplitSpec(
            name="id",
            forget_scope="identity",
            forget_identity_ids=("id0",),
            eval_fraction=0.3,
        )
        first = SplitBuilder(samples).build(spec)
        second = SplitBuilder(samples).build(spec)
        assert [s.source_sample_id for s in first.retain_train] == [
            s.source_sample_id for s in second.retain_train
        ]
        assert [s.source_sample_id for s in first.retain_eval] == [
            s.source_sample_id for s in second.retain_eval
        ]

    def test_identity_fact_scope_requires_matching_fact_and_task(self):
        fact = ProfileFact(
            fact_id="fairget_nationality", relation="nationality", value="Fjordmark"
        )
        samples = [
            _sample("f1", "alpha", facts=(fact,), task_type="identity_fact"),
            _sample("f2", "beta", facts=(fact,), task_type="visual_attribute"),
            _sample("f3", "gamma", task_type="identity_fact"),
        ]
        spec = SplitSpec(
            name="facts",
            forget_scope="identity_fact",
            forget_fact_ids=("fairget_nationality",),
        )
        result = SplitBuilder(samples).build(spec)
        assert [s.source_sample_id for s in result.forget] == ["f1"]

    def test_visual_identity_link_scope(self):
        samples = [
            _sample("v1", "alpha", modality="image_text", name="Ava Alpha"),
            _sample("v2", "beta", modality="image_text", name=None),
            _sample("v3", "gamma", modality="image_only", name="Gamma"),
        ]
        spec = SplitSpec(
            name="link",
            forget_scope="visual_identity_link",
            forget_identity_ids=("alpha",),
        )
        result = SplitBuilder(samples).build(spec)
        assert [s.source_sample_id for s in result.forget] == ["v1"]

    def test_global_attribute_scope(self):
        samples = [
            _sample("g1", "alpha", attrs={"Eyeglasses": True}),
            _sample("g2", "beta", attrs={"Smiling": True}),
        ]
        spec = SplitSpec(
            name="global", forget_scope="global_attribute", attribute="Eyeglasses"
        )
        result = SplitBuilder(samples).build(spec)
        assert [s.source_sample_id for s in result.forget] == ["g1"]


class TestInvariants:
    def test_well_formed_split_has_no_issues(self):
        samples = _balanced_population()
        spec = SplitSpec(
            name="id",
            forget_scope="identity",
            forget_identity_ids=("alpha",),
            eval_fraction=0.01,
        )
        result = SplitBuilder(samples).build(spec)
        assert validate_split_invariants(result, strict=True) == []

    def test_leaked_forget_sample_is_flagged(self):
        leaked = _sample("a1", "alpha", attrs={"Eyeglasses": True})
        spec = SplitSpec(
            name="id", forget_scope="identity", forget_identity_ids=("alpha",)
        )
        result = SplitResult(
            spec=spec,
            forget=[leaked],
            retain_train=[leaked],
            retain_eval=[],
            unassigned=[],
        )
        issues = validate_split_invariants(result, strict=False)
        assert any("leaked" in issue for issue in issues)
        with pytest.raises(SplitError):
            validate_split_invariants(result, strict=True)
