"""Canonical schema round-trips and fail-loud validation (plan 7)."""

from __future__ import annotations

import pytest

from route_data.data.schemas import (
    AttributeObservation,
    CanonicalSample,
    ProfileFact,
    Provenance,
    RouteProbe,
    SchemaError,
)


def _observation() -> AttributeObservation:
    return AttributeObservation(
        name="extended_attributes.celeba40.Glasses",
        label=True,
        score=0.93,
        source="source_model",
        confidence_band="high",
    ).validate()


def _fact() -> ProfileFact:
    return ProfileFact(
        fact_id="fairget_nationality",
        relation="nationality",
        value="Fjordmark",
        privacy_class="identity_textual",
        forgettable=True,
    ).validate()


def _sample() -> CanonicalSample:
    return CanonicalSample(
        benchmark="fairget",
        source_sample_id="gf_0001",
        identity_id="golden_alpha",
        identity_name="Ava Alpha",
        provenance=Provenance(source_dataset="fairget", adapter="fairget"),
        image_id="gf_0001",
        image_uri="images/gf_0001.png",
        visual_attributes={
            "extended_attributes.celeba40.Glasses": _observation(),
        },
        profile_facts=[_fact()],
        modality="image_text",
        task_type="visual_attribute",
        route_probe=RouteProbe(
            probe_family="direct_visual",
            expected_evidence_source="visual",
        ),
    ).validate()


class TestRoundTrips:
    def test_sample_round_trips_through_dict(self):
        sample = _sample()
        assert CanonicalSample.from_dict(sample.to_dict()) == sample

    def test_observation_round_trip(self):
        obs = _observation()
        assert AttributeObservation.from_dict(obs.to_dict()) == obs

    def test_fact_round_trip(self):
        fact = _fact()
        assert ProfileFact.from_dict(fact.to_dict()) == fact


class TestValidation:
    def test_empty_benchmark_rejected(self):
        sample = _sample()
        sample.benchmark = ""
        with pytest.raises(SchemaError, match="benchmark"):
            sample.validate()

    def test_bad_modality_rejected(self):
        sample = _sample()
        sample.modality = "audio"
        with pytest.raises(SchemaError, match="modality"):
            sample.validate()

    def test_bad_observation_source_rejected(self):
        with pytest.raises(SchemaError, match="source"):
            AttributeObservation(
                name="x", label=True, source="oracle"
            ).validate()

    def test_non_bool_label_rejected(self):
        with pytest.raises(SchemaError, match="label"):
            AttributeObservation(name="x", label="yes").validate()

    def test_unknown_probe_family_rejected(self):
        with pytest.raises(SchemaError, match="probe_family"):
            RouteProbe(probe_family="psychic").validate()

    def test_empty_source_dataset_rejected(self):
        with pytest.raises(SchemaError, match="source_dataset"):
            Provenance(source_dataset="").validate()
