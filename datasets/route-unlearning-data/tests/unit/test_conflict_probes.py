"""Route-conflict probe generation and pair manifest (plan 14)."""

from __future__ import annotations

import pytest

from route_data.build.conflict_generation import (
    ConflictError,
    RouteProbeBuilder,
    build_identity_probes,
    build_pair_manifest,
    make_pair,
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


@pytest.fixture()
def registry(repo_root) -> PromptRegistry:
    prompts = PromptsConfig(
        binary=str(repo_root / "configs/prompts/celeba_binary_v1.yaml"),
        grouped=str(repo_root / "configs/prompts/celeba_grouped_json_v1.yaml"),
        route_conflict=str(repo_root / "configs/prompts/route_conflict_v1.yaml"),
    )
    return PromptRegistry(prompts)


def _anchor(**overrides) -> CanonicalSample:
    fields = dict(
        benchmark="fairget",
        source_sample_id="golden_alpha_s1",
        identity_id="golden_alpha",
        identity_name="Ava Alpha",
        provenance=Provenance(source_dataset="fairget"),
        image_uri="images/gf_0001.png",
        modality="image_text",
        visual_attributes={
            EYEGLASSES_KEY: AttributeObservation(
                name=EYEGLASSES_KEY,
                label=True,
                source="source_model",
                confidence_band="high",
            )
        },
        profile_facts=[
            ProfileFact(
                fact_id="fairget_nationality",
                relation="nationality",
                value="Fjordmark",
            )
        ],
    )
    fields.update(overrides)
    return CanonicalSample(**fields)


class TestBuildIdentityProbes:
    def test_single_sample_yields_four_families(self, registry):
        probes = build_identity_probes([_anchor()], RouteProbeBuilder(registry))
        families = [p.route_probe.probe_family for p in probes]
        assert families == [
            "direct_visual",
            "name_only",
            "image_plus_name",
            "visual_text_conflict",
        ]
        assert all(p.task_type == "route_probe" for p in probes)

    def test_two_samples_with_wrong_name_yield_six_families(self, registry):
        second = _anchor(source_sample_id="golden_alpha_s2", image_uri="images/gf_0002.png")
        probes = build_identity_probes(
            [_anchor(), second],
            RouteProbeBuilder(registry),
            wrong_identity_name="Zed Zero",
        )
        families = {p.route_probe.probe_family for p in probes}
        assert len(probes) == 6
        assert families == {
            "direct_visual",
            "name_only",
            "image_plus_name",
            "visual_text_conflict",
            "wrong_name",
            "cross_image",
        }

    def test_empty_samples_raise(self, registry):
        with pytest.raises(ConflictError, match="at least one sample"):
            build_identity_probes([], RouteProbeBuilder(registry))

    def test_no_accepted_visible_attribute_raises(self, registry):
        weak = _anchor(
            visual_attributes={
                EYEGLASSES_KEY: AttributeObservation(
                    name=EYEGLASSES_KEY,
                    label=True,
                    source="derived",
                    confidence_band="medium",
                )
            }
        )
        with pytest.raises(ConflictError, match="no accepted visible attribute"):
            build_identity_probes([weak], RouteProbeBuilder(registry))

    def test_no_profile_facts_raises(self, registry):
        with pytest.raises(ConflictError, match="no profile facts"):
            build_identity_probes(
                [_anchor(profile_facts=[])], RouteProbeBuilder(registry)
            )

    def test_builder_inherits_registry_hash(self, registry):
        builder = RouteProbeBuilder(registry)
        assert builder.registry_hash == registry.registry_hash()
        assert len(builder.registry_hash) == 16


class TestProbeRows:
    def test_probe_row_fields(self, registry):
        builder = RouteProbeBuilder(registry)
        probes = build_identity_probes([_anchor()], builder)
        row = builder.probe_row(probes[0], attribute="Eyeglasses")
        assert row["probe_id"].startswith("probe_")
        assert row["expected_evidence_source"] == "visual"
        assert row["registry_hash"] == registry.registry_hash()


class TestPairs:
    def test_make_pair_assigns_id_and_effect(self):
        pair = make_pair(
            "correct_name_vs_wrong_name", "left", "right", index=3
        )
        assert pair["pair_id"] == "pair_000003"
        assert pair["expected_route_effect"] == "identity_mediation"

    def test_unknown_pair_type_raises(self):
        with pytest.raises(ConflictError, match="Unknown pair_type"):
            make_pair("nonsense", "a", "b")

    def test_pair_manifest_ids_are_sequential(self):
        pairs = [
            {"pair_type": "correct_name_vs_wrong_name", "left_sample_id": "a", "right_sample_id": "b"},
            {"pair_type": "visual_vs_fact_same_image", "left_sample_id": "c", "right_sample_id": "d"},
        ]
        manifest = build_pair_manifest(pairs)
        assert [p["pair_id"] for p in manifest] == ["pair_000000", "pair_000001"]
