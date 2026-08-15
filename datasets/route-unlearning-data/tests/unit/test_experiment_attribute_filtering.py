"""P0-3: experiment-attribute subset filtering in route-probe generation.

Verifies that ``build_identity_probes`` respects the
``experiment_attributes`` parameter so that excluded attributes
(e.g. Wearing_Hat, Wearing_Necktie, Sideburns) cannot become route
probe targets.
"""

from __future__ import annotations

import pytest

from route_data.build.conflict_generation import (
    ConflictError,
    RouteProbeBuilder,
    assign_balanced_route_attributes,
    build_identity_probes,
)
from route_data.config import PromptsConfig
from route_data.data.schemas import (
    AttributeObservation,
    CanonicalSample,
    ProfileFact,
    Provenance,
)
from route_data.prompts.registry import PromptRegistry

CELEBA_PREFIX = "extended_attributes.celeba40."


@pytest.fixture()
def registry(repo_root) -> PromptRegistry:
    prompts = PromptsConfig(
        binary=str(repo_root / "configs/prompts/celeba_binary_v1.yaml"),
        grouped=str(repo_root / "configs/prompts/celeba_grouped_json_v1.yaml"),
        route_conflict=str(repo_root / "configs/prompts/route_conflict_v1.yaml"),
    )
    return PromptRegistry(prompts)


def _make_attr(name: str, label: bool) -> AttributeObservation:
    key = f"{CELEBA_PREFIX}{name}"
    return AttributeObservation(
        name=key,
        label=label,
        source="source_model",
        confidence_band="high",
    )


def _anchor_with_attrs(*attr_names: str) -> CanonicalSample:
    """Create an anchor sample with the given attributes (all label=True)."""
    va = {}
    for name in attr_names:
        obs = _make_attr(name, True)
        va[obs.name] = obs
    return CanonicalSample(
        benchmark="fairget",
        source_sample_id="test_s1",
        identity_id="test_identity",
        identity_name="Test Person",
        provenance=Provenance(source_dataset="fairget"),
        image_uri="images/test.png",
        modality="image_text",
        visual_attributes=va,
        profile_facts=[
            ProfileFact(
                fact_id="fairget_nationality",
                relation="nationality",
                value="Testland",
            )
        ],
    )


class TestExperimentAttributeFiltering:
    """Verify that experiment_attributes restricts target selection."""

    def test_excluded_attribute_not_chosen_when_included_available(self, registry):
        """If identity has Wearing_Hat + Bald, and only Bald is in the
        experiment subset, the target must be Bald."""
        anchor = _anchor_with_attrs("Wearing_Hat", "Bald")
        builder = RouteProbeBuilder(registry)
        experiment = {"Bald"}
        probes = build_identity_probes(
            [anchor], builder, experiment_attributes=experiment,
        )
        # All visual-attribute probes should target Bald, not Wearing_Hat.
        for p in probes:
            if p.route_probe.probe_family == "name_only":
                continue
            target = p.source_metadata.get("target_attribute")
            assert target == "Bald", f"Expected Bald, got {target}"

    def test_raises_when_only_excluded_attributes_available(self, registry):
        """If identity has only Wearing_Hat (excluded), and experiment
        subset does not include it, raise ConflictError."""
        anchor = _anchor_with_attrs("Wearing_Hat")
        builder = RouteProbeBuilder(registry)
        experiment = {"Bald", "Bangs", "Smiling"}
        with pytest.raises(ConflictError, match="no accepted visible attribute.*experiment subset"):
            build_identity_probes(
                [anchor], builder, experiment_attributes=experiment,
            )

    def test_none_experiment_attributes_uses_all(self, registry):
        """When experiment_attributes is None, all accepted attributes
        are eligible (backward-compatible behavior)."""
        anchor = _anchor_with_attrs("Wearing_Hat", "Bald")
        builder = RouteProbeBuilder(registry)
        # No experiment_attributes → Wearing_Hat wins because min() picks
        # alphabetically first ("Bald" < "Wearing_Hat"), so Bald wins.
        probes = build_identity_probes([anchor], builder)
        for p in probes:
            if p.route_probe.probe_family == "name_only":
                continue
            target = p.source_metadata.get("target_attribute")
            # Bald < Wearing_Hat alphabetically, so Bald is chosen.
            assert target == "Bald"

    def test_all_three_excluded_attributes_blocked(self, registry):
        """Wearing_Hat, Wearing_Necktie, Sideburns are all blocked when
        the experiment subset excludes them."""
        anchor = _anchor_with_attrs(
            "Wearing_Hat", "Wearing_Necktie", "Sideburns", "Smiling",
        )
        builder = RouteProbeBuilder(registry)
        experiment = {"Bald", "Bangs", "Blond_Hair", "Smiling"}
        probes = build_identity_probes(
            [anchor], builder, experiment_attributes=experiment,
        )
        for p in probes:
            if p.route_probe.probe_family == "name_only":
                continue
            target = p.source_metadata.get("target_attribute")
            assert target == "Smiling"
            assert target not in {"Wearing_Hat", "Wearing_Necktie", "Sideburns"}

    def test_fallback_to_second_sample_when_first_excluded(self, registry):
        """If the first sample has only excluded attributes but the second
        sample has an included attribute, the second sample becomes anchor."""
        s1 = _anchor_with_attrs("Wearing_Hat")
        # Override source_sample_id so s2 is distinct
        s2_fields = {
            "source_sample_id": "test_s2",
            "image_uri": "images/test2.png",
        }
        s2_va = {_make_attr("Bald", True).name: _make_attr("Bald", True)}
        s2 = CanonicalSample(
            benchmark="fairget",
            identity_id="test_identity",
            identity_name="Test Person",
            provenance=Provenance(source_dataset="fairget"),
            modality="image_text",
            visual_attributes=s2_va,
            profile_facts=[
                ProfileFact(
                    fact_id="fairget_nationality",
                    relation="nationality",
                    value="Testland",
                )
            ],
            **s2_fields,
        )
        builder = RouteProbeBuilder(registry)
        experiment = {"Bald"}
        probes = build_identity_probes(
            [s1, s2], builder, experiment_attributes=experiment,
        )
        for p in probes:
            if p.route_probe.probe_family == "name_only":
                continue
            target = p.source_metadata.get("target_attribute")
            assert target == "Bald"


class TestBalancedRouteAttributeAssignment:
    """P0-2: state-balanced route-attribute assignment tests."""

    def test_seeds_positive_and_negative_states(self):
        """If an attribute has both positive and negative eligible
        identities, both states must appear in the assignment."""
        eligible = {
            "id_001": {"Bald": True},
            "id_002": {"Bald": False},
            "id_003": {"Bald": False},
            "id_004": {"Bald": True},
        }
        _, stats = assign_balanced_route_attributes(
            eligible, {"Bald"}, target_quota=10,
        )
        assert stats["Bald"]["positive"] >= 1
        assert stats["Bald"]["negative"] >= 1

    def test_deterministic(self):
        """Running the same input twice produces identical output."""
        eligible = {
            "id_b": {"Bald": True, "Smiling": False},
            "id_a": {"Bald": False, "Smiling": True},
            "id_c": {"Bald": True, "Smiling": True},
        }
        a1, s1 = assign_balanced_route_attributes(
            eligible, {"Bald", "Smiling"},
        )
        a2, s2 = assign_balanced_route_attributes(
            eligible, {"Bald", "Smiling"},
        )
        assert a1 == a2
        assert s1 == s2

    def test_respects_experiment_subset(self):
        """Only experiment attributes appear in the assignment."""
        eligible = {
            "id_001": {"Bald": True, "Sideburns": True},
            "id_002": {"Bald": False, "Sideburns": False},
        }
        assignment, stats = assign_balanced_route_attributes(
            eligible, {"Bald"}, target_quota=10,
        )
        for attr in stats:
            assert attr == "Bald"
        for _attr in assignment.values():
            assert _attr == "Bald"

    def test_handles_attribute_with_only_one_state(self):
        """If only positive identities exist for an attribute, no
        negative is seeded (but positives are still assigned)."""
        eligible = {
            "id_001": {"Mustache": True},
            "id_002": {"Mustache": True},
        }
        _, stats = assign_balanced_route_attributes(
            eligible, {"Mustache"}, target_quota=10,
        )
        assert set(stats.keys()) == {"Mustache"}
        assert stats["Mustache"]["positive"] == 2
        assert stats["Mustache"]["negative"] == 0

    def test_preserves_target_quota(self):
        """No attribute exceeds the target quota."""
        eligible = {}
        for i in range(30):
            eligible[f"id_{i:03d}"] = {"Bald": i % 2 == 0, "Smiling": i % 2 == 1}
        assignment, _ = assign_balanced_route_attributes(
            eligible, {"Bald", "Smiling"}, target_quota=10,
        )
        bald_count = sum(1 for _a in assignment.values() if _a == "Bald")
        smiling_count = sum(1 for _a in assignment.values() if _a == "Smiling")
        assert bald_count <= 10
        assert smiling_count <= 10

    def test_never_assigns_ineligible_attribute(self):
        """An identity is never assigned to an attribute it doesn't have."""
        eligible = {
            "id_001": {"Bald": True},
            "id_002": {"Smiling": False},
            "id_003": {"Bald": False, "Smiling": True},
        }
        assignment, _ = assign_balanced_route_attributes(
            eligible, {"Bald", "Smiling"}, target_quota=10,
        )
        for iid, attr in assignment.items():
            assert attr in eligible[iid], (
                f"{iid} assigned to {attr} but is not eligible"
            )
