"""E2C-v2 regression tests — validates all P0/P1 repairs.

Covers:
    P0-1: SHA-level leakage detection
    P0-2: source-render lineage isolation
    P0-3: VTC semantic correctness
    P0-4: calibration/experimental population isolation
    P0-5: calibration hard-stop logic
    P0-6: fail-closed identity audit
    P1-5: corrected failure taxonomy
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from route_data.e2c.dataset_builder import (
    build_condition_d,
    build_condition_m,
    build_condition_m_shuffled,
    validate_audit_completeness,
    validate_population_isolation,
)
from route_data.e2c.probe_builder import build_vtc_probes
from route_data.e2c.route_validation import (
    classify_failure,
    evaluate_calibration,
    validate_leakage,
    validate_vtc_semantics,
)
from route_data.e2c.synthetic_manifest import (
    assign_aliases,
    generate_identity_ids,
    generate_image_splits,
    generate_shuffled_mapping,
    generate_true_mapping,
    generate_wrong_name_pairs,
    get_prompt,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def identity_setup():
    exp_ids, cal_ids = generate_identity_ids()
    alias_map = assign_aliases(exp_ids, cal_ids)
    splits = generate_image_splits(exp_ids + cal_ids)
    true_mapping = generate_true_mapping(exp_ids)
    shuffled_mapping = generate_shuffled_mapping(true_mapping)
    return {
        "exp_ids": exp_ids,
        "cal_ids": cal_ids,
        "alias_map": alias_map,
        "splits": splits,
        "true_mapping": true_mapping,
        "shuffled_mapping": shuffled_mapping,
    }


def _make_image_splits_records(splits, all_ids, *, sha_suffix=""):
    """Helper to create image split records with optional SHA values."""
    records = []
    for id_ in all_ids:
        for split_name in ("train", "validation", "test"):
            for idx in splits[id_][split_name]:
                img_id = f"{id_}_img_{idx:03d}"
                records.append({
                    "identity_id": id_,
                    "image_id": img_id,
                    "image_path": f"e2c/data/processed/{id_}/{img_id}.png",
                    "image_sha256": f"sha_{id_}_{idx:03d}{sha_suffix}",
                    "source_render_id": f"render_{id_}_{idx:03d}",
                    "generation_type": "independent_render",
                    "augmentation_parent_id": None,
                    "split": split_name,
                })
    return records


def _make_train_records(identity_setup):
    """Build M/D/M_shuffled train records for testing."""
    s = identity_setup
    m = build_condition_m(
        alias_map=s["alias_map"],
        true_mapping=s["true_mapping"],
        image_splits=s["splits"],
        experimental_ids=s["exp_ids"],
        seed=17,
    )
    d = build_condition_d(
        alias_map=s["alias_map"],
        true_mapping=s["true_mapping"],
        image_splits=s["splits"],
        experimental_ids=s["exp_ids"],
        seed=17,
    )
    ms = build_condition_m_shuffled(
        alias_map=s["alias_map"],
        shuffled_mapping=s["shuffled_mapping"],
        image_splits=s["splits"],
        experimental_ids=s["exp_ids"],
        seed=17,
    )
    return m, d, ms


# --------------------------------------------------------------------------- #
# P0-1: SHA-level leakage detection
# --------------------------------------------------------------------------- #

class TestSHALeakage:
    def test_no_exact_sha_overlap_across_splits(self, identity_setup):
        s = identity_setup
        records = _make_image_splits_records(
            s["splits"], s["exp_ids"] + s["cal_ids"],
        )
        m, d, ms = _make_train_records(s)
        test_images = [r for r in records if r["split"] == "test"]

        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )

        # Should pass — unique SHAs per image
        report = validate_leakage(
            train_records={"M": m, "D": d, "M_shuffled": ms},
            test_images=test_images,
            image_splits=records,
            wn_pairs=wn_pairs,
            alias_map=s["alias_map"],
            true_mapping=s["true_mapping"],
            shuffled_mapping=s["shuffled_mapping"],
            experimental_ids=s["exp_ids"],
            calibration_ids=s["cal_ids"],
        )
        assert report["pass"]
        assert report["sha_overlap"]["train_test"] == []

    def test_same_sha_different_image_id_is_rejected(self, identity_setup):
        """Two different image_ids with same SHA across splits must fail."""
        s = identity_setup
        records = _make_image_splits_records(
            s["splits"], s["exp_ids"] + s["cal_ids"],
        )
        # Inject a SHA collision: train image and test image share same SHA
        duplicate_sha = "COLLISION_SHA_256"
        records[0]["image_sha256"] = duplicate_sha  # first train record
        # Find a test record and give it the same SHA
        for r in records:
            if r["split"] == "test":
                r["image_sha256"] = duplicate_sha
                break

        m, d, ms = _make_train_records(s)
        test_images = [r for r in records if r["split"] == "test"]
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )

        with pytest.raises(ValueError, match="SHA overlap"):
            validate_leakage(
                train_records={"M": m, "D": d, "M_shuffled": ms},
                test_images=test_images,
                image_splits=records,
                wn_pairs=wn_pairs,
                alias_map=s["alias_map"],
                true_mapping=s["true_mapping"],
                shuffled_mapping=s["shuffled_mapping"],
                experimental_ids=s["exp_ids"],
                calibration_ids=s["cal_ids"],
            )

    def test_missing_sha_is_hard_failure(self, identity_setup):
        """Empty image_sha256 must cause validation failure."""
        s = identity_setup
        records = _make_image_splits_records(
            s["splits"], s["exp_ids"] + s["cal_ids"],
        )
        # Set one SHA to empty
        records[0]["image_sha256"] = ""

        m, d, ms = _make_train_records(s)
        test_images = [r for r in records if r["split"] == "test"]
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )

        with pytest.raises(ValueError, match="image_sha256"):
            validate_leakage(
                train_records={"M": m, "D": d, "M_shuffled": ms},
                test_images=test_images,
                image_splits=records,
                wn_pairs=wn_pairs,
                alias_map=s["alias_map"],
                true_mapping=s["true_mapping"],
                shuffled_mapping=s["shuffled_mapping"],
                experimental_ids=s["exp_ids"],
                calibration_ids=s["cal_ids"],
            )

    def test_missing_source_render_id_is_hard_failure(self, identity_setup):
        """Empty source_render_id must cause validation failure."""
        s = identity_setup
        records = _make_image_splits_records(
            s["splits"], s["exp_ids"] + s["cal_ids"],
        )
        # Set one source_render_id to empty
        records[0]["source_render_id"] = ""

        m, d, ms = _make_train_records(s)
        test_images = [r for r in records if r["split"] == "test"]
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )

        with pytest.raises(ValueError, match="source_render_id"):
            validate_leakage(
                train_records={"M": m, "D": d, "M_shuffled": ms},
                test_images=test_images,
                image_splits=records,
                wn_pairs=wn_pairs,
                alias_map=s["alias_map"],
                true_mapping=s["true_mapping"],
                shuffled_mapping=s["shuffled_mapping"],
                experimental_ids=s["exp_ids"],
                calibration_ids=s["cal_ids"],
            )


# --------------------------------------------------------------------------- #
# P0-2: source-render lineage isolation
# --------------------------------------------------------------------------- #

class TestSourceRenderLineage:
    def test_source_render_family_does_not_cross_splits(self, identity_setup):
        s = identity_setup
        records = _make_image_splits_records(
            s["splits"], s["exp_ids"] + s["cal_ids"],
        )
        # All records have unique source_render_ids — should pass
        m, d, ms = _make_train_records(s)
        test_images = [r for r in records if r["split"] == "test"]
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )
        report = validate_leakage(
            train_records={"M": m, "D": d, "M_shuffled": ms},
            test_images=test_images,
            image_splits=records,
            wn_pairs=wn_pairs,
            alias_map=s["alias_map"],
            true_mapping=s["true_mapping"],
            shuffled_mapping=s["shuffled_mapping"],
            experimental_ids=s["exp_ids"],
            calibration_ids=s["cal_ids"],
        )
        assert report["pass"]

    def test_augmented_descendant_cannot_cross_splits(self, identity_setup):
        """Same source_render_id in train and test must fail."""
        s = identity_setup
        records = _make_image_splits_records(
            s["splits"], s["exp_ids"] + s["cal_ids"],
        )
        # Force a lineage violation: same source_render_id in train and test
        shared_render = "SHARED_RENDER_001"
        train_rec = None
        test_rec = None
        for r in records:
            if r["split"] == "train" and train_rec is None:
                r["source_render_id"] = shared_render
                train_rec = r
            elif r["split"] == "test" and test_rec is None:
                r["source_render_id"] = shared_render
                test_rec = r
            if train_rec and test_rec:
                break

        m, d, ms = _make_train_records(s)
        test_images = [r for r in records if r["split"] == "test"]
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )

        with pytest.raises(ValueError, match="source_render_id"):
            validate_leakage(
                train_records={"M": m, "D": d, "M_shuffled": ms},
                test_images=test_images,
                image_splits=records,
                wn_pairs=wn_pairs,
                alias_map=s["alias_map"],
                true_mapping=s["true_mapping"],
                shuffled_mapping=s["shuffled_mapping"],
                experimental_ids=s["exp_ids"],
                calibration_ids=s["cal_ids"],
            )


# --------------------------------------------------------------------------- #
# P0-3: VTC semantic correctness
# --------------------------------------------------------------------------- #

class TestVTCSemantics:
    def test_vtc_yes_claim_matches_metadata(self, identity_setup):
        """VTC probe with wrong_label=Yes must have rendered_claim_label=Yes."""
        s = identity_setup
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )
        test_images = [
            {"identity_id": id_, "image_id": f"{id_}_img_013",
             "image_path": "", "image_sha256": ""}
            for id_ in s["exp_ids"]
        ]
        probes = build_vtc_probes(
            test_images, wn_pairs, s["alias_map"], s["true_mapping"],
        )
        for p in probes:
            assert p["rendered_claim_label"] == p["presented_name_attribute"]

    def test_vtc_no_claim_matches_metadata(self, identity_setup):
        """VTC probe with wrong_label=No must have rendered_claim_label=No."""
        s = identity_setup
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )
        test_images = [
            {"identity_id": id_, "image_id": f"{id_}_img_013",
             "image_path": "", "image_sha256": ""}
            for id_ in s["exp_ids"]
        ]
        probes = build_vtc_probes(
            test_images, wn_pairs, s["alias_map"], s["true_mapping"],
        )
        no_probes = [p for p in probes if p["presented_name_attribute"] == "No"]
        assert len(no_probes) > 0  # at least some probes should have No
        for p in no_probes:
            assert p["rendered_claim_label"] == "No"

    def test_vtc_claim_is_opposite_to_true_image_fact(self, identity_setup):
        s = identity_setup
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )
        test_images = [
            {"identity_id": id_, "image_id": f"{id_}_img_013",
             "image_path": "", "image_sha256": ""}
            for id_ in s["exp_ids"]
        ]
        probes = build_vtc_probes(
            test_images, wn_pairs, s["alias_map"], s["true_mapping"],
        )
        for p in probes:
            assert p["presented_name_attribute"] != p["true_mapping"]

    def test_vtc_validator_rejects_text_metadata_mismatch(self):
        """Validator must reject probes where rendered != metadata."""
        bad_probes = [{
            "probe_id": "test_vtc_bad",
            "rendered_claim_label": "Yes",
            "presented_name_attribute": "No",
            "true_mapping": "No",
        }]
        report = validate_vtc_semantics(bad_probes)
        assert not report["pass"]

    def test_vtc_prompt_includes_wrong_label(self):
        """The VTC prompt template must accept wrong_label parameter."""
        prompt = get_prompt(
            "e2c_test_vtc_v1",
            wrong_alias="Bira",
            wrong_label="No",
        )
        assert "No" in prompt
        assert "Bira" in prompt
        assert "property-Z assignment" in prompt


# --------------------------------------------------------------------------- #
# P0-4: Calibration population isolation
# --------------------------------------------------------------------------- #

class TestPopulationIsolation:
    def test_calibration_dataset_contains_only_calibration_ids(self, identity_setup):
        s = identity_setup
        # Calibration needs a fact mapping for cal identities
        cal_mapping = {id_: "Yes" if i % 2 == 0 else "No"
                       for i, id_ in enumerate(s["cal_ids"])}
        cal_records = build_condition_m(
            alias_map=s["alias_map"],
            true_mapping=cal_mapping,
            image_splits=s["splits"],
            identity_ids=s["cal_ids"],
            seed=17,
        )
        cal_id_set = {r["identity_id"] for r in cal_records}
        assert cal_id_set == set(s["cal_ids"])

    def test_experimental_dataset_contains_only_experimental_ids(self, identity_setup):
        s = identity_setup
        exp_records = build_condition_m(
            alias_map=s["alias_map"],
            true_mapping=s["true_mapping"],
            image_splits=s["splits"],
            identity_ids=s["exp_ids"],
            seed=17,
        )
        exp_id_set = {r["identity_id"] for r in exp_records}
        assert exp_id_set == set(s["exp_ids"])

    def test_calibration_and_experimental_populations_disjoint(self, identity_setup):
        s = identity_setup
        cal_mapping = {id_: "Yes" if i % 2 == 0 else "No"
                       for i, id_ in enumerate(s["cal_ids"])}
        cal_records = build_condition_m(
            alias_map=s["alias_map"],
            true_mapping=cal_mapping,
            image_splits=s["splits"],
            identity_ids=s["cal_ids"],
            seed=17,
        )
        exp_records = build_condition_m(
            alias_map=s["alias_map"],
            true_mapping=s["true_mapping"],
            image_splits=s["splits"],
            identity_ids=s["exp_ids"],
            seed=17,
        )
        report = validate_population_isolation(
            calibration_records=cal_records,
            experimental_records=exp_records,
            calibration_ids=s["cal_ids"],
            experimental_ids=s["exp_ids"],
        )
        assert report["pass"]


# --------------------------------------------------------------------------- #
# P0-5: Calibration hard-stop
# --------------------------------------------------------------------------- #

class TestCalibrationHardStop:
    def test_all_pass_freezes_config(self):
        result = evaluate_calibration(
            i2n_calibration_m=0.95,
            name_calibration_m=0.95,
            dv_calibration_d=0.85,
            visual_control_results={
                "smiling": {"base_accuracy": 1.0, "trained_accuracy": 0.97},
            },
        )
        assert result["decision"] == "FREEZE_CANONICAL_CONFIG"

    def test_i2n_fail_stops(self):
        result = evaluate_calibration(
            i2n_calibration_m=0.50,
            name_calibration_m=0.95,
            dv_calibration_d=0.85,
            visual_control_results={},
        )
        assert result["decision"] == "STOP_REPAIR_OR_RECALIBRATE"
        assert not result["M1_pass"]

    def test_name_fail_stops(self):
        result = evaluate_calibration(
            i2n_calibration_m=0.95,
            name_calibration_m=0.50,
            dv_calibration_d=0.85,
            visual_control_results={},
        )
        assert result["decision"] == "STOP_REPAIR_OR_RECALIBRATE"
        assert not result["M2_pass"]

    def test_dv_fail_stops(self):
        result = evaluate_calibration(
            i2n_calibration_m=0.95,
            name_calibration_m=0.95,
            dv_calibration_d=0.50,
            visual_control_results={},
        )
        assert result["decision"] == "STOP_REPAIR_OR_RECALIBRATE"
        assert not result["D_pass"]

    def test_visual_control_fail_stops(self):
        result = evaluate_calibration(
            i2n_calibration_m=0.95,
            name_calibration_m=0.95,
            dv_calibration_d=0.85,
            visual_control_results={
                "smiling": {"base_accuracy": 1.0, "trained_accuracy": 0.80},
            },
        )
        assert result["decision"] == "STOP_REPAIR_OR_RECALIBRATE"
        assert not result["visual_control_pass"]


# --------------------------------------------------------------------------- #
# P0-6: Fail-closed identity audit
# --------------------------------------------------------------------------- #

class TestFailClosedAudit:
    def test_pending_audit_blocks_freeze(self):
        audit = [
            {"image_id": "img_000", "audit_status": "pending",
             "identity_consistent": None, "duplicate": None,
             "corrupted": None, "watermark": None,
             "alias_leakage": None, "target_fact_leakage": None},
        ]
        report = validate_audit_completeness(audit)
        assert not report["pass"]

    def test_failed_audit_blocks_freeze(self):
        audit = [
            {"image_id": "img_000", "audit_status": "fail",
             "identity_consistent": False, "duplicate": False,
             "corrupted": False, "watermark": False,
             "alias_leakage": False, "target_fact_leakage": False},
        ]
        report = validate_audit_completeness(audit)
        assert not report["pass"]

    def test_all_pass_audit_allows_freeze(self):
        audit = [
            {"image_id": "img_000", "audit_status": "pass",
             "identity_consistent": True, "duplicate": False,
             "corrupted": False, "watermark": False,
             "alias_leakage": False, "target_fact_leakage": False},
            {"image_id": "img_001", "audit_status": "pass",
             "identity_consistent": True, "duplicate": False,
             "corrupted": False, "watermark": False,
             "alias_leakage": False, "target_fact_leakage": False},
        ]
        report = validate_audit_completeness(audit)
        assert report["pass"]


# --------------------------------------------------------------------------- #
# P1-5: Corrected failure taxonomy
# --------------------------------------------------------------------------- #

class TestFailureTaxonomy:
    def test_c_not_called_composition_when_i2n_fails(self):
        """If I2N fails, failure C should be 'downstream DV failure'."""
        gates = {
            "R1": {"status": "FAIL", "value": 0.0},
            "R2": {"status": "FAIL", "value_m": 0.5},
            "R3": {"status": "FAIL", "value_m": 0.2, "value_d": 0.8},
            "R4": {"status": "PASS"},
            "R5": {"status": "PASS"},
            "R6": {"status": "PASS"},
            "R7": {"status": "PASS"},
        }
        failures = classify_failure(gates)
        c_failures = [f for f in failures if f["code"] == "C"]
        assert len(c_failures) == 1
        assert c_failures[0]["pattern"] == "Downstream DV failure"

    def test_c_not_called_composition_when_name_fails(self):
        """If NAME fails but I2N passes, still downstream DV failure."""
        gates = {
            "R1": {"status": "PASS", "value": 0.95},
            "R2": {"status": "FAIL", "value_m": 0.5},
            "R3": {"status": "FAIL", "value_m": 0.3, "value_d": 0.8},
            "R4": {"status": "PASS"},
            "R5": {"status": "PASS"},
            "R6": {"status": "PASS"},
            "R7": {"status": "PASS"},
        }
        failures = classify_failure(gates)
        c_failures = [f for f in failures if f["code"] == "C"]
        assert len(c_failures) == 1
        assert c_failures[0]["pattern"] == "Downstream DV failure"

    def test_c_called_composition_only_when_upstream_subtasks_pass(self):
        """If both I2N and NAME pass but DV fails, genuine composition failure."""
        gates = {
            "R1": {"status": "PASS", "value": 0.95},
            "R2": {"status": "PASS", "value_m": 0.95},
            "R3": {"status": "FAIL", "value_m": 0.5, "value_d": 0.85},
            "R4": {"status": "PASS"},
            "R5": {"status": "PASS"},
            "R6": {"status": "PASS"},
            "R7": {"status": "PASS"},
        }
        failures = classify_failure(gates)
        c_failures = [f for f in failures if f["code"] == "C"]
        assert len(c_failures) == 1
        assert c_failures[0]["pattern"] == "Composition failure"


# --------------------------------------------------------------------------- #
# Calibration mapping generation
# --------------------------------------------------------------------------- #

class TestCalibrationMappings:
    def test_calibration_mapping_balanced(self):
        from route_data.e2c.synthetic_manifest import generate_calibration_mapping
        cal_ids = ["syn_cal_00", "syn_cal_01"]
        true_map, shuf_map = generate_calibration_mapping(cal_ids)
        yes_count = sum(1 for v in true_map.values() if v == "Yes")
        no_count = sum(1 for v in true_map.values() if v == "No")
        assert yes_count == 1
        assert no_count == 1
        # Shuffled is opposite
        for id_ in cal_ids:
            assert shuf_map[id_] != true_map[id_]

    def test_calibration_mapping_deterministic(self):
        from route_data.e2c.synthetic_manifest import generate_calibration_mapping
        cal_ids = ["syn_cal_00", "syn_cal_01"]
        t1, s1 = generate_calibration_mapping(cal_ids, seed=17)
        t2, s2 = generate_calibration_mapping(cal_ids, seed=17)
        assert t1 == t2
        assert s1 == s2


# --------------------------------------------------------------------------- #
# Alias tokenization validation
# --------------------------------------------------------------------------- #

class TestAliasTokenization:
    def test_validate_complete_records(self):
        from route_data.e2c.synthetic_manifest import validate_alias_tokenization
        records = [
            {"identity_id": "syn_00", "alias": "Aven",
             "alias_token_ids": [32, 1002], "alias_token_count": 2,
             "tokenizer_id": "Qwen/Qwen3.5-9B", "tokenizer_revision": "abc"},
            {"identity_id": "syn_01", "alias": "Bira",
             "alias_token_ids": [33, 8565], "alias_token_count": 2,
             "tokenizer_id": "Qwen/Qwen3.5-9B", "tokenizer_revision": "abc"},
        ]
        report = validate_alias_tokenization(records)
        assert report["pass"]
        assert report["min_token_count"] == 2
        assert report["max_token_count"] == 2

    def test_validate_rejects_missing_token_ids(self):
        from route_data.e2c.synthetic_manifest import validate_alias_tokenization
        records = [
            {"identity_id": "syn_00", "alias": "Aven",
             "alias_token_ids": [], "alias_token_count": 0,
             "tokenizer_id": "Qwen/Qwen3.5-9B", "tokenizer_revision": "abc"},
        ]
        report = validate_alias_tokenization(records)
        assert not report["pass"]
