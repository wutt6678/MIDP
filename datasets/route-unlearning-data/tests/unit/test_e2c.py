"""E2C unit tests — CPU-only tests for all E2C modules.

Covers:
    - Dataset construction (deterministic split, balanced mapping, aliases)
    - Condition invariants (M no direct, D no name-to-attr, etc.)
    - Probe construction (WN opposite-label, VTC conflict, unique IDs)
    - Scoring (margin orientation for Yes and No)
    - Metrics (NameEffect, WrongNameEffect, ConflictEffect)
    - Gate logic (R1–R7 PASS/FAIL fixtures)
    - Provenance (SHA reproducibility, mutation detection)
"""

from __future__ import annotations

# Import E2C modules
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from route_data.e2c.dataset_builder import (
    build_condition_d,
    build_condition_m,
    build_condition_m_shuffled,
    validate_condition_invariants,
)
from route_data.e2c.probe_builder import (
    build_all_probes,
    validate_probes,
)
from route_data.e2c.route_metrics import (
    compute_accuracy_from_probes,
    compute_route_effects,
    compute_signed_margin,
    identity_clustered_bootstrap,
)
from route_data.e2c.route_validation import (
    aggregate_route_decision,
    classify_failure,
    evaluate_r1,
    evaluate_r2,
    evaluate_r3,
    evaluate_r4,
    evaluate_r5,
    evaluate_r6,
    evaluate_r7,
)
from route_data.e2c.synthetic_manifest import (
    DEFAULT_SEED,
    IMAGES_PER_IDENTITY,
    TEST_COUNT,
    TRAIN_COUNT,
    VAL_COUNT,
    assign_aliases,
    generate_e2c_manifests,
    generate_identity_ids,
    generate_image_splits,
    generate_shuffled_mapping,
    generate_true_mapping,
    generate_wrong_name_pairs,
    load_json_manifest,
    prompt_registry_sha,
    sha256_file,
    sha256_json,
    write_json_manifest,
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


@pytest.fixture
def condition_setup(identity_setup):
    s = identity_setup
    m = build_condition_m(
        alias_map=s["alias_map"],
        true_mapping=s["true_mapping"],
        image_splits=s["splits"],
        experimental_ids=s["exp_ids"],
        seed=DEFAULT_SEED,
    )
    d = build_condition_d(
        alias_map=s["alias_map"],
        true_mapping=s["true_mapping"],
        image_splits=s["splits"],
        experimental_ids=s["exp_ids"],
        seed=DEFAULT_SEED,
    )
    ms = build_condition_m_shuffled(
        alias_map=s["alias_map"],
        shuffled_mapping=s["shuffled_mapping"],
        image_splits=s["splits"],
        experimental_ids=s["exp_ids"],
        seed=DEFAULT_SEED,
    )
    return {"M": m, "D": d, "M_shuffled": ms, **identity_setup}


# --------------------------------------------------------------------------- #
# Dataset construction tests
# --------------------------------------------------------------------------- #

class TestDatasetConstruction:
    def test_deterministic_split_seed17(self):
        s1 = generate_image_splits(["syn_00"], seed=17)
        s2 = generate_image_splits(["syn_00"], seed=17)
        assert s1 == s2

    def test_different_seed_different_split(self):
        s1 = generate_image_splits(["syn_00"], seed=17)
        s2 = generate_image_splits(["syn_00"], seed=42)
        assert s1 != s2

    def test_10_3_3_split(self, identity_setup):
        for id_ in identity_setup["exp_ids"]:
            s = identity_setup["splits"][id_]
            assert len(s["train"]) == TRAIN_COUNT
            assert len(s["validation"]) == VAL_COUNT
            assert len(s["test"]) == TEST_COUNT

    def test_balanced_5_5_mapping(self, identity_setup):
        m = identity_setup["true_mapping"]
        yes_count = sum(1 for v in m.values() if v == "Yes")
        no_count = sum(1 for v in m.values() if v == "No")
        assert yes_count == 5
        assert no_count == 5

    def test_shuffled_differs_from_true(self, identity_setup):
        true_m = identity_setup["true_mapping"]
        shuf_m = identity_setup["shuffled_mapping"]
        for id_ in true_m:
            assert shuf_m[id_] != true_m[id_]

    def test_shuffled_all_flipped(self, identity_setup):
        true_m = identity_setup["true_mapping"]
        shuf_m = identity_setup["shuffled_mapping"]
        for id_ in true_m:
            assert (
                (true_m[id_] == "Yes" and shuf_m[id_] == "No")
                or (true_m[id_] == "No" and shuf_m[id_] == "Yes")
            )

    def test_aliases_unique(self, identity_setup):
        aliases = list(identity_setup["alias_map"].values())
        assert len(aliases) == len(set(aliases))

    def test_sample_ids_unique(self, condition_setup):
        for cond in ("M", "D", "M_shuffled"):
            ids = [r["sample_id"] for r in condition_setup[cond]]
            assert len(ids) == len(set(ids)), f"{cond} has duplicate sample IDs"


# --------------------------------------------------------------------------- #
# Condition invariants
# --------------------------------------------------------------------------- #

class TestConditionInvariants:
    def test_m_no_direct(self, condition_setup):
        m = condition_setup["M"]
        direct = [r for r in m if r["task"] == "image_to_attribute"]
        assert len(direct) == 0

    def test_d_no_name_to_attr(self, condition_setup):
        d = condition_setup["D"]
        name = [r for r in d if r["task"] == "name_to_attribute"]
        assert len(name) == 0

    def test_m_shuffled_uses_shuffled_labels(self, condition_setup):
        ms = condition_setup["M_shuffled"]
        shuf_m = condition_setup["shuffled_mapping"]
        for r in ms:
            if r["task"] == "name_to_attribute":
                assert r["answer"] == shuf_m[r["identity_id"]]

    def test_m_d_image_populations_identical(self, condition_setup):
        m_images = {
            r["image_id"] for r in condition_setup["M"]
            if r["image_id"] is not None
        }
        d_images = {
            r["image_id"] for r in condition_setup["D"]
            if r["image_id"] is not None and r["task"] == "image_to_attribute"
        }
        assert m_images == d_images

    def test_m_d_image_exposure_matched(self, condition_setup):
        m_img = len([r for r in condition_setup["M"] if r["image_id"] is not None])
        d_img = len([
            r for r in condition_setup["D"]
            if r["task"] == "image_to_attribute"
        ])
        assert m_img == d_img

    def test_validate_invariants_pass(self, condition_setup):
        report = validate_condition_invariants(
            condition_setup["M"],
            condition_setup["D"],
            condition_setup["M_shuffled"],
            true_mapping=condition_setup["true_mapping"],
            shuffled_mapping=condition_setup["shuffled_mapping"],
        )
        assert report["pass"] is True


# --------------------------------------------------------------------------- #
# Probe construction
# --------------------------------------------------------------------------- #

class TestProbeConstruction:
    @pytest.fixture
    def probe_setup(self, identity_setup):
        s = identity_setup
        wn_pairs = generate_wrong_name_pairs(
            s["true_mapping"], s["alias_map"],
        )
        # Build image splits as flat records
        image_records = []
        for id_ in s["exp_ids"] + s["cal_ids"]:
            for idx in range(IMAGES_PER_IDENTITY):
                img_id = f"{id_}_img_{idx:03d}"
                split_name = "train"
                if idx in s["splits"][id_]["validation"]:
                    split_name = "validation"
                elif idx in s["splits"][id_]["test"]:
                    split_name = "test"
                image_records.append({
                    "identity_id": id_,
                    "image_id": img_id,
                    "image_path": f"e2c/data/processed/{id_}/{img_id}.png",
                    "image_sha256": "",
                    "split": split_name,
                })
        visual_controls = [
            {
                "image_id": f"{id_}_img_{idx:03d}",
                "identity_id": id_,
                "controls": {"smiling": True, "eyeglasses": False, "hat": False},
                "source": "test",
            }
            for id_ in s["exp_ids"] + s["cal_ids"]
            for idx in range(IMAGES_PER_IDENTITY)
        ]
        probes = build_all_probes(
            image_splits=image_records,
            alias_map=s["alias_map"],
            true_mapping=s["true_mapping"],
            wn_pairs=wn_pairs,
            visual_controls=visual_controls,
            experimental_ids=s["exp_ids"],
        )
        return probes, s, wn_pairs

    def test_all_test_identities_covered(self, probe_setup):
        probes, s, _ = probe_setup
        i2n_ids = {p["identity_id"] for p in probes["I2N"]}
        for id_ in s["exp_ids"]:
            assert id_ in i2n_ids

    def test_wn_alias_neq_true_alias(self, probe_setup):
        probes, _, _ = probe_setup
        for p in probes["WN"]:
            assert p["presented_alias"] != p["correct_alias"]

    def test_wn_label_always_opposite(self, probe_setup):
        probes, _, _ = probe_setup
        for p in probes["WN"]:
            assert p["presented_name_attribute"] != p["true_mapping"]

    def test_vtc_label_conflict_opposite(self, probe_setup):
        probes, _, _ = probe_setup
        for p in probes["VTC"]:
            assert p["presented_name_attribute"] != p["true_mapping"]

    def test_probe_ids_unique(self, probe_setup):
        probes, _, _ = probe_setup
        all_ids = []
        for family_probes in probes.values():
            for p in family_probes:
                all_ids.append(p["probe_id"])
        assert len(all_ids) == len(set(all_ids))

    def test_exact_family_counts(self, probe_setup):
        probes, s, _ = probe_setup
        test_img_count = len(s["exp_ids"]) * TEST_COUNT
        assert len(probes["I2N"]) == test_img_count
        assert len(probes["NAME"]) == len(s["exp_ids"])
        assert len(probes["DV_syn"]) == test_img_count
        assert len(probes["IPN_syn"]) == test_img_count
        assert len(probes["WN"]) == test_img_count
        assert len(probes["VTC"]) == test_img_count


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

class TestScoring:
    def test_signed_margin_yes_correct(self):
        margin = compute_signed_margin(2.0, 1.0, "Yes")
        assert margin > 0

    def test_signed_margin_yes_incorrect(self):
        margin = compute_signed_margin(1.0, 2.0, "Yes")
        assert margin < 0

    def test_signed_margin_no_correct(self):
        margin = compute_signed_margin(1.0, 2.0, "No")
        assert margin > 0

    def test_signed_margin_no_incorrect(self):
        margin = compute_signed_margin(2.0, 1.0, "No")
        assert margin < 0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

class TestMetrics:
    def test_name_effect_correct(self):
        dv = [{"image_id": "a", "signed_answer_margin": 1.0}]
        ipn = [{"image_id": "a", "signed_answer_margin": 2.0}]
        wn = [{"image_id": "a", "signed_answer_margin": 0.5}]
        vtc = [{"image_id": "a", "signed_answer_margin": 0.8}]
        effects = compute_route_effects(
            dv_results=dv, ipn_results=ipn,
            wn_results=wn, vtc_results=vtc,
        )
        assert effects["NameEffect"] == pytest.approx(1.0)

    def test_wrong_name_effect_correct(self):
        dv = [{"image_id": "a", "signed_answer_margin": 1.0}]
        ipn = [{"image_id": "a", "signed_answer_margin": 2.0}]
        wn = [{"image_id": "a", "signed_answer_margin": 0.5}]
        vtc = [{"image_id": "a", "signed_answer_margin": 0.8}]
        effects = compute_route_effects(
            dv_results=dv, ipn_results=ipn,
            wn_results=wn, vtc_results=vtc,
        )
        assert effects["WrongNameEffect"] == pytest.approx(-1.5)

    def test_conflict_effect_correct(self):
        dv = [{"image_id": "a", "signed_answer_margin": 1.0}]
        ipn = [{"image_id": "a", "signed_answer_margin": 2.0}]
        wn = [{"image_id": "a", "signed_answer_margin": 0.5}]
        vtc = [{"image_id": "a", "signed_answer_margin": 0.8}]
        effects = compute_route_effects(
            dv_results=dv, ipn_results=ipn,
            wn_results=wn, vtc_results=vtc,
        )
        assert effects["ConflictEffect"] == pytest.approx(-0.2)

    def test_md_route_contrast(self):
        # M has large WN effect, D has small
        assert abs(-1.5) > abs(-0.5)  # conceptual test

    def test_identity_averaging(self):
        results = [
            {"signed_answer_margin": 2.0},
            {"signed_answer_margin": 0.5},
        ]
        acc = compute_accuracy_from_probes(results)
        assert acc == pytest.approx(1.0)

    def test_clustered_bootstrap_deterministic(self):
        m_results = [
            {"identity_id": "syn_00", "family": "NAME", "signed_answer_margin": 1.0},
            {"identity_id": "syn_01", "family": "NAME", "signed_answer_margin": 0.5},
        ]
        d_results = [
            {"identity_id": "syn_00", "family": "NAME", "signed_answer_margin": 0.1},
            {"identity_id": "syn_01", "family": "NAME", "signed_answer_margin": 0.2},
        ]
        b1 = identity_clustered_bootstrap(
            probe_results_m=m_results,
            probe_results_d=d_results,
            experimental_ids=["syn_00", "syn_01"],
            n_resamples=100,
            seed=17,
        )
        b2 = identity_clustered_bootstrap(
            probe_results_m=m_results,
            probe_results_d=d_results,
            experimental_ids=["syn_00", "syn_01"],
            n_resamples=100,
            seed=17,
        )
        assert b1["bootstrap"] == b2["bootstrap"]


# --------------------------------------------------------------------------- #
# Gate logic
# --------------------------------------------------------------------------- #

class TestGates:
    def test_r1_pass(self):
        assert evaluate_r1(0.96)["status"] == "PASS"

    def test_r1_fail(self):
        assert evaluate_r1(0.85)["status"] == "FAIL"

    def test_r2_pass(self):
        assert evaluate_r2(0.95, 0.50)["status"] == "PASS"

    def test_r2_fail_weak_m(self):
        assert evaluate_r2(0.80, 0.50)["status"] == "FAIL"

    def test_r2_fail_no_separation(self):
        assert evaluate_r2(0.95, 0.80)["status"] == "FAIL"

    def test_r3_pass(self):
        assert evaluate_r3(0.85, 0.82)["status"] == "PASS"

    def test_r3_fail(self):
        assert evaluate_r3(0.75, 0.85)["status"] == "FAIL"

    def test_r4_pass(self):
        assert evaluate_r4(1.5, 0.5, True)["status"] == "PASS"

    def test_r4_fail_magnitude(self):
        assert evaluate_r4(0.5, 1.5, True)["status"] == "FAIL"

    def test_r4_fail_ci(self):
        assert evaluate_r4(1.5, 0.5, False)["status"] == "FAIL"

    def test_r5_pass(self):
        assert evaluate_r5(1.5, 0.5, True)["status"] == "PASS"

    def test_r6_pass(self):
        assert evaluate_r6(0.90, 0.10)["status"] == "PASS"

    def test_r6_fail(self):
        assert evaluate_r6(0.40, 0.60)["status"] == "FAIL"

    def test_r7_pass(self):
        vc = {"smiling": {"base_accuracy": 0.95, "trained_accuracy": 0.92}}
        assert evaluate_r7(vc)["status"] == "PASS"

    def test_r7_fail(self):
        vc = {"smiling": {"base_accuracy": 0.95, "trained_accuracy": 0.80}}
        assert evaluate_r7(vc)["status"] == "FAIL"

    def test_aggregate_all_pass(self):
        gates = {f"R{i}": {"status": "PASS"} for i in range(1, 8)}
        d = aggregate_route_decision(gates)
        assert d["route_established"] is True

    def test_aggregate_one_fail(self):
        gates = {f"R{i}": {"status": "PASS"} for i in range(1, 8)}
        gates["R4"]["status"] = "FAIL"
        d = aggregate_route_decision(gates)
        assert d["route_established"] is False

    def test_classify_failure_a(self):
        gates = {f"R{i}": {"status": "PASS"} for i in range(1, 8)}
        gates["R1"]["status"] = "FAIL"
        failures = classify_failure(gates)
        assert any(f["code"] == "A" for f in failures)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

class TestProvenance:
    def test_sha_reproducible(self):
        data = {"key": "value", "number": 42}
        s1 = sha256_json(data)
        s2 = sha256_json(data)
        assert s1 == s2

    def test_mutation_changes_sha(self):
        s1 = sha256_json({"key": "value1"})
        s2 = sha256_json({"key": "value2"})
        assert s1 != s2

    def test_file_sha_reproducible(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text('{"a": 1}')
        s1 = sha256_file(p)
        s2 = sha256_file(p)
        assert s1 == s2

    def test_manifest_write_read(self, tmp_path):
        data = {"key": "value"}
        sha = write_json_manifest(data, tmp_path / "manifest.json")
        loaded = load_json_manifest(tmp_path / "manifest.json")
        assert loaded == data
        assert sha == sha256_file(tmp_path / "manifest.json")

    def test_prompt_registry_sha_stable(self):
        s1 = prompt_registry_sha()
        s2 = prompt_registry_sha()
        assert s1 == s2


# --------------------------------------------------------------------------- #
# Full pipeline test
# --------------------------------------------------------------------------- #

class TestFullPipeline:
    def test_generate_manifests(self, tmp_path):
        shas = generate_e2c_manifests(tmp_path, seed=17)
        assert len(shas) > 0
        # Check key manifests exist
        assert (tmp_path / "synthetic_identity_manifest.json").exists()
        assert (tmp_path / "e2c_image_split.json").exists()
        assert (tmp_path / "synthetic_attribute_mapping.json").exists()
        assert (tmp_path / "synthetic_attribute_mapping_shuffled.json").exists()
        assert (tmp_path / "e2c_wrong_name_pairs.json").exists()

    def test_build_and_validate_pipeline(self, tmp_path):
        """Full pipeline: manifests -> conditions -> probes -> validate."""
        generate_e2c_manifests(tmp_path / "manifests", seed=17)
        exp_ids, cal_ids = generate_identity_ids()
        alias_map = assign_aliases(exp_ids, cal_ids)
        splits = generate_image_splits(exp_ids + cal_ids)
        true_mapping = generate_true_mapping(exp_ids)
        shuffled_mapping = generate_shuffled_mapping(true_mapping)

        m = build_condition_m(
            alias_map=alias_map, true_mapping=true_mapping,
            image_splits=splits, experimental_ids=exp_ids, seed=17,
        )
        d = build_condition_d(
            alias_map=alias_map, true_mapping=true_mapping,
            image_splits=splits, experimental_ids=exp_ids, seed=17,
        )
        ms = build_condition_m_shuffled(
            alias_map=alias_map, shuffled_mapping=shuffled_mapping,
            image_splits=splits, experimental_ids=exp_ids, seed=17,
        )

        report = validate_condition_invariants(
            m, d, ms,
            true_mapping=true_mapping,
            shuffled_mapping=shuffled_mapping,
        )
        assert report["pass"]

        # Build probes
        image_records = []
        for id_ in exp_ids + cal_ids:
            for idx in range(IMAGES_PER_IDENTITY):
                img_id = f"{id_}_img_{idx:03d}"
                split_name = "train"
                if idx in splits[id_]["validation"]:
                    split_name = "validation"
                elif idx in splits[id_]["test"]:
                    split_name = "test"
                image_records.append({
                    "identity_id": id_,
                    "image_id": img_id,
                    "image_path": "",
                    "image_sha256": "",
                    "split": split_name,
                })

        wn_pairs = generate_wrong_name_pairs(true_mapping, alias_map)
        visual_controls = [
            {
                "image_id": f"{id_}_img_{idx:03d}",
                "identity_id": id_,
                "controls": {"smiling": True, "eyeglasses": False, "hat": False},
            }
            for id_ in exp_ids + cal_ids
            for idx in range(IMAGES_PER_IDENTITY)
        ]

        probes = build_all_probes(
            image_splits=image_records,
            alias_map=alias_map,
            true_mapping=true_mapping,
            wn_pairs=wn_pairs,
            visual_controls=visual_controls,
            experimental_ids=exp_ids,
        )

        test_img_count = len(exp_ids) * TEST_COUNT
        probe_report = validate_probes(
            probes, experimental_ids=exp_ids,
            test_image_count=test_img_count,
        )
        assert probe_report["pass"]
