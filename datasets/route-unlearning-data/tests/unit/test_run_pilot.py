"""Unit tests for run_pilot module (Commit 6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from route_data.eval.run_pilot import (
    PilotRunner,
    _build_validation_gates,
    generate_pilot_validation_report,
    load_experiment_config,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

FAMILIES = [
    "direct_visual",
    "image_plus_name",
    "wrong_name",
    "visual_text_conflict",
    "name_only",
]

ATTRIBUTES = [
    "hair_color", "eye_color", "skin_tone", "face_shape",
    "age_group", "gender", "height", "build",
    "clothing_style", "accessories",
]


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file (test helper)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_synthetic_data(tmp_path: Path, n_identities: int = 12) -> tuple[Path, Path]:
    """Create synthetic baseline results, route probes, and baseline manifest."""
    bl_path = tmp_path / "baseline_results.jsonl"
    rp_path = tmp_path / "route_probes.jsonl"

    probe_counter = 0
    bl_rows = []
    rp_rows = []
    seen_ids: set[str] = set()

    for i in range(n_identities):
        iid = f"identity_{i:03d}"
        attr = ATTRIBUTES[i % len(ATTRIBUTES)]
        is_positive = i % 2 == 0

        for fam in FAMILIES:
            pid = f"probe_{probe_counter:04d}"
            probe_counter += 1

            if fam == "name_only":
                bl_rows.append({
                    "probe_id": pid, "sample_id": f"s_{pid}",
                    "identity_id": iid, "probe_family": fam,
                    "modality": "text", "question": f"What is {iid}'s {attr}?",
                    "target_attribute": attr, "answer_label": None,
                    "protocol_role": "train", "predicted_label": None,
                    "correct": False, "signed_answer_margin": None,
                    "token_overlap": 0.3,
                })
            else:
                margin = 5.0 + (i * 0.1)
                if fam == "image_plus_name":
                    margin -= 0.5
                elif fam == "wrong_name":
                    margin -= 0.3
                elif fam == "visual_text_conflict":
                    margin -= 1.0

                bl_rows.append({
                    "probe_id": pid, "sample_id": f"s_{pid}",
                    "identity_id": iid, "probe_family": fam,
                    "modality": "image+text",
                    "question": f"Does {iid} have {attr}={is_positive}?",
                    "target_attribute": attr, "answer_label": is_positive,
                    "protocol_role": "train",
                    "predicted_label": "Yes" if is_positive else "No",
                    "correct": True, "signed_answer_margin": margin,
                    "token_overlap": None,
                })

                if iid not in seen_ids:
                    seen_ids.add(iid)
                    rp_rows.append({
                        "probe_id": f"route_{iid}", "identity_id": iid,
                        "probe_family": fam, "target_attribute": attr,
                        "answer_label": is_positive,
                        "question": f"Does {iid} have {attr}?",
                        "image_uri": f"images/{iid}.jpg",
                    })

    with bl_path.open("w") as fh:
        for r in bl_rows:
            fh.write(json.dumps(r) + "\n")
    with rp_path.open("w") as fh:
        for r in rp_rows:
            fh.write(json.dumps(r) + "\n")

    # P0-5: Create the baseline manifest file (required by frozen SHA preflight).
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"version": "test-v1", "n_rows": len(bl_rows)}))

    return bl_path, rp_path


def _make_config(
    tmp_path: Path,
    bl_path: Path,
    rp_path: Path,
) -> Path:
    """Create a minimal experiment config YAML."""
    import yaml

    # P0-5: Compute actual SHAs for frozen input files.
    manifest_path = tmp_path / "manifest.json"
    config = {
        "experiment_id": "test_pilot",
        "base_model": {
            "model_id": "test/model",
            "revision": "abc123",
            "dtype": "float32",
            "fingerprint_id": "test_fp",
            "model_config_path": str(tmp_path / "model_config.yaml"),
        },
        "baseline": {
            "version": "test-v1",
            "manifest_path": str(manifest_path),
            "results_path": str(bl_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "results_sha256": _sha256_file(bl_path),
        },
        "dataset": {
            "route_probe_path": str(rp_path),
            "route_probe_sha256": _sha256_file(rp_path),
            "processed_dataset_path": "",
            "processed_dataset_sha256": "",
            "research_manifest_path": str(tmp_path / "research_manifest.json"),
            "freeze_verification_path": str(tmp_path / "freeze_verification.json"),
        },
        "selection": {
            "target_identity_count": 2,
            "retain_identity_count": 2,
            "control_identity_count": 2,
            "seed": 17,
            "matching_criteria": ["protocol_role"],
            "preferred_role": "train",
        },
        "method": {
            "name": "lora_targeted_candidate_margin",
            "hyperparameters": {
                "lora_rank": 8,
                "lora_alpha": 16,
                "learning_rate": 1e-4,
                "num_optimizer_steps": 50,
                "retain_weight": 0.1,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 4,
            },
        },
        "evaluation": {
            "reuse_frozen_500_probes": True,
            "primary_metric": "signed_answer_margin",
            "preserve_visual_capability": True,
            "direct_visual_accuracy_gate": 0.98,
        },
        "runtime": {
            "seed": 17,
            "output_dir": str(tmp_path / "pilot_v1"),
        },
    }

    cfg_path = tmp_path / "experiment.yaml"
    with cfg_path.open("w") as fh:
        yaml.dump(config, fh)
    return cfg_path


# --------------------------------------------------------------------------- #
# Tests: load_experiment_config
# --------------------------------------------------------------------------- #

class TestLoadConfig:
    def test_load_yaml(self, tmp_path: Path) -> None:
        import yaml
        cfg_path = tmp_path / "test.yaml"
        data = {"experiment_id": "test", "seed": 17}
        with cfg_path.open("w") as fh:
            yaml.dump(data, fh)
        loaded = load_experiment_config(cfg_path)
        assert loaded["experiment_id"] == "test"
        assert loaded["seed"] == 17


# --------------------------------------------------------------------------- #
# Tests: PilotRunner
# --------------------------------------------------------------------------- #

class TestPilotRunner:
    def test_run_selection(self, tmp_path: Path) -> None:
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)

        runner = PilotRunner(cfg_path, base_dir=tmp_path)
        manifest = runner.run_selection()

        assert len(manifest["target_identities"]) == 2
        assert len(manifest["retain_identities"]) == 2
        assert len(manifest["control_identities"]) == 2

        # No overlap
        all_ids = (
            set(manifest["target_identities"])
            | set(manifest["retain_identities"])
            | set(manifest["control_identities"])
        )
        assert len(all_ids) == 6

        # Manifest file exists
        assert (runner.output_dir / "selection" / "pilot_identity_selection.json").exists()

    def test_get_training_config(self, tmp_path: Path) -> None:
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)

        runner = PilotRunner(cfg_path, base_dir=tmp_path)
        manifest = runner.run_selection()
        train_cfg = runner.get_training_config(manifest)

        assert train_cfg["forget_identity_ids"] == manifest["target_identities"]
        assert train_cfg["retain_identity_ids"] == manifest["retain_identities"]
        assert train_cfg["lora_rank"] == 8
        # P0-12: Key renamed from num_steps to num_optimizer_steps.
        assert train_cfg["num_optimizer_steps"] == 50

    def test_get_post_eval_config(self, tmp_path: Path) -> None:
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)

        runner = PilotRunner(cfg_path, base_dir=tmp_path)
        runner.run_selection()

        post_cfg = runner.get_post_eval_config(
            checkpoint_path="/fake/checkpoint",
            checkpoint_name="step_050",
        )
        assert post_cfg["checkpoint_path"] == "/fake/checkpoint"
        assert post_cfg["checkpoint_name"] == "step_050"
        assert "model_id" in post_cfg

    def test_run_paired_analysis(self, tmp_path: Path) -> None:
        import copy

        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)

        runner = PilotRunner(cfg_path, base_dir=tmp_path)
        manifest = runner.run_selection()

        # Create fake post results (shift target margins)
        bl_rows = [json.loads(line) for line in bl_path.open()]
        target_ids = set(manifest["target_identities"])
        post_rows = []
        for row in bl_rows:
            new_row = copy.deepcopy(row)
            if (
                row["identity_id"] in target_ids
                and row["probe_family"] != "name_only"
                and row.get("signed_answer_margin") is not None
            ):
                new_row["signed_answer_margin"] = row["signed_answer_margin"] - 2.0
            post_rows.append(new_row)

        post_path = tmp_path / "post_results.jsonl"
        with post_path.open("w") as fh:
            for r in post_rows:
                fh.write(json.dumps(r) + "\n")

        # P0-10: Bypass 500-row pairing requirement for synthetic test data.
        from route_data.eval.paired_analysis import PairedAnalysis

        def _bypass_pairing(self_pa):
            report = {
                "pass": True,
                "baseline_rows": len(self_pa.baseline_rows),
                "post_rows": len(self_pa.post_rows),
                "baseline_unique_ids": len({r["probe_id"] for r in self_pa.baseline_rows}),
                "post_unique_ids": len({r["probe_id"] for r in self_pa.post_rows}),
                "missing": [],
                "extra": [],
                "duplicates_baseline": [],
                "duplicates_post": [],
            }
            self_pa._pairing_validation = report
            return report

        with patch.object(PairedAnalysis, "validate_pairing", _bypass_pairing):
            results = runner.run_paired_analysis(post_path)

        assert "probe_deltas" in results
        assert "group_effects" in results
        assert (runner.output_dir / "analysis" / "paired_probe_deltas.jsonl").exists()

    def test_generate_validation_report(self, tmp_path: Path) -> None:
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)

        runner = PilotRunner(cfg_path, base_dir=tmp_path)
        runner.run_selection()

        # P0-9/P0-14: analysis_results uses overall_visual and
        # pairing_validation keys.
        report = runner.generate_validation_report(
            training_summary={"final_loss": 0.5, "steps": 50},
            post_eval_summary={"n_probes": 500, "inference_errors": 0},
            analysis_results={
                "group_effects": {
                    "target": {"overall_visual": {"mean": -1.5, "count": 10}},
                    "retain": {"overall_visual": {"mean": -0.1, "count": 10}},
                    "control": {"overall_visual": {"mean": 0.05, "count": 10}},
                },
                "preservation_report": {
                    "global_direct_visual": {"post_accuracy": 0.99},
                    "target_direct_visual": {
                        "pre_accuracy": 1.0, "post_accuracy": 0.5,
                        "pre_mean_margin": 5.0, "post_mean_margin": 3.0,
                    },
                    "retain_direct_visual": {"post_accuracy": 0.99},
                    "control_direct_visual": {"post_accuracy": 0.99},
                    "untargeted_direct_visual": {"post_accuracy": 0.99},
                },
                "pairing_validation": {"pass": True},
            },
        )

        assert report["stages"]["selection_completed"] is True
        assert report["stages"]["training_completed"] is True
        # P0-9/P0-14: New analysis_summary keys.
        assert report["analysis_summary"]["target_visual_delta_mean"] == -1.5
        # P0-14: New gate keys.
        assert report["gates"]["target_exceeds_retain_plus_tolerance"] is True
        assert report["gates"]["global_direct_visual_accuracy_gate"] is True
        assert report["gates"]["zero_post_eval_inference_errors"] is True
        assert report["gates"]["exact_pairing"] is True

        report_path = runner.output_dir / "evidence" / "pilot_validation_report.json"
        assert report_path.exists()


# --------------------------------------------------------------------------- #
# Tests: generate_pilot_validation_report (standalone)
# --------------------------------------------------------------------------- #

class TestStandaloneReport:
    def test_basic_report(self, tmp_path: Path) -> None:
        # Create selection dir so stage check passes
        sel_dir = tmp_path / "selection"
        sel_dir.mkdir()
        (sel_dir / "pilot_identity_selection.json").write_text("{}")

        # P0-9/P0-14: Use overall_visual and pairing_validation keys.
        report = generate_pilot_validation_report(
            tmp_path,
            experiment_id="test",
            analysis_results={
                "group_effects": {
                    "target": {"overall_visual": {"mean": -2.0}},
                    "retain": {"overall_visual": {"mean": 0.0}},
                    "control": {"overall_visual": {"mean": 0.1}},
                },
                "preservation_report": {
                    "global_direct_visual": {"post_accuracy": 0.98},
                    "target_direct_visual": {
                        "pre_accuracy": 1.0, "post_accuracy": 0.5,
                        "pre_mean_margin": 5.0, "post_mean_margin": 3.0,
                    },
                    "retain_direct_visual": {"post_accuracy": 0.99},
                    "control_direct_visual": {"post_accuracy": 0.99},
                    "untargeted_direct_visual": {"post_accuracy": 0.99},
                },
                "pairing_validation": {"pass": True},
            },
            post_eval_summary={"inference_errors": 0},
        )

        assert report["experiment_id"] == "test"
        assert report["stages"]["selection_completed"] is True
        # P0-9/P0-14: New analysis_summary key.
        assert report["analysis_summary"]["target_visual_delta_mean"] == -2.0
        # P0-14: New gate keys.
        assert report["gates"]["target_exceeds_retain_plus_tolerance"] is True
        assert report["gates"]["exact_pairing"] is True
        assert (tmp_path / "evidence" / "pilot_validation_report.json").exists()


# --------------------------------------------------------------------------- #
# Tests: Fix 1/2/3 regression tests (A–E)
# --------------------------------------------------------------------------- #

class TestGroupSpecificGateWiring:
    """Fix 1: gates must read group-specific direct_visual fields."""

    def test_a_group_specific_gates_ignore_legacy_group_fields(self) -> None:
        """Test A: gate reads retain/control/untargeted direct_visual,
        NOT retain_group/control_group."""
        analysis_results = {
            "group_effects": {
                "target": {"overall_visual": {"mean": -2.0}},
                "retain": {"overall_visual": {"mean": 0.0}},
                "control": {"overall_visual": {"mean": 0.0}},
            },
            "preservation_report": {
                "global_direct_visual": {"post_accuracy": 1.00},
                "target_direct_visual": {"post_accuracy": 0.50},
                "retain_direct_visual": {"post_accuracy": 1.00},
                "control_direct_visual": {"post_accuracy": 1.00},
                "untargeted_direct_visual": {"post_accuracy": 0.99},
                # Deliberately misleading legacy fields — must be ignored.
                "retain_group": {"post_accuracy": 0.10},
                "control_group": {"post_accuracy": 0.20},
            },
            "pairing_validation": {"pass": True},
        }
        _summary, gates = _build_validation_gates(
            analysis_results, None,
        )
        assert gates["retain_direct_visual_accuracy_gate"] is True
        assert gates["control_direct_visual_accuracy_gate"] is True
        assert gates["untargeted_direct_visual_accuracy_gate"] is True
        assert gates["global_direct_visual_accuracy_gate"] is True

    def test_b_target_values_are_not_global_values(self) -> None:
        """Test B: analysis_summary must use target_direct_visual,
        NOT global_direct_visual, for target_* fields."""
        analysis_results = {
            "group_effects": {
                "target": {"overall_visual": {"mean": -1.0}},
                "retain": {"overall_visual": {"mean": 0.0}},
                "control": {"overall_visual": {"mean": 0.0}},
            },
            "preservation_report": {
                "global_direct_visual": {
                    "pre_accuracy": 1.0, "post_accuracy": 1.0,
                },
                "target_direct_visual": {
                    "pre_accuracy": 0.8, "post_accuracy": 0.6,
                },
                "retain_direct_visual": {},
                "control_direct_visual": {},
                "untargeted_direct_visual": {},
            },
            "pairing_validation": {},
        }
        summary, _ = _build_validation_gates(analysis_results, None)
        assert summary["target_direct_visual_pre_accuracy"] == 0.8
        assert summary["target_direct_visual_post_accuracy"] == 0.6

    def test_c_untargeted_failure_causes_gate_failure(self) -> None:
        """Test C: untargeted post_accuracy below threshold => gate False."""
        analysis_results = {
            "group_effects": {},
            "preservation_report": {
                "global_direct_visual": {},
                "target_direct_visual": {},
                "retain_direct_visual": {},
                "control_direct_visual": {},
                "untargeted_direct_visual": {"post_accuracy": 0.97},
            },
            "pairing_validation": {},
        }
        _, gates = _build_validation_gates(
            analysis_results, None, dv_gate=0.98,
        )
        assert gates["untargeted_direct_visual_accuracy_gate"] is False


class TestPairingGateNaming:
    """Fix 3: exact_pairing (not exact_500_pairing) + expected_n."""

    def test_d_smoke_pairing_semantics(self) -> None:
        """Test D: smoke pairing expected_n=10, gate named exact_pairing."""
        analysis_results = {
            "group_effects": {},
            "preservation_report": {
                "global_direct_visual": {},
                "target_direct_visual": {},
                "retain_direct_visual": {},
                "control_direct_visual": {},
                "untargeted_direct_visual": {},
            },
            "pairing_validation": {
                "pass": True,
                "expected_n": 10,
                "baseline_rows": 10,
                "post_rows": 10,
            },
        }
        summary, gates = _build_validation_gates(
            analysis_results, None,
        )
        assert gates["exact_pairing"] is True
        assert "exact_500_pairing" not in gates
        assert summary["pairing_expected_n"] == 10
        assert summary["pairing_baseline_rows"] == 10
        assert summary["pairing_post_rows"] == 10

    def test_e_full_pairing_semantics(self) -> None:
        """Test E: full pairing expected_n=500."""
        analysis_results = {
            "group_effects": {},
            "preservation_report": {
                "global_direct_visual": {},
                "target_direct_visual": {},
                "retain_direct_visual": {},
                "control_direct_visual": {},
                "untargeted_direct_visual": {},
            },
            "pairing_validation": {
                "pass": True,
                "expected_n": 500,
                "baseline_rows": 500,
                "post_rows": 500,
            },
        }
        summary, gates = _build_validation_gates(
            analysis_results, None,
        )
        assert gates["exact_pairing"] is True
        assert summary["pairing_expected_n"] == 500


class TestSharedHelperEquivalence:
    """Fix 2: class method and standalone must produce identical gates."""

    def test_identical_gates_for_identical_inputs(
        self, tmp_path: Path,
    ) -> None:
        """PilotRunner and standalone must agree on gate semantics."""
        # Setup for PilotRunner
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)
        runner = PilotRunner(cfg_path, base_dir=tmp_path)
        runner.run_selection()

        analysis_results = {
            "group_effects": {
                "target": {"overall_visual": {"mean": -1.5}},
                "retain": {"overall_visual": {"mean": -0.1}},
                "control": {"overall_visual": {"mean": 0.05}},
            },
            "preservation_report": {
                "global_direct_visual": {"post_accuracy": 0.99},
                "target_direct_visual": {
                    "pre_accuracy": 1.0, "post_accuracy": 0.5,
                    "pre_mean_margin": 5.0, "post_mean_margin": 3.0,
                },
                "retain_direct_visual": {"post_accuracy": 0.99},
                "control_direct_visual": {"post_accuracy": 0.99},
                "untargeted_direct_visual": {"post_accuracy": 0.99},
            },
            "pairing_validation": {"pass": True, "expected_n": 500},
        }
        post_eval_summary = {"inference_errors": 0}

        # Class method
        class_report = runner.generate_validation_report(
            training_summary={"final_loss": 0.5},
            post_eval_summary=post_eval_summary,
            analysis_results=analysis_results,
        )

        # Standalone
        sel_dir = tmp_path / "standalone_out" / "selection"
        sel_dir.mkdir(parents=True)
        (sel_dir / "pilot_identity_selection.json").write_text("{}")
        standalone_report = generate_pilot_validation_report(
            tmp_path / "standalone_out",
            experiment_id="test",
            training_summary={"final_loss": 0.5},
            post_eval_summary=post_eval_summary,
            analysis_results=analysis_results,
        )

        # Gate semantics must match
        for key in class_report["gates"]:
            assert class_report["gates"][key] == standalone_report["gates"][key], (
                f"Gate mismatch for {key}: "
                f"class={class_report['gates'][key]} vs "
                f"standalone={standalone_report['gates'][key]}"
            )
