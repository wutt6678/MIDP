"""Unit tests for run_pilot module (Commit 6)."""

from __future__ import annotations

import json
from pathlib import Path

from route_data.eval.run_pilot import (
    PilotRunner,
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


def _make_synthetic_data(tmp_path: Path, n_identities: int = 12) -> tuple[Path, Path]:
    """Create synthetic baseline results and route probes."""
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

    return bl_path, rp_path


def _make_config(
    tmp_path: Path,
    bl_path: Path,
    rp_path: Path,
) -> Path:
    """Create a minimal experiment config YAML."""
    import yaml

    config = {
        "experiment_id": "test_pilot",
        "base_model": {
            "model_id": "test/model",
            "revision": "abc123",
            "dtype": "float32",
            "fingerprint_id": "test_fp",
        },
        "baseline": {
            "version": "test-v1",
            "manifest_path": str(tmp_path / "manifest.json"),
            "results_path": str(bl_path),
            "manifest_sha256": "",
            "results_sha256": "",
        },
        "dataset": {
            "route_probe_path": str(rp_path),
            "route_probe_sha256": "",
            "processed_dataset_path": "",
            "processed_dataset_sha256": "",
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
            "name": "lora_targeted_update",
            "hyperparameters": {
                "lora_rank": 8,
                "lora_alpha": 16,
                "learning_rate": 1e-4,
                "num_steps": 50,
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
        assert train_cfg["num_steps"] == 50

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

        results = runner.run_paired_analysis(post_path)
        assert "probe_deltas" in results
        assert "group_effects" in results
        assert (runner.output_dir / "analysis" / "paired_probe_deltas.jsonl").exists()

    def test_generate_validation_report(self, tmp_path: Path) -> None:
        bl_path, rp_path = _make_synthetic_data(tmp_path)
        cfg_path = _make_config(tmp_path, bl_path, rp_path)

        runner = PilotRunner(cfg_path, base_dir=tmp_path)
        runner.run_selection()

        report = runner.generate_validation_report(
            training_summary={"final_loss": 0.5, "steps": 50},
            post_eval_summary={"n_probes": 500, "n_errors": 0},
            analysis_results={
                "group_effects": {
                    "target": {"overall": {"mean": -1.5, "count": 10}},
                    "retain": {"overall": {"mean": -0.1, "count": 10}},
                    "control": {"overall": {"mean": 0.05, "count": 10}},
                },
                "preservation_report": {
                    "global_direct_visual": {"post_accuracy": 0.99},
                },
            },
        )

        assert report["stages"]["selection_completed"] is True
        assert report["stages"]["training_completed"] is True
        assert report["analysis_summary"]["target_mean_delta"] == -1.5
        assert report["gates"]["target_effect_visible"] is True
        assert report["gates"]["direct_visual_pass"] is True

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

        report = generate_pilot_validation_report(
            tmp_path,
            experiment_id="test",
            analysis_results={
                "group_effects": {
                    "target": {"overall": {"mean": -2.0}},
                    "retain": {"overall": {"mean": 0.0}},
                    "control": {"overall": {"mean": 0.1}},
                },
                "preservation_report": {
                    "global_direct_visual": {"post_accuracy": 0.98},
                },
            },
        )

        assert report["experiment_id"] == "test"
        assert report["stages"]["selection_completed"] is True
        assert report["analysis_summary"]["target_mean_delta"] == -2.0
        assert (tmp_path / "evidence" / "pilot_validation_report.json").exists()
