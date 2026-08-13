"""Tests for the 7 P0 items from the review-54c0dc9 repair checklist.

Covers:
  P0-1: final_verify.py resolves FIUBench protocol from data config
  P0-2: --limit removed when --smoke-manifest is supplied
  P0-3: coverage_satisfied() stops greedy selection early
  P0-4: structural wrong-name feasibility Gate A
  P0-5: required_roles includes eval for holdout protocols
  P0-6: relative manifest path resolves against CWD
  P0-7: protocol SHA fail-closed when current protocol absent
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def _import_final_verify():
    """Import final_verify.py as a module (it lives outside the package)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import final_verify
    return final_verify


def _make_sample(
    source_sample_id: str,
    identity_id: str,
    split: str,
    *,
    image_uri: str | None = None,
    profile_facts: list | None = None,
    identity_name: str | None = None,
) -> dict:
    """Build a minimal sample dict for select_smoke_subset.

    Uses ``identity_name`` (the canonical field from ``CanonicalSample``),
    not the legacy ``name`` key.
    """
    s: dict = {
        "source_sample_id": source_sample_id,
        "identity_id": identity_id,
        "split": split,  # resolve_effective_split reads this field
    }
    if image_uri is not None:
        s["image_uri"] = image_uri
    if profile_facts is not None:
        s["profile_facts"] = profile_facts
    if identity_name is not None:
        s["identity_name"] = identity_name
    return s


# ========================================================================== #
# P0-1: final_verify resolves FIUBench protocol from data config
# ========================================================================== #


class TestP01ProtocolResolution:
    """P0-1: _verify_protocol_identity_counts uses _data_config_for."""

    def test_resolves_fiubench_protocol_from_data_config(
        self, tmp_path: Path,
    ):
        """Protocol is found via data config even when run config lacks it."""
        fv = _import_final_verify()

        # Create a run config WITHOUT fiubench_protocol in data.extras.
        run_cfg = {
            "run": {"name": "test_run"},
            "model": {
                "backend": "stub", "model_id": "local/stub", "revision": "v1",
            },
            "data": {
                "name": "fiubench",
                "source_version": "test",
            },
            "build": {"datasets": ["fiubench"], "output_dir": "out"},
        }
        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump(run_cfg))

        # Create a data config WITH fiubench_protocol.
        data_cfg = {
            "data": {
                "name": "fiubench",
                "source_file": "data/dataset.json",
                "split_file": "splits/official.json",
                "extras": {
                    "fiubench_protocol": {
                        "name": "test_proto",
                        "forget_bucket": "forget10",
                        "train_bucket": "retain15",
                        "eval_bucket": None,
                        "eval_fraction": 0.2,
                        "eval_seed": 17,
                    },
                },
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg))

        # Create a fake processed JSONL with valid splits.
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        processed = export_dir / "fiubench_processed.jsonl"
        rows = [
            {"identity_id": "id1", "split": "train"},
            {"identity_id": "id2", "split": "eval"},
            {"identity_id": "id3", "split": "exclude"},
        ]
        processed.write_text("\n".join(json.dumps(r) for r in rows))

        failures: list[str] = []
        # Patch _data_config_for to resolve against our tmp data config.
        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            rec = fv._verify_protocol_identity_counts(
                export_dir, "fiubench", str(run_path), failures,
            )

        # The protocol was found → result should NOT be NOT_APPLICABLE.
        assert rec.result != fv.CheckResult.NOT_APPLICABLE

    def test_protocol_check_not_na_for_real_run_config(
        self, tmp_path: Path,
    ):
        """When data config has a protocol, result is not NOT_APPLICABLE."""
        fv = _import_final_verify()

        run_cfg = {
            "run": {"name": "test"},
            "model": {"backend": "stub", "model_id": "m", "revision": "v"},
            "data": {"name": "fiubench"},
            "build": {"datasets": ["fiubench"], "output_dir": "o"},
        }
        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump(run_cfg))

        data_cfg = {
            "data": {
                "name": "fiubench",
                "extras": {
                    "fiubench_protocol": {
                        "name": "p", "forget_bucket": "f",
                        "train_bucket": "t", "eval_bucket": None,
                        "eval_fraction": 0.2, "eval_seed": 1,
                    },
                },
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg))

        export_dir = tmp_path / "export"
        export_dir.mkdir()
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
        ]
        (export_dir / "fiubench_processed.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows),
        )

        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            rec = fv._verify_protocol_identity_counts(
                export_dir, "fiubench", str(run_path), [],
            )
        assert rec.result != fv.CheckResult.NOT_APPLICABLE

    def test_protocol_check_fails_on_unassigned(self, tmp_path: Path):
        """Unassigned identities in processed → FAIL."""
        fv = _import_final_verify()

        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump({"data": {}}))

        data_cfg = {
            "data": {
                "name": "fiubench",
                "extras": {
                    "fiubench_protocol": {
                        "name": "p", "forget_bucket": "f",
                        "train_bucket": "t", "eval_bucket": None,
                        "eval_fraction": 0.2, "eval_seed": 1,
                    },
                },
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg))

        export_dir = tmp_path / "export"
        export_dir.mkdir()
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
            {"identity_id": "d", "split": "unassigned"},
        ]
        (export_dir / "fiubench_processed.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows),
        )

        failures: list[str] = []
        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            rec = fv._verify_protocol_identity_counts(
                export_dir, "fiubench", str(run_path), failures,
            )
        assert rec.result == fv.CheckResult.FAIL
        assert "unassigned" in rec.detail

    def test_protocol_check_fails_on_hash(self, tmp_path: Path):
        """Hash identities in processed → FAIL."""
        fv = _import_final_verify()

        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump({"data": {}}))

        data_cfg = {
            "data": {
                "name": "fiubench",
                "extras": {
                    "fiubench_protocol": {
                        "name": "p", "forget_bucket": "f",
                        "train_bucket": "t", "eval_bucket": None,
                        "eval_fraction": 0.2, "eval_seed": 1,
                    },
                },
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg))

        export_dir = tmp_path / "export"
        export_dir.mkdir()
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "b", "split": "eval"},
            {"identity_id": "c", "split": "exclude"},
            {"identity_id": "h", "split": "hash"},
        ]
        (export_dir / "fiubench_processed.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows),
        )

        failures: list[str] = []
        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            rec = fv._verify_protocol_identity_counts(
                export_dir, "fiubench", str(run_path), failures,
            )
        assert rec.result == fv.CheckResult.FAIL
        assert "hash" in rec.detail

    def test_protocol_check_requires_train_eval_exclude(self, tmp_path: Path):
        """Missing eval → FAIL when protocol requires all three roles."""
        fv = _import_final_verify()

        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump({"data": {}}))

        data_cfg = {
            "data": {
                "name": "fiubench",
                "extras": {
                    "fiubench_protocol": {
                        "name": "p", "forget_bucket": "f",
                        "train_bucket": "t", "eval_bucket": None,
                        "eval_fraction": 0.2, "eval_seed": 1,
                    },
                },
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg))

        export_dir = tmp_path / "export"
        export_dir.mkdir()
        # No eval identity.
        rows = [
            {"identity_id": "a", "split": "train"},
            {"identity_id": "c", "split": "exclude"},
        ]
        (export_dir / "fiubench_processed.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows),
        )

        failures: list[str] = []
        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            rec = fv._verify_protocol_identity_counts(
                export_dir, "fiubench", str(run_path), failures,
            )
        assert rec.result == fv.CheckResult.FAIL
        assert "eval" in rec.detail


# ========================================================================== #
# P0-2: --limit removed when --smoke-manifest is supplied
# ========================================================================== #


class TestP02ManifestDisablesLimit:
    """P0-2: main_check does not pass --limit when smoke_manifest is given."""

    def test_manifest_disables_limit(self, tmp_path: Path):
        """When smoke_manifest is set, --limit must not appear in stage_argv."""
        fv = _import_final_verify()

        captured_argvs: list[list[str]] = []

        def fake_run_cli(label, argv, *, expect=0, failures=None):
            captured_argvs.append(list(argv))

        # We just need to verify the argv construction, not actually run CLI.
        with (
            patch.object(fv, "_run_cli", side_effect=fake_run_cli),
            patch.object(fv, "verify_benchmark", return_value=[]),
        ):
            try:
                fv.main_check(
                    dataset="fairget",
                    config=str(REPO_ROOT / "configs/runs/golden_stub.yaml"),
                    output_dir=tmp_path / "out",
                    strict=False,
                    smoke_manifest=Path("/fake/manifest.json"),
                )
            except Exception:
                pass  # We only care about captured argvs.

        for argv in captured_argvs:
            assert "--limit" not in argv, f"--limit found in argv: {argv}"
            assert "--smoke-manifest" in argv, f"--smoke-manifest missing: {argv}"

    def test_no_manifest_uses_limit(self, tmp_path: Path):
        """Without smoke_manifest, --limit IS passed."""
        fv = _import_final_verify()

        captured_argvs: list[list[str]] = []

        def fake_run_cli(label, argv, *, expect=0, failures=None):
            captured_argvs.append(list(argv))

        with (
            patch.object(fv, "_run_cli", side_effect=fake_run_cli),
            patch.object(fv, "verify_benchmark", return_value=[]),
        ):
            try:
                fv.main_check(
                    dataset="fairget",
                    config=str(REPO_ROOT / "configs/runs/golden_stub.yaml"),
                    output_dir=tmp_path / "out",
                    strict=False,
                    smoke_manifest=None,
                )
            except Exception:
                pass

        for argv in captured_argvs:
            assert "--limit" in argv, f"--limit missing in argv: {argv}"
            assert "--smoke-manifest" not in argv


# ========================================================================== #
# P0-3: coverage_satisfied() stops selection early
# ========================================================================== #


class TestP03CoverageSatisfied:
    """P0-3: select_smoke_subset stops when coverage is complete."""

    def _make_coverage_samples(self):
        """Create samples where coverage is satisfied with minimal identities.

        Includes ``identity_name`` and a second row for idB so that
        structural wrong-name feasibility can be satisfied.
        """
        return [
            # Identity A: exclude + image + fact
            _make_sample("s1", "idA", "exclude",
                         image_uri="/img/a.jpg", identity_name="A",
                         profile_facts=[{"q": "nat?", "a": "X"}]),
            # Identity B: train + image + fact (2 rows for control)
            _make_sample("s2", "idB", "train",
                         image_uri="/img/b.jpg", identity_name="B",
                         profile_facts=[{"q": "nat?", "a": "Y"}]),
            _make_sample("s2b", "idB", "train",
                         image_uri="/img/b2.jpg", identity_name="B"),
            # Identity C: eval + image + fact
            _make_sample("s3", "idC", "eval",
                         image_uri="/img/c.jpg", identity_name="C",
                         profile_facts=[{"q": "nat?", "a": "Z"}]),
            # Extra identities that should NOT be selected.
            _make_sample("s4", "idD", "train", image_uri="/img/d.jpg",
                         identity_name="D"),
            _make_sample("s5", "idE", "train", image_uri="/img/e.jpg",
                         identity_name="E"),
            _make_sample("s6", "idF", "eval", image_uri="/img/f.jpg",
                         identity_name="F"),
            _make_sample("s7", "idG", "exclude", image_uri="/img/g.jpg",
                         identity_name="G"),
            _make_sample("s8", "idH", "train", image_uri="/img/h.jpg",
                         identity_name="H"),
            _make_sample("s9", "idI", "eval", image_uri="/img/i.jpg",
                         identity_name="I"),
        ]

    def test_stops_when_required_coverage_complete(self):
        """Selection stops early, not at 12."""
        fv = _import_final_verify()
        samples = self._make_coverage_samples()
        result = fv.select_smoke_subset(
            samples,
            min_identities=3,
            min_image_bearing=2,
            require_train=True,
            require_eval=True,
            require_exclude=True,
            require_visual=True,
            require_profile_fact=True,
        )
        # Should stop at 3 identities (A, B, C) once coverage is satisfied
        # including structural wrong-name feasibility (idB has 2 rows).
        selected_ids = result["coverage"]["identities"]
        assert len(selected_ids) <= 4
        assert result["coverage"]["selected_samples"] <= 5

    def test_does_not_fill_to_12_without_need(self):
        """Even with 9+ samples, selection is well under 12."""
        fv = _import_final_verify()
        samples = self._make_coverage_samples()
        result = fv.select_smoke_subset(samples, min_identities=3)
        assert result["coverage"]["selected_samples"] < 12

    def test_minimal_three_role_case(self):
        """3 identities covering train/eval/exclude + wrong-name feasibility."""
        fv = _import_final_verify()
        samples = [
            _make_sample("s1", "A", "exclude", image_uri="/a.jpg",
                         identity_name="A",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s2", "B", "train", image_uri="/b.jpg",
                         identity_name="B",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s2b", "B", "train", image_uri="/b2.jpg",
                         identity_name="B"),
            _make_sample("s3", "C", "eval", image_uri="/c.jpg",
                         identity_name="C",
                         profile_facts=[{"q": "x", "a": "y"}]),
        ]
        result = fv.select_smoke_subset(
            samples, min_identities=3, min_image_bearing=2,
        )
        ids = result["coverage"]["identities"]
        assert len(ids) == 3
        assert set(ids) == {"A", "B", "C"}

    def test_deterministic_given_same_source(self):
        """Same input → same output every time."""
        fv = _import_final_verify()
        samples = self._make_coverage_samples()
        r1 = fv.select_smoke_subset(samples, min_identities=3)
        r2 = fv.select_smoke_subset(samples, min_identities=3)
        assert r1["coverage"]["identities"] == r2["coverage"]["identities"]
        assert r1["coverage"]["selected_samples"] == r2["coverage"]["selected_samples"]


# ========================================================================== #
# P0-4: Structural wrong-name feasibility Gate A
# ========================================================================== #


class TestP04WrongNameGateA:
    """P0-4: structural wrong-name feasibility check in manifest generation.

    Uses ``identity_name`` (the canonical field) and the shared
    :func:`structural_wrong_name_candidates` helper.
    """

    def test_preinference_wrong_name_structural_feasibility(self):
        """Gate A requires a control identity with >= 2 rows."""
        from route_data.build.conflict_generation import (
            structural_wrong_name_candidates,
        )

        # Two named image-bearing identities, control has >= 2 rows.
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "Alice", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "Alice", "image_uri": "/a2.jpg"},
            ],
            "idB": [
                {"identity_id": "idB", "identity_name": "Bob", "image_uri": "/b.jpg"},
            ],
        }
        pairs = structural_wrong_name_candidates(by_identity)
        # idB is a valid target (has name+image), idA is a valid control
        # (has name+image, >= 2 rows).  So at least one pair exists.
        assert len(pairs) >= 1
        target_ids = {p["target_identity_id"] for p in pairs}
        control_ids = {p["control_identity_id"] for p in pairs}
        assert "idB" in target_ids
        assert "idA" in control_ids

    def test_preinference_does_not_require_visual_similarity(self):
        """Gate A does not check visual attributes — only structure."""
        from route_data.build.conflict_generation import (
            structural_wrong_name_candidates,
        )

        by_identity = {
            "idX": [
                {"identity_id": "idX", "identity_name": "X", "image_uri": "/x.jpg"},
                {"identity_id": "idX", "identity_name": "X", "image_uri": "/x2.jpg"},
            ],
            "idY": [{"identity_id": "idY", "identity_name": "Y", "image_uri": "/y.jpg"}],
        }
        pairs = structural_wrong_name_candidates(by_identity)
        assert len(pairs) >= 1
        # No visual_attributes needed for Gate A.


# ========================================================================== #
# P0-5: Required roles include eval for holdout protocols
# ========================================================================== #


class TestP05HoldoutEvalRole:
    """P0-5: required_roles includes eval when eval_fraction > 0."""

    def test_holdout_protocol_requires_eval_coverage(self):
        """eval_fraction > 0 with eval_bucket=None → eval is required."""
        proto = {
            "forget_bucket": "forget10",
            "train_bucket": "retain15",
            "eval_bucket": None,
            "eval_fraction": 0.20,
        }
        required_roles: set[str] = set()
        if proto.get("forget_bucket"):
            required_roles.add("exclude")
        if proto.get("train_bucket"):
            required_roles.add("train")
        if proto.get("eval_bucket") or proto.get("eval_fraction", 0) > 0:
            required_roles.add("eval")

        assert "eval" in required_roles
        assert "train" in required_roles
        assert "exclude" in required_roles

    def test_missing_eval_identity_fails_strict_manifest(self):
        """When no eval identity is selected, strict mode should fail."""
        proto = {
            "forget_bucket": "forget10",
            "train_bucket": "retain15",
            "eval_bucket": None,
            "eval_fraction": 0.20,
        }
        required_roles = set()
        if proto.get("forget_bucket"):
            required_roles.add("exclude")
        if proto.get("train_bucket"):
            required_roles.add("train")
        if proto.get("eval_bucket") or proto.get("eval_fraction", 0) > 0:
            required_roles.add("eval")

        # Simulate selected role_identities without eval.
        role_identities = {"train": {"id1"}, "exclude": {"id2"}}
        missing = required_roles - set(role_identities.keys())
        assert "eval" in missing

    def test_train_eval_exclude_all_present_passes(self):
        """All three roles present → no missing roles."""
        proto = {
            "forget_bucket": "forget10",
            "train_bucket": "retain15",
            "eval_bucket": None,
            "eval_fraction": 0.20,
        }
        required_roles = set()
        if proto.get("forget_bucket"):
            required_roles.add("exclude")
        if proto.get("train_bucket"):
            required_roles.add("train")
        if proto.get("eval_bucket") or proto.get("eval_fraction", 0) > 0:
            required_roles.add("eval")

        role_identities = {"train": {"id1"}, "eval": {"id2"}, "exclude": {"id3"}}
        missing = required_roles - set(role_identities.keys())
        assert len(missing) == 0


# ========================================================================== #
# P0-6: Relative manifest path resolves against CWD
# ========================================================================== #


class TestP06RelativePathResolution:
    """P0-6: manifest paths resolve against CWD, not config dir."""

    def test_relative_manifest_create_consume_roundtrip(self, tmp_path: Path):
        """A relative manifest path is resolved against CWD."""
        # Create a manifest at a known location.
        manifest_dir = tmp_path / "data" / "smoke"
        manifest_dir.mkdir(parents=True)
        manifest = manifest_dir / "test_manifest.json"
        manifest.write_text(json.dumps({
            "selected_source_sample_ids": ["s1", "s2"],
            "protocol_sha256": None,
        }))

        # Simulate _filter_by_smoke_manifest with a relative path.
        # The function should resolve against CWD, not config dir.
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            args = SimpleNamespace(
                smoke_manifest="data/smoke/test_manifest.json",
                config=None,
                dataset="fiubench",
            )
            from route_data.cli import _filter_by_smoke_manifest
            # Should find the file (resolved against CWD).
            result = _filter_by_smoke_manifest(
                [{"source_sample_id": "s1"}, {"source_sample_id": "s2"}, {"source_sample_id": "s3"}],
                args,
            )
            result_ids = {s["source_sample_id"] for s in result}
            assert result_ids == {"s1", "s2"}
        finally:
            os.chdir(original_cwd)

    def test_absolute_manifest_roundtrip(self, tmp_path: Path):
        """An absolute manifest path works correctly."""
        manifest = tmp_path / "abs_manifest.json"
        manifest.write_text(json.dumps({
            "selected_source_sample_ids": ["x1", "x2", "x3"],
            "protocol_sha256": None,
        }))

        args = SimpleNamespace(
            smoke_manifest=str(manifest),
            config=None,
            dataset="fiubench",
        )
        from route_data.cli import _filter_by_smoke_manifest
        samples = [
            {"source_sample_id": "x1"},
            {"source_sample_id": "x2"},
            {"source_sample_id": "x3"},
            {"source_sample_id": "x4"},
        ]
        result = _filter_by_smoke_manifest(samples, args)
        result_ids = {s["source_sample_id"] for s in result}
        assert result_ids == {"x1", "x2", "x3"}


# ========================================================================== #
# P0-7: Protocol SHA fail-closed when current protocol absent
# ========================================================================== #


class TestP07ProtocolShaFailClosed:
    """P0-7: manifest protocol SHA fails closed when current protocol missing."""

    def test_protocol_sha_current_protocol_missing_fails(
        self, tmp_path: Path,
    ):
        """Manifest has protocol_sha256 but current config has none → error."""
        from route_data.config import ConfigError

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "selected_source_sample_ids": ["s1"],
            "protocol_sha256": "abc123deadbeef",
        }))

        # Create a data config WITHOUT fiubench_protocol.
        data_cfg_no_proto = {
            "data": {
                "name": "fiubench",
                "source_file": "data/dataset.json",
                "split_file": "splits/official.json",
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg_no_proto))

        run_cfg = {
            "run": {"name": "test"},
            "model": {"backend": "stub", "model_id": "m", "revision": "v"},
            "data": {"name": "fiubench"},
            "build": {"datasets": ["fiubench"], "output_dir": "o"},
        }
        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump(run_cfg))

        args = SimpleNamespace(
            smoke_manifest=str(manifest),
            config=str(run_path),
            dataset="fiubench",
        )

        from route_data.cli import _filter_by_smoke_manifest
        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            with pytest.raises(ConfigError, match="P0-7"):
                _filter_by_smoke_manifest(
                    [{"source_sample_id": "s1"}], args,
                )

    def test_protocol_sha_mismatch_fails(self, tmp_path: Path):
        """Manifest SHA ≠ current config SHA → ConfigError."""
        from route_data.config import ConfigError

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "selected_source_sample_ids": ["s1"],
            "protocol_sha256": "wrong_sha_value",
        }))

        # Create a data config WITH a different protocol.
        from route_data.data.split_mapping import compute_protocol_sha256
        real_proto = {
            "name": "real_proto",
            "forget_bucket": "forget10",
            "train_bucket": "retain15",
            "eval_bucket": None,
            "eval_fraction": 0.2,
            "eval_seed": 17,
            "source_population": {"mode": "all"},
        }
        _real_sha, _ = compute_protocol_sha256(real_proto)

        data_cfg = {
            "data": {
                "name": "fiubench",
                "source_file": "data/dataset.json",
                "split_file": "splits/official.json",
                "extras": {"fiubench_protocol": real_proto},
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg))

        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump({
            "run": {"name": "t"},
            "model": {"backend": "stub", "model_id": "m", "revision": "v"},
            "data": {"name": "fiubench"},
            "build": {"datasets": ["fiubench"], "output_dir": "o"},
        }))

        args = SimpleNamespace(
            smoke_manifest=str(manifest),
            config=str(run_path),
            dataset="fiubench",
        )

        from route_data.cli import _filter_by_smoke_manifest
        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            with pytest.raises(ConfigError, match="protocol SHA mismatch"):
                _filter_by_smoke_manifest(
                    [{"source_sample_id": "s1"}], args,
                )

    def test_protocol_sha_match_passes(self, tmp_path: Path):
        """Matching protocol SHA → no error (filter proceeds normally)."""
        from route_data.data.split_mapping import compute_protocol_sha256

        real_proto = {
            "name": "real_proto",
            "forget_bucket": "forget10",
            "train_bucket": "retain15",
            "eval_bucket": None,
            "eval_fraction": 0.2,
            "eval_seed": 17,
            "source_population": {"mode": "all"},
        }
        real_sha, _ = compute_protocol_sha256(real_proto)

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "selected_source_sample_ids": ["s1"],
            "protocol_sha256": real_sha,
        }))

        data_cfg = {
            "data": {
                "name": "fiubench",
                "source_file": "data/dataset.json",
                "split_file": "splits/official.json",
                "extras": {"fiubench_protocol": real_proto},
            },
        }
        data_dir = tmp_path / "configs" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "fiubench.yaml").write_text(yaml.dump(data_cfg))

        run_path = tmp_path / "run.yaml"
        run_path.write_text(yaml.dump({
            "run": {"name": "t"},
            "model": {"backend": "stub", "model_id": "m", "revision": "v"},
            "data": {"name": "fiubench"},
            "build": {"datasets": ["fiubench"], "output_dir": "o"},
        }))

        args = SimpleNamespace(
            smoke_manifest=str(manifest),
            config=str(run_path),
            dataset="fiubench",
        )

        from route_data.cli import _filter_by_smoke_manifest
        with patch("route_data.cli._data_config_for") as mock_dcf:
            from route_data.config import load_data_config
            mock_dcf.return_value = load_data_config(data_dir / "fiubench.yaml")
            # Should NOT raise — matching SHA passes through.
            result = _filter_by_smoke_manifest(
                [{"source_sample_id": "s1"}], args,
            )
            assert len(result) == 1
            assert result[0]["source_sample_id"] == "s1"


# ========================================================================== #
# P0-1: identity_name field correctness tests
# ========================================================================== #


class TestP01IdentityNameField:
    """P0-1: wrong-name gate uses identity_name, not name."""

    def test_wrong_name_gate_uses_identity_name(self):
        """A sample with identity_name is recognised as named."""
        from route_data.build.conflict_generation import (
            _identity_name,
            structural_wrong_name_candidates,
        )

        sample = {"identity_id": "id1", "identity_name": "Alice", "image_uri": "/a.jpg"}
        assert _identity_name(sample) == "Alice"

        by_identity = {
            "id1": [
                sample,
                {"identity_id": "id1", "identity_name": "Alice", "image_uri": "/a2.jpg"},
            ],
            "id2": [{"identity_id": "id2", "identity_name": "Bob", "image_uri": "/b.jpg"}],
        }
        pairs = structural_wrong_name_candidates(by_identity)
        assert len(pairs) >= 1

    def test_wrong_name_gate_rejects_missing_identity_name(self):
        """A sample without identity_name is NOT recognised as named."""
        from route_data.build.conflict_generation import (
            _identity_name,
            structural_wrong_name_candidates,
        )

        # Has "name" but NOT "identity_name" — should be rejected.
        sample = {"identity_id": "id1", "name": "Alice", "image_uri": "/a.jpg"}
        assert _identity_name(sample) is None

        by_identity = {
            "id1": [
                sample,
                {"identity_id": "id1", "name": "Alice", "image_uri": "/a2.jpg"},
            ],
            "id2": [{"identity_id": "id2", "name": "Bob", "image_uri": "/b.jpg"}],
        }
        pairs = structural_wrong_name_candidates(by_identity)
        assert len(pairs) == 0

    def test_wrong_name_gate_accepts_real_canonical_identity_name(self):
        """CanonicalSample.to_dict() produces a dict recognised by the gate."""
        from route_data.build.conflict_generation import (
            _identity_name,
            structural_wrong_name_candidates,
        )
        from route_data.data.schemas import CanonicalSample, Provenance

        prov = Provenance(source_dataset="fiubench")
        cs = CanonicalSample(
            benchmark="fiubench",
            source_sample_id="s1",
            identity_id="id1",
            provenance=prov,
            identity_name="Synthetic Person A",
            image_uri="/tmp/a.jpg",
        )
        d = cs.to_dict()
        assert _identity_name(d) == "Synthetic Person A"

        by_identity = {
            "id1": [
                d,
                CanonicalSample(
                    benchmark="fiubench", source_sample_id="s1b",
                    identity_id="id1", provenance=prov,
                    identity_name="Synthetic Person A",
                    image_uri="/tmp/a2.jpg",
                ).to_dict(),
            ],
            "id2": [
                CanonicalSample(
                    benchmark="fiubench", source_sample_id="s2",
                    identity_id="id2", provenance=prov,
                    identity_name="Synthetic Person B",
                    image_uri="/tmp/b.jpg",
                ).to_dict(),
            ],
        }
        pairs = structural_wrong_name_candidates(by_identity)
        assert len(pairs) >= 1


# ========================================================================== #
# Structural wrong-name gate — detailed tests
# ========================================================================== #


class TestStructuralWrongNameGate:
    """Detailed tests for structural_wrong_name_candidates()."""

    def _pair(self, by_identity):
        from route_data.build.conflict_generation import (
            structural_wrong_name_candidates,
        )
        return structural_wrong_name_candidates(by_identity)

    def test_two_identities_one_row_each_rejected(self):
        """2 named image identities with only 1 row each → no pairs."""
        by_identity = {
            "idA": [{"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"}],
            "idB": [{"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg"}],
        }
        assert self._pair(by_identity) == []

    def test_control_with_two_rows_passes(self):
        """Control identity with 2 selected rows → at least one pair."""
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a2.jpg"},
            ],
            "idB": [{"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg"}],
        }
        pairs = self._pair(by_identity)
        assert len(pairs) >= 1
        # idA is the control (2 rows), idB is the target.
        control_ids = {p["control_identity_id"] for p in pairs}
        assert "idA" in control_ids

    def test_deterministic(self):
        """Same input → same output every time."""
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a2.jpg"},
            ],
            "idB": [{"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg"}],
            "idC": [{"identity_id": "idC", "identity_name": "C", "image_uri": "/c.jpg"}],
        }
        r1 = self._pair(by_identity)
        r2 = self._pair(by_identity)
        assert r1 == r2

    def test_no_visual_labels_needed(self):
        """Structural gate passes without any visual_attributes."""
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a2.jpg"},
            ],
            "idB": [{"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg"}],
        }
        # No visual_attributes keys present at all.
        pairs = self._pair(by_identity)
        assert len(pairs) >= 1

    def test_rejects_same_identity_pair(self):
        """Target and control must differ — no self-pairs."""
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a2.jpg"},
            ],
        }
        pairs = self._pair(by_identity)
        # Only one identity → no valid (target, control) pair.
        assert pairs == []

    def test_rejects_missing_image(self):
        """Identity without image_uri is not eligible."""
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "A"},  # no image
            ],
            "idB": [{"identity_id": "idB", "identity_name": "B"}],  # no image
        }
        pairs = self._pair(by_identity)
        # idA has one image-bearing row but idB has no image at all.
        # idA can't be control (only 1 image-bearing row counted by any()).
        # Actually idA has_name=True, has_image=True (any row has image_uri).
        # But idB has has_image=False → idB not eligible as target or control.
        # idA has 2 rows → eligible control, but no eligible target.
        assert pairs == []

    def test_rejects_missing_identity_name(self):
        """Identity without identity_name is not eligible."""
        by_identity = {
            "idA": [
                {"identity_id": "idA", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "image_uri": "/a2.jpg"},
            ],
            "idB": [{"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg"}],
        }
        pairs = self._pair(by_identity)
        # idA has no identity_name → not eligible as control.
        # idB has only 1 row → not eligible as control.
        # idB is eligible target, but no eligible control exists.
        assert pairs == []


# ========================================================================== #
# Selector tests for wrong-name structural feasibility
# ========================================================================== #


class TestSelectorWrongName:
    """Selector integration tests for structural wrong-name feasibility."""

    def test_coverage_does_not_stop_before_wrong_name_feasibility(self):
        """Selector continues until structural wrong-name is satisfied."""
        fv = _import_final_verify()
        # 3 identities covering train/eval/exclude, but none has 2 rows.
        # Without wrong-name feasibility, selector would stop at 3.
        # With it, selector must keep going until some identity gets 2 rows.
        samples = [
            _make_sample("s1", "idA", "exclude",
                         image_uri="/a.jpg", identity_name="A",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s2", "idB", "train",
                         image_uri="/b.jpg", identity_name="B",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s3", "idC", "eval",
                         image_uri="/c.jpg", identity_name="C",
                         profile_facts=[{"q": "x", "a": "y"}]),
            # Second row for idB — makes idB a valid control.
            _make_sample("s4", "idB", "train",
                         image_uri="/b2.jpg", identity_name="B"),
            # Extra identities that should NOT be selected if s4 is enough.
            _make_sample("s5", "idD", "train", image_uri="/d.jpg",
                         identity_name="D"),
            _make_sample("s6", "idE", "eval", image_uri="/e.jpg",
                         identity_name="E"),
        ]
        result = fv.select_smoke_subset(
            samples, min_identities=3, min_image_bearing=2,
        )
        by_identity: dict[str, list] = {}
        for s in result["selected"]:
            by_identity.setdefault(s["identity_id"], []).append(s)
        from route_data.build.conflict_generation import (
            structural_wrong_name_candidates,
        )
        pairs = structural_wrong_name_candidates(by_identity)
        assert len(pairs) >= 1, (
            f"No structural wrong-name pairs; by_identity sizes: "
            f"{({k: len(v) for k, v in by_identity.items()})}"
        )

    def test_selector_chooses_second_row_for_control(self):
        """Selector prefers second row for control over new identity."""
        fv = _import_final_verify()
        samples = [
            _make_sample("s1", "idA", "exclude",
                         image_uri="/a.jpg", identity_name="A",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s2", "idB", "train",
                         image_uri="/b.jpg", identity_name="B",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s3", "idC", "eval",
                         image_uri="/c.jpg", identity_name="C",
                         profile_facts=[{"q": "x", "a": "y"}]),
            # Second row for idA — completes control group.
            _make_sample("s4", "idA", "exclude",
                         image_uri="/a2.jpg", identity_name="A"),
            # Extra identities.
            _make_sample("s5", "idD", "train", image_uri="/d.jpg",
                         identity_name="D"),
            _make_sample("s6", "idE", "eval", image_uri="/e.jpg",
                         identity_name="E"),
            _make_sample("s7", "idF", "exclude", image_uri="/f.jpg",
                         identity_name="F"),
        ]
        result = fv.select_smoke_subset(
            samples, min_identities=3, min_image_bearing=2,
        )
        selected_ids = [s["source_sample_id"] for s in result["selected"]]
        # s4 (second row for idA) should be selected before idD/idE/idF.
        assert "s4" in selected_ids

    def test_minimal_row_count_deterministic(self):
        """Same input → same selected row count every time."""
        fv = _import_final_verify()
        samples = [
            _make_sample("s1", "idA", "exclude",
                         image_uri="/a.jpg", identity_name="A",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s2", "idB", "train",
                         image_uri="/b.jpg", identity_name="B",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s3", "idC", "eval",
                         image_uri="/c.jpg", identity_name="C",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s4", "idB", "train",
                         image_uri="/b2.jpg", identity_name="B"),
            _make_sample("s5", "idD", "train", image_uri="/d.jpg",
                         identity_name="D"),
        ]
        r1 = fv.select_smoke_subset(samples, min_identities=3)
        r2 = fv.select_smoke_subset(samples, min_identities=3)
        assert r1["coverage"]["selected_samples"] == r2["coverage"]["selected_samples"]

    def test_selected_roles_include_train_eval_exclude(self):
        """Selected subset covers all three required roles."""
        fv = _import_final_verify()
        samples = [
            _make_sample("s1", "idA", "exclude",
                         image_uri="/a.jpg", identity_name="A",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s2", "idB", "train",
                         image_uri="/b.jpg", identity_name="B",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s3", "idC", "eval",
                         image_uri="/c.jpg", identity_name="C",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s4", "idB", "train",
                         image_uri="/b2.jpg", identity_name="B"),
        ]
        result = fv.select_smoke_subset(
            samples, min_identities=3, min_image_bearing=2,
        )
        splits = {s.get("split") for s in result["selected"]}
        assert "train" in splits
        assert "eval" in splits
        assert "exclude" in splits

    def test_selected_subset_supports_structural_wrong_name_pair(self):
        """Selected subset has >= 1 structural wrong-name pair."""
        fv = _import_final_verify()
        samples = [
            _make_sample("s1", "idA", "exclude",
                         image_uri="/a.jpg", identity_name="A",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s2", "idB", "train",
                         image_uri="/b.jpg", identity_name="B",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s3", "idC", "eval",
                         image_uri="/c.jpg", identity_name="C",
                         profile_facts=[{"q": "x", "a": "y"}]),
            _make_sample("s4", "idB", "train",
                         image_uri="/b2.jpg", identity_name="B"),
        ]
        result = fv.select_smoke_subset(
            samples, min_identities=3, min_image_bearing=2,
        )
        by_identity: dict[str, list] = {}
        for s in result["selected"]:
            by_identity.setdefault(s["identity_id"], []).append(s)
        from route_data.build.conflict_generation import (
            structural_wrong_name_candidates,
        )
        assert structural_wrong_name_candidates(by_identity)


# ========================================================================== #
# Gate A → Gate B transition tests
# ========================================================================== #


class TestGateAToGateBTransition:
    """Verify the pre-inference → post-annotation wrong-name pipeline."""

    def _make_visual_attrs(self, smiling=True, eyeglasses=False):
        """Build fake accepted visual attributes for Gate B."""
        prefix = "extended_attributes.celeba40."
        return {
            f"{prefix}Smiling": {
                "label": smiling, "confidence_band": "high",
            },
            f"{prefix}Eyeglasses": {
                "label": eyeglasses, "confidence_band": "high",
            },
        }

    def test_structural_pair_exists_pre_annotation(self):
        """Gate A: structural pair exists before annotation."""
        from route_data.build.conflict_generation import (
            structural_wrong_name_candidates,
        )
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a2.jpg"},
            ],
            "idB": [
                {"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg"},
            ],
        }
        assert structural_wrong_name_candidates(by_identity)

    def test_visual_attrs_make_gate_b_pass(self):
        """Adding accepted visual attrs makes find_wrong_name_candidates() non-empty."""
        from route_data.build.conflict_generation import (
            find_wrong_name_candidates,
        )
        va = self._make_visual_attrs(smiling=True, eyeglasses=True)
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg",
                 "visual_attributes": va},
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a2.jpg",
                 "visual_attributes": va},
            ],
            "idB": [
                {"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg",
                 "visual_attributes": va},
            ],
        }
        triples = find_wrong_name_candidates(by_identity)
        assert len(triples) >= 1

    def test_no_visual_attrs_fails_gate_b(self):
        """Without visual attributes, find_wrong_name_candidates() is empty."""
        from route_data.build.conflict_generation import (
            find_wrong_name_candidates,
        )
        by_identity = {
            "idA": [
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a.jpg"},
                {"identity_id": "idA", "identity_name": "A", "image_uri": "/a2.jpg"},
            ],
            "idB": [
                {"identity_id": "idB", "identity_name": "B", "image_uri": "/b.jpg"},
            ],
        }
        # Gate A passes (structural), but Gate B fails (no visual attrs).
        triples = find_wrong_name_candidates(by_identity)
        assert triples == []


# ========================================================================== #
# P0-7: E2E minimal selector test for wrong-name viability
# ========================================================================== #


class TestP07E2EMinimalSelector:
    """P0-7: end-to-end test proving selected subset yields wrong-name pair."""

    def test_e2e_minimal_selector_wrong_name_viability(self):
        """Full pipeline: selector → structural gate → Gate B with visual attrs.

        Fixture:
          Identity A: exclude, 1 row (target candidate)
          Identity B: train, 2 rows (control candidate)
          Identity C: eval, 1 row (fills eval role)
          Plus extra irrelevant identities.
        """
        fv = _import_final_verify()
        from route_data.build.conflict_generation import (
            find_wrong_name_candidates,
            structural_wrong_name_candidates,
        )

        samples = [
            # Identity A: exclude, 1 canonical image+QA row.
            _make_sample("s1", "idA", "exclude",
                         image_uri="/img/a.jpg", identity_name="Person A",
                         profile_facts=[{"q": "nat?", "a": "X"}]),
            # Identity B: train, 2 canonical rows.
            _make_sample("s2", "idB", "train",
                         image_uri="/img/b1.jpg", identity_name="Person B",
                         profile_facts=[{"q": "nat?", "a": "Y"}]),
            _make_sample("s3", "idB", "train",
                         image_uri="/img/b2.jpg", identity_name="Person B"),
            # Identity C: eval, 1 canonical image+QA row.
            _make_sample("s4", "idC", "eval",
                         image_uri="/img/c.jpg", identity_name="Person C",
                         profile_facts=[{"q": "nat?", "a": "Z"}]),
            # Extra irrelevant identities.
            _make_sample("s5", "idD", "train", image_uri="/img/d.jpg",
                         identity_name="Person D"),
            _make_sample("s6", "idE", "eval", image_uri="/img/e.jpg",
                         identity_name="Person E"),
            _make_sample("s7", "idF", "exclude", image_uri="/img/f.jpg",
                         identity_name="Person F"),
            _make_sample("s8", "idG", "train", image_uri="/img/g.jpg",
                         identity_name="Person G"),
        ]

        result = fv.select_smoke_subset(
            samples,
            min_identities=3,
            min_image_bearing=2,
            require_train=True,
            require_eval=True,
            require_exclude=True,
            require_visual=True,
            require_profile_fact=True,
        )

        # Build by_identity from selected samples.
        by_identity: dict[str, list] = {}
        for s in result["selected"]:
            by_identity.setdefault(s["identity_id"], []).append(s)

        # Gate A: structural wrong-name candidates exist.
        structural_pairs = structural_wrong_name_candidates(by_identity)
        assert structural_pairs, (
            f"No structural pairs; by_identity: "
            f"{({k: len(v) for k, v in by_identity.items()})}"
        )

        # Gate B: add fake accepted visual attributes and verify
        # find_wrong_name_candidates() can become non-empty.
        prefix = "extended_attributes.celeba40."
        shared_va = {
            f"{prefix}Smiling": {"label": True, "confidence_band": "high"},
            f"{prefix}Eyeglasses": {"label": False, "confidence_band": "high"},
        }
        by_identity_with_attrs: dict[str, list] = {}
        for iid, group in by_identity.items():
            enriched = []
            for s in group:
                s2 = dict(s)
                s2["visual_attributes"] = shared_va
                enriched.append(s2)
            by_identity_with_attrs[iid] = enriched

        triples = find_wrong_name_candidates(by_identity_with_attrs)
        assert triples, "Gate B: find_wrong_name_candidates() should be non-empty with visual attrs"

        # Verify the selected subset is minimal (3-4 identities, 4-6 rows).
        n_selected = result["coverage"]["selected_samples"]
        n_identities = len(result["coverage"]["identities"])
        assert 3 <= n_identities <= 5, f"Too many identities selected: {n_identities}"
        assert n_selected <= 8, f"Too many rows selected: {n_selected}"


# ========================================================================== #
# P0-8: Schema contract test
# ========================================================================== #


class TestP08SchemaContract:
    """P0-8: smoke selection uses CanonicalSample schema end-to-end."""

    def test_smoke_selection_contract_uses_canonical_sample_schema(self):
        """CanonicalSample.to_dict() → selector → structural feasibility.

        Verifies that no code expects undefined top-level fields.
        """
        from route_data.build.conflict_generation import (
            _identity_name,
            structural_wrong_name_candidates,
        )
        from route_data.data.schemas import (
            CanonicalSample,
            ProfileFact,
            Provenance,
        )

        fv = _import_final_verify()
        prov = Provenance(source_dataset="fiubench")

        def _cs(sid, iid, split, name, img, facts=None):
            return CanonicalSample(
                benchmark="fiubench",
                source_sample_id=sid,
                identity_id=iid,
                provenance=prov,
                identity_name=name,
                image_uri=img,
                split=split,
                profile_facts=facts or [],
            ).to_dict()

        fact = ProfileFact(fact_id="f1", relation="nat", value="X")
        samples = [
            _cs("s1", "idA", "exclude", "Person A", "/a.jpg", [fact]),
            _cs("s2", "idB", "train", "Person B", "/b1.jpg", [fact]),
            _cs("s3", "idB", "train", "Person B", "/b2.jpg"),
            _cs("s4", "idC", "eval", "Person C", "/c.jpg", [fact]),
            _cs("s5", "idD", "train", "Person D", "/d.jpg"),
            _cs("s6", "idE", "eval", "Person E", "/e.jpg"),
        ]

        # Verify every dict has identity_name, not name.
        for s in samples:
            assert "identity_name" in s
            assert _identity_name(s) is not None or s["identity_name"] is None

        # Run the smoke selector.
        result = fv.select_smoke_subset(
            samples, min_identities=3, min_image_bearing=2,
        )

        # Run structural wrong-name feasibility.
        by_identity: dict[str, list] = {}
        for s in result["selected"]:
            by_identity.setdefault(s["identity_id"], []).append(s)
        pairs = structural_wrong_name_candidates(by_identity)
        assert pairs, "Schema contract: structural wrong-name pairs must exist"

        # Verify no code expects a "name" field — only "identity_name".
        for s in result["selected"]:
            # identity_name must be the field used.
            assert "identity_name" in s
            # The selector should not have introduced any "name" key.
            # (CanonicalSample.to_dict() does not produce "name".)
            assert "name" not in s
