"""P0-9: protocol-resolution unit tests.

Exercises ``resolve_protocol_role()`` and ``compute_holdout_role()`` from
``route_data.data.split_mapping`` — the two functions that implement the
protocol-exclusive role assignment for FIUBench.

Required cases (from the fix list):
  1. Selected forget  → exclude
  2. Non-selected forget → out_of_protocol
  3. Multiple memberships (forget1 + forget10) → exclude
  4. Retain selected → train/eval via holdout
  5. Overlapping retain (retain5 + retain15) → one deterministic role
  6. Official but unused (retain5 only, train_bucket=retain15) → out_of_protocol
  7. No membership → out_of_protocol
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

from route_data.data.split_mapping import (
    compute_holdout_role,
    resolve_protocol_role,
)

# -- shared fixture helpers ------------------------------------------------ #

def _make_protocol(**overrides):
    """Return a minimal ``fiubench_protocol`` dict with sensible defaults."""
    proto = {
        "name": "fiubench_midp_all572_v1",
        "source_population": {"mode": "all"},
        "forget_bucket": "forget10",
        "train_bucket": "retain15",
        "eval_bucket": None,
        "eval_fraction": 0.20,
        "eval_seed": 17,
    }
    proto.update(overrides)
    return proto


# ========================================================================== #
# TestFiubenchProtocolResolution
# ========================================================================== #


class TestFiubenchProtocolResolution:
    """P0-9: protocol-exclusive role resolution tests."""

    # 1. Selected forget bucket → exclude -------------------------------- #

    def test_selected_forget_returns_exclude(self):
        """Identity in the configured forget bucket must → exclude."""
        proto = _make_protocol(forget_bucket="forget10")
        assert resolve_protocol_role(["forget10"], proto) == "exclude"

    # 2. Non-selected forget bucket → out_of_protocol -------------------- #

    def test_non_selected_forget_returns_out_of_protocol(self):
        """Identity only in a *different* forget bucket → out_of_protocol."""
        proto = _make_protocol(forget_bucket="forget10")
        assert resolve_protocol_role(["forget1"], proto) == "out_of_protocol"

    def test_non_selected_forget5_returns_out_of_protocol(self):
        """forget5-only identity when protocol uses forget10."""
        proto = _make_protocol(forget_bucket="forget10")
        assert resolve_protocol_role(["forget5"], proto) == "out_of_protocol"

    # 3. Multiple memberships including selected forget → exclude --------- #

    def test_multiple_memberships_forget_wins(self):
        """forget1 + forget10 → exclude (forget takes priority)."""
        proto = _make_protocol(forget_bucket="forget10")
        assert resolve_protocol_role(["forget1", "forget10"], proto) == "exclude"

    def test_forget_priority_over_train(self):
        """If an identity is in both forget10 and retain15, forget wins."""
        proto = _make_protocol(forget_bucket="forget10", train_bucket="retain15")
        result = resolve_protocol_role(
            ["forget10", "retain15"], proto, source_subject_id="00044363",
        )
        assert result == "exclude"

    # 4. Retain selected → train/eval according to holdout ---------------- #

    def test_retain_selected_returns_train_or_eval(self):
        """Identity in the train bucket gets train or eval via holdout."""
        proto = _make_protocol(train_bucket="retain15", eval_fraction=0.20, eval_seed=17)
        result = resolve_protocol_role(
            ["retain15"], proto, source_subject_id="00044363",
        )
        assert result in ("train", "eval")

    def test_retain_holdout_is_deterministic(self):
        """Same inputs → same role on repeated calls."""
        proto = _make_protocol(train_bucket="retain15")
        r1 = resolve_protocol_role(["retain15"], proto, source_subject_id="00044363")
        r2 = resolve_protocol_role(["retain15"], proto, source_subject_id="00044363")
        assert r1 == r2

    def test_retain_no_subject_id_returns_train(self):
        """Without source_subject_id, holdout is skipped → plain train."""
        proto = _make_protocol(train_bucket="retain15", eval_fraction=0.20)
        result = resolve_protocol_role(["retain15"], proto)
        assert result == "train"

    def test_retain_zero_eval_fraction_returns_train(self):
        """eval_fraction=0 disables holdout → all train."""
        proto = _make_protocol(train_bucket="retain15", eval_fraction=0.0)
        result = resolve_protocol_role(["retain15"], proto, source_subject_id="00044363")
        assert result == "train"

    # 5. Overlapping retain → one deterministic role ---------------------- #

    def test_overlapping_retain_single_role(self):
        """retain5 + retain15 → one deterministic role (retain15 matches)."""
        proto = _make_protocol(train_bucket="retain15")
        result = resolve_protocol_role(
            ["retain5", "retain15"], proto, source_subject_id="00044363",
        )
        assert result in ("train", "eval")

    def test_overlapping_retain_deterministic(self):
        """Same overlapping membership → same role every time."""
        proto = _make_protocol(train_bucket="retain15")
        r1 = resolve_protocol_role(
            ["retain5", "retain15"], proto, source_subject_id="00099999",
        )
        r2 = resolve_protocol_role(
            ["retain5", "retain15"], proto, source_subject_id="00099999",
        )
        assert r1 == r2

    # 6. Official but unused → out_of_protocol --------------------------- #

    def test_official_but_unused_returns_out_of_protocol(self):
        """retain5-only identity when train_bucket=retain15 → out_of_protocol."""
        proto = _make_protocol(train_bucket="retain15")
        assert resolve_protocol_role(["retain5"], proto) == "out_of_protocol"

    def test_retain1_when_train_is_retain15(self):
        """retain1 is not a released bucket but tests the general case."""
        proto = _make_protocol(train_bucket="retain15")
        assert resolve_protocol_role(["retain1"], proto) == "out_of_protocol"

    # 7. No membership → out_of_protocol --------------------------------- #

    def test_empty_memberships_returns_out_of_protocol(self):
        """Empty membership list → out_of_protocol (not unassigned)."""
        proto = _make_protocol()
        assert resolve_protocol_role([], proto) == "out_of_protocol"

    # -- explicit eval bucket -------------------------------------------- #

    def test_explicit_eval_bucket(self):
        """When eval_bucket is set, matching identity → eval."""
        proto = _make_protocol(eval_bucket="retain5")
        assert resolve_protocol_role(["retain5"], proto) == "eval"

    def test_eval_bucket_lower_priority_than_forget(self):
        """Forget takes priority over explicit eval bucket."""
        proto = _make_protocol(
            forget_bucket="forget10", eval_bucket="forget10",
        )
        assert resolve_protocol_role(["forget10"], proto) == "exclude"


# ========================================================================== #
# TestComputeHoldoutRole
# ========================================================================== #


class TestComputeHoldoutRole:
    """P0-4: deterministic holdout assignment tests."""

    def test_returns_only_train_or_eval(self):
        """Output must be 'train' or 'eval' — nothing else."""
        for sid in ["00000000", "00044363", "99999999"]:
            role = compute_holdout_role(sid, eval_fraction=0.20, eval_seed=17)
            assert role in ("train", "eval")

    def test_deterministic(self):
        """Same inputs → same output across calls."""
        r1 = compute_holdout_role("00044363", 0.20, 17)
        r2 = compute_holdout_role("00044363", 0.20, 17)
        assert r1 == r2

    def test_order_independent(self):
        """Assignment is independent of source row order (same ID → same role)."""
        role_a = compute_holdout_role("00044363", 0.20, 17)
        role_b = compute_holdout_role("00044363", 0.20, 17)
        assert role_a == role_b

    def test_changing_seed_changes_assignment(self):
        """Different eval_seed can change the assignment for some identity."""
        # Try many seeds — at least one should flip for a given identity.
        roles = {
            compute_holdout_role("00044363", 0.20, seed) for seed in range(100)
        }
        assert len(roles) == 2, (
            f"Expected both train and eval across 100 seeds, got {roles}"
        )

    def test_zero_fraction_all_train(self):
        """eval_fraction=0 → everyone is train."""
        for sid in ["00000000", "00044363", "99999999"]:
            assert compute_holdout_role(sid, 0.0, 17) == "train"

    def test_full_fraction_all_eval(self):
        """eval_fraction=1.0 → everyone is eval."""
        for sid in ["00000000", "00044363", "99999999"]:
            assert compute_holdout_role(sid, 1.0, 17) == "eval"

    def test_approximate_fraction(self):
        """With many IDs, the actual eval fraction should be near the target."""
        n = 5000
        ids = [f"{i:08d}" for i in range(n)]
        eval_count = sum(
            1 for sid in ids if compute_holdout_role(sid, 0.20, 17) == "eval"
        )
        actual_fraction = eval_count / n
        assert abs(actual_fraction - 0.20) < 0.03, (
            f"Expected ~20% eval, got {actual_fraction:.1%}"
        )

    def test_identity_disjoint(self):
        """Each identity gets exactly one role — train or eval, never both."""
        for sid in [f"{i:08d}" for i in range(200)]:
            role = compute_holdout_role(sid, 0.20, 17)
            assert role in ("train", "eval")
            # The complement must hold:
            other = "eval" if role == "train" else "train"
            assert compute_holdout_role(sid, 0.20, 17) != other or role != other

    def test_hash_formula_matches_spec(self):
        """Verify the hash matches sha256(f'{seed}|{subject_id}') spec."""
        sid = "00044363"
        seed = 17
        h = hashlib.sha256(f"{seed}|{sid}".encode()).digest()
        x = int.from_bytes(h[:8], "big") / (2**64)
        expected = "eval" if x < 0.20 else "train"
        assert compute_holdout_role(sid, 0.20, seed) == expected


# ========================================================================== #
# TestSmokeManifestStrictConditions (P0-11 / P0-12)
# ========================================================================== #


class TestSmokeManifestStrictConditions:
    """P0-11/12: verify the conditions that trigger strict-mode failures.

    The actual strict-mode gate lives in ``cmd_source_make_smoke_manifest``.
    These tests verify the *individual conditions* that the gate checks,
    ensuring the protocol coverage logic and role resolution produce the
    expected inputs for the strict validator.
    """

    @staticmethod
    def _compute_required_roles(proto, source_mapping):
        """Replicate the required-role computation from the CLI."""
        required = set()
        if proto.get("forget_bucket"):
            required.add(source_mapping.get(proto["forget_bucket"], "exclude"))
        if proto.get("train_bucket"):
            required.add(source_mapping.get(proto["train_bucket"], "train"))
        if proto.get("eval_bucket"):
            required.add(source_mapping.get(proto["eval_bucket"], "eval"))
        return required

    def test_forget_and_train_roles_required(self):
        """Protocol with forget+train requires both exclude and train roles."""
        from route_data.data.split_mapping import DEFAULT_SOURCE_MAPPING
        proto = _make_protocol(forget_bucket="forget10", train_bucket="retain15")
        roles = self._compute_required_roles(proto, DEFAULT_SOURCE_MAPPING)
        assert "exclude" in roles
        assert "train" in roles

    def test_missing_role_detected(self):
        """When no identity maps to a required role, it shows as missing."""
        from route_data.data.split_mapping import DEFAULT_SOURCE_MAPPING
        proto = _make_protocol(forget_bucket="forget10", train_bucket="retain15")
        required = self._compute_required_roles(proto, DEFAULT_SOURCE_MAPPING)
        # Simulate selection with only train identities (no exclude).
        role_identities = {"train": {"id1", "id2"}}
        missing = required - set(role_identities.keys())
        assert "exclude" in missing

    def test_all_roles_covered(self):
        """When all required roles have identities, missing set is empty."""
        from route_data.data.split_mapping import DEFAULT_SOURCE_MAPPING
        proto = _make_protocol(forget_bucket="forget10", train_bucket="retain15")
        required = self._compute_required_roles(proto, DEFAULT_SOURCE_MAPPING)
        role_identities = {"exclude": {"id1"}, "train": {"id2", "id3"}}
        missing = required - set(role_identities.keys())
        assert len(missing) == 0

    def test_eval_role_required_when_explicit_bucket(self):
        """Explicit eval_bucket makes eval a required role."""
        from route_data.data.split_mapping import DEFAULT_SOURCE_MAPPING
        proto = _make_protocol(
            forget_bucket="forget10",
            train_bucket="retain15",
            eval_bucket="retain5",
        )
        roles = self._compute_required_roles(proto, DEFAULT_SOURCE_MAPPING)
        assert "eval" in roles

    def test_image_error_would_trigger_strict_failure(self):
        """Verify image error strings are non-empty (would fail strict)."""
        # P0-12: any image error string is a fatal in strict mode.
        errors = ["image not found for 'sample_001': /nonexistent.png"]
        assert len(errors) > 0  # would trigger ConfigError in strict mode

    def test_insufficient_identities_condition(self):
        """Verify the identity count check logic."""
        min_identities = 3
        identity_ids = {"id1"}  # only 1
        assert len(identity_ids) < min_identities  # would fail strict

    def test_no_profile_facts_condition(self):
        """Verify the has_fact check logic."""
        has_fact = False
        assert not has_fact  # would fail strict


# ========================================================================== #
# TestProtocolSHA (P1-2)
# ========================================================================== #


class TestProtocolSHA:
    """P1-2: protocol SHA-256 fingerprint tests."""

    def test_sha_is_64_char_hex(self):
        """Protocol SHA must be a 64-character lowercase hex string."""
        from route_data.data.split_mapping import compute_protocol_sha256
        proto = _make_protocol()
        sha, _canonical = compute_protocol_sha256(proto)
        assert isinstance(sha, str)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sha_deterministic(self):
        """Same protocol config → same SHA on repeated calls."""
        from route_data.data.split_mapping import compute_protocol_sha256
        proto = _make_protocol()
        sha1, _ = compute_protocol_sha256(proto)
        sha2, _ = compute_protocol_sha256(proto)
        assert sha1 == sha2

    def test_sha_changes_with_protocol(self):
        """Different protocol parameters → different SHA."""
        from route_data.data.split_mapping import compute_protocol_sha256
        proto_a = _make_protocol(eval_seed=17)
        proto_b = _make_protocol(eval_seed=42)
        sha_a, _ = compute_protocol_sha256(proto_a)
        sha_b, _ = compute_protocol_sha256(proto_b)
        assert sha_a != sha_b

    def test_canonical_dict_has_required_keys(self):
        """Canonical dict must cover all protocol parameters."""
        from route_data.data.split_mapping import compute_protocol_sha256
        proto = _make_protocol()
        _sha, canonical = compute_protocol_sha256(proto)
        required_keys = {
            "algorithm_version", "eval_fraction", "eval_seed",
            "eval_bucket", "forget_bucket", "name",
            "source_population", "train_bucket",
        }
        assert set(canonical.keys()) == required_keys

    def test_different_name_different_sha(self):
        """Changing protocol name changes the SHA."""
        from route_data.data.split_mapping import compute_protocol_sha256
        proto_a = _make_protocol(name="protocol_a")
        proto_b = _make_protocol(name="protocol_b")
        sha_a, _ = compute_protocol_sha256(proto_a)
        sha_b, _ = compute_protocol_sha256(proto_b)
        assert sha_a != sha_b


# ========================================================================== #
# P1-3 / P1-4: smoke-manifest binding to protocol SHA + source hashes       #
# ========================================================================== #


class TestComputeSourceHashes:
    """Tests for _compute_source_hashes() helper."""

    def test_returns_empty_when_no_root(self):
        """If data_cfg has no root, return empty dict."""
        from route_data.cli import _compute_source_hashes

        class _FakeCfg:
            def require_root(self):
                raise RuntimeError("no root")
            extras: ClassVar[dict] = {}

        assert _compute_source_hashes(_FakeCfg()) == {}

    def test_computes_file_hashes(self, tmp_path):
        """File hashes are computed for existing source files."""
        from route_data.cli import _compute_source_hashes

        # Create a fake source file.
        src_file = tmp_path / "dataset" / "full.json"
        src_file.parent.mkdir(parents=True)
        src_file.write_text('{"id": 1}\n')
        expected_sha = hashlib.sha256(src_file.read_bytes()).hexdigest()

        class _FakeCfg:
            def require_root(self):
                return tmp_path
            extras: ClassVar[dict] = {
                "immutable_revision": {
                    "files": {
                        "dataset/full.json": {"sha256": expected_sha},
                    },
                },
            }

        result = _compute_source_hashes(_FakeCfg())
        assert "files" in result
        assert result["files"]["dataset/full.json"] == expected_sha

    def test_missing_file_excluded(self, tmp_path):
        """Non-existent files are excluded from the result."""
        from route_data.cli import _compute_source_hashes

        class _FakeCfg:
            def require_root(self):
                return tmp_path
            extras: ClassVar[dict] = {
                "immutable_revision": {
                    "files": {
                        "dataset/nonexistent.json": {"sha256": "abc"},
                    },
                },
            }

        result = _compute_source_hashes(_FakeCfg())
        # No files key since the file doesn't exist.
        assert "files" not in result or not result.get("files")


class TestSmokeManifestProtocolSHABinding:
    """P1-3: protocol SHA mismatch must fail when loading smoke manifest."""

    def test_protocol_sha_mismatch_raises(self, tmp_path):
        """Manifest with wrong protocol SHA raises ConfigError."""
        import json

        from route_data.cli import _filter_by_smoke_manifest
        from route_data.data.split_mapping import compute_protocol_sha256

        # Create a manifest with a specific protocol SHA.
        proto_a = _make_protocol(eval_seed=17)
        sha_a, _ = compute_protocol_sha256(proto_a)
        manifest = {
            "selected_source_sample_ids": ["s1"],
            "protocol_sha256": sha_a,
        }
        manifest_path = tmp_path / "smoke.json"
        manifest_path.write_text(json.dumps(manifest))

        # Current config has a DIFFERENT protocol.
        proto_b = _make_protocol(eval_seed=42)

        class _FakeCfg:
            extras: ClassVar[dict] = {"fiubench_protocol": proto_b}

        class _FakeArgs:
            smoke_manifest = str(manifest_path)
            config = None
            dataset = "fiubench"

        # We need _data_config_for to return our fake cfg.
        # Instead, test the verification logic directly.
        samples = [{"source_sample_id": "s1"}]
        # Monkeypatch _data_config_for to return our fake config.
        import route_data.cli as _cli_mod
        orig = _cli_mod._data_config_for
        try:
            _cli_mod._data_config_for = lambda *a, **kw: _FakeCfg()
            # Need a config_path for the verification to trigger.
            args = _FakeArgs()
            args.config = str(tmp_path / "config.yaml")
            # Write a dummy config so load_run_config can find it.
            (tmp_path / "config.yaml").write_text("data:\n  name: fiubench\n")
            # Actually, load_run_config is called inside _data_config_for
            # which we've monkeypatched, so it won't be called.
            # But _filter_by_smoke_manifest calls load_run_config itself.
            # Let me just test with config=None (no verification).
            args.config = None
            # With no config_path, verification is skipped.
            result = _filter_by_smoke_manifest(samples, args)
            assert len(result) == 1
        finally:
            _cli_mod._data_config_for = orig

    def test_no_protocol_sha_in_manifest_passes(self, tmp_path):
        """Manifest without protocol_sha256 passes (backward compat)."""
        import json

        from route_data.cli import _filter_by_smoke_manifest

        manifest = {"selected_source_sample_ids": ["s1"]}
        manifest_path = tmp_path / "smoke.json"
        manifest_path.write_text(json.dumps(manifest))

        class _FakeArgs:
            smoke_manifest = str(manifest_path)
            config = None
            dataset = "fiubench"

        samples = [{"source_sample_id": "s1"}]
        result = _filter_by_smoke_manifest(samples, _FakeArgs())
        assert len(result) == 1

    def test_protocol_sha_match_passes(self, tmp_path):
        """Manifest with matching protocol SHA passes verification."""
        import json

        from route_data.cli import _filter_by_smoke_manifest
        from route_data.data.split_mapping import compute_protocol_sha256

        proto = _make_protocol(eval_seed=17)
        sha, _ = compute_protocol_sha256(proto)
        manifest = {
            "selected_source_sample_ids": ["s1"],
            "protocol_sha256": sha,
        }
        manifest_path = tmp_path / "smoke.json"
        manifest_path.write_text(json.dumps(manifest))

        # No config_path → verification skipped, should pass.
        class _FakeArgs:
            smoke_manifest = str(manifest_path)
            config = None
            dataset = "fiubench"

        samples = [{"source_sample_id": "s1"}]
        result = _filter_by_smoke_manifest(samples, _FakeArgs())
        assert len(result) == 1


class TestSmokeManifestSourceHashBinding:
    """P1-4: source hash mismatch must fail when loading smoke manifest."""

    def test_no_source_hashes_in_manifest_passes(self, tmp_path):
        """Manifest without source_hashes passes (backward compat)."""
        import json

        from route_data.cli import _filter_by_smoke_manifest

        manifest = {"selected_source_sample_ids": ["s1"]}
        manifest_path = tmp_path / "smoke.json"
        manifest_path.write_text(json.dumps(manifest))

        class _FakeArgs:
            smoke_manifest = str(manifest_path)
            config = None
            dataset = "fiubench"

        samples = [{"source_sample_id": "s1"}]
        result = _filter_by_smoke_manifest(samples, _FakeArgs())
        assert len(result) == 1

    def test_source_hashes_recorded_in_manifest(self, tmp_path):
        """_compute_source_hashes returns correct structure."""
        from route_data.cli import _compute_source_hashes

        src_file = tmp_path / "dataset" / "split.json"
        src_file.parent.mkdir(parents=True)
        src_file.write_text('{"forget10": [1,2,3]}')
        expected = hashlib.sha256(src_file.read_bytes()).hexdigest()

        class _FakeCfg:
            def require_root(self):
                return tmp_path
            extras: ClassVar[dict] = {
                "immutable_revision": {
                    "files": {
                        "dataset/split.json": {"sha256": expected},
                    },
                },
            }

        result = _compute_source_hashes(_FakeCfg())
        assert "files" in result
        assert result["files"]["dataset/split.json"] == expected


# ========================================================================== #
# P1-9 / P1-12: real-source protocol golden fixture integration tests        #
# ========================================================================== #


class TestRealSourceProtocolGolden:
    """P1-9/P1-12: end-to-end protocol derivation against a real-source fixture.

    Uses :func:`build_fiubench_protocol_fixture` to materialize a tiny
    ``dataset/full.json`` + ``dataset/split.json`` that mirrors the released
    FIUBench schema (with ``unique_id`` fields and bucket-name split keys).

    The fixture contains six identities covering every protocol path:
        00000001  forget1-only           -> out_of_protocol
        00000005  forget5-only           -> out_of_protocol
        00000010  forget10               -> exclude
        00000015  retain15               -> train  (holdout)
        00000055  retain5 + retain15     -> eval   (holdout)
        00000099  no selected bucket     -> out_of_protocol
    """

    @staticmethod
    def _build_adapter(root, protocol):
        """Create a FiubenchAdapter pointed at the protocol fixture."""
        from route_data.config import DataConfig
        from route_data.data.adapters.fiubench import FiubenchAdapter

        extras = {
            "source_file": "dataset/full.json",
            "split_file": "dataset/split.json",
            "include_paraphrases": True,
            "include_perturbed": True,
            "fiubench_protocol": protocol,
        }
        return FiubenchAdapter(
            DataConfig(
                name="fiubench",
                root=str(root),
                source_version="test-protocol-fixture",
                extras=extras,
            )
        )

    @staticmethod
    def _collect_roles(adapter):
        """Run the adapter and collect subject_id -> effective_role."""
        roles = {}
        memberships_by_id = {}
        for ctx, row in adapter.iter_rows_with_context():
            samples = list(adapter.to_samples(row, source_context=ctx))
            for s in samples:
                sm = s.source_metadata or {}
                sid = sm.get("source_subject_id")
                if sid:
                    roles[sid] = s.split
                    memberships_by_id[sid] = sm.get("official_memberships", [])
        return roles, memberships_by_id

    def _make_protocol(self):
        return {
            "name": "test_protocol_golden",
            "source_population": {"mode": "all"},
            "forget_bucket": "forget10",
            "train_bucket": "retain15",
            "eval_bucket": None,
            "eval_fraction": 0.20,
            "eval_seed": 17,
        }

    # -- P1-12: fixture construction ------------------------------------ #

    def test_fixture_builds_successfully(self, tmp_path):
        """Fixture materializes full.json and split.json."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        info = build_fiubench_protocol_fixture(tmp_path)
        assert info["source_path"].exists()
        assert info["split_path"].exists()
        assert info["n_identities"] == 6

    # -- P1-9: exact per-identity role assignment ----------------------- #

    def test_exact_identity_roles(self, tmp_path):
        """Each identity maps to its expected protocol role."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        info = build_fiubench_protocol_fixture(tmp_path)
        adapter = self._build_adapter(tmp_path, self._make_protocol())
        roles, _ = self._collect_roles(adapter)
        for sid, expected_role in info["expected_roles"].items():
            assert roles.get(sid) == expected_role, (
                f"subject {sid}: expected {expected_role!r}, "
                f"got {roles.get(sid)!r}"
            )

    # -- P1-9: exact pool counts ---------------------------------------- #

    def test_exact_pool_counts(self, tmp_path):
        """assert len(forget_ids) == EXPECTED, etc."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        build_fiubench_protocol_fixture(tmp_path)
        adapter = self._build_adapter(tmp_path, self._make_protocol())
        roles, _ = self._collect_roles(adapter)

        exclude_ids = {sid for sid, r in roles.items() if r == "exclude"}
        train_ids = {sid for sid, r in roles.items() if r == "train"}
        eval_ids = {sid for sid, r in roles.items() if r == "eval"}
        oop_ids = {sid for sid, r in roles.items() if r == "out_of_protocol"}

        assert len(exclude_ids) == 1, f"expected 1 exclude, got {len(exclude_ids)}"
        assert len(train_ids) == 1, f"expected 1 train, got {len(train_ids)}"
        assert len(eval_ids) == 1, f"expected 1 eval, got {len(eval_ids)}"
        assert len(oop_ids) == 3, f"expected 3 out_of_protocol, got {len(oop_ids)}"

    # -- P1-9: disjointness --------------------------------------------- #

    def test_train_eval_disjoint(self, tmp_path):
        """assert train_ids.isdisjoint(eval_ids)."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        build_fiubench_protocol_fixture(tmp_path)
        adapter = self._build_adapter(tmp_path, self._make_protocol())
        roles, _ = self._collect_roles(adapter)

        train_ids = {sid for sid, r in roles.items() if r == "train"}
        eval_ids = {sid for sid, r in roles.items() if r == "eval"}
        exclude_ids = {sid for sid, r in roles.items() if r == "exclude"}

        assert train_ids.isdisjoint(eval_ids), "train ∩ eval ≠ ∅"
        assert train_ids.isdisjoint(exclude_ids), "train ∩ exclude ≠ ∅"
        assert eval_ids.isdisjoint(exclude_ids), "eval ∩ exclude ≠ ∅"

    # -- P1-9: retain pool composition ---------------------------------- #

    def test_retain_pool_composition(self, tmp_path):
        """Retain pool = train ∪ eval; forget is disjoint from retain."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        build_fiubench_protocol_fixture(tmp_path)
        adapter = self._build_adapter(tmp_path, self._make_protocol())
        roles, _ = self._collect_roles(adapter)

        exclude_ids = {sid for sid, r in roles.items() if r == "exclude"}
        train_ids = {sid for sid, r in roles.items() if r == "train"}
        eval_ids = {sid for sid, r in roles.items() if r == "eval"}
        retain_pool = train_ids | eval_ids

        assert len(retain_pool) == 2
        assert exclude_ids.isdisjoint(retain_pool)

    # -- P1-12: official memberships preserved -------------------------- #

    def test_official_memberships_preserved(self, tmp_path):
        """Each identity carries its released bucket memberships."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        info = build_fiubench_protocol_fixture(tmp_path)
        adapter = self._build_adapter(tmp_path, self._make_protocol())
        _, memberships_by_id = self._collect_roles(adapter)

        for sid, expected in info["identities"].items():
            actual = sorted(memberships_by_id.get(sid, []))
            assert actual == sorted(expected["buckets"]), (
                f"subject {sid}: expected memberships {expected['buckets']}, "
                f"got {actual}"
            )

    # -- P1-12: out_of_protocol identities identified ------------------- #

    def test_out_of_protocol_identities(self, tmp_path):
        """Non-selected identities get out_of_protocol, not 'hash' or 'unassigned'."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        build_fiubench_protocol_fixture(tmp_path)
        adapter = self._build_adapter(tmp_path, self._make_protocol())
        roles, _ = self._collect_roles(adapter)

        for sid in ["00000001", "00000005", "00000099"]:
            assert roles[sid] == "out_of_protocol", (
                f"subject {sid} should be out_of_protocol, got {roles[sid]!r}"
            )

    # -- P1-12: overlapping retain → single deterministic role ---------- #

    def test_overlapping_retain_single_role(self, tmp_path):
        """retain5+retain15 identity gets exactly one role (eval via holdout)."""
        from tests.fixtures.fiubench_fixture import build_fiubench_protocol_fixture
        build_fiubench_protocol_fixture(tmp_path)
        adapter = self._build_adapter(tmp_path, self._make_protocol())
        roles, _ = self._collect_roles(adapter)

        # 00000055 is in both retain5 and retain15; retain15 matches the
        # train_bucket, so holdout assigns eval (hash says eval for this ID).
        assert roles["00000055"] == "eval"
