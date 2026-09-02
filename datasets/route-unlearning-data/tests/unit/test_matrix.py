"""Unit tests for the E2C-v3 multi-seed / multi-forget-set matrix runner.

GPU-free tests covering the pure logic:
- balanced target selection per dataset (professions / taxonomy depth)
- determinism of the matrix build (same rules -> same sets)
- one-checkpoint-per-set simultaneity bookkeeping (set ids, oracle reuse)
- per-cell failure-criteria flags (leak / retain-broken / OOV / catastrophic)
- mean/std/min aggregation helpers
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

_MX_PATH = (Path(__file__).resolve().parents[2]
            / "scripts" / "e2c_v3_matrix.py")


def _load_mx_module():
    spec = importlib.util.spec_from_file_location("e2c_mx_under_test",
                                                  _MX_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mx = _load_mx_module()


def _args(dataset, tmp_path, **kw):
    a = argparse.Namespace(
        dataset=dataset, seeds=[17, 42, 123], smoke=False,
        ul_steps=500, ul_warmup=50, ul_lr=2e-5, ul_repeat=50,
        route_steps=3000, route_warmup=200, route_lr=2e-5, route_repeat=50,
        **kw)
    return a


def _redirect_manifest_dir(tmp_path):
    tmp = tmp_path / "manifests"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def _read_json(*parts):
    path = Path(__file__).resolve().parents[2].joinpath(*parts)
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# balanced selection rules
# ------------------------------------------------------------------ #
def test_mllmu_singles_one_per_profession():
    manifest = _read_json("e2c_mllmu", "manifests", "mllmu_manifest.json")
    singles = mx._balanced_mllmu_singles(manifest)
    assert len(singles) == 6
    profs = [manifest["alias_of"][i] for i in singles]
    assert len(set(profs)) == 6  # exactly one per profession


def test_mllmu_simultaneous_distinct_professions_no_reuse():
    manifest = _read_json("e2c_mllmu", "manifests", "mllmu_manifest.json")
    sets = mx._balanced_mllmu_simultaneous(manifest)
    assert len(sets) == 3 and all(len(s) == 3 for s in sets)
    used = [i for s in sets for i in s]
    assert len(set(used)) == 9  # no id reuse across sets
    for s in sets:
        profs = [manifest["alias_of"][i] for i in s]
        assert len(set(profs)) == 3  # each set spans distinct professions
    # determinism
    assert sets == mx._balanced_mllmu_simultaneous(manifest)


def test_salmu_singles_balanced_across_taxonomy():
    manifest = _read_json("e2c_salmu", "manifests", "salmu_manifest.json")
    singles = mx._balanced_salmu_singles(manifest)
    assert len(singles) == 6
    lv = manifest["job_levels"]
    l2 = {lv[i][2] for i in singles}
    l1 = {lv[i][1] for i in singles}
    assert len(l2) >= 5 and len(l1) >= 5  # spread over taxonomy groups
    assert singles == mx._balanced_salmu_singles(manifest)


def test_salmu_simultaneous_spans_groups_no_reuse():
    manifest = _read_json("e2c_salmu", "manifests", "salmu_manifest.json")
    sets = mx._balanced_salmu_simultaneous(manifest)
    lv = manifest["job_levels"]
    assert len(sets) == 3 and all(len(s) == 3 for s in sets)
    used = [i for s in sets for i in s]
    assert len(set(used)) == 9
    for s in sets:
        assert len({lv[i][2] for i in s}) >= 2  # >= 2 level-2 groups
    assert sets == mx._balanced_salmu_simultaneous(manifest)


def test_ppubench_all_rotations_and_three_pairs():
    manifest = _read_json("e2c_v3_real", "manifests",
                          "realdata_identity_mapping.json")
    singles, sim = mx._ppubench_sets(manifest)
    assert sorted(singles) == [["001"], ["002"], ["003"], ["004"]]
    assert len(sim) == 3 and all(len(s) == 2 for s in sim)
    assert len({tuple(s) for s in sim}) == 3  # distinct pairs
    assert sim == mx._ppubench_sets(manifest)[1]


# ------------------------------------------------------------------ #
# matrix build: ids, modes, oracle reuse, determinism
# ------------------------------------------------------------------ #
def test_build_matrix_ids_modes_and_oracle_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(mx, "MANIFEST_DIR", _redirect_manifest_dir(tmp_path))
    args = _args("ppubench", tmp_path)
    m1 = mx.build_matrix(args)
    m2 = mx.build_matrix(args)
    assert [e["set_id"] for e in m1["sets"]] == \
           [e["set_id"] for e in m2["sets"]]
    singles = [e for e in m1["sets"] if e["mode"] == "single"]
    sim = [e for e in m1["sets"] if e["mode"] == "simultaneous"]
    assert len(singles) == 4 and len(sim) == 3
    # the pilot oracle excluded exactly {001} -> reused for that set only
    reuse = {e["set_id"]: e["oracle_reuse"] for e in m1["sets"]}
    assert reuse["fs_001"] is not None
    assert all(v is None for k, v in reuse.items() if k != "fs_001")
    assert m1["edit_seeds"] == [17, 42, 123]
    assert m1["oracle_seed"] == 17
    # simultaneity guarantee recorded
    assert "one training run and one checkpoint per" in \
        m1["simultaneity_guarantee"]
    assert (mx.MANIFEST_DIR / "matrix_ppubench.json").exists()


def test_set_id_canonical():
    assert mx.set_id(["002", "001"]) == "fs_001-002"
    assert mx.set_id(["00040875"]) == "fs_00040875"


# ------------------------------------------------------------------ #
# per-cell failure criteria
# ------------------------------------------------------------------ #
def _fake_cell_inputs(manifest_ids, aliases, targets, *,
                      leaks=(), soft_leaks=(), retain_wrong=(),
                      oov=(), ambiguous=()):
    vocab = sorted(set(aliases.values()) | {mx.DELETED_LABEL})
    hard = []
    for iid in manifest_ids:
        role = "target" if iid in targets else "retain"
        expected = mx.DELETED_LABEL if role == "target" else aliases[iid]
        correct = expected if iid not in retain_wrong else "garbage"
        hard.append({
            "identity_id": iid, "raw": correct, "parsed_label": correct,
            "recognized_labels": ([aliases[iid]] if iid in leaks else
                                  [correct]),
            "multi_label_ambiguous": iid in ambiguous,
            "role": role, "expected_post_edit": expected,
            "correct_post_edit": correct == expected,
            "old_alias_leaked": iid in leaks,
        })
    soft = {}
    for iid in manifest_ids:
        p_alias = 0.9 if iid in soft_leaks else 1e-6
        probs = {aliases[iid]: p_alias if iid in targets else 0.999,
                 mx.DELETED_LABEL: 1e-6 if iid in targets else 0.0}
        soft[iid] = {"probs": probs,
                     "candidate_mass": 0.0001 if iid in oov else 0.999,
                     "other_mass": 0.9999 if iid in oov else 0.001}
    return hard, soft, vocab


_MANIFEST = {"identity_ids": ["a", "b", "c", "d"],
             "alias_of": {"a": "Aster", "b": "Briar", "c": "Clove",
                          "d": "Dune"}}


def test_failure_flags_clean_cell_passes():
    hard, soft, vocab = _fake_cell_inputs(
        _MANIFEST["identity_ids"], _MANIFEST["alias_of"], {"a"})
    f = mx._failure_flags(hard, soft, {"a"}, _MANIFEST, vocab)
    assert f["cell_pass"] and not f["leak"] and not f["catastrophic"]
    assert f["retain_acc"] == 1.0
    assert f["worst_retained_identity"] in {"b", "c", "d"}


def test_failure_flags_hard_leak_is_categorical():
    hard, soft, vocab = _fake_cell_inputs(
        _MANIFEST["identity_ids"], _MANIFEST["alias_of"], {"a"}, leaks=("a",))
    f = mx._failure_flags(hard, soft, {"a"}, _MANIFEST, vocab)
    assert f["leak"] and f["catastrophic"] and not f["cell_pass"]


def test_failure_flags_soft_leak_and_oov():
    hard, soft, vocab = _fake_cell_inputs(
        _MANIFEST["identity_ids"], _MANIFEST["alias_of"], {"a"},
        soft_leaks=("a",))
    f = mx._failure_flags(hard, soft, {"a"}, _MANIFEST, vocab)
    assert f["leak"] and not f["cell_pass"]
    hard, soft, vocab = _fake_cell_inputs(
        _MANIFEST["identity_ids"], _MANIFEST["alias_of"], {"a"}, oov=("c",))
    f = mx._failure_flags(hard, soft, {"a"}, _MANIFEST, vocab)
    assert f["oov_garbage_ids"] == ["c"] and f["catastrophic"]


def test_failure_flags_broken_retain_visible_but_not_leak():
    hard, soft, vocab = _fake_cell_inputs(
        _MANIFEST["identity_ids"], _MANIFEST["alias_of"], {"a"},
        retain_wrong=("b",))
    f = mx._failure_flags(hard, soft, {"a"}, _MANIFEST, vocab)
    assert f["retain_broken"] and not f["leak"] and not f["cell_pass"]
    assert pytest.approx(f["retain_acc"]) == 2 / 3


# ------------------------------------------------------------------ #
# aggregation helper
# ------------------------------------------------------------------ #
def test_msd_mean_std_min_max():
    s = mx._msd([1.0, 1.0, 0.0])
    assert pytest.approx(s["mean"]) == 2 / 3
    assert s["min"] == 0.0 and s["max"] == 1.0 and s["std"] > 0
    assert mx._msd([])["mean"] is None
    assert mx._msd([0.5])["std"] == 0.0
