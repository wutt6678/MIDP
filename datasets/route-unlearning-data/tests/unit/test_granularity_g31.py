"""CPU tests for the G3.1 oracle-family repair.

Covers the correction that the G3 "matched/LOO" oracles were continued
fine-tuning references (baseline-h init), not retraining references:
- fresh_reinit_lora / session_reset_fresh produce a FRESH init and never
  read (or preserve) any trained checkpoint -- proven by zeroing a
  pre-populated (simulated trained) lora_B;
- the retrain-family data recipe (matched_retrain sees the transformed
  targets at target_boost=5 plus every retained mapping at a uniform
  repeat -- the boost is mx.TARGET_BOOST from the EDIT/suppression recipe,
  NOT route h's, which rd.train_h builds with no boost at all; loo_retrain
  never sees the targets);
- oracle_family_distances computes Delta_FT and Delta_retrain separately
  with legacy aliases;
- GX2R rewrites the oracle block of stored cells from distributions only
  (edited checkpoints untouched);
- aggregation separates transformation targets from refusal controls,
  reports sibling coverage, configured-vs-executed seeds, and the G3.1
  promotion gate;
- the G3.1 gate decomposes a false verdict into COVERAGE (no LOO distance
  exists, so no comparison was made) and CONDITIONS (a comparison was made
  and failed), and the behavioral and oracle-separation conclusions are
  reported as two claims with two denominators;
- the reports name matched_retrain_weighted / matched_retrain_balanced
  rather than describing the x5 weighting as the ordinary route-h recipe,
  and keep the stored numeric keys stable;
- a CPU re-aggregation accumulates cost instead of overwriting the record
  of the GPU passes that produced the numbers;
- the dirty-code gate that lets GX2B/GX2S coexist with a running main
  matrix is scoped to the scripts this runner executes, and reports a
  declared script that is missing rather than widening to the worktree.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import time
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gx = _load("gx_lib_g31", "e2c_v3_granularity.py")
gxm = _load("gx_runner_g31", "e2c_v3_granularity_matrix.py")
rv = gxm.rv


# ------------------------------------------------------------------ #
# shared synthetic fixture (tiny taxonomic-style ctx)
# ------------------------------------------------------------------ #
VOCAB = ["alpha", "beta", "gamma", "delta", "Unknown"]


def _ctx():
    return {
        "kind": "taxonomic",
        "identity_ids": ["i1", "i2", "i3"],
        "code_of": {"i1": "ID_1", "i2": "ID_2", "i3": "ID_3"},
        "baseline_alias_of": {"i1": "alpha", "i2": "gamma", "i3": "delta"},
        "vocab": VOCAB,
        "dag": {"alpha": ["alpha"], "beta": ["beta"], "gamma": ["gamma"],
                "delta": ["delta"]},
        "hierarchy_of": {},
    }


def _entry():
    return {
        "set_id": "sX",
        "mode": "single_level1",
        "assignments": {
            "i1": {"source": "alpha", "target": "beta",
                   "operation": "taxonomic", "target_depth": 1},
        },
        "retain_ids": ["i2", "i3"],
        "controls": {},
    }


def _summary(one_hot_label, noise=0.0):
    probs = {l: noise for l in VOCAB}
    probs[one_hot_label] = 1.0 - noise * (len(VOCAB) - 1)
    return rv.build_candidate_summary(probs, VOCAB, gx.DELETED_LABEL)


# ------------------------------------------------------------------ #
# 1. fresh init never loads (or preserves) trained weights
# ------------------------------------------------------------------ #
def test_fresh_reinit_zeroes_trained_lora_B_and_rejects_bad_layout():
    a = torch.nn.Parameter(torch.full((4, 4), 0.3))
    b = torch.nn.Parameter(torch.full((4, 4), 0.7))  # simulated TRAINED
    params = [("m.lora_A.default.weight", a), ("m.lora_B.default.weight", b)]
    n_a, n_b = gxm.fresh_reinit_lora(iter(params), seed=17)
    assert (n_a, n_b) == (1, 1)
    # fresh-init signature: B exactly zero -> trained values were
    # OVERWRITTEN, not loaded or preserved
    assert torch.count_nonzero(b).item() == 0
    assert torch.count_nonzero(a).item() > 0
    assert not torch.allclose(a, torch.full((4, 4), 0.3))

    # deterministic under the same seed
    a3 = torch.nn.Parameter(torch.zeros(4, 4))
    b3 = torch.nn.Parameter(torch.zeros(4, 4))
    gxm.fresh_reinit_lora(
        iter([("x.lora_A.w", a3), ("x.lora_B.w", b3)]), seed=999)
    a4 = torch.nn.Parameter(torch.zeros(4, 4))
    b4 = torch.nn.Parameter(torch.zeros(4, 4))
    gxm.fresh_reinit_lora(
        iter([("x.lora_A.w", a4), ("x.lora_B.w", b4)]), seed=999)
    assert torch.allclose(a3, a4)

    # fail-closed on layout errors
    with pytest.raises(RuntimeError, match="unexpected LoRA layout"):
        gxm.fresh_reinit_lora(iter([("lora_A", torch.zeros(2, 2))]), 17)
    with pytest.raises(RuntimeError, match="unexpected LoRA layout"):
        gxm.fresh_reinit_lora(iter([]), 17)


def test_fresh_reinit_takes_no_checkpoint_argument():
    """The retrain families must not depend on any checkpoint path: the
    fresh-init helpers accept ONLY live parameters + a seed (the seed is an
    RNG seed for the fresh init, never a checkpoint)."""
    sig = inspect.signature(gxm.fresh_reinit_lora)
    assert list(sig.parameters) == ["named_params", "seed"]
    sig2 = inspect.signature(gxm.session_reset_fresh)
    assert list(sig2.parameters) == ["session", "seed"]
    src = inspect.getsource(gxm.session_reset_fresh)
    assert "safetensors" not in src and "reset_to" not in src


class _StubAdapterModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return iter(self._params)


class _StubSessionFresh:
    def __init__(self):
        self.a = torch.nn.Parameter(torch.full((2, 2), 0.4))
        self.b = torch.nn.Parameter(torch.full((2, 2), 0.9))
        # mx.ModelSession exposes the live PeftModel as .model
        self.model = _StubAdapterModel(
            [("l.lora_A.w", self.a), ("l.lora_B.w", self.b)])
        # attributes consumed by the (monkeypatched) training helpers
        self.adapter = object()
        self.processor = object()


def test_session_reset_fresh_overwrites_trained_state():
    s = _StubSessionFresh()
    gxm.session_reset_fresh(s)
    assert torch.count_nonzero(s.b).item() == 0
    assert torch.count_nonzero(s.a).item() > 0


def test_session_reset_fresh_fail_closed_without_lora():
    class Empty:
        model = _StubAdapterModel([("dense.weight", torch.zeros(2, 2))])
    with pytest.raises(RuntimeError, match="unexpected LoRA layout"):
        gxm.session_reset_fresh(Empty())


# ------------------------------------------------------------------ #
# 2. retrain-family data recipe + protocol (monkeypatched training)
# ------------------------------------------------------------------ #
class _FakeArgs:
    device = "cpu"
    max_gen_tokens = 12
    seed = 17


def test_train_oracle_retrain_recipe(monkeypatch, tmp_path):
    ctx, entry = _ctx(), _entry()
    calls = {"items": [], "train": None, "fresh": 0}

    def fake_build(adapter, processor, pairs, repeat):
        calls["items"].append(([p["answer"] for p in pairs], repeat))
        return [(p["answer"], repeat) for p in pairs]

    def fake_train(name, adapter, model, processor, items, out_dir, device,
                   steps, warmup, lr):
        calls["train"] = {"name": name, "items": items, "steps": steps,
                          "warmup": warmup, "lr": lr,
                          "out_dir": str(out_dir)}
        (Path(out_dir) / "adapter_final").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(gxm.rv, "build_supervised_items", fake_build)
    monkeypatch.setattr(gxm.rv, "train_supervised", fake_train)
    monkeypatch.setattr(gxm, "_soft_all",
                        lambda session, c, a: {
                            i: _summary(gxm.expected_label(c, entry, i))
                            for i in c["identity_ids"]})
    monkeypatch.setattr(gxm, "_strict_accuracy",
                        lambda session, c, exp, a: 1.0)
    monkeypatch.setattr(gxm, "session_reset_fresh",
                        lambda session, seed=17: calls.__setitem__(
                            "fresh", calls["fresh"] + 1))
    session = _StubSessionFresh()

    out = tmp_path / "matched_retrain_sX"
    rec = gxm.train_oracle_retrain(session, "salmu", ctx, entry,
                                   "matched_retrain", out, _FakeArgs())
    assert calls["fresh"] == 1                       # fresh init used
    assert calls["train"]["steps"] == gxm.RETRAIN_STEPS == 3000
    assert calls["train"]["warmup"] == gxm.RETRAIN_WARMUP == 200
    assert calls["train"]["lr"] == gxm.RETRAIN_LR == 2e-5
    answers = [ans for group, _ in calls["items"] for ans in group]
    # transformed target present, boosted x5; retained present
    assert answers.count("beta") == gxm.RETRAIN_TARGET_BOOST
    assert "alpha" not in answers                    # source label absent
    assert calls["items"][0][1] == gxm.RETRAIN_REPEAT == 50
    assert calls["items"][1][1] == gxm.RETRAIN_REPEAT
    assert rec["fit_ok"] and rec["mode"] == "trained_fresh"
    with open(out / "oracle_results.json") as f:
        res = json.load(f)
    assert res["family"] == "matched_retrain"
    assert res["init"] == "fresh_lora"
    assert res["strict_fit_scope"] == "all_transformed_and_retained"

    # loo_retrain: targets NEVER seen; fit scope retained-only
    calls["items"], calls["fresh"] = [], 0
    out2 = tmp_path / "loo_retrain_sX"
    rec2 = gxm.train_oracle_retrain(session, "salmu", ctx, entry,
                                    "loo_retrain", out2, _FakeArgs())
    answers2 = [ans for group, _ in calls["items"] for ans in group]
    assert "alpha" not in answers2 and "beta" not in answers2
    assert set(answers2) == {"gamma", "delta"}  # retained baselines only
    with open(out2 / "oracle_results.json") as f:
        res2 = json.load(f)
    assert res2["strict_fit_scope"].startswith("retained_only")
    assert rec2["fit_ok"]


# ------------------------------------------------------------------ #
# 3. family distances: separate deltas + legacy aliases
# ------------------------------------------------------------------ #
def _write_family(oracle_root, family, sid, label_of):
    d = oracle_root / gxm.ORACLE_DIR_BY_FAMILY[family].format(sid=sid)
    d.mkdir(parents=True, exist_ok=True)
    soft = {i: _summary(label_of[i]) for i in label_of}
    with open(d / "oracle_soft.json", "w") as f:
        json.dump(soft, f)


def test_oracle_family_distances_and_deltas(tmp_path):
    ctx, entry = _ctx(), _entry()
    oracle_root = tmp_path / "oracles"
    edit = {i: _summary(gxm.expected_label(ctx, entry, i))
            for i in ctx["identity_ids"]}
    # matched families ~ edit; loo families keep the ORIGINAL label
    _write_family(oracle_root, "matched_finetune", "sX",
                  {i: gxm.expected_label(ctx, entry, i)
                   for i in ctx["identity_ids"]})
    _write_family(oracle_root, "matched_retrain", "sX",
                  {i: gxm.expected_label(ctx, entry, i)
                   for i in ctx["identity_ids"]})
    _write_family(oracle_root, "loo_finetune", "sX",
                  dict(ctx["baseline_alias_of"]))
    _write_family(oracle_root, "loo_retrain", "sX",
                  dict(ctx["baseline_alias_of"]))

    fam = gxm.oracle_family_distances(edit, entry, ctx, oracle_root, "sX")
    t = fam["i1"]
    for f in gxm.ORACLE_FAMILIES:
        assert t[f]["reliable"] and t[f]["distance"]["l2"] is not None
    assert t["matched_finetune"]["distance"]["l2"] < 1e-6
    assert t["matched_retrain"]["distance"]["l2"] < 1e-6
    assert t["loo_finetune"]["distance"]["l2"] > 1.0
    assert t["delta_ft_l2"] > 1.0 and t["delta_retrain_l2"] > 1.0
    # legacy aliases (G3 reports)
    assert t["matched"] is t["matched_finetune"]
    assert t["loo"] is t["loo_finetune"]
    assert t["delta_oracle_l2"] == t["delta_ft_l2"]
    assert t["delta_reliable"] and t["delta_retrain_reliable"]
    # retained identity: all distances ~0 (all references agree there)
    assert fam["i2"]["loo_retrain"]["distance"]["l2"] < 1e-6

    # missing family -> None distances, no crash
    fam2 = gxm.oracle_family_distances(edit, entry, ctx,
                                       tmp_path / "empty", "sX")
    assert fam2["i1"]["matched_retrain"]["distance"] is None
    assert fam2["i1"]["delta_retrain_l2"] is None
    assert not fam2["i1"]["delta_retrain_reliable"]


# ------------------------------------------------------------------ #
# 4. GX2R: CPU re-evaluation rewrites the oracle block only
# ------------------------------------------------------------------ #
def test_reevaluate_oracles_cpu(tmp_path):
    ctx, entry = _ctx(), _entry()
    out_base = tmp_path
    oracle_root = out_base / "oracles"
    _write_family(oracle_root, "matched_finetune", "sX",
                  {i: gxm.expected_label(ctx, entry, i)
                   for i in ctx["identity_ids"]})
    cell_dir = out_base / "cells" / "sX" / "seed_17"
    cell_dir.mkdir(parents=True)
    cell = {
        "cell_id": "sX__seed17", "set_id": "sX", "seed": 17,
        "mode": "single_level1", "assignments": entry["assignments"],
        "soft_probs_full": {i: _summary(
            gxm.expected_label(ctx, entry, i))["probs"]
            for i in ctx["identity_ids"]},
        "dual_oracle": {"legacy": True},
        "checkpoint_sha256": "deadbeef",
    }
    p = cell_dir / "cell_results.json"
    with open(p, "w") as f:
        json.dump(cell, f)
    n = gxm.reevaluate_oracles_cpu("salmu", ctx, {"sets": [entry]},
                                   out_base)
    assert n == 1
    with open(p) as f:
        out = json.load(f)
    assert out["oracle_block_version"] == "g3_1"
    assert out["dual_oracle"] == {"legacy": True}   # legacy view kept
    fam = out["oracle_families"]["i1"]
    assert fam["matched_finetune"]["distance"]["l2"] < 1e-6
    assert fam["loo_finetune"]["distance"] is None  # family absent
    assert fam["delta_ft_l2"] is None
    assert "oracle_reevaluation_note" in out


# ------------------------------------------------------------------ #
# 5. aggregation: refusal separation, coverage, seeds, gate
# ------------------------------------------------------------------ #
def _mk_cell(seed=17, with_refusal=False, retrain=(0.001, 1.3)):
    entry = _entry()
    if with_refusal:
        entry = json.loads(json.dumps(entry))
        entry["mode"] = "simultaneous_mixed_depth"
        entry["assignments"]["i3"] = {
            "source": "delta", "target": gx.DELETED_LABEL,
            "operation": "refusal", "target_depth": None}
        entry["retain_ids"] = ["i2"]
    ctx = _ctx()

    def fam_block(iid, l2m, l2l, mr, lr):
        rec = {}
        for f, l2 in (("matched_finetune", l2m), ("loo_finetune", l2l),
                      ("matched_retrain", mr), ("loo_retrain", lr)):
            rec[f] = ({"distance": {"l2": l2, "js": l2 / 2, "cosine": 1.0},
                       "reliable": True, "reason": None}
                      if l2 is not None else
                      {"distance": None, "reliable": False,
                       "reason": "oracle not available"})
        rec["matched"] = rec["matched_finetune"]
        rec["loo"] = rec["loo_finetune"]
        rec["delta_ft_l2"] = (l2l - l2m) if l2m is not None \
            and l2l is not None else None
        rec["delta_retrain_l2"] = (lr - mr) if mr is not None \
            and lr is not None else None
        rec["delta_oracle_l2"] = rec["delta_ft_l2"]
        rec["delta_reliable"] = rec["delta_ft_l2"] is not None
        rec["delta_retrain_reliable"] = rec["delta_retrain_l2"] is not None
        return rec

    mr, lr = retrain
    ids = ctx["identity_ids"]
    fams = {}
    for iid in ids:
        if iid in entry["assignments"]:
            fams[iid] = fam_block(iid, 3e-06, 1.3, mr, lr)
        else:
            fams[iid] = fam_block(iid, 1e-06, 1e-06, 1e-06, 1e-06)
    hard = []
    for iid in ids:
        grp = "target" if iid in entry["assignments"] else "retain"
        hard.append({"identity_id": iid, "group": grp,
                     "classification": "correct" if grp == "target" else None,
                     "correct_post_edit": True, "boundary_tags": [],
                     "parsed_label": "x"})
    checks = {k: True for k in (
        "strict_expected_accuracy==1.0", "min_target_p_desired>=0.90",
        "max_target_p_source<=0.01", "min_candidate_mass>=0.99",
        "retained_strict_accuracy==1.0", "sibling_strict_accuracy==1.0",
        "wrong_branch_rate==0", "unparseable_outputs==0",
        "multi_label_outputs==0")}
    return {
        "cell_id": f"{entry['set_id']}__seed{seed}", "set_id":
            entry["set_id"], "mode": entry["mode"], "seed": seed,
        "assignments": entry["assignments"],
        "hard_preds": hard,
        "soft_probs_full": {i: {"alpha": 1.0} for i in ids},
        "oracle_families": fams,
        "checkpoint_sha256": "abc",
        "e2e": {"level": "association-only"},
        "criteria": {"cell_pass": True, "failed_criteria": [],
                     "checks": checks,
                     "strict_expected_accuracy": 1.0, "retain_acc": 1.0,
                     "sibling_acc": None, "sibling_ids": [],
                     "min_target_p_desired": 0.999,
                     "max_target_p_source": 1e-07,
                     "min_candidate_mass": 0.999},
    }, entry


def _aggregate(tmp_path, cells, matrix, ds="celeba_numeric"):
    out_base = tmp_path
    (out_base / "cells").mkdir(exist_ok=True)
    for c in cells:
        d = out_base / "cells" / c["set_id"] / f"seed_{c['seed']}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "cell_results.json", "w") as f:
            json.dump(c, f)
    return gxm.aggregate_gx(ds, out_base, matrix)


def test_aggregate_refusal_separation_and_seed_fields(tmp_path):
    c1, entry = _mk_cell(with_refusal=True)
    matrix = {"edit_seeds": [17, 42, 123], "n_sets": 1, "n_cells": 3,
              "sets": [entry]}
    s = _aggregate(tmp_path, [c1], matrix)
    mode = s["per_mode"]["simultaneous_mixed_depth"]
    # refusal target must NOT contribute to the transformation block
    tt = mode["transformation_targets"]
    assert tt["delta_ft_l2"]["n"] == 1          # i1 only
    assert abs(tt["delta_ft_l2"]["mean"] - (1.3 - 3e-06)) < 1e-3
    rc = mode["refusal_controls"]
    assert rc["delta_ft_l2"]["n"] == 1          # i3 only
    assert s["configured_full_seeds"] == [17, 42, 123]
    assert s["executed_seeds"] == [17]
    assert "pilot" in s["run_stage"]
    assert s["cells_evaluated"] == 1
    assert s["full_matrix_cells_expected"] == 3
    cov = mode["sibling_coverage"]
    assert cov["cells_with_sibling_controls"] == 0
    assert cov["sibling_controls_total"] == 0
    assert "null" in cov["note"]
    # retrain deltas present in this fixture -> gate evaluated
    gate = s["g3_1_gate"]
    assert gate["status"] == "evaluated"
    assert gate["n_transformation_targets"] == 1   # refusal excluded
    assert gate["per_target"][0]["target"] == "i1"
    assert "correction" in s and s["correction"]["corrects_commit"] \
        == "a1df9be"


def test_g3_1_gate_materiality_and_claims(tmp_path):
    # gate PASS: matched_retrain fit_ok + material delta
    c1, entry = _mk_cell(retrain=(0.001, 1.3))
    out_base = tmp_path
    (out_base / "cells").mkdir()
    d = out_base / "cells" / c1["set_id"] / "seed_17"
    d.mkdir(parents=True)
    with open(d / "cell_results.json", "w") as f:
        json.dump(c1, f)
    odir = out_base / "oracles" / f"matched_retrain_{c1['set_id']}"
    odir.mkdir(parents=True)
    with open(odir / "oracle_results.json", "w") as f:
        json.dump({"fit_ok": True, "strict_all_expected": 1.0,
                   "min_candidate_mass": 0.999}, f)
    matrix = {"edit_seeds": [17], "n_sets": 1, "n_cells": 1,
              "sets": [entry]}
    s = gxm.aggregate_gx("celeba_numeric", out_base, matrix)
    assert s["g3_1_gate"]["passed"] is True
    assert "supported" in s["claims"]["retraining_claim"]
    assert "NOT yet supported" not in s["claims"]["retraining_claim"]

    # gate FAIL: immaterial delta (0.1 < margin 0.5)
    c2, _ = _mk_cell(retrain=(0.001, 0.101))
    d2 = out_base / "cells" / c2["set_id"] / "seed_17"
    with open(d2 / "cell_results.json", "w") as f:
        json.dump(c2, f)
    s2 = gxm.aggregate_gx("celeba_numeric", out_base, matrix)
    assert s2["g3_1_gate"]["passed"] is False
    assert "NOT yet supported" in s2["claims"]["retraining_claim"]

    # gate PENDING: no retrain oracles
    c3, entry3 = _mk_cell(retrain=(None, None))
    tmp2 = tmp_path / "b"
    (tmp2).mkdir()
    matrix3 = {"edit_seeds": [17], "n_sets": 1, "n_cells": 1,
               "sets": [entry3]}
    (tmp2 / "cells").mkdir()
    d3 = tmp2 / "cells" / c3["set_id"] / "seed_17"
    d3.mkdir(parents=True)
    with open(d3 / "cell_results.json", "w") as f:
        json.dump(c3, f)
    s3 = gxm.aggregate_gx("celeba_numeric", tmp2, matrix3)
    assert s3["g3_1_gate"]["status"] == "retrain_oracles_not_yet_trained"
    assert "passed" not in s3["g3_1_gate"]


# ------------------------------------------------------------------ #
# the TWO conclusions: behavioral coverage vs oracle-separation coverage
# ------------------------------------------------------------------ #
#: the cause the frozen distance gate recorded on the real numeric run, where
#: loo_retrain_gx_num_mix_2 fell to a candidate score sum of 5.874e-03
_REAL_LOO_CAUSE = (
    "candidate mass too small to renormalize (model=9.998e-01, "
    "oracle=5.874e-03, threshold=1.000e-02); output lies outside the "
    "recognized label set, distance not established")


def _two_seed_run(out_base, retrain_by_seed, cause=None):
    """A matrix of one cell per given seed, with the LOO distances asked for."""
    (out_base / "cells").mkdir(parents=True, exist_ok=True)
    for seed, retrain in sorted(retrain_by_seed.items()):
        c = _mk_cell(seed=seed, retrain=retrain)[0]
        if cause is not None and retrain[1] is None:
            for fam in c["oracle_families"].values():
                if not fam["loo_retrain"]["reliable"]:
                    fam["loo_retrain"]["reason"] = cause
        d = out_base / "cells" / c["set_id"] / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "cell_results.json", "w") as f:
            json.dump(c, f)
    odir = out_base / "oracles" / f"matched_retrain_{_entry()['set_id']}"
    odir.mkdir(parents=True, exist_ok=True)
    with open(odir / "oracle_results.json", "w") as f:
        json.dump({"fit_ok": True, "strict_all_expected": 1.0,
                   "min_candidate_mass": 0.999}, f)
    return {"edit_seeds": sorted(retrain_by_seed), "n_sets": 1,
            "n_cells": len(retrain_by_seed), "sets": [_entry()]}


def test_a_coverage_gap_is_not_reported_as_a_failed_separation(tmp_path):
    """The numeric G5 situation, and the conflation it must not suffer.

    Every cell passes the behavioral criteria while one target-seed comparison
    has no LOO distance at all, because the frozen MIN_CANDIDATE_MASS gate
    refused to renormalize a loo_retrain distribution that left the recognized
    label set.  The G3.1 gate requires EVERY transformation target, so
    ``passed`` is false -- but nothing failed.  Reporting the two together
    reads as a broken separation and hides a complete behavioral matrix.
    """
    matrix = _two_seed_run(tmp_path, {17: (0.001, 1.3), 42: (0.001, None)},
                           cause=_REAL_LOO_CAUSE)
    s = gxm.aggregate_gx("celeba_numeric", tmp_path, matrix)
    gate = s["g3_1_gate"]
    assert gate["passed"] is False, "the frozen gate requires EVERY target"
    cov = gate["coverage"]
    assert cov["gate_fails_on"] == "coverage"
    assert cov["n_separation_established"] == 1
    assert cov["n_separation_not_established"] == 1
    assert cov["n_established_and_failing"] == 0
    assert cov["established_and_failing"] == []
    row = cov["not_established"][0]
    assert row["target"] == "i1" and row["seed"] == 42
    assert row["l2_to_loo_retrain"] is None
    # the matched side is fine: the edit IS extremely close to matched retrain,
    # which is why "not established" cannot be read as "the edit drifted"
    assert row["matched_retrain_reliable"] is True
    assert row["l2_to_matched_retrain"] == pytest.approx(0.001)
    # ... and the cause is the gate's OWN recorded reason, not a paraphrase
    assert row["why_no_loo_retrain_distance"] == _REAL_LOO_CAUSE
    assert "NOT 'failed'" in cov["meaning"]
    assert "IS trained and committed at the frozen protocol" \
        in cov["not_a_pending_job"]
    assert "not of an unfinished job" in cov["not_a_pending_job"]
    assert "never be substituted into this primary gate" \
        in cov["no_seed_substitution"]
    # per-target rows carry the same decomposition
    per = {p["seed"]: p for p in gate["per_target"]}
    assert per[17]["delta_retrain_established"] is True
    assert per[42]["delta_retrain_established"] is False
    assert per[42]["loo_retrain_reliable"] is False

    claims = s["claims"]
    assert "All 2 cells of the full 2-cell celeba_numeric matrix satisfy the " \
        "frozen behavioral criteria" in claims["behavioral_claim"]
    assert "BEHAVIORAL conclusion" in claims["behavioral_claim"]
    sep = claims["oracle_separation_claim"]
    assert "established for 1 of 2 target-seed comparisons" in sep
    assert "It is NOT established for the remaining 1" in sep
    assert "target i1 in sX at edit seed(s) [42]" in sep
    assert "candidate mass too small to renormalize" in sep
    assert "did not fail on any measured comparison" in sep
    assert "until it happens to pass is not a repair" in sep
    rt = claims["retraining_claim"]
    assert "supported WHERE IT IS ESTABLISHED" in rt
    assert "passed=false on COVERAGE" in rt
    assert "NOT yet supported" not in rt, \
        "no oracle is missing and no comparison failed; coverage is short"


def test_a_conditions_failure_is_named_as_such_and_not_as_a_coverage_gap(
        tmp_path):
    """The other side: an established comparison that FAILS must say so.

    The coverage vocabulary is only honest if it is not also used for a real
    failure, so the same block has to name which of the two broke the gate.
    """
    matrix = _two_seed_run(tmp_path, {17: (0.001, 1.3), 42: (0.001, 0.101)})
    s = gxm.aggregate_gx("celeba_numeric", tmp_path, matrix)
    cov = s["g3_1_gate"]["coverage"]
    assert s["g3_1_gate"]["passed"] is False
    assert cov["gate_fails_on"] == "conditions"
    assert cov["n_established_and_failing"] == 1
    assert cov["n_separation_not_established"] == 0
    assert cov["not_established"] == []
    assert cov["established_and_failing"][0]["target"] == "i1"
    assert cov["established_and_failing"][0]["seed"] == 42
    claims = s["claims"]
    assert "NOT yet supported" in claims["retraining_claim"]
    assert "FAIL the conditions" in claims["oracle_separation_claim"]
    assert "did not fail on any measured comparison" \
        not in claims["oracle_separation_claim"]


def test_a_passing_gate_reports_full_coverage(tmp_path):
    matrix = _two_seed_run(tmp_path, {17: (0.001, 1.3), 42: (0.001, 1.3)})
    s = gxm.aggregate_gx("celeba_numeric", tmp_path, matrix)
    gate = s["g3_1_gate"]
    assert gate["passed"] is True
    cov = gate["coverage"]
    assert cov["gate_fails_on"] == "nothing"
    assert cov["n_separation_established"] == 2
    assert cov["n_separation_not_established"] == 0
    assert cov["n_established_and_failing"] == 0
    assert "established for 2 of 2 target-seed comparisons" \
        in s["claims"]["oracle_separation_claim"]
    assert "every transformation target evaluated" \
        in s["claims"]["oracle_separation_claim"]


# ------------------------------------------------------------------ #
# naming: the two matched references, and the x5 that is not route h's
# ------------------------------------------------------------------ #
def test_the_summary_never_calls_the_x5_weighting_the_route_h_recipe(tmp_path):
    """The terminology defect, pinned against the artifact that ships it.

    ``rd.train_h`` builds every code->alias pair at a uniform
    ``repeat=ROUTE_REPEAT`` with no target boost anywhere in the module, so
    ``matched_retrain``'s x5 is ``mx.TARGET_BOOST`` from the EDIT/suppression
    recipe.  Calling it "the original route-h protocol" makes the weighted
    reference read as the neutral one.
    """
    matrix = _two_seed_run(tmp_path, {17: (0.001, 1.3)})
    s = gxm.aggregate_gx("celeba_numeric", tmp_path, matrix)
    defs = s["oracle_family_definitions"]
    assert gxm.WEIGHTED_LABEL == "matched_retrain_weighted"
    assert gxm.BALANCED_LABEL == "matched_retrain_balanced"
    assert gxm.WEIGHTED_LABEL in defs["matched_retrain"]
    assert "NOT route h's recipe" in defs["matched_retrain"]
    assert "rd.train_h" in defs["matched_retrain"]
    assert defs["matched_retrain_balanced"].startswith(
        "as matched_retrain but target_boost=1")
    assert "which is how route h itself is trained" \
        in defs["matched_retrain_balanced"]
    blob = json.dumps(s)
    for banned in ("targets x5", "ORIGINAL route-h protocol",
                   "original route-h protocol", "relabeled"):
        assert banned not in blob, f"the summary still says {banned!r}"
    naming = s["reference_naming"]
    assert naming["key_to_name"]["x5"] == "matched_retrain_weighted"
    assert naming["key_to_name"]["x1"] == "matched_retrain_balanced"
    assert naming["key_to_name"]["L"] == "loo_retrain"


def test_the_ablation_report_names_both_references_and_keeps_stored_keys(
        tmp_path):
    """Prose is renamed; stored numeric keys are NOT.

    Renaming ``d_E_Mx5`` would break comparability with the committed
    ablation reports and with the cells already measured under those keys, so
    the report carries a naming block that maps each shorthand instead -- the
    same split the prompt panel uses for its score-sum fields.
    """
    ctx, entry = _ctx(), _entry()
    matrix = {"sets": [entry]}
    exp = {i: gxm.expected_label(ctx, entry, i) for i in ctx["identity_ids"]}
    base = dict(ctx["baseline_alias_of"])
    out_base = tmp_path
    (out_base / "cells" / "sX" / "seed_17").mkdir(parents=True)
    with open(out_base / "cells" / "sX" / "seed_17" / "cell_results.json",
              "w") as f:
        json.dump({"cell_id": "sX__seed17", "set_id": "sX", "seed": 17,
                   "soft_probs_full": {i: _summary(exp[i])["probs"]
                                       for i in ctx["identity_ids"]}}, f)
    matched_soft = {i: _summary(exp[i]) for i in ctx["identity_ids"]}
    loo_soft = {i: _summary(base[i]) for i in ctx["identity_ids"]}
    fit_ok = {"strict_all_expected": 1.0, "min_candidate_mass": 0.999,
              "fit_ok": True, "protocol": {"target_boost": 1}}
    _place_oracle(out_base, "matched_retrain_balanced_sX", matched_soft,
                  fit_ok)
    _place_oracle(out_base, "matched_retrain_sX", matched_soft)
    _place_oracle(out_base, "loo_retrain_sX", loo_soft)
    cmp = gxm.compare_matched_boost("salmu", ctx, matrix, out_base, ["sX"])
    assert cmp["ablation"] == (
        "matched_retrain_balanced (target_boost=1) vs "
        "matched_retrain_weighted (target_boost=5) vs loo_retrain; same "
        "edited cells E")
    assert cmp["protocol_identical"]["only_difference"] == (
        "target_boost 1 (matched_retrain_balanced) vs 5 "
        "(matched_retrain_weighted)")
    conds = " ".join(cmp["promotion_conditions"])
    assert "D(E, matched_retrain_balanced) < D(E, loo_retrain)" in conds
    assert "Delta_balanced = D(E, loo_retrain) - " \
        "D(E, matched_retrain_balanced)" in conds
    assert "M_x1" not in conds and "M_x5" not in conds
    assert cmp["reference_naming"]["key_to_name"]["Mx5"] == \
        "matched_retrain_weighted"
    assert "UNCHANGED" in cmp["reference_naming"][
        "stored_numeric_keys_keep_the_shorthand"]
    # the stored numeric keys survive, and still hold the same numbers
    row = cmp["per_set"]["sX"]["transformation_targets"][0]
    for key in ("d_E_Mx1", "d_E_Mx5", "d_E_L", "delta_x1", "delta_x5",
                "Mx1_closer_than_L", "delta_x1_material", "Mx1_vs_Mx5"):
        assert key in row, f"a stored numeric key was renamed: {key}"
    assert row["d_E_Mx1"] < 1e-6 and row["d_E_Mx5"] < 1e-6
    assert cmp["promotes_all"] is True
    assert "x5" not in json.dumps(cmp["promotion_conditions"])


# ------------------------------------------------------------------ #
# cost of a re-aggregation must not erase cost of the run before it
# ------------------------------------------------------------------ #
def test_cumulative_elapsed_never_erases_the_cost_of_an_earlier_pass(tmp_path):
    """A CPU re-aggregation must not overwrite a GPU run's cost.

    ``elapsed_sec`` used to be this pass alone, so re-deriving a finished
    matrix on CPU replaced the manifest's record of a 244,305s run with the
    handful of seconds the re-derivation took.  The field describes what
    producing this state cost, so it accumulates -- and a pass that cannot
    read its predecessor says so instead of silently restarting from zero.
    """
    path = tmp_path / "run_manifest.json"
    # no predecessor: this pass is the whole total, and it says so
    out = gxm._cumulative_elapsed(path, time.time() - 12.5, "cpu")
    assert out["elapsed_prior_passes_sec"] == 0.0
    assert out["elapsed_this_pass_sec"] == pytest.approx(12.5, abs=1.0)
    assert out["elapsed_sec"] == pytest.approx(12.5, abs=1.0)
    assert "no earlier artifact" in out["elapsed_accumulation"]
    assert out["devices_used"] == ["cpu"]
    assert "elapsed_restoration" not in out
    # a predecessor holding a finished GPU run
    with open(path, "w") as f:
        json.dump({"elapsed_sec": 244305.5,
                   "gpu": "NVIDIA RTX 6000 Ada Generation"}, f)
    out2 = gxm._cumulative_elapsed(path, time.time() - 3.0, "cpu")
    assert out2["elapsed_prior_passes_sec"] == 244305.5
    assert out2["elapsed_sec"] == pytest.approx(244308.5, abs=1.0)
    assert out2["elapsed_sec"] > 244305.5, "the GPU cost survives"
    assert out2["devices_used"] == ["NVIDIA RTX 6000 Ada Generation", "cpu"]
    assert "prior elapsed_sec read from the artifact" \
        in out2["elapsed_accumulation"]
    assert "erases nothing" in out2["elapsed_covers"]
    # an unreadable predecessor is reported loudly, not silently zeroed
    path.write_text("{not json", encoding="utf-8")
    out3 = gxm._cumulative_elapsed(path, time.time() - 1.0, "cpu")
    assert out3["elapsed_prior_passes_sec"] == 0.0
    assert "UNREADABLE" in out3["elapsed_accumulation"]
    assert "UNDERSTATES every run before it" in out3["elapsed_accumulation"]
    # a restoration note is carried forward rather than dropped
    with open(path, "w") as f:
        json.dump({"elapsed_sec": 100.0,
                   "elapsed_restoration": {"restored_from": "run log"}}, f)
    out4 = gxm._cumulative_elapsed(path, time.time(), "cpu")
    assert out4["elapsed_restoration"] == {"restored_from": "run log"}
    assert out4["elapsed_prior_passes_sec"] == 100.0


def test_every_cost_bearing_artifact_accumulates_rather_than_overwriting():
    """No site may still write a per-pass elapsed_sec.

    Three artifacts carry a cost -- the GX2B report, the GX2S report and the
    run manifest -- and fixing two of the three would leave the third erasing
    its predecessor on the next CPU re-aggregation.
    """
    src = inspect.getsource(gxm)
    assert src.count("_cumulative_elapsed(") >= 4, \
        "the definition plus GX2B, GX2S and the run manifest"
    assert '"elapsed_sec": round(time.time() - t_start, 1)' not in src, \
        "a per-pass elapsed_sec is still being written somewhere"


def test_fit_metrics_from_soft():
    ctx, entry = _ctx(), _entry()
    soft = {i: _summary(gxm.expected_label(ctx, entry, i))
            for i in ctx["identity_ids"]}
    m = gxm._fit_metrics_from_soft(soft, ctx, entry)
    assert m["fit_proxy_argmax"] == 1.0 and m["fit_proxy_n"] == 3
    assert m["min_candidate_mass_proxy"] > 0.99
    # break the target -> proxy drops
    soft["i1"] = _summary("alpha")
    m2 = gxm._fit_metrics_from_soft(soft, ctx, entry)
    assert abs(m2["fit_proxy_argmax"] - 2 / 3) < 1e-9


def test_ft_protocol_recorded_for_finetune_families():
    assert gxm.FT_PROTOCOL["steps"] == 3000
    assert "baseline h" in gxm.FT_PROTOCOL["note"]
    assert set(gxm.ORACLE_FAMILIES) == {
        "matched_finetune", "loo_finetune", "matched_retrain",
        "loo_retrain"}
    # directory naming keeps the G3 finetune artifacts addressable
    assert gxm.ORACLE_DIR_BY_FAMILY["matched_finetune"] == "matched_{sid}"
    assert gxm.ORACLE_DIR_BY_FAMILY["loo_finetune"] == "loo_{sid}"
    assert gxm.ORACLE_DIR_BY_FAMILY["matched_retrain"] \
        == "matched_retrain_{sid}"


# ------------------------------------------------------------------ #
# GX2B balanced matched-retrain ablation (target_boost=1)
# ------------------------------------------------------------------ #
def test_balanced_constants_and_rep_sets_cover_all_types():
    assert gxm.RETRAIN_TARGET_BOOST_BALANCED == 1
    assert gxm.RETRAIN_TARGET_BOOST == 5
    assert gxm.BALANCED_FAMILY == "matched_retrain_balanced"
    assert gxm.WEIGHTED_FAMILY == "matched_retrain"
    reps = gxm.BALANCED_REP_SETS
    assert len(reps["salmu"]) == 3 and len(reps["celeba_numeric"]) == 2
    # representative sets must exist in the frozen matrices and span the
    # transformation types (L1 single, L2 single, mixed; narrow, broad)
    sm = json.loads((gxm.MANIFEST_DIR / "matrix_salmu.json").read_text())
    nm = gx.build_numeric_matrix(gx.build_numeric_manifest())
    sal_modes = {e["set_id"]: e["mode"] for e in sm["sets"]}
    num_modes = {e["set_id"]: e["mode"] for e in nm["sets"]}
    for sid in reps["salmu"]:
        assert sid in sal_modes
    assert {sal_modes[s] for s in reps["salmu"]} >= {
        "single_level1", "single_level2", "simultaneous_mixed_depth"}
    for sid in reps["celeba_numeric"]:
        assert sid in num_modes
    assert {num_modes[s] for s in reps["celeba_numeric"]} >= {
        "single_exact_to_narrow", "single_exact_to_broad"}


def test_train_oracle_retrain_balanced_boost_is_one(monkeypatch, tmp_path):
    ctx, entry = _ctx(), _entry()
    calls = {"items": [], "train": None, "fresh": 0}

    def fake_build(adapter, processor, pairs, repeat):
        calls["items"].append(([p["answer"] for p in pairs], repeat))
        return [(p["answer"], repeat) for p in pairs]

    def fake_train(name, adapter, model, processor, items, out_dir, device,
                   steps, warmup, lr):
        calls["train"] = {"name": name, "steps": steps, "warmup": warmup,
                          "lr": lr}
        (Path(out_dir) / "adapter_final").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(gxm.rv, "build_supervised_items", fake_build)
    monkeypatch.setattr(gxm.rv, "train_supervised", fake_train)
    monkeypatch.setattr(gxm, "_soft_all", lambda s, c, a: {
        i: _summary(gxm.expected_label(c, entry, i))
        for i in c["identity_ids"]})
    monkeypatch.setattr(gxm, "_strict_accuracy", lambda s, c, e, a: 1.0)
    monkeypatch.setattr(gxm, "session_reset_fresh",
                        lambda s, seed=17: calls.__setitem__(
                            "fresh", calls["fresh"] + 1))
    session = _StubSessionFresh()
    out = tmp_path / "matched_retrain_balanced_sX"
    rec = gxm.train_oracle_retrain(
        session, "salmu", ctx, entry, gxm.BALANCED_FAMILY, out,
        _FakeArgs(), target_boost=gxm.RETRAIN_TARGET_BOOST_BALANCED)
    answers = [ans for group, _ in calls["items"] for ans in group]
    # BALANCED: transformed target appears exactly ONCE (boost=1), not x5
    assert answers.count("beta") == 1
    assert "alpha" not in answers                 # source label absent
    # protocol IDENTICAL to weighted: steps/warmup/lr/repeat unchanged
    assert calls["train"]["steps"] == gxm.RETRAIN_STEPS == 3000
    assert calls["train"]["warmup"] == gxm.RETRAIN_WARMUP == 200
    assert calls["train"]["lr"] == gxm.RETRAIN_LR == 2e-5
    assert calls["items"][0][1] == gxm.RETRAIN_REPEAT == 50
    assert calls["fresh"] == 1 and rec["target_boost"] == 1
    # matched-type fit scope (fits transformed + retained), fresh init
    with open(out / "oracle_results.json") as f:
        res = json.load(f)
    assert res["family"] == gxm.BALANCED_FAMILY
    assert res["init"] == "fresh_lora"
    assert res["protocol"]["target_boost"] == 1
    assert res["strict_fit_scope"] == "all_transformed_and_retained"
    assert res["fit_ok"] is True


def _place_oracle(out_base, dir_name, soft, res=None):
    d = out_base / "oracles" / dir_name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "oracle_soft.json", "w") as f:
        json.dump(soft, f)
    if res is not None:
        with open(d / "oracle_results.json", "w") as f:
            json.dump(res, f)


def test_compare_matched_boost_promotion_and_negative(tmp_path):
    ctx, entry = _ctx(), _entry()
    matrix = {"sets": [entry]}
    exp = {i: gxm.expected_label(ctx, entry, i) for i in ctx["identity_ids"]}
    base = dict(ctx["baseline_alias_of"])
    # edited cell E: target transformed, retained at baseline
    out_base = tmp_path
    cell_dir = out_base / "cells" / "sX" / "seed_17"
    cell_dir.mkdir(parents=True)
    with open(cell_dir / "cell_results.json", "w") as f:
        json.dump({"cell_id": "sX__seed17", "set_id": "sX", "seed": 17,
                   "soft_probs_full": {i: _summary(exp[i])["probs"]
                                       for i in ctx["identity_ids"]}}, f)
    matched_soft = {i: _summary(exp[i]) for i in ctx["identity_ids"]}
    loo_soft = {i: _summary(base[i]) for i in ctx["identity_ids"]}
    fit_ok = {"strict_all_expected": 1.0, "min_candidate_mass": 0.999,
              "fit_ok": True, "protocol": {"target_boost": 1}}

    # ---- PROMOTION: balanced fits, M_x1 ~ E, L far from E ----
    _place_oracle(out_base, "matched_retrain_balanced_sX", matched_soft,
                  fit_ok)
    _place_oracle(out_base, "matched_retrain_sX", matched_soft)
    _place_oracle(out_base, "loo_retrain_sX", loo_soft)
    cmp = gxm.compare_matched_boost("salmu", ctx, matrix, out_base, ["sX"])
    assert cmp["promotes_all"] is True
    s = cmp["per_set"]["sX"]
    assert s["set_promotes"] is True and s["failed_conditions"] == []
    row = s["transformation_targets"][0]
    assert row["d_E_Mx1"] < 1e-6 and row["d_E_Mx5"] < 1e-6
    assert row["d_E_L"] > 1.0
    assert row["delta_x1"] >= gxm.DELTA_RETRAIN_MIN_MARGIN
    assert row["delta_x5"] >= gxm.DELTA_RETRAIN_MIN_MARGIN
    assert row["Mx1_closer_than_L"] and row["delta_x1_material"]

    # ---- NEGATIVE: balanced does NOT fit (M_x1 == L far from E) ----
    tmp2 = tmp_path / "neg"
    (tmp2 / "cells" / "sX" / "seed_17").mkdir(parents=True)
    with open(tmp2 / "cells" / "sX" / "seed_17" / "cell_results.json",
              "w") as f:
        json.dump({"cell_id": "sX__seed17", "set_id": "sX", "seed": 17,
                   "soft_probs_full": {i: _summary(exp[i])["probs"]
                                       for i in ctx["identity_ids"]}}, f)
    bad_fit = {"strict_all_expected": 0.0, "min_candidate_mass": 0.4,
               "fit_ok": False, "protocol": {"target_boost": 1}}
    _place_oracle(tmp2, "matched_retrain_balanced_sX", loo_soft, bad_fit)
    _place_oracle(tmp2, "matched_retrain_sX", matched_soft)
    _place_oracle(tmp2, "loo_retrain_sX", loo_soft)
    cmp2 = gxm.compare_matched_boost("salmu", ctx, matrix, tmp2, ["sX"])
    assert cmp2["promotes_all"] is False
    s2 = cmp2["per_set"]["sX"]
    assert s2["set_promotes"] is False
    assert "balanced_fit" in s2["failed_conditions"]
    assert any("delta_x1_below_margin" in c or "Mx1_not_closer_than_L" in c
               for c in s2["failed_conditions"])


def test_compare_matched_boost_excludes_refusal_from_verdict(tmp_path):
    # mixed set: 1 taxonomic target + 1 refusal control; refusal is
    # reported but never gates the promotion verdict
    ctx = _ctx()
    entry = json.loads(json.dumps(_entry()))
    entry["mode"] = "simultaneous_mixed_depth"
    entry["assignments"]["i3"] = {"source": "delta",
                                  "target": gx.DELETED_LABEL,
                                  "operation": "refusal",
                                  "target_depth": None}
    entry["retain_ids"] = ["i2"]
    matrix = {"sets": [entry]}
    exp = {i: gxm.expected_label(ctx, entry, i) for i in ctx["identity_ids"]}
    base = dict(ctx["baseline_alias_of"])
    out_base = tmp_path
    (out_base / "cells" / "sX" / "seed_17").mkdir(parents=True)
    with open(out_base / "cells" / "sX" / "seed_17" / "cell_results.json",
              "w") as f:
        json.dump({"cell_id": "sX__seed17", "set_id": "sX", "seed": 17,
                   "soft_probs_full": {i: _summary(exp[i])["probs"]
                                       for i in ctx["identity_ids"]}}, f)
    matched_soft = {i: _summary(exp[i]) for i in ctx["identity_ids"]}
    loo_soft = {i: _summary(base[i]) for i in ctx["identity_ids"]}
    fit_ok = {"strict_all_expected": 1.0, "min_candidate_mass": 0.999,
              "fit_ok": True, "protocol": {"target_boost": 1}}
    _place_oracle(out_base, "matched_retrain_balanced_sX", matched_soft,
                  fit_ok)
    _place_oracle(out_base, "matched_retrain_sX", matched_soft)
    _place_oracle(out_base, "loo_retrain_sX", loo_soft)
    cmp = gxm.compare_matched_boost("salmu", ctx, matrix, out_base, ["sX"])
    s = cmp["per_set"]["sX"]
    # verdict driven by the taxonomic target only
    assert [r["target"] for r in s["transformation_targets"]] == ["i1"]
    assert [r["target"] for r in s["refusal_controls"]] == ["i3"]
    assert s["set_promotes"] is True


# ------------------------------------------------------------------ #
# GX2S oracle-seed sensitivity (matched/LOO retrain at seeds 42, 123)
# ------------------------------------------------------------------ #
def test_retrain_oracle_dir_naming_and_seeds():
    assert gxm.ORACLE_SEEDS_SENSITIVITY == (17, 42, 123)
    # seed 17 keeps the EXISTING unsuffixed dir (main-matrix reuse)
    assert gxm.retrain_oracle_dir("matched_retrain", "sX", 17) \
        == "matched_retrain_sX"
    assert gxm.retrain_oracle_dir("loo_retrain", "sX", 17) \
        == "loo_retrain_sX"
    # other seeds get an explicit suffix
    assert gxm.retrain_oracle_dir("matched_retrain", "sX", 42) \
        == "matched_retrain_sX__oseed42"
    assert gxm.retrain_oracle_dir("loo_retrain", "sX", 123) \
        == "loo_retrain_sX__oseed123"


def test_train_oracle_retrain_threads_oracle_seed(monkeypatch, tmp_path):
    ctx, entry = _ctx(), _entry()
    seen = {"fresh_seed": []}
    monkeypatch.setattr(gxm.rv, "build_supervised_items",
                        lambda a, p, pairs, repeat:
                        [(x["answer"], repeat) for x in pairs])
    monkeypatch.setattr(gxm.rv, "train_supervised",
                        lambda name, a, m, p, items, out_dir, device,
                        steps, warmup, lr:
                        (Path(out_dir) / "adapter_final").mkdir(
                            parents=True, exist_ok=True))
    monkeypatch.setattr(gxm, "_soft_all", lambda s, c, a: {
        i: _summary(gxm.expected_label(c, entry, i))
        for i in c["identity_ids"]})
    monkeypatch.setattr(gxm, "_strict_accuracy", lambda s, c, e, a: 1.0)
    monkeypatch.setattr(gxm, "session_reset_fresh",
                        lambda s, seed=17: seen["fresh_seed"].append(seed))
    session = _StubSessionFresh()
    out = tmp_path / "matched_retrain_sX__oseed42"
    rec = gxm.train_oracle_retrain(
        session, "salmu", ctx, entry, "matched_retrain", out, _FakeArgs(),
        oracle_seed=42)
    # the oracle seed drives BOTH the fresh init and the recorded protocol
    assert seen["fresh_seed"] == [42]
    assert rec["oracle_seed"] == 42
    with open(out / "oracle_results.json") as f:
        res = json.load(f)
    assert res["protocol"]["seed"] == 42
    # weighted (x5) by default -- GX2S varies the SEED, not the boost
    assert res["protocol"]["target_boost"] == gxm.RETRAIN_TARGET_BOOST == 5


def _place_cell(out_base, sid, seed, probs_by_iid):
    d = out_base / "cells" / sid / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cell_results.json", "w") as f:
        json.dump({"cell_id": f"{sid}__seed{seed}", "set_id": sid,
                   "seed": seed, "soft_probs_full": probs_by_iid}, f)


def test_compare_oracle_seed_sensitivity_paired_table_and_gate(tmp_path):
    ctx, entry = _ctx(), _entry()
    matrix = {"sets": [entry]}
    exp = {i: gxm.expected_label(ctx, entry, i) for i in ctx["identity_ids"]}
    base = dict(ctx["baseline_alias_of"])
    out_base = tmp_path
    # edited cells at seeds 17 and 42 (123 pending -> coverage note)
    for es in (17, 42):
        _place_cell(out_base, "sX", es,
                    {i: _summary(exp[i])["probs"]
                     for i in ctx["identity_ids"]})
    matched = {i: _summary(exp[i]) for i in ctx["identity_ids"]}
    loo = {i: _summary(base[i]) for i in ctx["identity_ids"]}
    for os_ in (17, 42):
        _place_oracle(out_base, gxm.retrain_oracle_dir(
            "matched_retrain", "sX", os_), matched)
        _place_oracle(out_base, gxm.retrain_oracle_dir(
            "loo_retrain", "sX", os_), loo)
    cmp = gxm.compare_oracle_seed_sensitivity(
        "salmu", ctx, matrix, out_base, ["sX"], oracle_seeds=(17, 42))
    # 2 edit seeds x 2 oracle seeds = 4 paired rows
    assert len(cmp["paired_table"]) == 4
    assert all(r["status"] == "ok" for r in cmp["paired_table"])
    for r in cmp["rows"]:
        assert r["d_matched"] < 1e-6 and r["d_loo"] > 1.0
        assert r["delta"] > gxm.DELTA_RETRAIN_MIN_MARGIN
    g = cmp["gate"]
    assert g["status"] == "evaluated"
    assert g["passed"] and g["passed_strong"]
    assert g["all_edit_seeds_closer_to_all_matched_oracle_seeds"] is True
    assert g["worst_case_delta"] >= gxm.DELTA_RETRAIN_MIN_MARGIN
    assert g["worst_case_delta_ge_margin"] is True
    # coverage: edit seed 123 not yet present
    assert cmp["pending_edit_seeds"]["sX"] == [123]


def test_compare_oracle_seed_sensitivity_missing_oracle_and_negative(tmp_path):
    ctx, entry = _ctx(), _entry()
    matrix = {"sets": [entry]}
    exp = {i: gxm.expected_label(ctx, entry, i) for i in ctx["identity_ids"]}
    base = dict(ctx["baseline_alias_of"])
    out_base = tmp_path
    _place_cell(out_base, "sX", 17,
                {i: _summary(exp[i])["probs"] for i in ctx["identity_ids"]})
    matched = {i: _summary(exp[i]) for i in ctx["identity_ids"]}
    loo = {i: _summary(base[i]) for i in ctx["identity_ids"]}
    # seed-17 present (matched close, LOO far -> positive delta);
    # seed-42 oracles absent -> oracle_missing row
    _place_oracle(out_base, gxm.retrain_oracle_dir(
        "matched_retrain", "sX", 17), matched)
    _place_oracle(out_base, gxm.retrain_oracle_dir(
        "loo_retrain", "sX", 17), loo)
    cmp = gxm.compare_oracle_seed_sensitivity(
        "salmu", ctx, matrix, out_base, ["sX"], oracle_seeds=(17, 42))
    status = {(r["edit_seed"], r["oracle_seed"]): r["status"]
              for r in cmp["paired_table"]}
    assert status[(17, 17)] == "ok"
    assert status[(17, 42)] == "oracle_missing"
    # gate evaluated on the single ok row only
    assert cmp["gate"]["n_pairs"] == 1 and cmp["gate"]["passed"] is True


# ------------------------------------------------------------------ #
# the dirty-code gate: what a parallel ablation refuses to run against
# ------------------------------------------------------------------ #
def test_the_dirty_code_gate_is_scoped_to_the_executed_scripts(monkeypatch):
    monkeypatch.chdir(_ROOT)
    monkeypatch.setattr(gxm, "GX_CODE", ["scripts/e2c_v3_granularity.py",
                                         "scripts/e2c_v3_matrix.py"])
    assert gxm._dirty_tracked_code() == []          # committed, so clean
    # A missing declared script is reported instead of silently widening the
    # pathspec to the whole worktree, which would blame this ablation for the
    # edits the parallel method-baseline/panel/RG runs are making -- the whole
    # point of scoping the gate, since GX2B/GX2S run BESIDE a main matrix that
    # legitimately dirties tracked result files.
    monkeypatch.setattr(gxm, "GX_CODE", ["scripts/does_not_exist.py"])
    assert gxm._dirty_tracked_code() == [
        "<declared executed code missing: scripts/does_not_exist.py>"]
    monkeypatch.setattr(gxm, "GX_CODE", ["scripts/e2c_v3_granularity.py",
                                         "scripts/also_missing.py"])
    assert gxm._dirty_tracked_code() == [
        "<declared executed code missing: scripts/also_missing.py>"]


def test_every_script_the_matrix_runner_executes_is_declared():
    for path in ("scripts/e2c_v3_granularity_matrix.py",
                 "scripts/e2c_v3_granularity.py",
                 "scripts/e2c_v3_research_validity.py",
                 "scripts/e2c_v3_matrix.py",
                 "scripts/e2c_v3_realdata.py"):
        assert path in gxm.GX_CODE, path
        assert (_ROOT / path).exists(), path
    # the runner itself is first: an ablation that forgot to declare its own
    # script would gate on everything except the code it is running
    assert gxm.GX_CODE[0] == "scripts/e2c_v3_granularity_matrix.py"
    # The other three runners declare their own script first and then the same
    # five shared ones in the same order.  This runner IS granularity_matrix,
    # so that entry is its own and the remaining four follow in the same order:
    # four runners, one rule, and a shared script renamed or dropped anywhere
    # fails in all four.
    assert gxm.GX_CODE == [
        "scripts/e2c_v3_granularity_matrix.py",
        "scripts/e2c_v3_research_validity.py",
        "scripts/e2c_v3_granularity.py",
        "scripts/e2c_v3_matrix.py",
        "scripts/e2c_v3_realdata.py"]
