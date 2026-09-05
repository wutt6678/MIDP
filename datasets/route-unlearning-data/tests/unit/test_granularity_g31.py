"""CPU tests for the G3.1 oracle-family repair.

Covers the correction that the G3 "matched/LOO" oracles were continued
fine-tuning references (baseline-h init), not retraining references:
- fresh_reinit_lora / session_reset_fresh produce a FRESH init and never
  read (or preserve) any trained checkpoint -- proven by zeroing a
  pre-populated (simulated trained) lora_B;
- the retrain-family data recipe (matched_retrain sees transformed
  targets x5 + retained x50 under the route protocol; loo_retrain never
  sees the targets);
- oracle_family_distances computes Delta_FT and Delta_retrain separately
  with legacy aliases;
- GX2R rewrites the oracle block of stored cells from distributions only
  (edited checkpoints untouched);
- aggregation separates transformation targets from refusal controls,
  reports sibling coverage, configured-vs-executed seeds, and the G3.1
  promotion gate.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
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
    fresh-init helpers accept ONLY live parameters + a seed."""
    sig = inspect.signature(gxm.fresh_reinit_lora)
    assert list(sig.parameters) == ["named_params", "seed"]
    sig2 = inspect.signature(gxm.session_reset_fresh)
    assert list(sig2.parameters) == ["session"]
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
        self.adapter_model = _StubAdapterModel(
            [("l.lora_A.w", self.a), ("l.lora_B.w", self.b)])
        # attributes consumed by the (monkeypatched) training helpers
        self.adapter = object()
        self.processor = object()
        self.model = object()


def test_session_reset_fresh_overwrites_trained_state():
    s = _StubSessionFresh()
    gxm.session_reset_fresh(s)
    assert torch.count_nonzero(s.b).item() == 0
    assert torch.count_nonzero(s.a).item() > 0


def test_session_reset_fresh_fail_closed_without_lora():
    class Empty:
        adapter_model = _StubAdapterModel([("dense.weight",
                                            torch.zeros(2, 2))])
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
                        lambda session: calls.__setitem__(
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
