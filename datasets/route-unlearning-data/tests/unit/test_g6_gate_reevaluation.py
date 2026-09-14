"""The gate re-derivation must not silently report "no difference".

This tool exists to file the repaired gate criteria BESIDE the sealed pilot
reports rather than over them, so its whole output is a comparison.  A
comparison that reads the wrong path does not fail loudly -- it reports that
nothing differs, which reads as a finding rather than as a bug.  That happened
once while writing it: ``evaluate_gates`` nests the per-gate results under a key
also called ``gates``, and the sealed file wraps that result under its own
``gates`` key, so unwrapping one level too many produced an empty dict and a
delta claiming the matched-oracle gate was unchanged -- when in fact its
criteria, its oracle families and its unit of aggregation had all been replaced.

The tests below pin the comparison against that failure mode, and pin the two
properties that make the artifact trustworthy: that it re-derives from stored
evidence without touching a sealed byte, and that it does so without importing
torch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
TOOL = _SCRIPTS / "e2c_v3_g6_gate_reevaluation.py"
OUT_BASE = _ROOT / "e2c_granularity" / "outputs" / "mllmu"
SEALED = ("g6_pilot_gates.json", "granularity_summary.json",
          "run_manifest.json")

#: Adapters ``verify_content_bindings`` recomputes: the pilot's own cells and
#: the two GATE oracle families.
#:
#: The weights are gitignored and live in the HF revision-pinned archive, so a
#: fresh clone -- which is what CI checks out -- carries the sealed JSON evidence
#: (cell_results.json, run_manifest.json, the oracle summaries, adapter_config.json)
#: and NONE of the adapter bytes.  ``OUT_BASE.is_dir()`` is therefore true there,
#: so a guard on the directory alone lets these tests run against a tree that
#: cannot satisfy them, and the binding correctly refuses: an absent adapter
#: cannot be hashed, and "14 of 15 bound" is indistinguishable from having bound
#: the evidence.
#:
#: Enumerated from the TRACKED file beside each adapter, so the list is derived
#: from the repository and picks up a new cell without being edited.  It
#: deliberately does NOT glob every ``adapter_final`` under the output tree: that
#: tree also holds the protocol-study cells, the two non-gate oracle families and
#: an unrelated ``route_h`` adapter, none of which this binding touches, and a
#: guard that counted them would skip these tests whenever any of THOSE lost its
#: weights.  Under-enumerating is the safe direction -- it fails loudly in CI
#: rather than skipping silently.
_GATE_ORACLE_FAMILIES = ("matched_retrain", "loo_retrain")


def _sealed_adapters_the_binding_needs():
    paths = [cr.parent / "edited_h" / "adapter_final"
             / "adapter_model.safetensors"
             for cr in sorted(OUT_BASE.glob("cells/*/seed_*/cell_results.json"))]
    for fam in _GATE_ORACLE_FAMILIES:
        paths += [cfg.parent / "adapter_model.safetensors"
                  for cfg in sorted(OUT_BASE.glob(
                      f"oracles/{fam}_*/adapter_final/adapter_config.json"))]
    return paths


_SEALED_ADAPTERS = _sealed_adapters_the_binding_needs()
_ABSENT_ADAPTERS = [str(p.relative_to(OUT_BASE)) for p in _SEALED_ADAPTERS
                    if not p.is_file()]

_needs_sealed_evidence = pytest.mark.skipif(
    not OUT_BASE.is_dir() or bool(_ABSENT_ADAPTERS),
    reason=("runs against the REAL sealed pilot evidence, which needs the "
            "adapter weights beside it; those are gitignored and live in the HF "
            "revision-pinned archive, so a fresh clone has the committed JSON "
            "evidence but not the bytes"
            + (f" -- {len(_ABSENT_ADAPTERS)} of {len(_SEALED_ADAPTERS)} absent, "
               f"e.g. {_ABSENT_ADAPTERS[0]}" if _ABSENT_ADAPTERS else "")
            + ". Skipping is not a weakened check: the binding's BEHAVIOUR is "
              "pinned without the weights by the hermetic tests over a synthetic "
              "pilot tree, which forge an adapter, hand-edit a cell record and "
              "delete an adapter, and those do run everywhere"))


def _load(name, filename=None):
    path = _SCRIPTS / (filename or Path(name).name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load("g6_reeval_under_test", "e2c_v3_g6_gate_reevaluation.py")


def _evaluation(per_gate, passed=True, failed=()):
    """An ``evaluate_gates``-shaped result: the per-gate dict sits under a key
    that is itself called ``gates``."""
    return {"passed": passed, "failed_gates": list(failed),
            "coverage_defects": [], "gates": per_gate,
            "coverage": {"exact": True},
            "proceed_to_full_matrix": passed}


def _sealed_file(tmp_path, block, n_cells_adapted=15):
    """A ``g6_pilot_gates.json``-shaped FILE: it wraps the evaluation once
    more under its own ``gates`` key."""
    p = tmp_path / "g6_pilot_gates.json"
    p.write_text(json.dumps({"gates": block, "sets": [],
                             "n_cells_adapted": n_cells_adapted,
                             "provenance": {}}), encoding="utf-8")
    return p


def test_a_criteria_change_is_reported_even_when_the_verdict_does_not_move(
        tool, tmp_path):
    """The load-bearing case, and the one the nesting bug hid.

    ``matched_oracle`` PASSED under the sealed criteria and passes under the
    repaired ones, so a delta that compared pass flags would report no change
    across a gate whose evidence was replaced wholesale: averaged
    ``*_finetune`` deltas per set, over zero per-target rows, becoming
    every-target-per-seed ``*_retrain`` conditions that also check oracle fit.
    """
    old = _evaluation({"matched_oracle": {
        "name": "matched_oracle", "passed": True, "failures": [],
        "criteria": {"min_mean_delta_oracle": 0.0,
                     "aggregation": "mean over edit seeds, per set"},
        "per_set": {}}})
    new = _evaluation({"matched_oracle": {
        "name": "matched_oracle", "passed": True, "failures": [],
        "criteria": {"oracle_families": ["matched_retrain", "loo_retrain"],
                     "aggregation": "every target identity of every edit seed "
                                    "must pass; nothing is averaged"},
        "per_set": {"gx_mll_151252": {"n_target_rows": 6,
                                      "worst_case_delta_retrain": 1.2088}}}})
    delta = tool.delta_vs_sealed(new, _sealed_file(tmp_path, old))

    assert delta["available"] is True
    # the verdict did not move, and the delta must say so rather than invent
    # a change
    assert delta["verdict_fields"] == {}
    # but the gate is not the same gate, and that MUST be reported
    assert "matched_oracle" in delta["gates_that_differ"]
    assert "passed" not in delta["gates_that_differ"]["matched_oracle"]
    crit = delta["gates_that_differ"]["matched_oracle"]["criteria"]
    assert crit["keys_added"] == ["oracle_families"]
    assert crit["keys_removed"] == ["min_mean_delta_oracle"]
    mo = delta["matched_oracle"]
    assert mo["sealed"]["oracle_families"] is None
    assert mo["reevaluated"]["oracle_families"] == ["matched_retrain",
                                                    "loo_retrain"]
    assert mo["sealed"]["n_target_rows"] == 0
    assert mo["reevaluated"]["n_target_rows"] == 6
    assert mo["sealed"]["aggregation"] == "mean over edit seeds, per set"
    assert mo["reevaluated"]["worst_case_delta_retrain"] == {
        "gx_mll_151252": 1.2088}


def test_a_verdict_change_is_also_reported(tool, tmp_path):
    """The other direction: a gate that flips must be named as flipped."""
    old = _evaluation({"retention": {"name": "retention", "passed": False,
                                     "failures": ["a", "b"], "criteria": {}}},
                      passed=False, failed=["retention"])
    new = _evaluation({"retention": {"name": "retention", "passed": True,
                                     "failures": [], "criteria": {}}})
    delta = tool.delta_vs_sealed(new, _sealed_file(tmp_path, old))
    assert delta["verdict_fields"]["passed"] == {"sealed": False,
                                                 "reevaluated": True}
    assert delta["gates_that_differ"]["retention"]["passed"] == {
        "sealed": False, "reevaluated": True}
    assert delta["gates_that_differ"]["retention"]["n_failures"] == {
        "sealed": 2, "reevaluated": 0}


def test_an_empty_per_gate_read_raises_instead_of_reporting_no_difference(
        tool, tmp_path):
    """The regression guard for the bug this module was written around.

    If the per-gate path resolves to nothing, "nothing differs" is not a
    finding -- it is the tool failing to find the data.  Both sides are
    checked, because either one going empty produces the same false all-clear.
    """
    empty = {"passed": True, "failed_gates": [], "coverage_defects": [],
             "coverage": {"exact": True}}
    with pytest.raises(RuntimeError, match="no per-gate results"):
        tool.delta_vs_sealed(empty, _sealed_file(tmp_path, _evaluation({})))

    good_new = _evaluation({"retention": {"name": "retention",
                                          "passed": True, "failures": [],
                                          "criteria": {}}})
    sealed = _sealed_file(tmp_path, {"passed": True, "failed_gates": [],
                                     "not_the_gates_key": {}})
    with pytest.raises(RuntimeError, match="no per-gate results"):
        tool.delta_vs_sealed(good_new, sealed)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pilot_tree(tmp_path, sids=("gx_mll_151252", "gx_mll_254012"),
                seeds=(17,), families=("matched_retrain", "loo_retrain")):
    """A complete miniature of the pilot output tree.

    Each cell gets a ``cell_results.json`` and an ``adapter_model.safetensors``;
    each gate oracle gets the same pair; and ``run_manifest.json`` records the
    digests of those very bytes.  Writing the manifest from the same bytes the
    cells record is what makes the three-way check pass, and it is also what
    makes a tamper detectable: change any byte and one of the three disagrees.

    Returns ``(cells, manifest)`` as the loaded records.
    """
    ckpts = {}
    cells = []
    for sid in sids:
        for seed in seeds:
            cell_id = f"{sid}__seed{seed}"
            d = tmp_path / "cells" / sid / f"seed_{seed}"
            (d / "edited_h" / "adapter_final").mkdir(parents=True)
            ad = d / "edited_h" / "adapter_final" / "adapter_model.safetensors"
            ad.write_bytes(b"adapter:" + cell_id.encode())
            ckpts[f"cell_{cell_id}"] = _sha(ad)
            rec = {"cell_id": cell_id, "set_id": sid, "seed": seed,
                   "checkpoint_sha256": ckpts[f"cell_{cell_id}"]}
            (d / "cell_results.json").write_text(json.dumps(rec),
                                                 encoding="utf-8")
            cells.append(rec)
        for fam in families:
            od = tmp_path / "oracles" / f"{fam}_{sid}" / "adapter_final"
            od.mkdir(parents=True)
            oa = od / "adapter_model.safetensors"
            oa.write_bytes(b"oracle:" + f"{fam}_{sid}".encode())
            ckpts[f"oracle_{fam}_{sid}"] = _sha(oa)
            (od.parent / "oracle_results.json").write_text(
                json.dumps({"family": fam, "set_id": sid, "fit_ok": True}),
                encoding="utf-8")

    oracle_fit = {
        sid: {fam: {"fit_ok": True, "strict_all_expected": 1.0,
                    "min_candidate_mass": 0.999,
                    "sha256": ckpts[f"oracle_{fam}_{sid}"]}
              for fam in families}
        for sid in sids}
    (tmp_path / "oracles").mkdir(exist_ok=True)
    (tmp_path / "oracles" / "oracles_summary.json").write_text(
        json.dumps(oracle_fit), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"checkpoints_sha256": ckpts}), encoding="utf-8")
    (tmp_path / "g6_pilot_gates.json").write_text(
        json.dumps({"n_cells_adapted": len(cells), "gates": {}}),
        encoding="utf-8")
    return cells, {"checkpoints_sha256": ckpts}


def test_the_cell_cross_check_refuses_a_different_cell_set(tool, tmp_path):
    """Identity, not count: a run that swapped one cell for another must not
    be compared against the pilot it claims to re-derive."""
    cells, _ = _pilot_tree(tmp_path)
    fit = json.loads((tmp_path / "oracles" / "oracles_summary.json")
                     .read_text(encoding="utf-8"))
    got = tool.cross_check_cells(cells, tmp_path, _sha, fit)
    assert got["checked"] is True
    assert got["identity"]["method"] == "cell_id set equality"
    assert got["identity"]["n_cells"] == 2
    assert got["content"]["verified"] is True

    # same COUNT, different identity -- a count check would have passed this
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"checkpoints_sha256": {
            "cell_gx_mll_151252__seed17": "deadbeef",
            "cell_gx_mll_191023__seed123": "deadbeef"}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cell identity disagrees"):
        tool.cross_check_cells(cells, tmp_path, _sha, fit)

    # and it refuses to degrade to "could not check"
    (tmp_path / "run_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="cannot cross-check"):
        tool.cross_check_cells(cells, tmp_path, _sha, fit)


def test_content_binding_verifies_bytes_not_just_cell_names(tool, tmp_path):
    """The point of the binding: same cell IDs, same recorded digest, but the
    evidence bytes differ -- which an identity-only check cannot see.

    A cell that keeps its name and its ``checkpoint_sha256`` while its adapter
    is replaced is exactly the forgery the binding exists to stop, and the
    sealed manifest is what catches it: the recomputed adapter digest no longer
    matches the digest the run itself filed.
    """
    cells, _ = _pilot_tree(tmp_path)
    fit = json.loads((tmp_path / "oracles" / "oracles_summary.json")
                     .read_text(encoding="utf-8"))

    ad = (tmp_path / "cells" / "gx_mll_151252" / "seed_17" / "edited_h"
          / "adapter_final" / "adapter_model.safetensors")
    ad.write_bytes(b"adapter:RETRAINED-BEHIND-AN-UNCHANGED-NAME")
    with pytest.raises(RuntimeError, match="content binding failed"):
        tool.cross_check_cells(cells, tmp_path, _sha, fit)


def test_content_binding_catches_a_hand_edited_cell_record(tool, tmp_path):
    """Editing the JSON while leaving the weights alone must also fail.

    This is the more likely accident -- a re-score written straight over a
    committed cell -- and it is invisible to a checkpoint-only check, because
    the adapter still hashes to what the manifest recorded.
    """
    cells, _ = _pilot_tree(tmp_path)
    fit = json.loads((tmp_path / "oracles" / "oracles_summary.json")
                     .read_text(encoding="utf-8"))

    p = tmp_path / "cells" / "gx_mll_254012" / "seed_17" / "cell_results.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["checkpoint_sha256"] = "0" * 64          # plausible shape, wrong value
    p.write_text(json.dumps(rec), encoding="utf-8")
    cells = [rec if c["cell_id"] == rec["cell_id"] else c for c in cells]
    with pytest.raises(RuntimeError, match="content binding failed"):
        tool.cross_check_cells(cells, tmp_path, _sha, fit)


def test_a_missing_adapter_is_a_refusal_not_a_partial_pass(tool, tmp_path):
    """An adapter that is gone cannot be recomputed, so its digest is
    unverified.  Reporting "14 of 15 bound" and continuing would let a reader
    take the artifact as fully bound."""
    cells, _ = _pilot_tree(tmp_path)
    fit = json.loads((tmp_path / "oracles" / "oracles_summary.json")
                     .read_text(encoding="utf-8"))
    (tmp_path / "oracles" / "loo_retrain_gx_mll_151252" / "adapter_final"
     / "adapter_model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="cannot be recomputed"):
        tool.cross_check_cells(cells, tmp_path, _sha, fit)


def test_the_binding_hashes_every_cell_and_gate_oracle_file(tool, tmp_path):
    """Coverage of the binding itself: a per-file hash table that silently
    omitted the oracles would still report ``verified: True``."""
    cells, _ = _pilot_tree(tmp_path, seeds=(17, 42))
    fit = json.loads((tmp_path / "oracles" / "oracles_summary.json")
                     .read_text(encoding="utf-8"))
    got = tool.cross_check_cells(cells, tmp_path, _sha, fit)

    assert len(got["cell_files"]) == 4                       # 2 sets x 2 seeds
    assert len(got["oracle_files"]) == 4                     # 2 sets x 2 fams
    assert got["content"]["n_cells_bound"] == 4
    assert got["content"]["n_oracles_bound"] == 4
    for entry in got["cell_files"].values():
        assert len(entry["cell_results_sha256"]) == 64
        assert entry["adapter_present"] is True
        assert entry["adapter_sha256_recomputed"] == \
            entry["checkpoint_sha256_recorded_by_cell"]
    for entry in got["oracle_files"].values():
        assert entry["oracle_results_present"] is True
        assert len(entry["oracle_results_sha256"]) == 64
    # only the two families the repaired gate actually reads are bound; the
    # *_finetune families are not this gate's evidence
    assert sorted(got["content"]["oracle_families_bound"]) == \
        ["loo_retrain", "matched_retrain"]


def test_the_score_sum_reading_is_recomputed_and_refuses_to_guess(tool):
    """Every figure in the terminology correction comes from the cells.

    The correction exists because ``candidate_mass`` was read as a probability,
    so a hardcoded "6 of 6" beside it would be the same class of error one
    level up: prose that survives the evidence moving.
    """
    def cell(seed, mass, ok=True, failed=("min_candidate_mass>=0.99",)):
        return {"set_id": tool.SCORE_SUM_SET, "seed": seed,
                "criteria": {"min_candidate_mass": mass,
                             "strict_expected_accuracy": 1.0 if ok else 0.5,
                             "cell_pass": not failed,
                             "failed_criteria": list(failed)},
                "hard_preds": [{"group": "target", "correct_post_edit": ok},
                               {"group": "target", "correct_post_edit": ok}]}

    got = tool.candidate_score_sum_reading(
        [cell(17, 0.9845), cell(42, 0.9812), cell(123, 0.9786)], 0.99)
    assert got["target_rows_correct"] == 6
    assert got["target_rows"] == 6
    assert got["all_target_rows_strictly_correct"] is True
    assert got["only_failed_criterion_is_the_score_sum"] is True
    assert got["candidate_mass_floor_cleared_on_any_seed"] is False
    assert got["min_candidate_mass_over_seeds"] == 0.9786
    assert "6 of 6" in got["verdict"]
    assert "NOT evidence that the transformation" in got["interpretation"]

    # a seed that clears the floor must move the reading, not be ignored
    cleared = tool.candidate_score_sum_reading(
        [cell(17, 0.995), cell(42, 0.9812)], 0.99)
    assert cleared["candidate_mass_floor_cleared_on_any_seed"] is True

    # and a wrong target output must break the "only the score sum" claim
    broke = tool.candidate_score_sum_reading(
        [cell(17, 0.9845, ok=False, failed=("strict", "min_candidate_mass>=0.99"))],
        0.99)
    assert broke["all_target_rows_strictly_correct"] is False
    assert broke["only_failed_criterion_is_the_score_sum"] is False

    with pytest.raises(RuntimeError, match="cannot be recomputed"):
        # no cell for the set at all -- an empty reading would report 0 of 0
        # strictly correct, which is the vacuous pass this guard exists to stop
        other = cell(17, 0.98)
        other["set_id"] = "gx_mll_191023"
        tool.candidate_score_sum_reading([other], 0.99)
    with pytest.raises(RuntimeError, match="cannot be recomputed"):
        tool.candidate_score_sum_reading([], 0.99)


def test_the_terminology_note_carries_no_measured_figures(tool):
    """The qualitative definition must stay free of MEASUREMENTS, so every
    number in the artifact traces to the cells it was computed from.

    The definition legitimately contains the literal 1 -- it states the formula
    ``other_mass = 1 - candidate_mass`` and that the sum is not constrained to
    1 -- so what is forbidden is a decimal, a count of rows, or the floor
    itself: those are measurements, and a measurement in prose outlives the
    evidence it was read from.
    """
    text = json.dumps(tool.METRIC_SEMANTICS)
    assert not re.search(r"\d+\.\d+", text), \
        "METRIC_SEMANTICS quotes a decimal measurement"
    assert not re.search(r"\d+\s+of\s+\d+", text), \
        "METRIC_SEMANTICS quotes a row count"
    assert "0.99" not in text, "the floor belongs to gx.PASS_CRITERIA"
    assert "probability" in tool.METRIC_SEMANTICS["candidate_mass"]["is_not"]
    assert "termination" in tool.METRIC_SEMANTICS["candidate_mass"][
        "why_it_need_not_reach_one"]
    # the score-sum set is identified by ID, and its figures are recomputed
    assert tool.SCORE_SUM_SET == "gx_mll_254012"


@_needs_sealed_evidence
def test_the_real_binding_covers_every_pilot_cell_and_gate_oracle(tool):
    """On the committed evidence: all fifteen cells and all ten gate oracles
    (five sets x two retrain families) bind, and the enforced side is the
    sealed manifest rather than a value this tool could have written."""
    block = tool.build(OUT_BASE)
    cc = block["inputs"]["cell_identity_and_content_cross_check"]
    assert cc["content"]["verified"] is True
    assert cc["content"]["n_cells_bound"] == 15
    assert cc["content"]["n_oracles_bound"] == 10
    assert len(cc["cell_files"]) == 15
    # and the guard deciding whether this test runs enumerates exactly what the
    # binding checked -- so it cannot drift into skipping these tests because
    # some unrelated adapter elsewhere in the tree lost its weights
    assert len(_SEALED_ADAPTERS) == (cc["content"]["n_cells_bound"]
                                     + cc["content"]["n_oracles_bound"])
    # the artifact must say WHICH side enforces: the sealed manifest, because
    # this tool never writes it.  Without that a reader cannot tell a binding
    # from a self-consistent re-derivation.
    why = cc["content"]["why_the_sealed_manifest_is_the_enforcing_side"]
    assert "never written by this tool" in why
    assert "three-way" in cc["content"]["method"]

    reading = block["metric_semantics"]["candidate_score_sum_reading"]
    assert reading["set_id"] == "gx_mll_254012"
    assert reading["target_rows_correct"] == reading["target_rows"] == 6
    assert reading["only_failed_criterion_is_the_score_sum"] is True
    assert reading["candidate_mass_floor_cleared_on_any_seed"] is False
    # the floor is read from the frozen criteria, and the artifact says so
    assert reading["candidate_mass_floor"] == 0.99
    assert block["metric_semantics"]["criteria_source"].startswith(
        "gx.PASS_CRITERIA")


def test_a_sealed_destination_is_refused_by_absolute_path_too(tool):
    """``--dest`` must not become a way to write over a sealed report by
    absolute path, which a name-only guard would permit.

    Deliberately unguarded: the refusal fires on the DESTINATION alone, before
    anything is re-derived, so it needs neither the adapter weights nor the
    sealed bytes and can be covered on a fresh clone as well as here.
    """
    for sealed_name in SEALED:
        with pytest.raises(RuntimeError, match="never over"):
            tool.main(["--out-base", str(OUT_BASE),
                       "--dest", str(OUT_BASE / sealed_name), "--dry-run"])
    # the default destination is still inside the tree and still not sealed
    assert (OUT_BASE / tool.OUTPUT_NAME).resolve() not in {
        (OUT_BASE / n).resolve() for n in SEALED}


@_needs_sealed_evidence
def test_dest_outside_the_tree_is_allowed_but_a_sealed_path_is_not(tool,
                                                                   tmp_path):
    """``--dest`` exists so the artifact can be derived at a clean worktree,
    which means it has to be able to write OUTSIDE the tracked output path."""
    ok = tmp_path / "elsewhere" / "reeval.json"
    assert tool.main(["--out-base", str(OUT_BASE), "--dest", str(ok)]) == 0
    assert ok.is_file()
    assert json.loads(ok.read_text(encoding="utf-8"))["kind"] == tool.KIND


def test_it_hashes_the_scoring_script_without_importing_it(tool):
    """The provenance digests must equal rv's own, computed without paying for
    rv's module-scope ``import torch``.  A sha256 is a function of the bytes,
    so reading the file directly is the same number -- asserted here rather
    than assumed, because a typo in the path would silently pin the wrong
    file's digest into a provenance block."""
    rv = _load("rv_for_digest_equality", "e2c_v3_research_validity.py")
    g6m = _load("g6m_for_digest_equality", "e2c_v3_mllmu_matrix.py")
    assert g6m.sha256_file(tool.RESEARCH_VALIDITY_PATH) == rv.script_sha256()
    assert g6m.sha256_file(tool.LABEL_PARSER_PATH) == rv.label_parser_sha256()
    assert rv.script_sha256() != rv.label_parser_sha256()


def test_the_tool_needs_no_torch():
    """A CPU re-derivation that imports the training stack is not the
    lightweight path Step 3c established.  Proven in a subprocess: pytest has
    already imported torch in-process, so the dependency would be invisible."""
    body = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('t', {str(TOOL)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        # the tool's own loader, on the module that owns the gates: __name__
        # would just echo back the label passed to spec_from_file_location, so
        # the proof is that the gate API is reachable through it
        "g6m = m._load_sibling('g6m_no_torch_probe',\n"
        "                      'e2c_v3_mllmu_matrix.py')\n"
        "print(json.dumps({\n"
        "    'gate_api_reachable': all(hasattr(g6m, a) for a in (\n"
        "        'evaluate_gates', 'cells_for_gates', 'load_frozen_g6',\n"
        "        'matched_oracle_gate', 'sha256_file')),\n"
        "    'sha_helper_hashes_rv': g6m.sha256_file(\n"
        "        m.RESEARCH_VALIDITY_PATH) != 'unknown',\n"
        "    'torch_still_blocked': sys.modules.get('torch') is None,\n"
        "    'no_torch_submodule': not any(\n"
        "        k.startswith('torch.') and sys.modules[k] is not None\n"
        "        for k in sys.modules)}))\n"
    )
    preamble = (
        "import sys\n"
        "for _m in ('torch', 'torch.nn', 'torch.utils', 'torch.utils.data',\n"
        "           'transformers'):\n"
        "    sys.modules[_m] = None\n"
    )
    proc = subprocess.run([sys.executable, "-c", preamble + body],
                          capture_output=True, text=True, cwd=str(_ROOT),
                          timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip())
    assert got["gate_api_reachable"] is True
    assert got["sha_helper_hashes_rv"] is True
    assert got["torch_still_blocked"] is True
    assert got["no_torch_submodule"] is True


@_needs_sealed_evidence
def test_the_re_derivation_is_deterministic_apart_from_the_worktree_flag(tool):
    """Building twice must give the same artifact, except for the one field
    that regenerating necessarily changes.

    ``provenance.clean_worktree`` records the tracked-only worktree
    determination unscoped -- that is what ``g6m.worktree_state`` intends, its
    ``exclude_prefixes`` reaching only the untracked accounting -- so writing
    this tool's own output makes the NEXT run see a modified tracked file.
    That is a property of regenerating, not drift, and the artifact says so in
    ``regeneration_note``.  Everything that is actually a re-derivation
    (inputs, gates, sets, delta_vs_sealed) has to be identical, or the filed
    JSON could not be checked against the commit that claims to have produced
    it.
    """
    def comparable(block):
        prov = {k: v for k, v in block["provenance"].items()
                if k != "clean_worktree"}
        return json.dumps({**{k: v for k, v in block.items()
                              if k != "provenance"},
                           "provenance": prov}, sort_keys=True)

    first = tool.build(OUT_BASE)
    second = tool.build(OUT_BASE)
    assert comparable(first) == comparable(second)
    assert first["provenance"]["regeneration_note"] == tool.REGENERATION_NOTE
    assert "dirty_tracked_only" in first["provenance"]["clean_worktree"]


def test_it_writes_a_new_file_and_never_a_sealed_one(tool):
    """The decision was to file the correction beside the sealed reports, so
    the output name must not be one of them -- enforced in code, and pinned
    here because a rename would defeat the guard."""
    assert tool.OUTPUT_NAME not in tool.SEALED_REPORTS
    assert sorted(tool.SEALED_REPORTS) == sorted(SEALED)
    assert tool.OUTPUT_NAME == "g6_pilot_gates_reevaluated.json"


@_needs_sealed_evidence
def test_the_real_re_derivation_leaves_every_sealed_byte_untouched(tool):
    """End to end on the committed evidence: the re-derivation must succeed,
    the cells it used must be the pilot's own, and the three sealed reports
    must be byte-identical afterwards."""
    before = {n: (OUT_BASE / n).read_bytes() for n in SEALED
              if (OUT_BASE / n).is_file()}
    assert before, "no sealed reports to protect"

    block = tool.build(OUT_BASE)

    after = {n: (OUT_BASE / n).read_bytes() for n in SEALED
             if (OUT_BASE / n).is_file()}
    assert after == before, "a sealed report changed during re-derivation"

    assert block["kind"] == tool.KIND
    assert block["n_cells_adapted"] == 15
    cc = block["inputs"]["cell_identity_and_content_cross_check"]
    assert cc["checked"] is True
    assert cc["content"]["verified"] is True
    assert block["gates"]["coverage"]["exact"] is True
    assert block["delta_vs_sealed"]["available"] is True
    # the delta must have found the per-gate results rather than reporting an
    # empty comparison -- the sealed side read zero target rows because the old
    # gate had no per-target rows at all, and the repaired side reads all 30
    mo = block["delta_vs_sealed"]["matched_oracle"]
    assert mo["reevaluated"]["oracle_families"] == ["matched_retrain",
                                                    "loo_retrain"]
    assert mo["reevaluated"]["n_target_rows"] == 30
    assert mo["sealed"]["n_target_rows"] == 0
