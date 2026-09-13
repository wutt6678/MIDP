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

import importlib.util
import json
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


def test_the_cell_cross_check_refuses_a_different_cell_set(tool, tmp_path):
    """Identity, not count: a run that swapped one cell for another must not
    be compared against the pilot it claims to re-derive."""
    cells = [{"cell_id": "gx_mll_151252__seed17"},
             {"cell_id": "gx_mll_254012__seed42"}]

    def write_manifest(ids, n_adapted=2):
        (tmp_path / "run_manifest.json").write_text(json.dumps(
            {"checkpoints_sha256": {f"cell_{i}": "deadbeef" for i in ids}}),
            encoding="utf-8")
        (tmp_path / "g6_pilot_gates.json").write_text(json.dumps(
            {"n_cells_adapted": n_adapted, "gates": {}}), encoding="utf-8")

    write_manifest(["gx_mll_151252__seed17", "gx_mll_254012__seed42"])
    got = tool.cross_check_cells(cells, tmp_path)
    assert got["checked"] is True
    assert got["method"] == "cell_id set equality"
    assert got["n_cells"] == 2

    # same COUNT, different identity -- a count check would have passed this
    write_manifest(["gx_mll_151252__seed17", "gx_mll_191023__seed123"])
    with pytest.raises(RuntimeError, match="cell identity disagrees"):
        tool.cross_check_cells(cells, tmp_path)

    # and it refuses to degrade to "could not check"
    (tmp_path / "run_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="cannot cross-check"):
        tool.cross_check_cells(cells, tmp_path)


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


@pytest.mark.skipif(not OUT_BASE.is_dir(),
                    reason="sealed pilot evidence not present")
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


@pytest.mark.skipif(not OUT_BASE.is_dir(),
                    reason="sealed pilot evidence not present")
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
    assert block["inputs"]["cell_identity_cross_check"]["checked"] is True
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
