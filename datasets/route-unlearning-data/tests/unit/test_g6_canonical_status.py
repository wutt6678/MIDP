"""The canonical status record must derive what it declares.

This file exists to supersede reports that were read as current after they went
stale, so its own failure mode is the one it was written to fix: a summary that
restates a verdict from prose keeps asserting it after the evidence moves.
Every figure here is therefore read from the artifact it describes, and the two
explanations it offers -- why the sealed summary's oracle reading went stale,
and why the manifest's pinned digest is still accepted -- are CHECKED rather
than narrated, so an incomplete explanation refuses to file at all.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
OUT_BASE = _ROOT / "e2c_granularity" / "outputs" / "mllmu"
SEALED = ("g6_pilot_gates.json", "granularity_summary.json",
          "run_manifest.json")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load("g6_status_under_test", "e2c_v3_g6_canonical_status.py")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# digest acceptance, re-derived from rv's own source
# --------------------------------------------------------------------------

_RV_TEMPLATE = '''
SCORING_DIGEST_HISTORY = (
    {{
        "previous_script_sha256": "{prev}",
        "superseded_by_commit": "992efa9",
        "reason": "{reason}",
    }},
)
'''


def test_digest_acceptance_is_re_derived_and_only_for_provenance_only(
        tool, tmp_path):
    """Acceptance is read out of ``rv`` by AST, and the reason filter is what
    stops a genuine scoring change from being excused as provenance.

    Excusing a digest whose measurement content differs would be the blanket
    amnesty the filter exists to prevent, so a ``scoring_change`` entry must
    drop out of the accepted set even though it is declared in the same tuple.
    """
    prev = "92859f2de357e4b7" + "0" * 48
    for reason, expected in (("provenance_only", True),
                             ("scoring_change", False)):
        p = tmp_path / f"rv_{reason}.py"
        p.write_text(_RV_TEMPLATE.format(prev=prev, reason=reason),
                     encoding="utf-8")
        accepted, history = tool.accepted_scoring_digests_without_torch(p)
        assert len(history) == 1
        assert (prev in accepted) is expected
        # the file's own current digest is always accepted: it is what rv hashes
        assert _sha(p) in accepted


def test_digest_acceptance_refuses_a_history_it_cannot_read_literally(
        tool, tmp_path):
    """A history built at runtime cannot be read without importing torch, and
    silently accepting nothing would report the pinned digest as unaccepted.

    Three distinct refusals, each naming its own cause: a computed value, a
    literal of the wrong shape, and no assignment at all.  Collapsing them into
    one message would make the fix guesswork.
    """
    cases = (
        ("rv_dynamic.py", "SCORING_DIGEST_HISTORY = tuple(build_history())\n",
         "is not a literal"),
        ("rv_dict.py", 'SCORING_DIGEST_HISTORY = {"previous_script_sha256": "a"}\n',
         "not a sequence of entries"),
        ("rv_absent.py", "SOMETHING_ELSE = 1\n",
         "no longer a literal assignment"),
    )
    for name, source, expected in cases:
        p = tmp_path / name
        p.write_text(source, encoding="utf-8")
        with pytest.raises(RuntimeError, match=expected):
            tool.accepted_scoring_digests_without_torch(p)


# --------------------------------------------------------------------------
# the oracle-fit supersession, and the causal story built on it
# --------------------------------------------------------------------------

def _oracle_tree(tmp_path, flips, sids=("gx_mll_151252", "gx_mll_191023"),
                 families=("matched_retrain", "loo_retrain")):
    """Oracle dirs whose pre-GX2H fit is broken exactly for ``flips``.

    ``families`` is spelled out rather than read off the module under test: a
    helper that takes its own expectation from the code it is checking cannot
    notice that code changing.
    """
    for sid in sids:
        for fam in families:
            d = tmp_path / "oracles" / f"{fam}_{sid}"
            d.mkdir(parents=True)
            broken = (sid, fam) in flips
            (d / "oracle_results.json").write_text(json.dumps(
                {"fit_ok": True, "strict_all_expected": 1.0,
                 "min_candidate_mass": 0.999}), encoding="utf-8")
            if broken:
                (d / "oracle_results.pre_hard_reeval.json").write_text(
                    json.dumps({"fit_ok": False,
                                "strict_all_expected": 33 / 35,
                                "min_candidate_mass": 0.999}), encoding="utf-8")
    return sids


def test_supersession_counts_only_false_to_true_flips(tool, tmp_path):
    """A pair with no pre-GX2H backup was never broken, and must not be
    counted as repaired; a pair that was already fine must not be either."""
    assert list(tool.GATE_ORACLE_FAMILIES) == ["matched_retrain",
                                               "loo_retrain"]
    assert tool.PRE_REEVAL_SUFFIX == ".pre_hard_reeval.json"

    sids = _oracle_tree(tmp_path, {("gx_mll_151252", "matched_retrain")})
    got = tool.oracle_fit_supersession(tmp_path, sids)

    assert got["n_pairs"] == 4                     # 2 sets x 2 families
    assert got["n_fit_ok_flipped_false_to_true"] == 1
    by_key = {(p["family"], p["set_id"]): p for p in got["pairs"]}
    assert by_key[("matched_retrain", "gx_mll_151252")][
        "fit_ok_flipped_false_to_true"] is True
    assert by_key[("matched_retrain", "gx_mll_151252")]["pre_gx2h"][
        "fit_ok"] is False
    assert by_key[("loo_retrain", "gx_mll_191023")]["pre_gx2h"] is None
    assert by_key[("loo_retrain", "gx_mll_191023")][
        "fit_ok_flipped_false_to_true"] is False


def test_supersession_refuses_a_missing_current_oracle(tool, tmp_path):
    """The current fit is the side the authoritative report reads; without it
    there is nothing to compare and no verdict to summarise."""
    sids = _oracle_tree(tmp_path, set())
    (tmp_path / "oracles" / "matched_retrain_gx_mll_151252"
     / "oracle_results.json").unlink()
    with pytest.raises(RuntimeError, match="oracle results absent"):
        tool.oracle_fit_supersession(tmp_path, sids)


_MATRIX = {"sets": [
    {"set_id": "gx_mll_151252",
     "coarser_label": "Software and Web Developers, Programmers, and Testers"},
    {"set_id": "gx_mll_254012",
     "coarser_label": "Archivists, Curators, and Museum Technicians"},
    {"set_id": "gx_mll_191023", "coarser_label": "Biological Scientists"},
]}


def _supersession(flipped):
    return {"pairs": [{"set_id": sid, "family": "matched_retrain",
                       "fit_ok_flipped_false_to_true": sid in flipped}
                      for sid in sorted({s["set_id"] for s in _MATRIX["sets"]})]}


def _coverage(failing_sets, per_set=2):
    return {"established_and_failing": [
        {"cell_id": f"{sid}__seed{s}", "target": t, "seed": s}
        for sid in failing_sets for s in (17, 42, 123)
        for t in range(per_set)]}


def test_staleness_explanation_passes_when_the_story_holds(tool):
    """The real shape: the twelve rows the sealed summary called failing sit in
    the two comma-bearing sets, and those are exactly the sets GX2H repaired."""
    flipped = {"gx_mll_151252", "gx_mll_254012"}
    got = tool.staleness_explanation(
        _supersession(flipped), _coverage(flipped), _MATRIX)

    assert got["checked_not_asserted"] is True
    assert got["agrees"] is True
    assert got["n_rows_called_failing"] == 12          # 2 sets x 3 seeds x 2
    assert got["sets_those_rows_are_in"] == sorted(flipped)
    assert got["sets_whose_oracle_fit_flipped"] == sorted(flipped)
    mech = got["mechanism_also_derived"]
    assert mech["matches_the_flipped_sets"] is True
    assert mech["sets_whose_coarsened_label_has_a_comma"] == sorted(flipped)
    assert "Biological Scientists" in mech["coarser_labels"].values()


def test_staleness_explanation_refuses_an_unexplained_failing_row(tool):
    """The load-bearing refusal.

    If a row failed in a set whose oracle fit was never broken, the oracle-fit
    story does not account for it, and the authoritative reading would be
    disagreeing with the sealed one without a reason.  Filing that would replace
    one unexplained report with another.
    """
    flipped = {"gx_mll_151252"}
    failing = {"gx_mll_151252", "gx_mll_191023"}
    with pytest.raises(RuntimeError, match="Unexplained by the oracle-fit"):
        tool.staleness_explanation(
            _supersession(flipped), _coverage(failing), _MATRIX)


def test_staleness_explanation_refuses_a_mechanism_that_does_not_fit(tool):
    """Rows confined to the repaired sets, but the repaired sets are not the
    comma-bearing ones -- so the punctuation-matcher cause is contradicted even
    though the row-level check passed.  Both halves have to hold."""
    flipped = {"gx_mll_151252", "gx_mll_191023"}   # 191023 has no comma
    with pytest.raises(RuntimeError, match="punctuation-matcher explanation"):
        tool.staleness_explanation(
            _supersession(flipped), _coverage(flipped), _MATRIX)


# --------------------------------------------------------------------------
# the record itself
# --------------------------------------------------------------------------

def test_dest_refuses_the_artifacts_it_only_reads(tool):
    """This tool reads the sealed reports and the authoritative artifact.  It
    must not be able to write any of them, including by absolute path."""
    for name in (*SEALED, tool.AUTHORITATIVE_NAME):
        with pytest.raises(RuntimeError, match="only reads"):
            tool.main(["--out-base", str(OUT_BASE),
                       "--dest", str(OUT_BASE / name), "--dry-run"])
    assert tool.STATUS_NAME not in SEALED
    assert tool.STATUS_NAME != tool.AUTHORITATIVE_NAME


@pytest.mark.skipif(not (OUT_BASE / "g6_pilot_gates_reevaluated.json").is_file(),
                    reason="authoritative re-derived report not present")
def test_the_record_declares_everything_a_reader_needs(tool):
    """The five declarations this record exists to make, present and mutually
    consistent -- a status file that omits the role statement would leave the
    pilot free to be cited as a headline result."""
    block = tool.build(OUT_BASE)

    assert block["authoritative_result"]["file"] == tool.AUTHORITATIVE_NAME
    assert block["authoritative_result"]["sha256"] == _sha(
        OUT_BASE / tool.AUTHORITATIVE_NAME)

    v = block["pilot_verdict"]
    assert v["passed"] is False
    assert v["failed_gates"] and v["proceed_to_full_matrix"] is False

    fm = block["full_matrix"]
    assert fm["launched"] is False and fm["will_not_proceed"] is True
    assert fm["cells_outside_the_pilot_design"] == []

    role = block["g6_role"]
    assert role["decision_not_measurement"] is True
    assert "main claim" in role["is_not"]

    hist = block["historical_sealed_artifacts"]
    assert sorted(hist["files"]) == sorted(SEALED)
    assert all(f["present"] and len(f["sha256"]) == 64
               for f in hist["files"].values())
    assert "historical sealed artifacts" in hist["policy"]

    tuning = block["protocol_study"]["no_further_tuning"]
    assert tuning["planned"] is False
    assert tuning["decision_not_measurement"] is True
    assert tuning["sets_it_covers"], "the decision names no sets"
    labels = {s["target_label"] for s in tuning["sets_it_covers"]}
    assert labels == {"Museum Curator", "Software Engineer"}


@pytest.mark.skipif(not (OUT_BASE / "g6_pilot_gates_reevaluated.json").is_file(),
                    reason="authoritative re-derived report not present")
def test_the_record_restates_no_verdict_of_its_own(tool):
    """The summary's verdict must equal the artifact's, field for field.

    Retyping it is how a status record survives the evidence changing: the
    pilot could be re-derived to a pass and this file would still say failed.
    """
    block = tool.build(OUT_BASE)
    auth = json.loads((OUT_BASE / tool.AUTHORITATIVE_NAME)
                      .read_text(encoding="utf-8"))
    gates = auth["gates"]
    v = block["pilot_verdict"]

    assert v["passed"] == gates["passed"]
    assert v["failed_gates"] == list(gates["failed_gates"])
    assert v["proceed_to_full_matrix"] == gates["proceed_to_full_matrix"]
    assert v["coverage_exact"] == gates["coverage"]["exact"]
    assert v["n_cells_evaluated"] == gates["n_cells_evaluated"]
    assert v["per_gate_passed"] == {
        k: bool(g.get("passed")) for k, g in sorted(gates["gates"].items())}
    assert v["source"].startswith("read from")

    # the sealed overall verdict is compared, not assumed to agree
    sealed = json.loads((OUT_BASE / "g6_pilot_gates.json")
                        .read_text(encoding="utf-8"))
    agree = block["historical_sealed_artifacts"]["stale_claims"][
        "g6_pilot_gates.json"]["overall_verdict"]["agrees_with_authoritative"]
    assert agree is ((sealed["gates"]["passed"] == gates["passed"])
                     and list(sealed["gates"]["failed_gates"])
                     == list(gates["failed_gates"]))


@pytest.mark.skipif(not (OUT_BASE / "g6_pilot_gates_reevaluated.json").is_file(),
                    reason="authoritative re-derived report not present")
def test_the_stale_claims_are_read_from_the_files_they_describe(tool):
    """Each staleness entry quotes the value AS WRITTEN, so the record can be
    checked against the artifact rather than trusted."""
    block = tool.build(OUT_BASE)
    stale = block["historical_sealed_artifacts"]["stale_claims"]
    gs = json.loads((OUT_BASE / "granularity_summary.json")
                    .read_text(encoding="utf-8"))
    rm = json.loads((OUT_BASE / "run_manifest.json").read_text(encoding="utf-8"))

    assert stale["granularity_summary.json"]["run_stage"][
        "value_as_written"] == gs["run_stage"] == "full_matrix"
    g31 = stale["granularity_summary.json"]["g3_1_gate"]
    assert g31["value_as_written"]["passed"] == gs["g3_1_gate"]["passed"]
    assert g31["value_as_written"]["n_established_and_failing"] == \
        gs["g3_1_gate"]["coverage"]["n_established_and_failing"]
    assert g31["causal_check"]["agrees"] is True
    assert g31["current_reading"]["matched_oracle_passed"] is True

    pinned = stale["run_manifest.json"]["shared_scoring_script_sha256"]
    assert pinned["value_as_written"] == \
        rm["provenance"]["shared_scoring_script_sha256"]
    assert pinned["accepted_by_rv"] is True
    assert pinned["value_as_written"] in pinned["accepted_digests"]


def test_a_missing_authoritative_report_is_a_refusal(tool, tmp_path):
    """Without the authoritative artifact there is nothing to summarise, and a
    status record built from the sealed reports alone would restate exactly the
    reading it exists to supersede."""
    with pytest.raises(RuntimeError, match="authoritative re-derived gate "
                                           "report is absent"):
        tool.build(tmp_path)


@pytest.mark.skipif(not (OUT_BASE / "g6_pilot_gates_reevaluated.json").is_file(),
                    reason="authoritative re-derived report not present")
def test_a_staged_report_is_named_by_its_destination_not_its_staging_path(
        tool, tmp_path):
    """Both artifacts have to be derivable at one clean worktree.

    The report must be written outside the tree to record a clean worktree, and
    this record must hash the report -- so it has to be able to read it from
    there.  What it RECORDS is the destination name, because the record
    describes where the artifact lives for every later reader, not where it
    happened to be generated; the staging path is disclosed separately so a
    hash that will not match after placement is visible as such.
    """
    staged = tmp_path / "staged" / "reeval.json"
    staged.parent.mkdir(parents=True)
    staged.write_bytes((OUT_BASE / tool.AUTHORITATIVE_NAME).read_bytes())

    block = tool.build(OUT_BASE, authoritative_path=staged)
    ar = block["authoritative_result"]
    assert ar["file"] == tool.AUTHORITATIVE_NAME
    assert ar["path"] == tool.AUTHORITATIVE_NAME
    assert str(staged) not in ar["path"]
    assert ar["read_from"] == str(staged)
    assert ar["sha256"] == _sha(staged)

    # reading the in-tree copy instead discloses no staging path at all
    in_tree = tool.build(OUT_BASE)["authoritative_result"]
    assert in_tree["read_from"] == tool.AUTHORITATIVE_NAME
    assert in_tree["sha256"] == _sha(OUT_BASE / tool.AUTHORITATIVE_NAME)
