"""CPU tests for the G6.0 externally audited MLLMU profession hierarchy.

The point of G6.0 is that no parent comes from a model or from arithmetic on a
damaged cell, so these tests concentrate on the source-table loader, which is
where a wrong guess would look like a success and silently mis-parent every
profession:

- SOC parent CODES are arithmetic on the published numbering, and parent
  TITLES must come from the table (G1) -- both directions are pinned;
- the official download is a WIDE table (one column per level, code written in
  whichever column matches that row's depth), and a hand-made code+title CSV
  also works;
- a code cell Excel mangled into a date ("Nov-21") is NOT recoverable: a
  two-digit year keeps only the low digits, so the loader enumerates every
  candidate and refuses, rather than picking one.  An earlier version assumed
  the year was 20XX and produced 11-2021 "Natural Sciences Managers" (really
  11-9121), so the enumeration itself is pinned here;
- quarantine is opt-in, is recorded per row, and keeps the damaged code out of
  the index so no chain can resolve through it;
- ``singularize`` widens matching for PROPOSALS only; the gates keep comparing
  published titles verbatim.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mh = _load("mllmu_hierarchy_under_test", "e2c_v3_mllmu_hierarchy.py")

# The official wide layout, reduced to one clean branch.  Note the quoted
# title: it contains a comma, which is why the loader must use a real CSV
# reader rather than splitting on commas.
WIDE_CLEAN = """\
Major Group,Minor Group,Broad Occupation,Detailed Occupation,Detailed O*NET-SOC,SOC or O*NET-SOC 2019 Title
15-0000,,,,,Computer and Mathematical Occupations
,15-1200,,,,Computer Occupations
,,15-1250,,,"Software and Web Developers, Programmers, and Testers"
,,,15-1252,,Software Developers
,,,,15-1252.01,Video Game Designers
19-0000,,,,,"Life, Physical, and Social Science Occupations"
,19-1000,,,,Life Scientists
,,19-1030,,,Conservation Scientists and Foresters
,,,19-1031,,Conservation Scientists
"""

# The same shape, but with the minor-group code damaged by Excel exactly as
# the real download is: 11-2000/11-3000/.../11-9000 all render as "Nov-00".
WIDE_MANGLED = """\
Major Group,Minor Group,Broad Occupation,Detailed Occupation,Detailed O*NET-SOC,SOC or O*NET-SOC 2019 Title
11-0000,,,,,Management Occupations
,Nov-00,,,,Advertising and Promotions Managers
,,Nov-10,,,Advertising Managers
,,,Nov-11,,Advertising Managers
"""

LONG_CLEAN = """\
soc_code,title
15-0000,Computer and Mathematical Occupations
15-1200,Computer Occupations
15-1250,Software and Web Developers
15-1252,Software Developers
"""


def _write(tmp_path, text, name="table.csv"):
    p = tmp_path / name
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------- #
# SOC code arithmetic
# ---------------------------------------------------------------------- #
def test_parent_codes_are_arithmetic_on_the_published_numbering():
    """A detailed code implies its broad, minor and major group codes.

    This is the one thing G6.0 is allowed to compute: the numbering is
    published and hierarchical, so 15-1252's parents are 15-1250, 15-1200 and
    15-0000.  The TITLES of those parents are never computed -- see G1.
    """
    assert mh.soc_levels("15-1252") == {
        "major_group": "15-0000", "minor_group": "15-1200",
        "broad_occupation": "15-1250", "detailed_occupation": "15-1252"}
    # an O*NET-SOC extension keeps the same SOC parents
    assert mh.soc_levels("15-1252.01")["broad_occupation"] == "15-1250"
    # a non-code is not a code
    assert mh.soc_levels("Software Engineer") is None


def test_normalize_label_is_conservative_and_singularize_is_only_a_proposal():
    """"Retired Architect" must not merge into "Architect".

    Merging them would put an identity in the wrong branch, so the gate
    normalization keeps qualifiers.  ``singularize`` exists only because SOC
    titles are plural ("Software Developers") while MLLMU professions are
    singular ("Software Developer"); it widens the search a reviewer sees and
    is never used by a gate.
    """
    assert mh.normalize_label("  Software   Developer ") == "software developer"
    assert mh.normalize_label("Retired Architect") != mh.normalize_label("Architect")
    assert mh.singularize("Software Developers") == mh.singularize("Software Developer")
    assert mh.singularize("Architects, Except Landscape and Naval").startswith("architect")
    # qualifiers still survive the proposal-only widening
    assert mh.singularize("Retired Architect") != mh.singularize("Architect")


# ---------------------------------------------------------------------- #
# loader: fail closed
# ---------------------------------------------------------------------- #
def test_the_loader_refuses_an_absent_table_and_an_excel_download(tmp_path):
    """No table, no hierarchy -- and an XLSX is refused, not half-parsed.

    This environment has no openpyxl, so an official Excel download cannot be
    read; the message says what to do instead of raising an ImportError deep
    inside a freeze.
    """
    with pytest.raises(RuntimeError) as exc:
        mh.load_taxonomy_table(tmp_path / "missing.csv")
    assert "taxonomy table not found" in str(exc.value)
    assert "sha256" in str(exc.value)          # why the file must exist

    xlsx = _write(tmp_path, b"not really a workbook", name="official.xlsx")
    with pytest.raises(RuntimeError) as exc:
        mh.load_taxonomy_table(xlsx)
    assert "no openpyxl" in str(exc.value)


def test_the_loader_refuses_to_guess_a_column(tmp_path):
    """An unrecognizable header is refused, listing the columns it saw.

    Picking the wrong column would mis-parent every profession while still
    producing a file that looks like a successful freeze.
    """
    p = _write(tmp_path, "identifier,blurb\n15-1252,Software Developers\n")
    with pytest.raises(RuntimeError) as exc:
        mh.load_taxonomy_table(p)
    msg = str(exc.value)
    assert "cannot identify the SOC code column" in msg
    assert "identifier" in msg and "blurb" in msg      # it reports what it saw
    assert "Refusing to guess" in msg


# ---------------------------------------------------------------------- #
# loader: both layouts
# ---------------------------------------------------------------------- #
def test_the_official_wide_layout_is_read_with_one_code_per_level(tmp_path):
    """The real download puts the code in whichever column matches the depth.

    Every level must be present, because a level-1 parent's TITLE has to come
    from the table rather than from arithmetic on its code.
    """
    t = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    assert t["layout"] == "wide_official"
    assert t["title_column"] == "SOC or O*NET-SOC 2019 Title"
    assert t["by_code"]["15-1252"] == "Software Developers"
    assert t["by_code"]["15-1250"] == (
        "Software and Web Developers, Programmers, and Testers")   # comma kept
    assert t["level_of_code"]["15-0000"] == "major group"
    assert t["level_of_code"]["15-1252.01"] == "detailed o*net-soc"
    for level in ("major_group", "minor_group", "broad_occupation",
                  "detailed_occupation"):
        assert t["levels_present"][level] >= 1, level
    # The census comes from the table's own level columns, so an O*NET
    # extension is not counted as the SOC detailed occupation and a major
    # group is not counted at all four depths.  Arithmetic alone would say 8.
    assert t["levels_present"]["detailed_occupation"] == 2
    assert t["levels_by_arithmetic"]["detailed_occupation"] > 2
    assert "level columns" in t["levels_present_source"]
    assert t["malformed_rows"] == [] and t["quarantined_rows"] == []


def test_a_hand_made_code_and_title_csv_also_loads(tmp_path):
    """The wide layout is the official one, not the only accepted one."""
    t = mh.load_taxonomy_table(_write(tmp_path, LONG_CLEAN))
    assert t["layout"] == "long_code_and_title"
    assert t["code_column"] == "soc_code"
    assert t["by_code"]["15-1250"] == "Software and Web Developers"


# ---------------------------------------------------------------------- #
# loader: damaged cells are not repairable
# ---------------------------------------------------------------------- #
def test_an_excel_mangled_code_is_refused_rather_than_reconstructed(tmp_path):
    """"Nov-00" cannot be turned back into a code, so the loader refuses.

    The message must name the clean fix (re-export with the code columns as
    text) and must not offer an arithmetic repair, because there is no
    arithmetic that recovers a century Excel threw away.
    """
    p = _write(tmp_path, WIDE_MANGLED)
    with pytest.raises(RuntimeError) as exc:
        mh.load_taxonomy_table(p)
    msg = str(exc.value)
    assert "3 code cell(s) are not SOC codes" in msg
    assert "Nov-00" in msg and "Nov-11" in msg
    assert "NOT recoverable" in msg
    assert "re-export the CSV" in msg
    assert "formatted as TEXT" in msg
    assert "11-2000, 11-3000" in msg          # it says why: several fit
    assert "repair" not in msg.lower().replace("re-export", "")


def test_the_candidate_enumeration_covers_every_century_excel_could_have_kept():
    """A two-digit year keeps the low digits only, so all centuries fit.

    Regression test for the first wrong version of this function, which
    assumed the year was 20XX and built three-digit tails such as "11-100".
    Every candidate must be a well-formed SOC code, and the row's own level
    narrows the list.
    """
    minor = mh.excel_date_candidates("Nov-00", "11", "minor group")
    assert minor == [f"11-{y}000" for y in range(2, 10)]      # 11-2000..11-9000
    assert "11-2000" in minor and "11-9000" in minor
    broad = mh.excel_date_candidates("Nov-10", "11", "broad occupation")
    assert len(broad) > len(minor)                            # shape is looser
    assert all(c.endswith("0") for c in broad)
    for cand in minor + broad:
        assert mh.soc_levels(cand) is not None, cand          # never "11-100"
    # a month token that disagrees with the enclosing branch proposes nothing
    assert mh.excel_date_candidates("Mar-00", "11", "minor group") == []
    assert mh.excel_date_candidates("Nov-00", None, "minor group") == []
    assert mh.excel_date_candidates("15-1252", "15", "detailed occupation") == []


def test_quarantine_is_opt_in_recorded_per_row_and_keeps_the_code_unresolvable(tmp_path):
    """With quarantine the run proceeds, but the damage stays visible.

    The damaged code must NOT enter the index: if it did, a record could name
    it as a parent and G1 would find it in the table, turning a corrupted cell
    into an apparently sourced parent.
    """
    p = _write(tmp_path, WIDE_MANGLED)
    t = mh.load_taxonomy_table(p, allow_quarantine=True)
    assert len(t["quarantined_rows"]) == 3
    assert {q["major_group"] for q in t["quarantined_rows"]} == {"11"}
    for q in t["quarantined_rows"]:
        assert q["excel_date_candidates"]                     # never empty
        assert q["line"] >= 2 and q["title"]
    for code in ("11-2000", "11-3000", "11-9000"):
        assert code not in t["by_code"], code
    assert t["by_code"].get("Nov-00") is None
    # the only clean row still resolves
    assert t["by_code"]["11-0000"] == "Management Occupations"
    assert t["quarantined_codes"] == sorted(
        {c for q in t["quarantined_rows"] for c in q["excel_date_candidates"]})


def test_a_damaged_cell_in_a_long_layout_is_still_refused(tmp_path):
    """Quarantine reasoning needs the enclosing major group, so a long table
    with a damaged code has no candidates and must simply fail."""
    p = _write(tmp_path, "soc_code,title\n15-1252,Software Developers\n"
                         "Nov-21,Some Manager\n")
    with pytest.raises(RuntimeError) as exc:
        mh.load_taxonomy_table(p)
    assert "1 code cell(s) are not SOC codes" in str(exc.value)


# ---------------------------------------------------------------------- #
# freeze / verify
# ---------------------------------------------------------------------- #
def test_a_hierarchy_that_fails_its_own_gates_is_never_written(tmp_path, monkeypatch):
    """``freeze_hierarchy`` refuses to produce a committed input from a failure.

    An ambiguous profession must be excluded (G5) and a parent title must come
    from the table (G1); if either fails, nothing is written.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    records = [mh.build_record(
        "Software Developer", table, "synonym",
        external_code="15-1252", confidence=0.95,
        reviewer_decision="retained", reviewer="test")]
    gates = mh.run_gates(records, table)
    target = tmp_path / "hier.json"
    monkeypatch.setattr(mh, "HIERARCHY_PATH", target)
    broken = dict(gates)
    broken["passed"] = False
    with pytest.raises(RuntimeError) as exc:
        # path= must be passed explicitly: freeze_hierarchy's default binds
        # HIERARCHY_PATH at def time, so monkeypatching the module global
        # would NOT redirect the write and a test could touch the real
        # manifests directory.
        mh.freeze_hierarchy(table, records, broken, path=target)
    assert "refusing to freeze" in str(exc.value)
    assert not target.exists()


def test_a_unit_test_never_writes_the_real_manifest_path(tmp_path):
    """Freezing in a test must not leave an artifact in the repository.

    This happened: ``freeze_hierarchy``'s default binds ``HIERARCHY_PATH`` at
    def time, so monkeypatching the module global did NOT redirect the write,
    and a test left ``e2c_mllmu/manifests/mllmu_hierarchy.json`` behind -- a
    file that would later look exactly like a genuine committed hierarchy and
    would make G7's tracked check pass for the wrong reason.
    """
    real = Path(mh.HIERARCHY_PATH)
    before = real.exists()
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    records = [mh.build_record(
        "Software Developer", table, "synonym",
        external_code="15-1252", confidence=0.95,
        reviewer_decision="retained", reviewer="test")]
    mh.freeze_hierarchy(table, records,
                        {"passed": True, "gates": {}, "warnings": [],
                         "failed_gates": [], "hierarchy_of": {}},
                        path=tmp_path / "elsewhere.json")
    assert (tmp_path / "elsewhere.json").exists()
    assert real.exists() == before, (
        f"a test wrote the real frozen-hierarchy path {real}")


def test_g7_fails_until_the_artifact_is_committed(tmp_path, monkeypatch):
    """The hierarchy must be committed BEFORE any G6 training.

    An untracked hierarchy is not an input anyone can audit, so G7 refuses it
    and a downstream builder refuses to read it.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    records = [mh.build_record(
        "Software Developer", table, "synonym",
        external_code="15-1252", confidence=0.95,
        reviewer_decision="retained", reviewer="test")]
    monkeypatch.setattr(mh, "HIERARCHY_PATH", tmp_path / "hier.json")
    monkeypatch.setattr(mh, "_is_tracked", lambda path: False)
    gates = mh.run_gates(records, table)
    assert gates["failed_gates"] == ["G7_committed_before_training"]
    assert "not tracked by git" in gates["gates"][
        "G7_committed_before_training"]["issues"][0]

    monkeypatch.setattr(mh, "_is_tracked", lambda path: True)
    assert mh.run_gates(records, table)["passed"] is True


def test_verify_re_reads_the_table_and_fails_if_the_hash_moved(tmp_path, monkeypatch):
    """A frozen hierarchy is bound to the exact bytes of its source table.

    Swapping the external authority underneath a committed artifact must be
    caught, because the recorded parent titles would then be unverifiable.
    """
    p = _write(tmp_path, WIDE_CLEAN)
    table = mh.load_taxonomy_table(p)
    records = [mh.build_record(
        "Software Developer", table, "synonym",
        external_code="15-1252", confidence=0.95,
        reviewer_decision="retained", reviewer="test")]
    # G7 is tested separately; here the artifact stands in for a committed one
    monkeypatch.setattr(mh, "_is_tracked", lambda path: True)
    gates = mh.run_gates(records, table)
    assert gates["passed"], gates["failed_gates"]
    target = tmp_path / "hier.json"
    art = mh.freeze_hierarchy(table, records, gates, path=target)
    assert target.exists(), "freeze_hierarchy must write where it was told"
    target.write_text(json.dumps(art, default=str), encoding="utf-8")
    ok = mh.verify_hierarchy(target)
    assert ok["gates_passed"] is True
    assert ok["content_sha256"] == art["content_sha256"]

    # same path, different bytes -> the external authority moved
    p.write_text(WIDE_CLEAN.replace("Computer Occupations", "Renamed"),
                 encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        mh.verify_hierarchy(target)
    assert "changed underneath a committed artifact" in str(exc.value)
