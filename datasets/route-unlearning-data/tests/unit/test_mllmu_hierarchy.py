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

import csv
import importlib.util
import json
import subprocess
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


def _git_stub(dirty=(), absent=(), untracked=()):
    """A ``git_state`` replacement, keyed by substring of the path.

    ``git_state`` replaced ``_is_tracked``, so stubbing the OLD name would leave
    G7 and G8 calling the real one and the tests would assert against the
    repository's actual state -- passing or failing depending on whether the
    working tree happened to be clean at that moment.  Substrings rather than
    exact paths because both the real module constants and a tmp_path fixture
    have to be addressable from one stub.

    The three failure modes are kept distinct because they are genuinely
    different answers: absent, present-but-untracked, and tracked-but-modified.
    """
    def fake(path):
        p = str(path)
        st = {"path": p, "absolute_path": p, "exists": True,
              "in_repository": True, "repo_root": "/stub", "tracked": True,
              "matches_index": True, "matches_head": True, "clean": True,
              "reason": None}
        if any(s in p for s in absent):
            st.update(exists=False, tracked=False, matches_index=False,
                      matches_head=False, clean=False,
                      reason="file does not exist")
        elif any(s in p for s in untracked):
            st.update(tracked=False, matches_index=False, matches_head=False,
                      clean=False, reason="not tracked by git")
        elif any(s in p for s in dirty):
            st.update(matches_index=False, clean=False,
                      reason="tracked but modified in the working tree")
        return st
    return fake


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
# locating the header in an official spreadsheet export
# ---------------------------------------------------------------------- #
# The delivered re-export carries two sheet-title lines and a blank one above
# the real header, exactly as the O*NET download does.
PREAMBLE = """\
O*NET-SOC 2019 Occupation Listings,,,,,
O*NET-SOC and SOC structure,,,,,
,,,,,
"""


def test_a_sheet_title_preamble_is_skipped_and_the_header_is_located(tmp_path):
    """The header is found, not assumed to be line 1.

    Treating a title row as the header is not a harmless off-by-three: every
    real column name becomes a data value, so the loader either refuses on an
    unidentifiable column or finds a plausible-looking wrong one.
    """
    plain = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN, "plain.csv"))
    with_pre = mh.load_taxonomy_table(
        _write(tmp_path, PREAMBLE + WIDE_CLEAN, "pre.csv"))
    assert plain["header_line"] == 1 and plain["n_preamble_rows_skipped"] == 0
    assert with_pre["header_line"] == 4
    assert with_pre["n_preamble_rows_skipped"] == 3
    # the data is identical either way
    assert with_pre["by_code"] == plain["by_code"]
    assert with_pre["level_of_code"] == plain["level_of_code"]
    assert with_pre["levels_present"] == plain["levels_present"]


def test_a_reported_line_number_points_at_the_real_line_in_the_file(tmp_path):
    """Damaged-cell line numbers survive preamble and blank-row skipping.

    These numbers are recorded in the frozen artifact, so a reviewer must be
    able to open the file at that line and see the cell being described.
    """
    text = PREAMBLE + """\
Major Group,Minor Group,Broad Occupation,Detailed Occupation,Detailed O*NET-SOC,SOC or O*NET-SOC 2019 Title
11-0000,,,,,Management Occupations
,,,,,
,Nov-00,,,,Advertising and Promotions Managers
"""
    t = mh.load_taxonomy_table(_write(tmp_path, text), allow_quarantine=True)
    assert t["header_line"] == 4
    assert len(t["quarantined_rows"]) == 1
    q = t["quarantined_rows"][0]
    # line 5 = 11-0000, line 6 = the blank spacer, line 7 = the damaged cell
    assert q["line"] == 7, q
    with open(tmp_path / "table.csv", encoding="utf-8") as f:
        assert f.read().splitlines()[6].startswith(",Nov-00")


def test_a_file_with_no_header_row_is_refused_and_never_promotes_data(tmp_path):
    """Data-only rows are refused, and the message reports what it saw.

    The loader falls back to line 1 when nothing looks like a header, but only
    so the column-identification check can name the columns it actually found.
    A data row must never end up serving as a working header.
    """
    p = _write(tmp_path, "15-0000,,,,,Computer and Mathematical Occupations\n"
                         ",15-1200,,,,Computer Occupations\n")
    _rows, _linenos, header_line, found = mh._read_delimited(p)
    assert found is False, "a row of codes is not a header"
    assert header_line == 1
    with pytest.raises(RuntimeError) as exc:
        mh.load_taxonomy_table(p)
    msg = str(exc.value)
    assert "cannot identify" in msg
    assert "Refusing to guess" in msg
    # it reports the values it was handed, so the cause is visible
    assert "15-0000" in msg


def test_an_identified_header_is_recorded_as_identified(tmp_path):
    """The provenance distinguishes a located header from a line-1 fallback."""
    t = mh.load_taxonomy_table(_write(tmp_path, PREAMBLE + WIDE_CLEAN))
    assert t["header_identified"] is True and t["header_line"] == 4
    t2 = mh.load_taxonomy_table(_write(tmp_path, LONG_CLEAN, "long.csv"))
    assert t2["header_identified"] is True and t2["header_line"] == 1


def test_a_data_row_is_not_mistaken_for_the_header(tmp_path):
    """The header test looks for column NAMES, so a row of codes is data.

    A detailed-occupation row happens to contain the text of a title, which
    must not be enough to make it the header.
    """
    p = _write(tmp_path, "15-1252,,,,,Software Developers\n" + WIDE_CLEAN)
    t = mh.load_taxonomy_table(p)
    assert t["header_line"] == 2
    assert t["by_code"]["15-1252"] == "Software Developers"
    # the stray row sits above the header, so it is preamble and is not read
    # as data; the code that IS in the index comes from the real table below
    assert t["n_preamble_rows_skipped"] == 1


# ---------------------------------------------------------------------- #
# reviewer-facing proposals
# ---------------------------------------------------------------------- #
# Deliberately contains the two traps that produced wrong proposals: a
# qualifier-heavy correct title ("Architects, Except Landscape and Naval") that
# a Jaccard-style score ranks BELOW a wrong one ("Database Architects"), and a
# pluralized multi-word head ("Environmental Scientists and Specialists") that
# whole-phrase singularization fails to match.
PROPOSAL_TABLE = """\
Major Group,Minor Group,Broad Occupation,Detailed Occupation,Detailed O*NET-SOC,SOC or O*NET-SOC 2019 Title
17-0000,,,,,Architecture and Engineering Occupations
,17-1000,,,,Architects and Surveyors
,,17-1010,,,Architects
,,,17-1011,,"Architects, Except Landscape and Naval"
,,,17-1012,,Landscape Architects
15-0000,,,,,Computer and Mathematical Occupations
,15-1200,,,,Computer Occupations
,,15-1240,,,Database and Network Administrators and Architects
,,,15-1243,,Database Architects
,,,15-2051,,Data Scientists
19-0000,,,,,"Life, Physical, and Social Science Occupations"
,19-2000,,,,Physical Scientists
,,19-2040,,,Environmental Scientists and Geoscientists
,,,19-2041,,"Environmental Scientists and Specialists, Including Health"
,,,19-3091,,Anthropologists and Archeologists
29-0000,,,,,Healthcare Practitioners and Technical Occupations
,29-1200,,,,Health Diagnosing or Treating Practitioners
,,29-1220,,,Podiatrists
,,,29-1221,,Podiatrists
,,,,29-1221.01,"Surgeons, Cardiovascular"
"""


def _proposal_table(tmp_path):
    return mh.load_taxonomy_table(_write(tmp_path, PROPOSAL_TABLE, "prop.csv"))


def test_an_exact_head_match_outranks_a_qualifier_free_but_wrong_branch(tmp_path):
    """"Architect" is 17-1011, not 15-1243 "Database Architects".

    A Jaccard-style score ranks the WRONG one first, because the correct title
    carries the qualifier "Except Landscape and Naval" and so has more tokens to
    divide by.  That would place an architect in the Computer branch.  Ranking
    on match kind first, and comparing the title's pre-comma head, fixes it.
    """
    t = _proposal_table(tmp_path)
    c = mh.proposal_candidates("Architect", t)
    assert c["best_match_kind"] == "exact_head"
    assert c["best_candidate"] == "17-1011"
    assert c["candidates"][0]["title"] == "Architects, Except Landscape and Naval"
    codes = [x["code"] for x in c["candidates"]]
    assert codes.index("17-1011") < codes.index("15-1243")
    # the qualifier clause is what makes the head match, so it must be found
    assert mh._title_head("Architects, Except Landscape and Naval") == "Architects"


def test_tokens_are_singularized_per_word_not_per_phrase(tmp_path):
    """"Environmental Scientist" is 19-2041, not 15-2051 "Data Scientists".

    Whole-phrase singularization only strips the LAST word's trailing s, so
    "Environmental Scientists and Specialists" produced the token
    ``scientists`` and never matched ``scientist``.  The label then fell back to
    a single-token overlap with Data Scientists -- a wrong branch.
    """
    t = _proposal_table(tmp_path)
    c = mh.proposal_candidates("Environmental Scientist", t)
    assert c["best_candidate"] == "19-2041"
    assert c["best_match_kind"] == "head_tokens"
    assert c["best_recall"] == 1.0
    assert mh._tokens("Environmental Scientists and Specialists") == {
        "environmental", "scientist", "specialist"}
    assert "scientist" in mh._tokens("Environmental Scientists")


def test_ambiguity_is_a_tie_at_the_top_not_the_presence_of_weaker_candidates(tmp_path):
    """A confident exact match stays confident.

    An earlier version flagged ANY exact match that had further candidates as
    ambiguous, which wrongly demoted Architect, Graphic Designer and Civil
    Engineer out of the pool merely because unrelated occupations share a token
    with them.
    """
    t = _proposal_table(tmp_path)
    arch = mh.proposal_candidates("Architect", t)
    assert arch["n_candidates"] > 1
    assert arch["ambiguous"] is False, "weaker candidates are not a tie"
    # a genuine tie: two detailed occupations sharing the head token
    tie = mh.proposal_candidates("Podiatrist", t)
    assert tie["ambiguous"] is False        # only one detailed Podiatrists row
    landscape = mh.proposal_candidates("Landscape Architect", t)
    assert landscape["best_candidate"] == "17-1012"
    assert landscape["ambiguous"] is False


def test_onet_extensions_are_not_proposed_as_mapping_targets(tmp_path):
    """A person's profession sits at the SOC detailed-occupation level.

    The 149 O*NET-SOC extensions are deeper than that and would add a fifth
    depth to a chain built for specific -> L2 -> L1.
    """
    t = _proposal_table(tmp_path)
    assert t["by_code"]["29-1221.01"]           # it is in the table
    for label in ("Podiatrist", "Surgeon", "Cardiovascular Surgeon"):
        codes = [x["code"] for x in mh.proposal_candidates(label, t)["candidates"]]
        assert "29-1221.01" not in codes, label
        assert all("." not in c for c in codes), codes


def test_a_spelling_variant_is_surfaced_but_never_applied(tmp_path):
    """SOC writes "Archeologists"; MLLMU writes "Archaeologist".

    The variant is reported so the reviewer can see it, and the match kind stays
    ``none`` so nothing is decided automatically.
    """
    t = _proposal_table(tmp_path)
    c = mh.proposal_candidates("Archaeologist", t)
    assert c["best_match_kind"] == "none"
    assert c["best_candidate"] is None
    assert "archeologist" in c["spelling_variants_in_table"]
    assert c["authority"] == "source_table"


def _cand(best_group, kind="exact_head", recall=1.0, extra=(), ambiguous=False):
    """A minimal candidate dict, for testing pool selection directly.

    Carries the same keys ``proposal_candidates`` produces; ``select_balanced_pool``
    reads ``title`` and deliberately raises rather than reporting a titleless
    code, so a candidate shape regression surfaces here.
    """
    def one(g):
        return {"code": f"{g}-1011", "major_group": g,
                "title": f"Occupation {g}-1011"}
    return {"best_match_kind": kind, "best_recall": recall,
            "ambiguous": ambiguous,
            "candidates": [one(best_group)] + [one(g) for g in extra]}


def test_pool_coverage_comes_from_the_best_candidate_not_the_union():
    """Scattered weak candidates must not be paid as branch diversity.

    The first version scored the UNION of a profession's candidates, so a
    profession whose weak matches reached four branches was picked first --
    including 53-5011 "Sailors and Marine Oilers" as a Marine Biologist
    candidate.  Coverage now comes from the best candidate only, and an
    ambiguous top rank is ineligible outright.
    """
    counts = {"scattered": 84, "clean_19": 5, "clean_25": 4}
    cands = {
        # best candidate is in 19, but weak candidates reach 17, 25 and 53
        "scattered": _cand("19", kind="tokens", recall=0.5,
                           extra=("17", "25", "53"), ambiguous=True),
        "clean_19": _cand("19"),
        "clean_25": _cand("25"),
    }
    pool = mh.select_balanced_pool(cands, counts, n_professions=5,
                                   min_major_groups=2, min_identities=2)
    picked = [p["original_label"] for p in pool["pool"]]
    assert "scattered" not in picked
    assert pool["major_groups_covered"] == ["19", "25"]
    assert pool["per_major_group"] == {"19": 1, "25": 1}
    reasons = {x["original_label"]: x["reason"] for x in pool["ineligible"]}
    assert "ambiguous" in reasons["scattered"]
    assert reasons["scattered"]            # an explicit reason, never silent


def test_the_branch_cap_spreads_the_pool_without_starving_it():
    """max_per_major_group is a soft cap: it defers, it never truncates.

    With nothing covered yet every profession ties on branch novelty, so the
    opening pick is the most frequent one; the cap then binds from the second
    pick onward and a new branch jumps the queue.
    """
    counts = {"a": 10, "b": 9, "c": 8, "d": 7}
    cands = {"a": _cand("17"), "b": _cand("17"), "c": _cand("17"),
             "d": _cand("19")}
    capped = mh.select_balanced_pool(cands, counts, n_professions=4,
                                     min_major_groups=2, min_identities=2,
                                     max_per_major_group=1)
    order = [p["original_label"] for p in capped["pool"]]
    assert order == ["a", "d", "b", "c"], order
    assert len(capped["pool"]) == 4, "the cap must not shrink the pool"
    assert capped["major_groups_covered"] == ["17", "19"]
    assert capped["per_major_group"] == {"17": 3, "19": 1}
    # d opens a new branch, so it is taken second despite the lowest count
    assert capped["pool"][1]["major_group"] == "19"


def test_the_proposal_csv_leaves_every_decision_blank(tmp_path, monkeypatch):
    """A proposal is not a decision, and one row per profession.

    Every decision column must be empty, so build_hierarchy sees no
    mapping_type, treats the row as ambiguous and G5 excludes it: the file
    cannot produce a retained record until a human fills it in.  One row per
    profession (not per candidate) because several rows with the same label
    would build several records and fail G4.
    """
    monkeypatch.setattr(mh, "professions_in_source",
                        lambda path=None: {"Architect": 41,
                                           "Data Scientist": 2,
                                           "Archaeologist": 4})
    out = tmp_path / "proposal.csv"
    res = mh.build_proposal_csv(_write(tmp_path, PROPOSAL_TABLE, "prop.csv"),
                                out, min_major_groups=2, min_identities=2)
    assert res["n_rows"] == 3
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["original_label"] for r in rows].count("Architect") == 1
    for col in ("mapping_type", "external_code", "confidence",
                "reviewer_decision", "reviewer", "exclusion_reason"):
        assert {r[col] for r in rows} == {""}, col
    assert {r["proposed_by_model"] for r in rows} == {"false"}
    by_label = {r["original_label"]: r for r in rows}
    assert "17-1011" in by_label["Architect"]["proposed_candidates"]
    assert by_label["Archaeologist"]["proposed_candidates"] == ""
    assert "NO candidate" in by_label["Archaeologist"]["proposal_note"]
    # the file is directly usable as a --mapping input
    assert set(mh.PROPOSAL_COLUMNS) <= set(rows[0])


def test_g4b_is_scoped_to_selected_targets_and_is_a_hard_gate(tmp_path, monkeypatch):
    """Duplicates are legitimate in the table; they are invalid only as targets.

    Two raw professions sharing one SOC leaf is expected -- the lower-frequency
    one is a same-leaf retention control.  What must fail, hard, is selecting
    BOTH as confirmatory targets, because that counts one transformation twice.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "git_state", _git_stub())

    def rec(label, role="target_eligible", mtype="synonym"):
        return mh.build_record(label, table, mtype, external_code="15-1252",
                               confidence=0.95, reviewer_decision="retained",
                               reviewer="test", matrix_role=role)

    both = [rec("Software Developer"), rec("Software Engineer", mtype="manual")]

    # neither selected: recorded, not a failure
    g = mh.run_gates(both, table, selected_targets=[])["gates"][
        "G4b_no_duplicate_targets"]
    assert g["warn_only"] is False, "G4b is a hard gate for matrix generation"
    assert g["passed"] is True and g["n_issues"] == 0
    assert g["duplicate_targets"] == {"15-1252": ["software developer",
                                                  "software engineer"]}
    assert g["scope"].startswith("selected transformation targets")

    # both selected as targets: the hard gate fires
    g = mh.run_gates(both, table,
                     selected_targets=["Software Developer",
                                       "Software Engineer"])["gates"][
        "G4b_no_duplicate_targets"]
    assert g["passed"] is False
    assert g["n_targets_selected"] == 2 and g["n_distinct_leaf_codes"] == 1
    assert "target collision on SOC leaf 15-1252" in g["issues"][0]
    assert "same_leaf_retention_control" in g["issues"][0]

    # one demoted to a same-leaf control and not selected: clean
    fixed = [rec("Software Developer", role="same_leaf_retention_control"),
             rec("Software Engineer", mtype="manual")]
    g = mh.run_gates(fixed, table,
                     selected_targets=["Software Engineer"])["gates"][
        "G4b_no_duplicate_targets"]
    assert g["passed"] is True
    assert g["n_targets_selected"] == 1 == g["n_distinct_leaf_codes"]
    # a control must never be selected as a target either
    g = mh.run_gates(fixed, table, selected_targets=["Software Developer",
                                                     "Software Engineer"])[
        "gates"]["G4b_no_duplicate_targets"]
    assert g["passed"] is False
    assert "a same-leaf duplicate is a control, never a target" in g["issues"][0]


def test_g4b_does_not_reject_different_occupations_sharing_a_parent(tmp_path, monkeypatch):
    """Coarsening sibling occupations to a common parent is the experiment.

    17-1011 Architects and 17-1012 Landscape Architects are different leaves
    under one broad occupation.  Rejecting that would forbid the sibling
    transformation G6 exists to test, so it is recorded as legitimate.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, PROPOSAL_TABLE, "p2.csv"))
    monkeypatch.setattr(mh, "git_state", _git_stub())
    recs = [mh.build_record("Architect", table, "exact", 0.95, "retained",
                            external_code="17-1011", reviewer="t",
                            matrix_role="target_eligible"),
            mh.build_record("Landscape Architect", table, "exact", 0.95,
                            "retained", external_code="17-1012", reviewer="t",
                            matrix_role="target_eligible")]
    g = mh.run_gates(recs, table, selected_targets=["Architect",
                                                    "Landscape Architect"])[
        "gates"]["G4b_no_duplicate_targets"]
    assert g["passed"] is True and g["n_issues"] == 0
    assert g["n_targets_selected"] == 2 == g["n_distinct_leaf_codes"]
    assert g["different_leaves_sharing_a_parent"] == {"17-1010": ["17-1011",
                                                                   "17-1012"]}
    assert "must not be rejected" in g["shared_parent_is_not_a_collision"]
    assert g["duplicate_targets"] == {}


def test_duplicate_targets_are_carried_into_the_frozen_artifact(tmp_path, monkeypatch):
    """The constraint travels with the artifact, not just the run log."""
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "git_state", _git_stub())

    def rec(label, mtype, adjudication=None):
        return mh.build_record(label, table, mtype, external_code="15-1252",
                               confidence=0.9, reviewer_decision="retained",
                               reviewer="test", adjudication=adjudication)

    # G8 fails a manual mapping that carries no adjudication, so the override has
    # to travel with the record -- an override is not an unexplained code edit.
    adj = {"raw_label": "Software Engineer", "decision": "manual_override",
           "chosen_soc": "15-1252", "chosen_title": "Software Developers",
           "rejected_candidates": [{"soc": "17-2199",
                                    "reason": "wrong occupational branch"}],
           "target_eligible": True, "decided_before_experiment": True}
    records = [rec("Software Developer", "synonym"),
               rec("Software Engineer", "manual", adj)]
    gates = mh.run_gates(records, table)
    assert gates["gates"]["G8_pre_freeze_checklist"]["checks"][
        "manual_overrides_and_rejected_alternatives_recorded"] is True
    art = mh.freeze_hierarchy(table, records, gates,
                              path=tmp_path / "hier.json")
    assert art["duplicate_targets"] == {"15-1252": ["software developer",
                                                     "software engineer"]}
    # an unadjudicated manual mapping is a hard pre-freeze failure, not a warning
    gates2 = mh.run_gates([rec("Software Engineer", "manual")], table)
    with pytest.raises(RuntimeError, match="refusing to freeze"):
        mh.freeze_hierarchy(table, [rec("Software Engineer", "manual")], gates2,
                            path=tmp_path / "hier2.json")
    assert not (tmp_path / "hier2.json").exists()


def test_parents_come_from_the_table_not_from_arithmetic(tmp_path):
    """29-1221 sits under broad 29-1210, and 29-1220 does not exist.

    Arithmetic on the numbering says the broad parent of 29-1221 is 29-1220.
    The pinned release has no such row -- "Pediatricians, General" sits directly
    under "Physicians", which also holds 29-1211..29-1229.  Deriving the parent
    would have meant inventing its title, which is what G1 forbids; resolving it
    from the table's own level rows gives the real one and records the
    disagreement instead of hiding it.
    """
    text = """\
Major Group,Minor Group,Broad Occupation,Detailed Occupation,Detailed O*NET-SOC,SOC or O*NET-SOC 2019 Title
29-0000,,,,,Healthcare Practitioners and Technical Occupations
,29-1000,,,,Healthcare Diagnosing or Treating Practitioners
,,29-1210,,,Physicians
,,,29-1216,,General Internal Medicine Physicians
,,,29-1221,,"Pediatricians, General"
"""
    table = mh.load_taxonomy_table(_write(tmp_path, text, "phys.csv"))
    assert mh.soc_levels("29-1221")["broad_occupation"] == "29-1220"
    assert "29-1220" not in table["by_code"], "the arithmetic parent is absent"
    parents = mh.parents_from_table("29-1221", table)
    assert parents["broad_occupation"] == "29-1210"
    assert parents["minor_group"] == "29-1000"
    assert parents["arithmetic_disagreements"] == ["broad_occupation",
                                                   "minor_group"]
    r = mh.build_record("Pediatrician", table, "manual", 0.95, "retained",
                        external_code="29-1221", reviewer="t")
    assert r["level2_parent"] == {"code": "29-1210", "title": "Physicians",
                                  "level": "broad_occupation"}
    assert r["parents_resolved_from"] == "source_table"
    assert r["parent_arithmetic_disagreements"]["broad_occupation"] == {
        "arithmetic": "29-1220", "source_table": "29-1210"}
    # a code the table does not carry resolves to nothing rather than a guess
    assert mh.parents_from_table("29-9999", table) == {}


def test_the_confirmatory_selection_is_capped_and_defers_with_reasons(tmp_path):
    """Deferred is not rejected: the role stays target_eligible.

    A target with a single identity would make a strict-accuracy gate rest on one
    observation, and a branch holding every target would leave no sibling to
    preserve -- so both are capped, and each omission records why.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, PROPOSAL_TABLE, "p3.csv"))
    recs = []
    for label, code, role in (("Architect", "17-1011", "primary_target"),
                              ("Landscape Architect", "17-1012", "target_eligible"),
                              ("Database Architects Person", "15-1243",
                               "target_eligible"),
                              ("Data Scientist", "15-2051", "target_eligible")):
        recs.append(mh.build_record(label, table, "exact", 0.95, "retained",
                                    external_code=code, reviewer="t",
                                    matrix_role=role))
    counts = {"Architect": 41, "Landscape Architect": 2,
              "Database Architects Person": 5, "Data Scientist": 2}
    out = mh.select_confirmatory_targets(recs, counts, max_per_major_group=1,
                                         min_identities=2)
    # frequency outranks lexical order: "Data Scientist" sorts first but has 2
    # identities against "Database Architects Person"'s 5, so it is the one that
    # misses the single-per-branch cap.  Lexical order is only the LAST criterion.
    assert out["selected_targets"] == ["Architect", "Database Architects Person"]
    assert out["per_major_group"] == {"15": 1, "17": 1}
    deferred = {d["original_label"]: d["reason"] for d in out["deferred"]}
    assert "already has 1 targets" in deferred["Landscape Architect"]
    assert "already has 1 targets" in deferred["Data Scientist"]
    assert all(d["still"] == "target_eligible, promotable"
               for d in out["deferred"])
    # the branch is recorded structurally, not only inside the reason text
    by_lab = {d["original_label"]: d for d in out["deferred"]}
    assert by_lab["Landscape Architect"]["major_group"] == "17"
    assert by_lab["Data Scientist"]["major_group"] == "15"
    # equal frequency falls through to the deterministic lexical tie-breaker
    tie = dict(counts, **{"Database Architects Person": 2})
    out3 = mh.select_confirmatory_targets(recs, tie, max_per_major_group=1,
                                          min_identities=2)
    assert out3["selected_targets"] == ["Architect", "Data Scientist"]
    # a single-identity profession is deferred on evidence, not on branch
    counts2 = dict(counts, **{"Data Scientist": 1})
    out2 = mh.select_confirmatory_targets(recs, counts2, max_per_major_group=3,
                                          min_identities=2)
    assert "Data Scientist" not in out2["selected_targets"]
    assert "one observation" in {
        d["original_label"]: d["reason"] for d in out2["deferred"]}["Data Scientist"]
    # an eligible record with no SOC leaf is deferred with a reason, not crashed
    orphan = mh.build_record("Ghost", table, "ambiguous", 0.5, "retained",
                             reviewer="t", matrix_role="target_eligible")
    assert orphan["external_occupation_id"] is None
    out4 = mh.select_confirmatory_targets([orphan], {"Ghost": 9})
    assert out4["selected_targets"] == []
    assert "no canonical SOC leaf" in out4["deferred"][0]["reason"]


# ---------------------------------------------------------------------- #
# adjudication: the override registry and its precedence
# ---------------------------------------------------------------------- #
_OV = {"raw_label": "Software Engineer", "decision": "manual_override",
       "chosen_soc": "15-1252", "chosen_title": "Software Developers",
       "rejected_candidates": [{"soc": "17-2199", "reason": "wrong branch"}],
       "target_eligible": True,
       "rationale": "software engineering is the software-developer occupation",
       "decided_before_experiment": True}


def _registry(tmp_path, entries, name="ov.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"overrides": entries}), encoding="utf-8")
    return p


def test_the_override_registry_refuses_an_unexplained_or_unfrozen_decision(tmp_path):
    """A registry that cannot be audited is refused outright, not defaulted.

    Each refusal names the field, because "the registry failed to load" tells a
    reviewer nothing about which adjudication to fix.  A MISSING registry is
    different from a malformed one: no overrides yet is a legitimate state.
    """
    assert mh.load_manual_overrides(tmp_path / "absent.json") == {}
    ok = mh.load_manual_overrides(_registry(tmp_path, [_OV]))
    assert ok["Software Engineer"]["chosen_soc"] == "15-1252"

    for entries, needle in (
            ([{k: v for k, v in _OV.items() if k != "rationale"}], "rationale"),
            ([dict(_OV, decided_before_experiment=False)],
             "decided_before_experiment"),
            ([dict(_OV, decision="guessed")], "not one of"),
            # only an exclusion may omit the code it maps to
            ([dict(_OV, decision="same_mapping", chosen_soc="")], "chosen_soc"),
    ):
        with pytest.raises(RuntimeError) as exc:
            mh.load_manual_overrides(_registry(tmp_path, entries))
        assert needle in str(exc.value), needle
    # an exclusion needs no code, and still needs a rationale
    exc = dict(_OV, raw_label="Software Architect", decision="exclude",
               chosen_soc=None, chosen_title=None, target_eligible=False)
    assert "Software Architect" in mh.load_manual_overrides(
        _registry(tmp_path, [exc]))


def test_a_manual_override_outranks_a_higher_scoring_automatic_proposal(tmp_path):
    """manual > exact title > automatic > exclusion, and nothing reverses it.

    The precedence matters precisely when the automatic ranker disagrees: for
    Software Engineer the lexical matcher's best recall reaches the generic
    residual 17-2199, and a frozen adjudication must still win.  A model
    proposal is data here, not authority.
    """
    ovs = mh.load_manual_overrides(_registry(tmp_path, [_OV]))
    auto = {"best_match_kind": "exact_title", "best_recall": 1.0,
            "ambiguous": False,
            "candidates": [{"code": "17-2199", "major_group": "17",
                            "title": "Engineers, All Other"}]}
    dec, prec = mh.resolve_mapping("Software Engineer", ovs, auto)
    assert prec == "manual_override" and dec["chosen_soc"] == "15-1252"
    # with no override the same proposal is accepted at the exact-title rung
    dec2, prec2 = mh.resolve_mapping("Software Engineer", {}, auto)
    assert (prec2, dec2["chosen_soc"]) == ("exact_title", "17-2199")
    # an ambiguous top rank never resolves automatically, it excludes with why
    amb = dict(auto, ambiguous=True)
    dec3, prec3 = mh.resolve_mapping("Architect", {}, amb)
    assert prec3 == "exclusion" and dec3["decision"] == "exclude"
    assert dec3["target_eligible"] is False and "ambiguous=True" in dec3["rationale"]
    # a partial recall is not a mapping either
    weak = dict(auto, best_match_kind="head_tokens", best_recall=0.5,
                ambiguous=False)
    assert mh.resolve_mapping("X", {}, weak)[1] == "exclusion"


def test_semantic_cleanliness_only_separates_what_the_first_two_criteria_cannot():
    """Criterion 3 is a tie-break of last resort before lexical order.

    It is a deterministic proxy, not a judgement: catch-all leaf, then a
    non-occupational qualifier, then fewer modifiers.  It must never outrank
    confidence or frequency, so ``select_target_representatives`` only reaches
    it when both are tied.
    """
    c = mh._semantic_cleanliness
    # each arm pinned directly: a comparison alone can be carried by a LATER arm,
    # and "all"/"other" are stopwords so "Biological Scientists, All Other" has
    # the same token count as many a clean label.  "Engineers, All Other" and
    # "Biotechnologist" both reduce to one token, so ONLY the catch-all arm can
    # order them -- which is what makes this pair a real test of that arm.
    assert c("Engineers, All Other")[0] == 1 and c("Biotechnologist")[0] == 0
    assert c("Engineers, All Other")[2] == c("Biotechnologist")[2] == 1
    assert c("Biotechnologist") < c("Engineers, All Other")
    assert c("Retired Architect")[1] == 1 and c("Architect")[1] == 0
    assert c("Architect") < c("Retired Architect")
    assert c("Curator") < c("Museum Art Curator")
    assert c("Art Curator")[:3] == c("Museum Curator")[:3], \
        "the real duplicate pairs are separated by frequency, not by semantics"
    # the arms are ordered: a catch-all outranks a qualifier, which outranks
    # modifier count, so a clean two-word label still beats a residual leaf
    assert c("Museum Art Curator") < c("Engineers, All Other")


def test_the_representative_rule_picks_the_named_winners_on_frequency(tmp_path):
    """The frozen rule reproduces the adjudicated pairs from counts alone.

    Software Engineer over Software Developer (117 v 15), Museum Curator over
    Art Curator (10 v 6), Cultural Anthropologist over Archaeologist (5 v 4) --
    all three decided by identity frequency at equal confidence, and the losers
    become same-leaf retention controls rather than being dropped.  Confidence
    still outranks frequency when it differs.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))

    def rec(label, code, conf=0.95, role=None):
        return mh.build_record(label, table, "manual", conf, "retained",
                               external_code=code, reviewer="t",
                               matrix_role=role)

    counts = {"Software Engineer": 117, "Software Developer": 15,
              "Museum Curator": 10, "Art Curator": 6,
              "Cultural Anthropologist": 5, "Archaeologist": 4}
    recs = [rec("Software Developer", "15-1252"), rec("Software Engineer", "15-1252"),
            rec("Art Curator", "25-4012"), rec("Museum Curator", "25-4012"),
            rec("Archaeologist", "19-3091"), rec("Cultural Anthropologist", "19-3091")]
    sel = mh.select_target_representatives(recs, counts)
    assert [sel["15-1252"]["representative"], sel["25-4012"]["representative"],
            sel["19-3091"]["representative"]] == [
                "Software Engineer", "Museum Curator", "Cultural Anthropologist"]
    assert sel["15-1252"]["same_leaf_controls"] == ["Software Developer"]
    assert all(v["decided_by"] == "identity frequency" for v in sel.values())
    assert all(v["n_raw_labels"] == 2 for v in sel.values())
    # a lower-frequency label with strictly higher confidence wins instead
    conf = mh.select_target_representatives(
        [rec("Software Developer", "15-1252", conf=0.99),
         rec("Software Engineer", "15-1252", conf=0.90)], counts)
    assert conf["15-1252"]["representative"] == "Software Developer"
    assert conf["15-1252"]["decided_by"] == "mapping confidence"


# ---------------------------------------------------------------------- #
# freeze / verify
# ---------------------------------------------------------------------- #
def test_recorded_paths_are_portable_and_tracking_is_cwd_independent(
        tmp_path, monkeypatch):
    """A committed artifact must verify from another clone and another cwd.

    ``verify_hierarchy`` re-opens the table the artifact names, so an absolute
    recorded path made the frozen hierarchy unverifiable anywhere but the
    machine that wrote it.  Separately, ``git ls-files`` resolves a relative
    pathspec against the process cwd, so the tracked check could report a
    committed file as untracked when the CLI was run from elsewhere.
    """
    tracked = mh.SCRIPT_DIR / "e2c_v3_mllmu_hierarchy.py"
    rel = mh.record_path(tracked)
    assert not rel.startswith("/"), rel
    assert mh.resolve_path(rel) == tracked.resolve()
    # a path outside the repository stays absolute: there is nothing to anchor to
    outside = tmp_path / "x.csv"
    outside.write_text("a\n", encoding="utf-8")
    assert mh.record_path(outside).startswith("/")
    assert mh.resolve_path(mh.record_path(outside)) == outside.resolve()
    assert mh.git_state(outside)["in_repository"] is False
    assert mh.git_state(outside)["clean"] is False

    here = Path.cwd()
    # tracked-ness and repo membership are asserted, but NOT cleanliness: this
    # file is edited constantly, so asserting clean here would make the test
    # depend on the live state of the working tree rather than on the code.
    assert mh.git_state(tracked)["in_repository"] is True
    assert mh.git_state(tracked)["tracked"] is True
    before = mh.git_state(tracked)
    try:
        monkeypatch.chdir(tmp_path)
        assert mh.git_state(tracked) == before, \
            "the committed-state check must not depend on the process cwd"
        assert mh._is_tracked(tracked) is True, \
            "the tracked check must not depend on the process cwd"
        # REPO_ROOT wins over cwd when the same relative path exists under both:
        # resolving against cwd here would silently read a DIFFERENT file that
        # merely happens to share a layout, and the sha256 check would then
        # compare the wrong bytes.
        decoy = tmp_path / rel
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_text("not the real table\n", encoding="utf-8")
        assert mh.resolve_path(rel) == tracked.resolve()
        assert mh.resolve_path(rel) != decoy.resolve()
    finally:
        monkeypatch.chdir(here)

    # and a frozen artifact carries no machine-specific prefix for a repo file
    table = mh.load_taxonomy_table(tracked.parent.parent /
                                   "e2c_mllmu" / "external" /
                                   "soc_2019_structure.csv")
    assert not table["path"].startswith("/")
    assert table["path_absolute_at_freeze"].startswith("/")
    rec = mh.build_record("Software Developer", table, "synonym",
                          external_code="15-1252", confidence=0.95,
                          reviewer_decision="retained", reviewer="t")
    assert not rec["source_ref"]["path"].startswith("/")


def test_the_official_title_is_verbatim_and_only_the_output_label_may_shorten(
        tmp_path, monkeypatch):
    """The authority field keeps the release's own words, including the long ones.

    19-2041 is officially "Environmental Scientists and Specialists, Including
    Health".  Shortening ``external_title`` would make the mapping unauditable
    against the pinned release, while forcing the long form into the model's
    output vocabulary would make a strict-accuracy gate measure transcription
    instead of the transformation -- so the two fields are separate, and G8
    fails a shortened authority field specifically.
    """
    # isolate the spelling check: G8's other conditions are satisfied here, so a
    # failure can only come from the field under test.
    monkeypatch.setattr(mh, "git_state", _git_stub())
    table = mh.load_taxonomy_table(_write(tmp_path, PROPOSAL_TABLE, "t.csv"))
    long_title = "Environmental Scientists and Specialists, Including Health"
    assert table["by_code"]["19-2041"] == long_title
    adj = {"raw_label": "Ecologist", "decision": "manual_override",
           "chosen_soc": "19-2041", "chosen_title": long_title,
           "rejected_candidates": [], "target_eligible": True,
           "rationale": "ecology is environmental science",
           "decided_before_experiment": True}

    r = mh.build_record("Ecologist", table, "manual", 0.95, "retained",
                        external_code="19-2041", reviewer="t", adjudication=adj,
                        output_label="Environmental Scientists and Specialists")
    assert r["external_title"] == long_title
    assert r["output_label"] == "Environmental Scientists and Specialists"
    assert r["output_label_is_shortened"] is True
    g8 = mh.run_gates([r], table)["gates"]["G8_pre_freeze_checklist"]
    assert g8["checks"]["official_spelling_preserved_in_external_title"] is True
    assert g8["passed"] is True, g8["issues"]

    # without an explicit output_label the official title IS the output label
    plain = mh.build_record("Ecologist", table, "manual", 0.95, "retained",
                            external_code="19-2041", reviewer="t")
    assert plain["output_label"] == long_title
    assert "output_label_is_shortened" not in plain

    # shortening the AUTHORITY field instead is a hard pre-freeze failure, and
    # it is the ONLY thing that changed between the two records
    bad = dict(r, external_title="Environmental Scientists and Specialists")
    g8b = mh.run_gates([bad], table)["gates"]["G8_pre_freeze_checklist"]
    assert g8b["checks"]["official_spelling_preserved_in_external_title"] is False
    assert g8b["passed"] is False
    assert len(g8b["issues"]) == 1 and "verbatim" in g8b["issues"][0]

    # the release's own spelling survives even where the dataset differs: SOC
    # writes "Archeologists", the source label writes "Archaeologist"
    a = mh.build_record("Archaeologist", table, "manual", 0.95, "retained",
                        external_code="19-3091", reviewer="t")
    assert a["external_title"] == "Anthropologists and Archeologists"
    assert a["original_label"] == "Archaeologist", \
        "the source label keeps the dataset spelling; only the external title " \
        "follows the release"


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
    and a downstream builder refuses to read it.  G8 fails alongside it, because
    the same commitment requirement covers the source SOC table.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    records = [mh.build_record(
        "Software Developer", table, "synonym",
        external_code="15-1252", confidence=0.95,
        reviewer_decision="retained", reviewer="test")]
    monkeypatch.setattr(mh, "HIERARCHY_PATH", tmp_path / "hier.json")
    monkeypatch.setattr(mh, "git_state", _git_stub(untracked=("",)))
    gates = mh.run_gates(records, table)
    assert gates["failed_gates"] == ["G7_committed_before_training",
                                     "G8_pre_freeze_checklist"]
    g7 = gates["gates"]["G7_committed_before_training"]
    assert "not tracked by git" in g7["issues"][0]
    # all four files are checked, not just the hierarchy
    assert g7["checked"] == ["hierarchy_artifact",
                             "manual_override_registry", "source_soc_table",
                             "target_selection_manifest"]
    assert all(not v["clean"] for v in g7["files"].values())
    g8 = gates["gates"]["G8_pre_freeze_checklist"]
    assert g8["checks"]["source_soc_file_is_committed"] is False
    assert "source SOC table" in g8["issues"][0]

    monkeypatch.setattr(mh, "git_state", _git_stub())
    ok = mh.run_gates(records, table)
    assert ok["passed"] is True and ok["failed_gates"] == []


def test_a_tracked_but_modified_artifact_does_not_pass_g7(tmp_path, monkeypatch):
    """"Tracked" is not "committed", and G7 must know the difference.

    ``git ls-files`` is satisfied by a file modified since its commit, so the
    gate this replaces would certify an artifact whose bytes nobody has
    committed -- and a builder would then train against bytes the repository
    does not contain.  The three states are distinct: absent, untracked, and
    tracked-but-dirty, and only the last of those is the silent one.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    records = [mh.build_record(
        "Software Developer", table, "synonym", external_code="15-1252",
        confidence=0.95, reviewer_decision="retained", reviewer="test")]
    monkeypatch.setattr(mh, "HIERARCHY_PATH", tmp_path / "hier.json")

    for stub, needle in ((_git_stub(dirty=("",)),
                          "modified in the working tree"),
                         (_git_stub(absent=("",)), "does not exist"),
                         (_git_stub(untracked=("",)), "not tracked by git")):
        monkeypatch.setattr(mh, "git_state", stub)
        gates = mh.run_gates(records, table)
        assert "G7_committed_before_training" in gates["failed_gates"]
        assert needle in " ".join(
            gates["gates"]["G7_committed_before_training"]["issues"]), needle

    # a DIRTY hierarchy alone is enough: the other three files being clean must
    # not average out into a pass.  "hier.json" is the substring that matches
    # the monkeypatched HIERARCHY_PATH, not the real artifact's filename.
    monkeypatch.setattr(mh, "git_state",
                        _git_stub(dirty=("hier.json",)))
    gates = mh.run_gates(records, table)
    g7 = gates["gates"]["G7_committed_before_training"]
    assert g7["passed"] is False and g7["n_issues"] == 1
    assert "hier.json" in g7["issues"][0]
    assert "modified in the working tree" in g7["issues"][0]
    # and G8's output-commitment entry is a boolean that agrees with G7
    g8 = gates["gates"]["G8_pre_freeze_checklist"]
    assert g8["checks"]["artifact_and_selection_manifest_commitment"] is False
    assert isinstance(g8["checks"]["source_soc_file_is_committed"], bool)


def test_the_first_write_may_leave_only_g7_outstanding(tmp_path, monkeypatch):
    """G7 cannot be satisfied by the write that creates the file.

    The artifact is necessarily untracked until the commit that follows the
    freeze, so refusing to write on G7 alone would make the first freeze
    impossible.  It is recorded as pending instead of being waived, and any
    other failing gate still blocks the write absolutely.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    records = [mh.build_record(
        "Software Developer", table, "synonym", external_code="15-1252",
        confidence=0.95, reviewer_decision="retained", reviewer="test")]
    monkeypatch.setattr(mh, "git_state", _git_stub(untracked=("",)))
    gates = mh.run_gates(records, table)
    assert gates["failed_gates"] == ["G7_committed_before_training",
                                     "G8_pre_freeze_checklist"]
    target = tmp_path / "hier.json"
    # two gates outstanding, one of them not G7 -> refused
    with pytest.raises(RuntimeError) as exc:
        mh.freeze_hierarchy(table, records, gates, path=target)
    assert "refusing to freeze" in str(exc.value)
    assert not target.exists()

    # G7 alone outstanding -> written, and the pending state is recorded.  Both
    # OUTPUT files are absent, which is what a genuine first freeze looks like,
    # so G8's commitment boolean is satisfied by first_write while G7 still
    # reports the truth.
    monkeypatch.setattr(mh, "git_state", _git_stub(
        absent=("mllmu_hierarchy.json", "mllmu_target_selection.json")))
    gates = mh.run_gates(records, table)
    assert gates["failed_gates"] == ["G7_committed_before_training"]
    assert gates["pre_freeze_checks"][
        "artifact_and_selection_manifest_commitment"] is True
    art = mh.freeze_hierarchy(table, records, gates, path=target)
    assert target.exists()
    assert art["g7_status"].startswith("pending_commit")
    assert "before any G6 training" in art["g7_status"]


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
    monkeypatch.setattr(mh, "git_state", _git_stub())
    _forbid_source_dataset(monkeypatch)
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


def _forbid_source_dataset(monkeypatch):
    """Make any read of the MLLMU source dataset an immediate test failure.

    The dataset lives outside the repository, so it is absent in a bare clone
    and present here -- which means the authoring machine CANNOT reproduce the
    CI failure by running the suite.  The dependency has to be manufactured:
    patching the reader to raise turns "works on my machine" into a failure
    wherever the suite runs, and names the regression instead of letting a bare
    FileNotFoundError surface from inside an unrelated assertion.
    """
    def _refused(path=None):
        raise AssertionError(
            "verify_hierarchy must not read the MLLMU source dataset: it is "
            "outside the repository, so depending on it makes a committed "
            "artifact verifiable only on the machine that froze it")
    monkeypatch.setattr(mh, "professions_in_source", _refused)


def test_verify_reproduces_the_selection_from_the_artifact_alone(tmp_path,
                                                                 monkeypatch):
    """The frozen selection must be re-derivable from committed state.

    ``identity_counts`` is recorded in the artifact precisely so verification
    needs no dataset.  Recording it is not enough: the recomputed selection has
    to be COMPARED against the frozen one, or the counts are an unverified
    input and the comparison that justifies recording them never happens.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "git_state", _git_stub())
    _forbid_source_dataset(monkeypatch)
    def adj(label, role, eligible):
        # G8 fails a manual mapping with no recorded adjudication, so both
        # records need one for this test to reach the selection check at all.
        return {"raw_label": label, "decision": "manual_override",
                "chosen_soc": "15-1252", "chosen_title": "Software Developers",
                "rejected_candidates": [], "target_eligible": eligible,
                "rationale": f"{label} maps to the software-developer leaf",
                "decided_before_experiment": True}

    records = [
        mh.build_record("Software Developer", table, "manual", 0.95, "retained",
                        external_code="15-1252", reviewer="t",
                        matrix_role="same_leaf_retention_control",
                        adjudication=adj("Software Developer", "control", False)),
        mh.build_record("Software Engineer", table, "manual", 0.95, "retained",
                        external_code="15-1252", reviewer="t",
                        matrix_role="primary_target",
                        adjudication=adj("Software Engineer", "primary", True)),
    ]
    counts = {"Software Engineer": 117, "Software Developer": 15}
    gates = mh.run_gates(records, table, selected_targets=["Software Engineer"],
                         identity_counts=counts)
    target = tmp_path / "hier.json"
    art = mh.freeze_hierarchy(table, records, gates,
                              selected_targets=["Software Engineer"],
                              identity_counts=counts, path=target)
    assert art["identity_counts"] == counts, "the counts must be committed"
    assert art["target_selection"]["15-1252"]["representative"] == \
        "Software Engineer"
    ok = mh.verify_hierarchy(target)
    assert ok["selection_reproduced_from_artifact"] is True
    assert ok["n_identity_counts"] == 2

    # Tamper with the recorded counts so the rule now picks the other label.
    # This runs BEFORE the content_sha256 check on purpose, so the failure names
    # the selection rather than reporting a generic digest mismatch.
    bad = json.loads(target.read_text(encoding="utf-8"))
    bad["identity_counts"] = {"Software Engineer": 1, "Software Developer": 99}
    target.write_text(json.dumps(bad, default=str), encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        mh.verify_hierarchy(target)
    msg = str(exc.value)
    assert "not reproducible from committed state" in msg
    assert "15-1252" in msg


def test_a_missing_source_dataset_is_reported_not_raised_as_filenotfound(tmp_path):
    """Freezing needs the dataset; verifying does not, and each says so.

    A bare FileNotFoundError from inside ``verify_hierarchy`` gave no hint that
    the dataset is optional for that command, which is how this reached CI.
    """
    with pytest.raises(RuntimeError) as exc:
        mh.professions_in_source(tmp_path / "absent" / "Full_Set.jsonl")
    msg = str(exc.value)
    assert "not part of this repository" in msg
    assert "--verify does not need it" in msg


# ---------------------------------------------------------------------- #
# exact committed state, against real git
# ---------------------------------------------------------------------- #
def _throwaway_repo(tmp_path):
    """A real git repository under tmp_path, for testing ``git_state`` itself.

    The stubbed tests above only exercise the GATE WIRING -- they would pass
    even if ``git_state`` were wrong, because they replace it.  Testing the real
    thing needs real git, and it has to be a throwaway repository: dirtying a
    tracked file in the actual checkout would fail CI's post-preflight
    cleanliness gate, and a test that restores it afterwards still fails every
    run that crashes before the restore.
    """
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=root, check=True,
                       capture_output=True, text=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "test")
    return root, git


def test_git_state_separates_tracked_from_identical_to_index_and_head(tmp_path):
    """Three different states, three different answers.

    The bug this replaces: ``git ls-files`` reports "tracked" for a file that
    has been modified since its commit, so a gate built on it certified an
    artifact whose bytes nobody had committed.  Only the worktree-vs-index and
    index-vs-HEAD comparisons can tell those apart.
    """
    root, git = _throwaway_repo(tmp_path)
    (root / "sub").mkdir()
    p = root / "sub" / "a.json"
    p.write_text("v1\n", encoding="utf-8")
    git("add", "sub/a.json")
    git("commit", "-qm", "init")

    st = mh.git_state(p)
    assert (st["tracked"], st["matches_index"], st["matches_head"],
            st["clean"]) == (True, True, True, True)
    assert st["path"] == "sub/a.json", "reported relative to its OWN repository"
    assert st["repo_root"] == str(root.resolve())

    p.write_text("v2\n", encoding="utf-8")
    st = mh.git_state(p)
    assert st["tracked"] is True, "still tracked: this is the silent failure"
    assert st["matches_index"] is False and st["clean"] is False
    assert "modified in the working tree" in st["reason"]

    git("add", "sub/a.json")
    st = mh.git_state(p)
    assert st["matches_index"] is True and st["matches_head"] is False
    assert st["clean"] is False and "not identical to HEAD" in st["reason"]

    git("commit", "-qm", "v2")
    assert mh.git_state(p)["clean"] is True

    # untracked and absent are distinct answers, not one "not committed"
    untracked = root / "sub" / "new.json"
    untracked.write_text("x\n", encoding="utf-8")
    st = mh.git_state(untracked)
    assert st["exists"] and not st["tracked"]
    assert st["reason"] == "not tracked by git"
    st = mh.git_state(root / "sub" / "nope.json")
    assert not st["exists"] and st["reason"] == "file does not exist"
    # and a file in no repository at all says so rather than guessing
    loose = tmp_path / "loose.json"
    loose.write_text("x\n", encoding="utf-8")
    st = mh.git_state(loose)
    assert st["in_repository"] is False and st["clean"] is False
    assert st["reason"] == "not inside a git repository"


def test_verify_refuses_a_dirty_committed_artifact(tmp_path, monkeypatch):
    """--verify is not satisfied by re-running the gates on the working copy.

    Passing the audit proves the artifact is self-consistent; it does not prove
    the repository CONTAINS it.  A hierarchy edited after its commit must be
    refused, because that is exactly the state in which a builder would train
    against bytes no commit records.
    """
    root, git = _throwaway_repo(tmp_path)
    table_path = root / "soc.csv"
    table_path.write_text(WIDE_CLEAN, encoding="utf-8")
    hier = root / "hier.json"
    monkeypatch.setattr(mh, "HIERARCHY_PATH", hier)
    monkeypatch.setattr(mh, "SELECTION_PATH", root / "sel.json")
    monkeypatch.setattr(mh, "OVERRIDE_PATH", root / "ov.json")
    _forbid_source_dataset(monkeypatch)
    # The source table is an INPUT, so G8 requires it committed before anything
    # is frozen against it.  Committing it first is not test scaffolding, it is
    # the order the gates demand.
    git("add", "soc.csv")
    git("commit", "-qm", "pin the external authority")

    table = mh.load_taxonomy_table(table_path)
    records = [mh.build_record("Software Developer", table, "synonym",
                               external_code="15-1252", confidence=0.95,
                               reviewer_decision="retained", reviewer="t")]
    gates = mh.run_gates(records, table)
    # the two OUTPUT files do not exist yet, so G7 alone is outstanding
    assert gates["failed_gates"] == ["G7_committed_before_training"]
    art = mh.freeze_hierarchy(table, records, gates, path=hier)
    hier.write_text(json.dumps(art, default=str), encoding="utf-8")
    # G7 covers the manifest as well as the hierarchy, because freeze_decisions
    # always writes both; committing only one is the partial freeze this catches.
    (root / "sel.json").write_text("{}\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "freeze")

    ok = mh.verify_hierarchy(hier)
    assert ok["ok"] is True
    assert ok["committed_state"]["hierarchy_artifact"]["clean"] is True

    # edit the committed artifact in place: still tracked, no longer committed
    tampered = json.loads(hier.read_text(encoding="utf-8"))
    tampered["n_records"] = 999
    hier.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    assert mh.git_state(hier)["tracked"] is True, "the old gate would pass here"
    with pytest.raises(RuntimeError) as exc:
        mh.verify_hierarchy(hier)
    assert "committed exactly" in str(exc.value)
    assert "modified in the working tree" in str(exc.value)
    git("checkout", "--", "hier.json")
    assert mh.verify_hierarchy(hier)["ok"] is True, "restoring makes it pass"


# ---------------------------------------------------------------------- #
# the target-selection manifest is validated, not ignored
# ---------------------------------------------------------------------- #
def _freeze_pair(tmp_path, monkeypatch):
    """A hierarchy artifact and a target-selection manifest that agree."""
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "git_state", _git_stub())
    _forbid_source_dataset(monkeypatch)
    records = [mh.build_record("Software Developer", table, "synonym",
                               external_code="15-1252", confidence=0.95,
                               reviewer_decision="retained", reviewer="t",
                               matrix_role="primary_target")]
    counts = {"Software Developer": 15}
    gates = mh.run_gates(records, table,
                         selected_targets=["Software Developer"],
                         identity_counts=counts)
    hier, sel_path = tmp_path / "hier.json", tmp_path / "sel.json"
    art = mh.freeze_hierarchy(table, records, gates,
                              selected_targets=["Software Developer"],
                              identity_counts=counts,
                              selection_manifest_path=sel_path, path=hier)
    sel = {"kind": "mllmu_g6_target_selection",
           "hierarchy_artifact": {"path": mh.record_path(hier),
                                  "content_sha256": art["content_sha256"]},
           "selected_targets": ["Software Developer"],
           "n_selected_targets": 1, "n_distinct_leaf_codes": 1,
           "identity_counts": counts}
    sel["selection_sha256"] = mh.manifest_digest(sel)
    sel_path.write_text(json.dumps(sel, indent=2, sort_keys=True),
                        encoding="utf-8")
    return hier, sel_path, art, sel


def test_verify_validates_the_selection_manifest_against_the_hierarchy(
        tmp_path, monkeypatch):
    """Two files describe one design, so neither may drift alone.

    verify_hierarchy never opened the manifest before, so a hand-edited or
    stale one would sit beside a valid hierarchy and be trusted.  Every field
    the checklist depends on is compared, and the manifest carries its own
    digest -- computed before writing, since a file cannot contain a hash of
    itself.
    """
    hier, sel_path, _art, sel = _freeze_pair(tmp_path, monkeypatch)
    ok = mh.verify_hierarchy(hier)
    assert ok["selection_manifest"]["expected"] is True
    assert ok["selection_manifest"]["ok"] is True
    assert all(v is True
               for v in ok["selection_manifest"]["checks"].values())
    assert sel["selection_sha256"] == json.loads(
        sel_path.read_text(encoding="utf-8"))["selection_sha256"], \
        "the digest must be INSIDE the committed file, not only in the return"

    for field, value, check in (
            ("selected_targets", ["Architect"], "selected_targets_match"),
            ("n_selected_targets", 2, "target_count_matches"),
            ("n_distinct_leaf_codes", 7, "distinct_leaf_count_matches"),
            ("identity_counts", {"Software Developer": 999},
             "identity_counts_match"),
            ("selection_sha256", "0" * 64, "selection_digest_matches"),
    ):
        bad = dict(sel)
        bad[field] = value
        sel_path.write_text(json.dumps(bad, indent=2, sort_keys=True),
                            encoding="utf-8")
        with pytest.raises(RuntimeError) as exc:
            mh.verify_hierarchy(hier)
        assert "disagrees with the hierarchy" in str(exc.value)
        assert check in str(exc.value), field

    # a manifest naming a DIFFERENT hierarchy is the drift that matters most
    bad = dict(sel)
    bad["hierarchy_artifact"] = dict(sel["hierarchy_artifact"],
                                     content_sha256="f" * 64)
    bad["selection_sha256"] = mh.manifest_digest(
        {k: v for k, v in bad.items() if k != "selection_sha256"})
    sel_path.write_text(json.dumps(bad, indent=2, sort_keys=True),
                        encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        mh.verify_hierarchy(hier)
    assert "hierarchy_digest_matches" in str(exc.value)

    # a named manifest that is absent is a failure, not a silent pass
    sel_path.unlink()
    with pytest.raises(RuntimeError) as exc:
        mh.verify_hierarchy(hier)
    assert "does not exist" in str(exc.value)


def test_an_artifact_with_no_manifest_says_so_instead_of_claiming_a_check(
        tmp_path, monkeypatch):
    """freeze_hierarchy alone produces no manifest, and verify must not pretend.

    Reporting ok:True with a manifest check silently absent would be the same
    shape as the discarded-recount bug: a check that looks performed.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "git_state", _git_stub())
    _forbid_source_dataset(monkeypatch)
    records = [mh.build_record("Software Developer", table, "synonym",
                               external_code="15-1252", confidence=0.95,
                               reviewer_decision="retained", reviewer="t")]
    gates = mh.run_gates(records, table)
    hier = tmp_path / "hier.json"
    art = mh.freeze_hierarchy(table, records, gates, path=hier)
    assert art["selection_manifest_path"] is None
    hier.write_text(json.dumps(art, default=str), encoding="utf-8")
    ok = mh.verify_hierarchy(hier)
    assert ok["selection_manifest"]["expected"] is False
    assert "nothing to cross-check" in ok["selection_manifest"]["reason"]


# ---------------------------------------------------------------------- #
# the frozen counts are bound to the bytes they were counted from
# ---------------------------------------------------------------------- #
def _fake_dataset(tmp_path, counts):
    """A JSONL dataset with the nested shape the real one has.

    ``Employment`` is inside ``biography``, itself a JSON STRING -- not a
    top-level key -- so a fixture with a flat shape would let a counting bug
    pass unnoticed.
    """
    p = tmp_path / "Full_Set.jsonl"
    lines = [json.dumps({"biography": json.dumps({"Employment": lab})})
             for lab, n in counts.items() for _ in range(n)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_the_frozen_counts_are_bound_to_the_dataset_they_came_from(tmp_path,
                                                                   monkeypatch):
    """``identity_counts`` is an assertion until it names the bytes it counted.

    Without the source hash a later recount cannot distinguish "the dataset
    changed" from "the counting was wrong", which is the only reason to offer a
    recount at all.  The provenance carries no timestamp, because a freeze has
    to be a fixed point: a field that differs every run would move
    content_sha256 even when nothing about the mapping did.
    """
    ds = _fake_dataset(tmp_path, {"Software Engineer": 117,
                                  "Software Developer": 15})
    prov = mh.source_dataset_provenance(ds)
    assert prov["sha256"] == mh.sha256_file(ds)
    assert prov["n_identities"] == 132
    assert prov["n_distinct_professions"] == 2
    assert prov["outside_repository"] is True
    assert "Employment" in prov["count_semantics"]
    assert mh.source_dataset_provenance(ds) == prov, \
        "provenance must be deterministic or the freeze is not a fixed point"
    assert mh.source_dataset_provenance(tmp_path / "absent.jsonl") is None

    # and it travels into the artifact beside the counts
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "git_state", _git_stub())
    records = [mh.build_record("Software Developer", table, "synonym",
                               external_code="15-1252", confidence=0.95,
                               reviewer_decision="retained", reviewer="t")]
    gates = mh.run_gates(records, table)
    art = mh.freeze_hierarchy(table, records, gates, path=tmp_path / "h.json",
                              identity_counts={"Software Developer": 15},
                              source_dataset=prov)
    assert art["source_dataset"]["sha256"] == prov["sha256"]
    assert art["identity_counts"] == {"Software Developer": 15}


def test_the_opt_in_recount_separates_a_changed_dataset_from_a_wrong_count(
        tmp_path, monkeypatch):
    """Recounting is opt-in, and it reports WHICH of the two went wrong.

    Ordinary --verify must stay runnable in a bare clone, so the recount is a
    separate mode for the machine that staged the data.  A hash mismatch means
    the bytes changed; matching hashes with different counts means the counting
    did.  Collapsing those into one error would send the next reader to look in
    the wrong place.
    """
    ds = _fake_dataset(tmp_path, {"Software Engineer": 117})
    prov = mh.source_dataset_provenance(ds)
    # Both provenances are computed BEFORE anything is patched: patching
    # source_dataset_provenance first and then calling it to build the "changed
    # dataset" would just return the original, and the test would pass by
    # comparing a value with itself.
    changed_dir = tmp_path / "other"
    changed_dir.mkdir()
    other = mh.source_dataset_provenance(
        _fake_dataset(changed_dir, {"Software Engineer": 200}))
    assert other["sha256"] != prov["sha256"], "the fixture must really differ"

    frozen = {"Software Engineer": 117}
    monkeypatch.setattr(mh, "source_dataset_provenance", lambda path=None: prov)
    monkeypatch.setattr(mh, "professions_in_source",
                        lambda path=None: dict(frozen))

    art = {"identity_counts": frozen, "source_dataset": prov}
    out = mh._recount_source(frozen, art)
    assert out["source_sha256_matches"] is True
    assert out["n_drifted_professions"] == 0 and out["drift"] == {}

    # the dataset changed underneath the frozen counts
    monkeypatch.setattr(mh, "source_dataset_provenance",
                        lambda path=None: other)
    with pytest.raises(RuntimeError) as exc:
        mh._recount_source(frozen, art)
    assert "different bytes" in str(exc.value)

    # same bytes, different counts: a counting bug, reported with both numbers
    monkeypatch.setattr(mh, "source_dataset_provenance", lambda path=None: prov)
    monkeypatch.setattr(mh, "professions_in_source",
                        lambda path=None: {"Software Engineer": 99})
    with pytest.raises(RuntimeError) as exc:
        mh._recount_source(frozen, art)
    msg = str(exc.value)
    assert "no longer match the frozen identity counts" in msg
    assert "unchanged" in msg

    # and a machine with no dataset gets told that, not a FileNotFoundError
    monkeypatch.setattr(mh, "source_dataset_provenance", lambda path=None: None)
    with pytest.raises(RuntimeError) as exc:
        mh._recount_source(frozen, art)
    assert "--recount-source needs the MLLMU dataset" in str(exc.value)
