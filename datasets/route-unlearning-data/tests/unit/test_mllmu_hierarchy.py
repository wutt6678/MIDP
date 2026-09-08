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


def test_g4b_reports_two_professions_collapsing_onto_one_code(tmp_path, monkeypatch):
    """Synonyms are legitimate, but they are ONE target, not two.

    G4 keys on the normalized label, so it cannot see "Software Engineer" and
    "Software Developer" both mapped to 15-1252.  Left unreported, a matrix
    would test the same transformation twice and count it as two professions.
    """
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "_is_tracked", lambda path: True)

    def rec(label, mtype):
        return mh.build_record(label, table, mtype, external_code="15-1252",
                               confidence=0.9, reviewer_decision="retained",
                               reviewer="test")

    gates = mh.run_gates([rec("Software Developer", "synonym")], table)
    assert gates["gates"]["G4b_no_duplicate_targets"]["shared_codes"] == {}
    assert gates["gates"]["G4b_no_duplicate_targets"]["warnings"] == []

    both = [rec("Software Developer", "synonym"), rec("Software Engineer", "manual")]
    gates = mh.run_gates(both, table)
    g4b = gates["gates"]["G4b_no_duplicate_targets"]
    assert g4b["shared_codes"] == {"15-1252": ["software developer",
                                               "software engineer"]}
    assert g4b["warn_only"] is True
    assert g4b["passed"] is True, "a warning must not fail the freeze"
    assert "ONE transformation target, not 2" in g4b["warnings"][0]
    assert "overstates" in g4b["consequence"]
    assert any("15-1252" in w for w in gates["warnings"])
    # G4 itself stays silent: the labels differ, so it genuinely cannot see this
    assert gates["gates"]["G4_no_incompatible_branches"]["passed"] is True

    # the guard is not vacuous: as a hard gate it fails
    monkeypatch.setattr(mh, "G4B_IS_A_WARN", False)
    hard = mh.run_gates(both, table)["gates"]["G4b_no_duplicate_targets"]
    assert hard["passed"] is False and hard["issues"]


def test_duplicate_targets_are_carried_into_the_frozen_artifact(tmp_path, monkeypatch):
    """The constraint travels with the artifact, not just the run log."""
    table = mh.load_taxonomy_table(_write(tmp_path, WIDE_CLEAN))
    monkeypatch.setattr(mh, "_is_tracked", lambda path: True)

    def rec(label, mtype):
        return mh.build_record(label, table, mtype, external_code="15-1252",
                               confidence=0.9, reviewer_decision="retained",
                               reviewer="test")

    records = [rec("Software Developer", "synonym"),
               rec("Software Engineer", "manual")]
    gates = mh.run_gates(records, table)
    art = mh.freeze_hierarchy(table, records, gates,
                              path=tmp_path / "hier.json")
    assert art["duplicate_targets"] == {"15-1252": ["software developer",
                                                     "software engineer"]}


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
