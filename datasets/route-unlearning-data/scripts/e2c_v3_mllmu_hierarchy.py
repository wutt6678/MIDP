"""E2C-v3 MLLMU externally audited profession hierarchy (G6.0).

Freezes, BEFORE any experiment runs, the mapping

    specific MLLMU profession -> occupational family (L2) -> broad sector (L1)

for every profession INCLUDED in a granularity matrix, against an external
occupational taxonomy (2018 SOC / O*NET-SOC 2019).  Nothing here is inferred
from model output: a parent that is not a row in the retrieved source table
does not exist, and the builder fails closed rather than inventing one.

Why an external taxonomy at all.  SALMU's job_levels were declared by the
project; MLLMU's professions are benchmark-native strings, and a hierarchy
built by reading those strings and grouping them by hand would make the
coarsening experiment's own target labels a product of the investigator's
intuition.  The gates below are what turns "a hierarchy" into "an audited
hierarchy".

THE SEVEN GATES (all fail-closed unless a gate says WARN):
  G1  no parent inferred from model output alone -- every L1/L2 code AND title
      must be a row in the retrieved source table, and no record may take its
      authority from a model proposal;
  G2  no target and parent with identical text (after normalization);
  G3  no cyclic hierarchy -- delegated to gx.build_label_dag, which already
      rejects a label that is its own ancestor and a repeated label in a chain;
  G4  no identity assigned to incompatible branches -- one profession has
      exactly one chain, and no identity's chain contradicts another's;
  G5  ambiguous professions excluded from confirmatory cells -- mapping_type
      "ambiguous" REQUIRES reviewer_decision "excluded" plus a reason;
  G6  each evaluated parent has at least one retained sibling WHEN POSSIBLE --
      reported per parent, a WARN where structurally impossible, a FAIL where
      it was possible and not achieved;
  G7  hierarchy and selected targets committed before training -- the artifact
      is not accepted by a downstream builder unless git already tracks it.

Model use is permitted ONLY to propose a mapping or to flag ambiguity.  A
proposal lands in ``proposed_by_model`` and never in the authoritative fields;
``mapping_authority`` records which of {source_table, reviewer} decided, and
G1 fails on "model".

Source table.  A CSV or TSV with at least a code column and a title column,
covering the detailed occupations AND their broad-occupation and major-group
parents (titles for parents are required -- deriving a parent's CODE from the
SOC numbering is arithmetic, but its TITLE must come from the table).  XLSX is
not readable in this environment (openpyxl is absent), so an official XLSX/ZIP
download must be converted to CSV first.  The file's sha256 is recorded per
record, which is the "source-table hash" the artifact requires.

Usage
-----
    # freeze (writes e2c_mllmu/manifests/mllmu_hierarchy.json)
    python scripts/e2c_v3_mllmu_hierarchy.py \\
        --taxonomy-table /path/to/soc_structure.csv \\
        --mapping proposals.csv --freeze
    # verify a committed artifact against the table, changing nothing
    python scripts/e2c_v3_mllmu_hierarchy.py --verify
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e2c_v3_granularity as gx
import e2c_v3_research_validity as rv

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v3_mllmu_hier")

MANIFEST_DIR = Path("e2c_mllmu/manifests")
HIERARCHY_PATH = MANIFEST_DIR / "mllmu_hierarchy.json"
MLLMU_MANIFEST = MANIFEST_DIR / "mllmu_manifest.json"
FULL_SET = Path("/scratch/wutiantong/datasets/MLLMU-Bench/data/Full_Set.jsonl")

#: The external authority.  Recorded verbatim in every record.
TAXONOMY_NAME = "2018 SOC / O*NET-SOC 2019"
TAXONOMY_PUBLISHER = ("U.S. Bureau of Labor Statistics (2018 SOC); O*NET "
                      "Resource Center (O*NET-SOC 2019)")
TAXONOMY_URL = "https://www.onetcenter.org/taxonomy.html"
#: Recorded beside the source table's sha256, because the table itself is
#: committed: redistribution needs the licence next to the bytes, not in a
#: commit message nobody will find later.
TAXONOMY_LICENCE = (
    "2018 SOC structure: U.S. government work, public domain.  O*NET-SOC 2019 "
    "content: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), "
    "attribution to the O*NET Resource Center.  Committed here as the frozen "
    "external authority for G6.0; see source_table.sha256.")

MAPPING_TYPES = ("exact", "synonym", "manual", "ambiguous")
REVIEW_DECISIONS = ("retained", "excluded")
#: What may decide a mapping.  "model" is NOT here: a model may propose, and
#: G1 fails any record whose authority is a model.
MAPPING_AUTHORITIES = ("source_table", "reviewer")

#: The 11 fields the hierarchy artifact must record per profession, in the
#: order the plan lists them.  A record missing one is incomplete, not
#: "probably fine".
REQUIRED_RECORD_FIELDS = (
    "original_label", "normalized_label", "taxonomy", "taxonomy_version",
    "external_occupation_id", "level1_parent", "level2_parent",
    "mapping_type", "source_ref", "confidence", "reviewer_decision",
)

#: WARN, not FAIL: a parent with no retained sibling is a coverage limit the
#: matrix has to report, exactly as the numeric G5 run reported its 117/120.
G6_IS_A_WARN = True


# ====================================================================== #
# SOC code arithmetic
# ====================================================================== #
_SOC_RE = re.compile(r"^(\d{2})-(\d{4})(?:\.\d+)?$")


def soc_levels(code):
    """The four SOC aggregation levels implied by a code's own numbering.

    SOC 2018 codes are hierarchical by construction: for detailed 15-1252 the
    broad occupation is 15-1250, the minor group 15-1200 and the major group
    15-0000.  Deriving the parent CODES this way is arithmetic on the
    published numbering, not inference -- but the parent TITLES must still come
    from the source table, which is what G1 checks.
    """
    m = _SOC_RE.match(str(code).strip())
    if not m:
        return None
    major_digits, rest = m.group(1), m.group(2)
    return {
        "major_group": f"{major_digits}-0000",
        "minor_group": f"{major_digits}-{rest[:2]}00",
        "broad_occupation": f"{major_digits}-{rest[:3]}0",
        "detailed_occupation": f"{major_digits}-{rest}",
    }


# The official "O*NET-SOC and SOC structure" download is a WIDE table: one
# column per level, the code written in whichever column matches that row's
# depth, and the title always in the last column.  A hand-made CSV with a
# single code column is also accepted, so both layouts work.
SOC_LEVEL_COLUMNS = ("Major Group", "Minor Group", "Broad Occupation",
                     "Detailed Occupation", "Detailed O*NET-SOC")
SOC_TITLE_COLUMN = "SOC or O*NET-SOC 2019 Title"

_EXCEL_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_EXCEL_DATE_RE = re.compile(r"^([A-Za-z]{3})-(\d{2})$")
# What the last four digits of a code must look like at each level.  Used only
# to narrow the candidate list for a damaged cell, never to invent a code.
_SOC_LEVEL_SHAPE = {
    "major group": lambda n: n == "0000",
    "minor group": lambda n: n[1:] == "000",
    "broad occupation": lambda n: n[3] == "0",
    "detailed occupation": lambda n: True,
}
_EXCEL_MIN_YEAR, _EXCEL_MAX_YEAR = 1900, 9999   # Excel's own date range


def excel_date_candidates(raw, major_group_digits, level_column=None):
    """Every SOC code an Excel-mangled cell could have been, or [].

    Opening the official XLSX in Excel and re-saving as CSV makes Excel read a
    code like ``11-2021`` as *November of the year 2021* and re-render it with
    a two-digit year, so the cell becomes ``Nov-21``.  Codes whose last four
    digits are not a plausible year (``11-1010``) survive untouched, which is
    why only some rows are damaged.

    This is NOT invertible, and an earlier version of this module wrongly
    assumed the year was 20XX.  A two-digit year keeps only the low digits, so
    ``Nov-00`` is equally consistent with 11-2000, 11-3000, ... 11-9000 --
    three different minor groups.  Reconstructing one of them would have
    silently re-titled a whole branch (it produced 11-2021 "Natural Sciences
    Managers", which is really 11-9121).  So the candidates are enumerated --
    every year in Excel's own range whose last two digits match, filtered by
    the shape the row's level requires -- and the row is quarantined.  The fix
    is a clean re-export, not arithmetic.
    """
    m = _EXCEL_DATE_RE.match(str(raw or "").strip())
    if not m or not major_group_digits:
        return []
    month = _EXCEL_MONTHS.get(m.group(1).lower())
    if month is None or month != int(major_group_digits):
        return []                       # the month disagrees with the branch
    yy = m.group(2)
    shape = _SOC_LEVEL_SHAPE.get(str(level_column or "").lower(),
                                 lambda n: True)
    return sorted(
        f"{major_group_digits}-{year:04d}"
        for year in range(_EXCEL_MIN_YEAR, _EXCEL_MAX_YEAR + 1)
        if f"{year:04d}"[-2:] == yy and shape(f"{year:04d}"))


def normalize_label(text):
    """Casefold + collapse whitespace + strip punctuation at the edges.

    Used for the identical-text gate and for matching a profession against the
    table's titles.  Deliberately conservative: it does not stem, drop
    qualifiers or reorder words, because "Retired Architect" and "Architect"
    are different labels and silently merging them would put an identity in the
    wrong branch.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    return s.strip(" .,;:-")


def singularize(text):
    """Plural-tolerant key, for PROPOSING a mapping only -- never for a gate.

    SOC titles are plural ("Software Developers") while MLLMU professions are
    singular ("Software Developer"), so exact matching alone finds almost
    nothing.  This widens the search for the reviewer's benefit; every gate
    still compares against the published title verbatim.

    Only a single trailing ``s`` is removed (plus ``ies`` -> ``y``).  An
    earlier version also stripped ``ians`` and ``es``, which over-stemmed:
    "Veterinarians" became "veterinar" and so never matched "Veterinarian",
    and "Pediatricians" became "pediatric" and never matched "Pediatrician".
    Stripping one ``s`` handles both, and handles "Databases" -> "database"
    without a separate ``es`` rule.
    """
    s = re.sub(r"[^a-z0-9 ]", " ", normalize_label(text))
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("ies") and len(s) > 5:
        return s[:-3] + "y"
    if s.endswith("s") and len(s) > 3 and not s.endswith("ss"):
        return s[:-1]
    return s


# ====================================================================== #
# source table
# ====================================================================== #
_HEADER_SCAN_LIMIT = 40


def _looks_like_header(cells):
    """Does this row name the columns, rather than hold data?"""
    norm = [re.sub(r"\s+", " ", str(c or "")).strip().lower() for c in cells]
    levels = sum(1 for c in SOC_LEVEL_COLUMNS if c.lower() in norm)
    if levels >= 3:
        return True                     # the official wide layout
    has_code = any(c and ("code" in c) for c in norm)
    has_title = any(c and ("title" in c or c == "name") for c in norm)
    return has_code and has_title       # a hand-made code + title CSV


def _read_delimited(path):
    """Data rows, their true 1-based file lines, the header line, and whether
    the header was positively identified.

    Official spreadsheet exports commonly carry sheet-title preamble above the
    real header (the O*NET download has two title lines and a blank one), so
    the header is LOCATED rather than assumed to be line 1.  Taking a title row
    as the header is not a harmless off-by-three: it turns every real column
    name into a data value, so nothing matches and the loader either refuses on
    an unidentifiable column or, worse, finds a plausible-looking one.

    When no row positively looks like a header, line 1 is used anyway and
    ``header_found`` is False.  That is deliberate: a hand-made CSV with odd
    column names should reach the column-identification check, which reports
    the columns it actually saw, rather than being told only that no header was
    found.  A data row is never *promoted* to a header, because the fallback
    fires only when nothing in the scan window matches.

    Line numbers are carried per row rather than enumerated, because blank
    spacer rows are dropped and a reported line must still point at the real
    line in the file -- these numbers end up in the frozen artifact.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel
        raw = list(csv.reader(f, dialect=dialect))
    if not raw:
        raise RuntimeError(f"{path} is empty")
    idx = next((i for i, cells in enumerate(raw[:_HEADER_SCAN_LIMIT])
                if _looks_like_header(cells)), None)
    header_found = idx is not None
    if idx is None:
        idx = 0                             # fall back, and say so
    header = [str(c or "").strip() for c in raw[idx]]
    rows, linenos = [], []
    for offset, values in enumerate(raw[idx + 1:]):
        if not any(str(v or "").strip() for v in values):
            continue                        # blank spacer row
        padded = list(values) + [""] * (len(header) - len(values))
        rows.append(dict(zip(header, padded[:len(header)])))
        linenos.append(idx + 2 + offset)    # true 1-based file line
    return rows, linenos, idx + 1, header_found


def load_taxonomy_table(path, allow_quarantine=False):
    """Load the external table into ``{code: title}`` plus a normalized index.

    Fails closed, with an actionable message, when the table is absent, when
    its columns cannot be identified, or when any code cell is damaged.  Both
    the official wide layout and a hand-made code+title CSV are accepted.
    Guessing a column -- or quietly dropping or "repairing" a damaged row --
    would put every downstream parent in the wrong place while still looking
    like a success.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"taxonomy table not found: {path}.  G6.0 requires the official "
            f"{TAXONOMY_NAME} structure as a CSV or TSV on disk.  XLSX cannot "
            f"be read here (openpyxl is not installed), so convert the "
            f"official download first.  The file's sha256 is recorded in every "
            f"record, and no parent may be inferred without it.")
    if path.suffix.lower() in (".xlsx", ".xls", ".zip"):
        raise RuntimeError(
            f"{path.name} is {path.suffix}: this environment has no openpyxl, "
            f"so an Excel/ZIP taxonomy download cannot be parsed.  Convert it "
            f"to CSV (code + title columns) and re-run.")
    rows, linenos, header_line, header_found = _read_delimited(path)
    if not rows:
        raise RuntimeError(f"{path} parsed to zero data rows below its header "
                           f"on line {header_line}")
    cols = {c.strip().lower(): c for c in rows[0] if c is not None}
    level_cols = [cols[c.lower()] for c in SOC_LEVEL_COLUMNS if c.lower() in cols]
    wide = len(level_cols) >= 3

    def pick(candidates, what):
        for cand in candidates:
            for low, orig in cols.items():
                if low == cand or cand in low:
                    return orig
        raise RuntimeError(
            f"{path}: cannot identify the {what} column among "
            f"{sorted(cols)}; expected one of {candidates}.  Refusing to "
            f"guess, because a mis-picked column silently mis-parents every "
            f"profession.")

    if wide:
        code_col = " + ".join(level_cols)
        title_col = pick((SOC_TITLE_COLUMN.lower(), "title", "name",
                          "occupation_title"), "occupation title")
    else:
        code_col = pick(("code", "soc_code", "onet_soc_code", "occupation_code",
                         "o*net-soc code", "soc code"), "SOC code")
        title_col = pick(("title", "name", "occupation_title", "description"),
                         "occupation title")

    by_code, by_title, level_of_code = {}, {}, {}
    malformed, quarantined = [], []
    major_digits = None
    for lineno, row in zip(linenos, rows):
        title = str(row.get(title_col) or "").strip()
        if wide:
            filled = [c for c in level_cols if str(row.get(c) or "").strip()]
            if len(filled) != 1 or not title:
                continue                            # header/blank spacer row
            lvl_col, code = filled[0], str(row[filled[0]]).strip()
            if lvl_col.lower() == "major group" and _SOC_RE.match(code):
                major_digits = code.split("-")[0]
            level_name = lvl_col.lower()
        else:
            code = str(row.get(code_col) or "").strip()
            if not code or not title:
                continue
            level_name = None
        if not _SOC_RE.match(code):
            entry = {"line": lineno, "level_column": level_name,
                     "value_in_file": code, "title": title,
                     "major_group": major_digits,
                     "excel_date_candidates":
                         excel_date_candidates(code, major_digits, level_name)}
            if allow_quarantine:
                quarantined.append(entry)
                continue
            malformed.append(entry)
        else:
            by_code[code] = title
            if level_name:
                level_of_code[code] = level_name
            by_title.setdefault(normalize_label(title), []).append(code)
    if malformed:
        shown = "; ".join(f"line {m['line']} {m['level_column']} "
                          f"{m['value_in_file']!r} ({m['title'][:40]!r})"
                          for m in malformed[:6])
        ambiguous = sum(1 for m in malformed
                        if len(m["excel_date_candidates"]) > 1)
        raise RuntimeError(
            f"{path}: {len(malformed)} code cell(s) are not SOC codes -- "
            f"{shown}{'...' if len(malformed) > 6 else ''}.  "
            f"{ambiguous} are Excel date mangling and are NOT recoverable from "
            f"this file: a two-digit year keeps only the low digits, so e.g. "
            f"'Nov-00' is equally consistent with 11-2000, 11-3000 ... "
            f"11-9000.  Refusing to guess (picking one silently re-titles a "
            f"whole branch) and refusing to drop them silently (a dropped row "
            f"cannot serve as a parent).  PREFERRED FIX: re-export the CSV "
            f"with the code columns formatted as TEXT, so no cell is "
            f"date-coerced.  FALLBACK, only if no selected target needs a "
            f"quarantined code: allow_quarantine=True, which records every "
            f"unreadable row with its candidate codes in the frozen artifact "
            f"and makes G1 fail any record whose parent chain reaches one.")
    if not by_code:
        raise RuntimeError(f"{path}: no row had both a code and a title")
    # Two different censuses, kept apart because they disagree.  The wide
    # layout states each row's level in its own column, which is authoritative.
    # Arithmetic cannot: soc_levels("15-0000") returns 15-0000 at all four
    # depths, so a purely arithmetic census counts a major group as a detailed
    # occupation too and over-reports every level.
    levels_by_arithmetic = {}
    for code in by_code:
        lv = soc_levels(code)
        if lv:
            for name, c in lv.items():
                if c == code:
                    levels_by_arithmetic[name] = (
                        levels_by_arithmetic.get(name, 0) + 1)
    _ARITH_TO_COLUMN = {"major_group": "major group", "minor_group": "minor group",
                        "broad_occupation": "broad occupation",
                        "detailed_occupation": "detailed occupation"}
    if level_of_code:
        counts = {}
        for code, lvl in level_of_code.items():
            for name, col in _ARITH_TO_COLUMN.items():
                if lvl == col:
                    counts[name] = counts.get(name, 0) + 1
        levels_present = counts
    else:
        levels_present = levels_by_arithmetic
    return {
        "path": str(path.resolve()),
        "sha256": rv.sha256_file(path),
        "layout": "wide_official" if wide else "long_code_and_title",
        "header_line": header_line,
        "header_identified": header_found,
        "n_preamble_rows_skipped": header_line - 1,
        "code_column": code_col,
        "title_column": title_col,
        "n_rows": len(by_code),
        "by_code": by_code,
        "by_title": by_title,
        "level_of_code": level_of_code,
        "levels_present": levels_present,
        "levels_present_source": ("the table's own level columns"
                                  if level_of_code else
                                  "arithmetic on the code (over-counts: a "
                                  "major group satisfies every level)"),
        "levels_by_arithmetic": levels_by_arithmetic,
        "malformed_rows": malformed,
        "quarantined_rows": quarantined,
        "quarantined_codes": sorted(
            {c for q in quarantined for c in q["excel_date_candidates"]}),
    }


def lookup_title(table, code):
    """Title for a SOC code, from the table only.  Never invented."""
    return table["by_code"].get(code)


def lookup_code(table, label):
    """SOC code(s) whose published title matches ``label`` after normalization.

    Returns a list: more than one match is genuine ambiguity in the source and
    must be surfaced, not resolved by taking the first.
    """
    return list(table["by_title"].get(normalize_label(label), []))


# ====================================================================== #
# records
# ====================================================================== #
def build_record(original_label, table, mapping_type, confidence,
                 reviewer_decision, exclusion_reason=None,
                 external_code=None, proposed_by_model=None,
                 reviewer=None, l2_level="broad_occupation",
                 taxonomy_version="2018"):
    """One audited profession record, with parents taken from the table.

    ``external_code`` is the SOC detailed/broad occupation the reviewer
    accepted.  When it is None the code is looked up by title, and a title that
    matches zero or several rows is recorded as ambiguous rather than guessed.
    """
    norm = normalize_label(original_label)
    codes = [external_code] if external_code else lookup_code(table,
                                                             original_label)
    rec = {
        "original_label": original_label,
        "normalized_label": norm,
        "taxonomy": TAXONOMY_NAME,
        "taxonomy_version": taxonomy_version,
        "taxonomy_publisher": TAXONOMY_PUBLISHER,
        "taxonomy_url": TAXONOMY_URL,
        "external_occupation_id": codes[0] if len(codes) == 1 else None,
        "candidate_external_ids": codes,
        "mapping_type": mapping_type,
        "mapping_authority": "reviewer" if reviewer else "source_table",
        "source_ref": {"kind": "source_table_sha256",
                       "sha256": table["sha256"],
                       "path": table["path"],
                       "url": TAXONOMY_URL},
        "confidence": confidence,
        "reviewer_decision": reviewer_decision,
        "reviewer": reviewer,
        "exclusion_reason": exclusion_reason,
        "l2_level_used": l2_level,
    }
    if proposed_by_model is not None:
        # Recorded, and explicitly NOT authoritative.  G1 fails if a record's
        # authority is the model rather than the table or a reviewer.
        rec["proposed_by_model"] = proposed_by_model
    code = rec["external_occupation_id"]
    if code is None:
        rec["unmapped_reason"] = (
            f"{len(codes)} published title(s) match {original_label!r} after "
            f"normalization: {codes}.  Zero matches means the taxonomy has no "
            f"such title; several means the source itself is ambiguous.  "
            f"Neither is resolved by guessing."
            if codes else
            f"no published title matches {original_label!r} after "
            f"normalization, so there is no external identifier to anchor a "
            f"parent chain")
        rec["mapping_type"] = "ambiguous"
        rec["level1_parent"] = None
        rec["level2_parent"] = None
        return rec
    lv = soc_levels(code)
    if lv is None:
        raise RuntimeError(f"{original_label!r}: {code!r} is not a SOC code")
    l2_code = lv[l2_level]
    l1_code = lv["major_group"]
    l2_title, l1_title = lookup_title(table, l2_code), lookup_title(table,
                                                                    l1_code)
    rec["level2_parent"] = ({"code": l2_code, "title": l2_title,
                             "level": l2_level} if l2_title else
                            {"code": l2_code, "title": None, "level": l2_level})
    rec["level1_parent"] = ({"code": l1_code, "title": l1_title,
                             "level": "major_group"} if l1_title else
                            {"code": l1_code, "title": None,
                             "level": "major_group"})
    if l2_title is None or l1_title is None:
        rec["missing_parent_titles"] = [
            c for c, t in ((l2_code, l2_title), (l1_code, l1_title))
            if t is None]
    return rec


def chain_for(record):
    """The [specific, L2, L1] chain gx.build_label_dag consumes, or None."""
    l1, l2 = record.get("level1_parent"), record.get("level2_parent")
    if not (l1 and l2 and l1.get("title") and l2.get("title")):
        return None
    specific = record["original_label"]
    if specific == l2["title"] or specific == l1["title"]:
        return None
    return [specific, l2["title"], l1["title"]]


# ====================================================================== #
# the seven gates
# ====================================================================== #
def run_gates(records, table, selected_targets=None, retained_ids=None):
    """All seven gates, each with its own verdict and reasons.

    Returns ``{"passed": bool, "gates": {name: {...}}, "hierarchy_of": ...,
    "warnings": [...]}``.  ``passed`` is the conjunction of the FAIL-class
    gates; G6 contributes a warning where a sibling is structurally
    unavailable, because that is a coverage limit to report, not a defect to
    hide -- the lesson the numeric G5 run paid for.
    """
    gates, warnings = {}, []
    by_norm = {}
    for r in records:
        by_norm.setdefault(r["normalized_label"], []).append(r)

    # ---- G1: no parent inferred from model output alone ---------------- #
    g1 = []
    for r in records:
        if r["mapping_authority"] not in MAPPING_AUTHORITIES:
            g1.append(f"{r['original_label']!r}: mapping_authority "
                      f"{r['mapping_authority']!r} is not one of "
                      f"{MAPPING_AUTHORITIES} -- a model may propose but never "
                      f"decide")
        for key in ("level1_parent", "level2_parent"):
            p = r.get(key)
            if not p:
                continue
            if not p.get("title"):
                g1.append(f"{r['original_label']!r}: {key} code "
                          f"{p.get('code')} has NO TITLE in the source table, "
                          f"so accepting it would mean inventing the parent's "
                          f"name")
            elif p["code"] not in table["by_code"]:
                g1.append(f"{r['original_label']!r}: {key} code "
                          f"{p['code']} is not a row in the source table")
            elif table["by_code"][p["code"]] != p["title"]:
                g1.append(f"{r['original_label']!r}: {key} title "
                          f"{p['title']!r} disagrees with the source table's "
                          f"{table['by_code'][p['code']]!r}")
        if r.get("source_ref", {}).get("sha256") != table["sha256"]:
            g1.append(f"{r['original_label']!r}: source_ref sha256 does not "
                      f"match the table this artifact was built from")
    gates["G1_no_parent_from_model_output"] = {
        "passed": not g1, "n_issues": len(g1), "issues": g1,
        "criterion": ("every level-1 and level-2 code AND title is a row in "
                      "the retrieved source table, mapping_authority is "
                      "source_table or reviewer, and the record's source hash "
                      "matches the table"),
    }

    # ---- G2: no target and parent with identical text ------------------ #
    g2 = []
    for r in records:
        spec = normalize_label(r["original_label"])
        for key in ("level1_parent", "level2_parent"):
            p = r.get(key) or {}
            if p.get("title") and normalize_label(p["title"]) == spec:
                g2.append(f"{r['original_label']!r}: {key} has the identical "
                          f"text {p['title']!r}, so the transformation would "
                          f"be a no-op labelled as a coarsening")
    gates["G2_target_parent_not_identical"] = {
        "passed": not g2, "n_issues": len(g2), "issues": g2,
        "criterion": ("no target and no parent share the same normalized "
                      "text"),
    }

    # ---- G3: no cyclic hierarchy (delegated to gx) --------------------- #
    hierarchy_of, g3 = {}, []
    for r in records:
        ch = chain_for(r)
        if ch:
            hierarchy_of[r["original_label"]] = ch
    try:
        gx.build_label_dag(hierarchy_of)
    except ValueError as exc:
        g3.append(f"gx.build_label_dag rejected the hierarchy: {exc}")
    for label, ch in hierarchy_of.items():
        if len(set(ch)) != len(ch):
            g3.append(f"{label!r}: repeated label in its own chain {ch}")
    gates["G3_no_cycle"] = {
        "passed": not g3, "n_issues": len(g3), "issues": g3,
        "n_chains": len(hierarchy_of),
        "criterion": ("gx.build_label_dag accepts the derived chains: no label "
                      "is its own ancestor and no chain repeats a label"),
    }

    # ---- G4: no identity in incompatible branches ---------------------- #
    g4 = []
    for group in by_norm.values():
        chains = {json.dumps(chain_for(r), sort_keys=True) for r in group}
        chains.discard("null")
        if len(chains) > 1:
            g4.append(f"{group[0]['original_label']!r}: {len(group)} records "
                      f"for one normalized label disagree on their chain "
                      f"({sorted(chains)}), so the same profession would sit "
                      f"on two branches")
        l1s = {(r.get("level1_parent") or {}).get("code") for r in group}
        l1s.discard(None)
        if len(l1s) > 1:
            g4.append(f"{group[0]['original_label']!r}: assigned to "
                      f"{len(l1s)} different level-1 branches {sorted(l1s)}")
    gates["G4_no_incompatible_branches"] = {
        "passed": not g4, "n_issues": len(g4), "issues": g4,
        "criterion": ("one profession has exactly one chain, and no profession "
                      "is assigned to two level-1 branches"),
    }

    # ---- G5: ambiguous excluded from confirmatory cells ---------------- #
    g5 = []
    ambiguous = [r for r in records if r["mapping_type"] == "ambiguous"]
    for r in ambiguous:
        if r["reviewer_decision"] != "excluded":
            g5.append(f"{r['original_label']!r}: mapping_type is ambiguous but "
                      f"reviewer_decision is {r['reviewer_decision']!r}; an "
                      f"ambiguous profession cannot enter a confirmatory cell")
        if not r.get("exclusion_reason"):
            g5.append(f"{r['original_label']!r}: excluded but no "
                      f"exclusion_reason recorded")
        if r.get("external_occupation_id"):
            g5.append(f"{r['original_label']!r}: ambiguous yet carries an "
                      f"external id {r['external_occupation_id']}, which a "
                      f"downstream builder would treat as settled")
    if selected_targets:
        amb_norm = {r["normalized_label"] for r in ambiguous}
        for t in selected_targets:
            if normalize_label(t) in amb_norm:
                g5.append(f"selected target {t!r} is an ambiguous profession")
    gates["G5_ambiguous_excluded"] = {
        "passed": not g5, "n_issues": len(g5), "issues": g5,
        "n_ambiguous": len(ambiguous),
        "ambiguous_labels": sorted(r["original_label"] for r in ambiguous),
        "criterion": ("mapping_type 'ambiguous' requires reviewer_decision "
                      "'excluded', a recorded reason, no external id, and "
                      "absence from the selected confirmatory targets"),
    }

    # ---- G6: a retained sibling per evaluated parent, when possible ---- #
    g6, g6w = [], []
    retained = retained_ids if retained_ids is not None else [
        r["original_label"] for r in records
        if r["reviewer_decision"] == "retained"]
    targets = set(selected_targets or [])
    parents = {}
    for r in records:
        if r["original_label"] not in retained:
            continue
        for key in ("level2_parent", "level1_parent"):
            p = r.get(key) or {}
            if p.get("title"):
                parents.setdefault((key, p["title"]), []).append(
                    r["original_label"])
    for (key, title), members in sorted(parents.items()):
        if not any(m in targets for m in members):
            continue                      # not an evaluated parent
        siblings = [m for m in members if m not in targets]
        if siblings:
            continue
        pool = [r["original_label"] for r in records
                if ((r.get(key) or {}).get("title") == title
                    and r["original_label"] not in members)]
        if pool:
            g6.append(f"{key} {title!r} is a target with NO retained sibling, "
                      f"although {sorted(pool)} could have been retained")
        else:
            g6w.append(f"{key} {title!r} has no retained sibling and none was "
                       f"available in the included set: sibling control is "
                       f"NULL for this parent, never a vacuous 1.0")
    gates["G6_retained_sibling_where_possible"] = {
        "passed": not g6, "n_issues": len(g6), "issues": g6,
        "warnings": g6w, "warn_only": G6_IS_A_WARN,
        "criterion": ("each evaluated parent keeps at least one retained "
                      "sibling when the included set makes that possible; "
                      "where it is impossible the control is reported null "
                      "rather than vacuous"),
    }
    warnings.extend(g6w)

    # ---- G7: committed before training -------------------------------- #
    tracked = _is_tracked(HIERARCHY_PATH)
    gates["G7_committed_before_training"] = {
        "passed": tracked, "n_issues": 0 if tracked else 1,
        "issues": [] if tracked else [
            (f"{HIERARCHY_PATH} is not tracked by git.  The hierarchy and the "
             f"selected targets must be committed BEFORE any training phase; "
             f"a downstream builder refuses an untracked hierarchy.")],
        "criterion": ("the frozen hierarchy artifact is committed before any "
                      "G6 training or matrix build"),
    }

    passed = all(g["passed"] for g in gates.values())
    return {"passed": passed, "gates": gates, "warnings": warnings,
            "failed_gates": sorted(n for n, g in gates.items()
                                   if not g["passed"]),
            "hierarchy_of": hierarchy_of}


def _is_tracked(path):
    """Is ``path`` tracked by git?  G7's check, and a builder's precondition."""
    import subprocess
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", str(path)],
            stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ====================================================================== #
# freeze
# ====================================================================== #
def content_sha(obj):
    """Digest over the measurement content, excluding the digest field."""
    body = {k: v for k, v in obj.items() if k != "content_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_hierarchy(table_path, mapping_rows, selected_targets=None,
                    taxonomy_version="2018", allow_quarantine=False):
    """Build every record from a table plus reviewer decisions.

    ``mapping_rows`` is a list of dicts with at least ``original_label`` and
    ``mapping_type``; ``confidence``, ``reviewer_decision``, ``reviewer``,
    ``exclusion_reason``, ``external_code`` and ``proposed_by_model`` are
    optional.  A row whose type is not declared is treated as ambiguous, so an
    undecided profession cannot slip in as settled.
    """
    table = load_taxonomy_table(table_path, allow_quarantine=allow_quarantine)
    records = []
    for row in mapping_rows:
        label = row["original_label"]
        mtype = row.get("mapping_type") or "ambiguous"
        if mtype not in MAPPING_TYPES:
            raise RuntimeError(f"{label!r}: mapping_type {mtype!r} is not one "
                               f"of {MAPPING_TYPES}")
        decision = row.get("reviewer_decision") or (
            "excluded" if mtype == "ambiguous" else "retained")
        if decision not in REVIEW_DECISIONS:
            raise RuntimeError(f"{label!r}: reviewer_decision {decision!r} is "
                               f"not one of {REVIEW_DECISIONS}")
        records.append(build_record(
            label, table, mtype,
            row.get("confidence"), decision,
            exclusion_reason=row.get("exclusion_reason"),
            external_code=row.get("external_code"),
            proposed_by_model=row.get("proposed_by_model"),
            reviewer=row.get("reviewer"),
            l2_level=row.get("l2_level", "broad_occupation"),
            taxonomy_version=taxonomy_version))
    return table, records


def freeze_hierarchy(table, records, gates, selected_targets=None,
                     identity_ids=None, path=HIERARCHY_PATH):
    """Write the frozen artifact.  Refuses to write a hierarchy that fails."""
    if not gates["passed"]:
        failed = sorted(k for k, g in gates["gates"].items()
                        if not g["passed"])
        raise RuntimeError(
            f"refusing to freeze: gate(s) {failed} did not pass.  A hierarchy "
            f"that fails its own audit must not become a committed input.")
    retained = [r for r in records if r["reviewer_decision"] == "retained"]
    artifact = {
        "kind": "mllmu_external_profession_hierarchy",
        "audited": True,
        "taxonomy": TAXONOMY_NAME,
        "taxonomy_publisher": TAXONOMY_PUBLISHER,
        "taxonomy_url": TAXONOMY_URL,
        "source_table": {"path": table["path"], "sha256": table["sha256"],
                         "n_rows": table["n_rows"],
                         "layout": table["layout"],
                         "header_line": table["header_line"],
                         "n_preamble_rows_skipped":
                             table["n_preamble_rows_skipped"],
                         "licence": TAXONOMY_LICENCE,
                         "code_column": table["code_column"],
                         "title_column": table["title_column"],
                         "levels_present": table["levels_present"],
                         # Recorded so --verify re-reads the table exactly as
                         # this freeze did, and so a quarantined cell is never
                         # silently absent from the committed evidence.
                         "allow_quarantine": bool(table["quarantined_rows"]),
                         "n_quarantined_rows": len(table["quarantined_rows"]),
                         "quarantined_rows": table["quarantined_rows"],
                         "quarantine_note": (
                             "These rows' code cells were date-mangled by "
                             "Excel and are NOT recoverable from this file; "
                             "they are recorded with every code they could "
                             "have been, never repaired.  Any record whose "
                             "parent chain reaches a quarantined code fails "
                             "G1.  The clean fix is a re-export with the code "
                             "columns formatted as text."
                             if table["quarantined_rows"] else None)},
        "required_record_fields": list(REQUIRED_RECORD_FIELDS),
        "records": records,
        "n_records": len(records),
        "n_retained": len(retained),
        "n_ambiguous_excluded": sum(
            1 for r in records if r["mapping_type"] == "ambiguous"),
        "retained_labels": sorted(r["original_label"] for r in retained),
        "selected_targets": sorted(selected_targets or []),
        "identity_ids": sorted(identity_ids or []),
        # gx.build_label_dag's own input shape, keyed by profession label;
        # the matrix builder expands it to identity ids.
        "hierarchy_of": gates["hierarchy_of"],
        "gates": gates["gates"],
        "gates_passed": True,
        "warnings": gates["warnings"],
        "committed_before_training_required": True,
        "model_use_scope": ("a model may PROPOSE a mapping or FLAG ambiguity; "
                            "it is never the authority.  proposals are stored "
                            "in proposed_by_model and G1 fails any record "
                            "whose mapping_authority is not source_table or "
                            "reviewer"),
    }
    artifact["content_sha256"] = content_sha(artifact)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    logger.info("G6.0: froze %s (%d records, %d retained, content_sha256 %s)",
                path, len(records), len(retained),
                artifact["content_sha256"][:16])
    return artifact


def verify_hierarchy(path=HIERARCHY_PATH):
    """Re-run every gate against a committed artifact, changing nothing.

    The artifact's own parents are re-checked against the table it names, so a
    table swapped underneath a frozen hierarchy is caught rather than trusted.
    """
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    # The quarantine decision belongs to the artifact, not to the caller: a
    # verification pass must read the table exactly as the freeze did.
    table = load_taxonomy_table(
        art["source_table"]["path"],
        allow_quarantine=bool(art["source_table"].get("allow_quarantine")))
    if table["sha256"] != art["source_table"]["sha256"]:
        raise RuntimeError(
            f"the source table at {table['path']} has sha256 "
            f"{table['sha256'][:16]}, but the frozen hierarchy records "
            f"{art['source_table']['sha256'][:16]}: the external authority "
            f"changed underneath a committed artifact")
    gates = run_gates(art["records"], table,
                      selected_targets=art.get("selected_targets"))
    if not gates["passed"]:
        failed = sorted(k for k, g in gates["gates"].items()
                        if not g["passed"])
        raise RuntimeError(f"the committed hierarchy no longer passes: "
                           f"{failed}")
    if content_sha(art) != art["content_sha256"]:
        raise RuntimeError("content_sha256 does not match the artifact's own "
                           "content: the file was edited after freezing")
    return {"ok": True, "gates_passed": True,
            "content_sha256": art["content_sha256"],
            "n_records": art["n_records"],
            "warnings": gates["warnings"]}


def professions_in_source(path=FULL_SET):
    """Every Employment value in MLLMU-Bench with its identity count.

    The frozen manifest took the six most frequent; the source has many more,
    and G6 needs branches, not just frequency.
    """
    counts = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts[json.loads(row["biography"])["Employment"]] += 1
    return dict(counts.most_common())


def audit_table(table_path, allow_quarantine=True):
    """Report on the external CSV itself.  Changes nothing, freezes nothing.

    Answers, before any reviewer time is spent: is this the right file, does it
    carry every level G6 needs as a parent, are any cells damaged, and which
    MLLMU professions match a published SOC title (exactly, and modulo plural).
    A match here is a PROPOSAL for the reviewer; it is never written into a
    record by this function.
    """
    try:
        table = load_taxonomy_table(table_path, allow_quarantine=False)
        damaged = []
    except RuntimeError as exc:
        table = load_taxonomy_table(table_path,
                                   allow_quarantine=allow_quarantine)
        damaged = [{"message": str(exc)}]
    by_singular = collections.defaultdict(list)
    for title, codes in table["by_title"].items():
        by_singular[singularize(title)].extend(
            (table["by_code"][c], c) for c in codes)
    matched, unmatched = [], []
    for label, n in professions_in_source().items():
        exact = table["by_title"].get(normalize_label(label), [])
        loose = by_singular.get(singularize(label), [])
        (matched if (exact or loose) else unmatched).append(
            {"profession": label, "n_identities": n,
             "exact_title_codes": exact,
             "plural_tolerant_candidates": loose,
             "n_candidates": len(loose),
             "ambiguous": len(loose) > 1})
    groups = collections.Counter(
        c.split("-")[0] for c in table["by_code"]
        if table.get("level_of_code", {}).get(c) == "detailed occupation")
    q_groups = collections.Counter(
        str(q["major_group"]) for q in table["quarantined_rows"])
    return {
        "path": table["path"],
        "sha256": table["sha256"],
        "layout": table["layout"],
        "header_line": table["header_line"],
        "n_preamble_rows_skipped": table["n_preamble_rows_skipped"],
        "columns": {"code": table["code_column"],
                    "title": table["title_column"]},
        "n_codes": table["n_rows"],
        "rows_per_level": table["level_of_code"]
        and collections.Counter(table["level_of_code"].values()) or {},
        "levels_present_by_arithmetic": table["levels_present"],
        "major_groups_covered": len(groups),
        "damaged_cells": (len(table["malformed_rows"])
                          + len(table["quarantined_rows"])),
        "n_candidates_if_ambiguous": sorted(
            {len(q["excel_date_candidates"]) for q in table["quarantined_rows"]}
            | {len(m["excel_date_candidates"]) for m in table["malformed_rows"]}),
        "quarantined_rows": table["quarantined_rows"],
        "quarantined_by_major_group": dict(q_groups),
        "quarantine_is_ambiguous": all(
            len(q["excel_date_candidates"]) > 1
            for q in table["quarantined_rows"]) if table["quarantined_rows"]
        else None,
        "load_refusal": damaged,
        "professions_matched": len(matched),
        "professions_unmatched": len(unmatched),
        "identities_matched": sum(m["n_identities"] for m in matched),
        "identities_unmatched": sum(m["n_identities"] for m in unmatched),
        "matched": sorted(matched, key=lambda m: -m["n_identities"]),
        "unmatched": sorted(unmatched, key=lambda m: -m["n_identities"]),
        "meaning": (
            "A match is a PROPOSAL only: mapping_type, confidence and the "
            "reviewer decision are still required per record, and gates G1-G7 "
            "run on the reviewer's choice, never on this index."),
    }


# ====================================================================== #
# reviewer-facing proposals: deterministic, from the table, never a decision
# ====================================================================== #
#: A person's profession sits at the SOC detailed-occupation level.  The 149
#: O*NET-SOC extensions (17-2199.08) are deeper than that and would add a
#: fifth depth to a chain built for specific -> L2 -> L1, so they are not
#: proposed as targets.
PROPOSAL_LEVEL = "detailed occupation"
PROPOSAL_STOPWORDS = frozenset({
    "and", "of", "the", "a", "an", "all", "other", "except", "not",
    "including", "include", "postsecondary", "elementary", "middle",
    "secondary", "special", "various", "related"})
MATCH_KINDS = ("exact_title", "exact_head", "head_tokens", "tokens", "none")


def _title_head(title):
    """The occupation name before SOC's qualifier clause.

    SOC appends ", Except Naval" / ", All Other" / ", Including Health" after a
    comma, and that qualifier is what breaks naive similarity: "Architects,
    Except Landscape and Naval" carries two extra tokens, so a Jaccard-style
    score ranks "Database Architects" ABOVE the correct 17-1011 and would put
    an architect in the Computer branch.  Comparing the head segment first
    removes that failure without weakening anything.
    """
    return str(title or "").split(",")[0].strip()


def _tokens(text):
    """Content tokens of a label or title, each singularized, stopwords dropped.

    Singularization must be PER TOKEN.  Applied to the whole phrase it only
    strips the final word's trailing ``s``, so "Environmental Scientists and
    Specialists" yielded the token ``scientists`` and never matched the label
    token ``scientist`` -- which cost 19-2041 its head_tokens match and let
    15-2051 "Data Scientists" rank first for Environmental Scientist, a wrong
    branch.
    """
    s = re.sub(r"[^a-z0-9 ]", " ", normalize_label(text))
    out = set()
    for word in s.split():
        stem = singularize(word)
        if stem and stem not in PROPOSAL_STOPWORDS:
            out.add(stem)
    return out


def _edit_distance_at_most_one(a, b):
    """True when two tokens differ by a single insertion, deletion or swap."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    short, long_ = sorted((a, b), key=len)
    diffs = 0
    i = j = 0
    while i < len(short) and j < len(long_):
        if short[i] == long_[j]:
            i += 1
            j += 1
            continue
        diffs += 1
        if diffs > 1:
            return False
        j += 1                       # one extra character in the longer token
    return True


def proposal_candidates(label, table, max_candidates=4):
    """Rank the table's own detailed occupations against one MLLMU profession.

    Deterministic and derived only from published titles -- no model, no
    invented codes.  Scoring is RECALL of the label's tokens, not Jaccard, so a
    correct title carrying qualifiers is not penalized for carrying them; ties
    break on fewer extra title tokens, then on code, so the ranking is stable.

    ``match_kind`` says how strong the proposal is, and ``none`` means the
    table offers nothing: that is a finding, not a gap to paper over.
    """
    want = _tokens(label)
    label_norm = normalize_label(label)
    scored, table_tokens = [], set()
    for code, title in table["by_code"].items():
        if table.get("level_of_code", {}).get(code) != PROPOSAL_LEVEL:
            continue
        head = _title_head(title)
        head_toks = _tokens(head)
        full_toks = _tokens(title)
        table_tokens |= full_toks
        if normalize_label(title) == label_norm:
            kind, recall = "exact_title", 1.0
        elif singularize(head) == singularize(label):
            kind, recall = "exact_head", 1.0
        else:
            overlap = want & full_toks
            if not overlap:
                continue
            recall = len(overlap) / len(want) if want else 0.0
            kind = "head_tokens" if want <= head_toks else "tokens"
        scored.append({
            "code": code, "title": title,
            "level": PROPOSAL_LEVEL,
            "major_group": code.split("-")[0],
            "match_kind": kind, "recall": round(recall, 4),
            "n_extra_title_tokens": len(_tokens(title) - want),
            "chain": soc_levels(code),
        })
    order = {k: i for i, k in enumerate(MATCH_KINDS)}
    scored.sort(key=lambda c: (order[c["match_kind"]], -c["recall"],
                               c["n_extra_title_tokens"], c["code"]))
    kind = scored[0]["match_kind"] if scored else "none"
    # Ambiguous means a genuine TIE AT THE TOP RANK, not merely that weaker
    # candidates exist.  An earlier version flagged any exact match with more
    # than one scored candidate, which wrongly called Architect -> 17-1011,
    # Graphic Designer -> 27-1024 and Civil Engineer -> 17-2051 ambiguous just
    # because unrelated occupations share a token with them.
    tie_at_top = (len(scored) > 1
                  and scored[1]["match_kind"] == kind
                  and scored[1]["recall"] == scored[0]["recall"])
    # A spelling variant is reported, never applied: SOC writes "Archeologists"
    # where MLLMU writes "Archaeologist", and deciding that is the reviewer's.
    near = (sorted({w for w in table_tokens
                    if any(_edit_distance_at_most_one(w, x) for x in want)}
                   - want) if want else [])
    return {
        "original_label": label,
        "best_match_kind": kind,
        "best_recall": scored[0]["recall"] if scored else 0.0,
        "best_candidate": scored[0]["code"] if scored else None,
        "n_candidates": len(scored),
        "ambiguous": tie_at_top,
        "spelling_variants_in_table": near,
        "candidates": scored[:max_candidates],
        "level_proposed": PROPOSAL_LEVEL,
        "authority": "source_table",
    }


def select_balanced_pool(cands_by_label, counts, n_professions=12,
                         min_major_groups=6, min_identities=2,
                         min_recall=0.5, max_per_major_group=2):
    """Pick professions for BRANCH SPREAD, not for frequency.

    The frozen MLLMU manifest took the six most common professions, which is
    why Software Engineer and Software Developer sit side by side in the same
    branch.  This picks greedily on new major groups covered, tie-breaking on
    identity count then label, so the pool is deterministic and spread across
    the taxonomy.

    Only professions the table can actually place are eligible: a best match of
    ``none``, a recall below ``min_recall``, too few identities, or an AMBIGUOUS
    top rank is reported as ineligible rather than silently included.

    Coverage is measured on each profession's BEST candidate, never on the union
    of its candidates.  Scoring the union rewards exactly the wrong thing: a
    profession whose candidates are scattered across many branches looks like
    the most valuable pick, so the first version of this function chose Marine
    Biologist first because its weak candidates reached groups 17, 19, 25 and 53
    -- including 53-5011 "Sailors and Marine Oilers".  That is ambiguity being
    paid as diversity, and it would have biased the whole matrix toward the
    least mappable professions.
    """
    eligible, ineligible = {}, []
    for label, c in cands_by_label.items():
        n = counts.get(label, 0)
        why = None
        if c["best_match_kind"] == "none":
            why = "no SOC detailed-occupation title matches"
        elif c["best_recall"] < min_recall:
            why = f"best recall {c['best_recall']} is below {min_recall}"
        elif n < min_identities:
            why = f"only {n} identity/identities, below {min_identities}"
        elif c["ambiguous"]:
            why = ("top rank is ambiguous: several candidates tie, so the "
                   "branch is not determined by the table")
        best_group = (c["candidates"][0]["major_group"]
                      if c["candidates"] else None)
        if why or best_group is None:
            ineligible.append({"original_label": label, "n_identities": n,
                               "reason": why or "no candidate",
                               "best_match_kind": c["best_match_kind"],
                               "best_recall": c["best_recall"],
                               "best_candidate": (c["candidates"][0]["code"]
                                                  if c["candidates"] else None),
                               "best_major_group": best_group})
        else:
            eligible[label] = c
    chosen, covered = [], set()
    taken = collections.Counter()
    while eligible and len(chosen) < n_professions:
        def key(item, covered=covered, taken=taken):
            # bound as defaults: `covered` and `taken` grow each iteration and a
            # late binding would rank against the final state, not this round's
            label, c = item
            group = c["candidates"][0]["major_group"]
            return (group in covered,                 # new branch first
                    taken[group] >= max_per_major_group,   # then spread within
                    -counts.get(label, 0), label)     # then frequency
        label, c = min(eligible.items(), key=key)
        best = c["candidates"][0]
        chosen.append({"original_label": label,
                       "n_identities": counts.get(label, 0),
                       "best_candidate": best["code"],
                       "best_candidate_title": best["title"],
                       "best_match_kind": c["best_match_kind"],
                       "best_recall": c["best_recall"],
                       "major_group": best["major_group"]})
        covered.add(best["major_group"])
        taken[best["major_group"]] += 1
        del eligible[label]
    return {
        "pool": chosen,
        "n_pool": len(chosen),
        "major_groups_covered": sorted(covered),
        "n_major_groups": len(covered),
        "per_major_group": {g: taken[g] for g in sorted(covered)},
        "meets_min_major_groups": len(covered) >= min_major_groups,
        "ineligible": sorted(ineligible, key=lambda x: (-x["n_identities"],
                                                        x["original_label"])),
        "criteria": {"n_professions": n_professions,
                     "min_major_groups": min_major_groups,
                     "min_identities": min_identities,
                     "min_recall": min_recall,
                     "max_per_major_group": max_per_major_group,
                     "selection_rule": (
                         "greedy: a profession whose BEST candidate opens a new "
                         "SOC major group is taken first, then one from a group "
                         "still under max_per_major_group, then identity count "
                         "descending, then label -- branch spread first, "
                         "frequency only as a tie-break")},
    }


PROPOSAL_COLUMNS = (
    "original_label", "n_identities", "in_balanced_pool", "pool_major_groups",
    "proposed_candidates", "proposal_note",
    # Everything below is a REVIEWER decision and is written blank: an
    # unreviewed row has no mapping_type, so build_hierarchy treats it as
    # ambiguous and G5 excludes it.  Nothing can be retained without a human.
    "mapping_type", "external_code", "confidence", "reviewer_decision",
    "reviewer", "exclusion_reason", "proposed_by_model")


def build_proposal_csv(table_path, out_path, n_professions=12,
                       min_major_groups=6, min_identities=2, min_recall=0.5,
                       max_per_major_group=2):
    """Write the reviewer-facing mapping proposal.  Freezes nothing.

    One row per profession (not one per candidate) so the file can be edited
    into a ``--mapping`` input directly: a profession appearing on several rows
    would build several records with the same normalized label and fail G4.
    """
    table = load_taxonomy_table(table_path)
    counts = professions_in_source()
    cands = {label: proposal_candidates(label, table) for label in counts}
    pool = select_balanced_pool(cands, counts, n_professions=n_professions,
                                min_major_groups=min_major_groups,
                                min_identities=min_identities,
                                min_recall=min_recall,
                                max_per_major_group=max_per_major_group)
    in_pool = {p["original_label"]: p for p in pool["pool"]}
    rows = []
    for label in sorted(counts, key=lambda x: (x not in in_pool, -counts[x], x)):
        c = cands[label]
        note = []
        if c["best_match_kind"] == "none":
            note.append("NO candidate in the table at level "
                        f"{PROPOSAL_LEVEL}")
        if c["spelling_variants_in_table"]:
            note.append("spelling variant(s) present in the table: "
                        + ", ".join(c["spelling_variants_in_table"]))
        if c["ambiguous"]:
            note.append("AMBIGUOUS: several equally ranked candidates")
        rows.append({
            "original_label": label,
            "n_identities": counts[label],
            "in_balanced_pool": "true" if label in in_pool else "false",
            "pool_major_groups": ",".join(
                [in_pool[label]["major_group"]]) if label in in_pool else "",
            "proposed_candidates": "; ".join(
                f"{x['code']}|{x['title']}|{x['match_kind']}|recall={x['recall']}"
                for x in c["candidates"]),
            "proposal_note": "; ".join(note),
            "mapping_type": "", "external_code": "", "confidence": "",
            "reviewer_decision": "", "reviewer": "", "exclusion_reason": "",
            "proposed_by_model": "false",
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(PROPOSAL_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    return {"path": str(out_path.resolve()), "n_rows": len(rows),
            "columns": list(PROPOSAL_COLUMNS), "pool": pool,
            "source_table_sha256": table["sha256"],
            "decision_columns_written_blank": [
                "mapping_type", "external_code", "confidence",
                "reviewer_decision", "reviewer", "exclusion_reason"],
            "why_blank": (
                "A proposal is not a decision.  build_hierarchy treats a row "
                "with no mapping_type as ambiguous and G5 excludes it, so this "
                "file cannot produce a retained record until a reviewer fills "
                "it in -- and MAPPING_AUTHORITIES has no 'model' member."),
            }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taxonomy-table", type=Path, default=None,
                   help="CSV/TSV of the official SOC structure (code + title)")
    p.add_argument("--mapping", type=Path, default=None,
                   help="CSV of reviewer decisions: original_label, "
                        "mapping_type, external_code, confidence, "
                        "reviewer_decision, reviewer, exclusion_reason, "
                        "proposed_by_model")
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--taxonomy-version", default="2018")
    p.add_argument("--allow-quarantine", action="store_true",
                   help="tolerate code cells Excel mangled into dates "
                        "('Nov-21'), recording each one with its candidate "
                        "codes instead of repairing it; only sound if no "
                        "selected target needs a quarantined code")
    p.add_argument("--audit-table", action="store_true",
                   help="report on the CSV only: layout, rows per level, "
                        "damaged cells, and which MLLMU professions match a "
                        "published title.  Changes nothing.")
    p.add_argument("--propose-mapping", type=Path, default=None,
                   help="write a reviewer-facing mapping proposal CSV: ranked "
                        "candidates from the table's own titles, a "
                        "branch-balanced pool marked, and every decision "
                        "column left blank.  Freezes nothing.")
    p.add_argument("--pool-size", type=int, default=12,
                   help="professions to mark as the balanced pool")
    p.add_argument("--min-major-groups", type=int, default=6,
                   help="SOC major groups the pool should span")
    p.add_argument("--min-identities", type=int, default=2,
                   help="identities a profession needs to be pool-eligible")
    p.add_argument("--min-recall", type=float, default=0.5,
                   help="token recall a candidate needs to be pool-eligible")
    p.add_argument("--max-per-major-group", type=int, default=2,
                   help="soft cap on professions taken from one SOC major "
                        "group, so the pool does not pile into one branch")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--list-professions", action="store_true")
    args = p.parse_args(argv)

    if args.list_professions:
        for label, n in professions_in_source().items():
            print(f"{n:4d}  {label}")
        return 0
    if args.audit_table:
        if not args.taxonomy_table:
            p.error("--audit-table needs --taxonomy-table")
        print(json.dumps(audit_table(args.taxonomy_table), indent=2,
                         default=str))
        return 0
    if args.propose_mapping:
        if not args.taxonomy_table:
            p.error("--propose-mapping needs --taxonomy-table")
        out = build_proposal_csv(
            args.taxonomy_table, args.propose_mapping,
            n_professions=args.pool_size,
            min_major_groups=args.min_major_groups,
            min_identities=args.min_identities,
            min_recall=args.min_recall,
            max_per_major_group=args.max_per_major_group)
        print(json.dumps(out, indent=2, default=str))
        return 0
    if args.verify:
        out = verify_hierarchy()
        print(json.dumps(out, indent=2))
        return 0
    if not (args.taxonomy_table and args.mapping and args.freeze):
        p.error("--freeze needs --taxonomy-table and --mapping "
                "(or use --verify / --list-professions / --audit-table)")
    with open(args.mapping, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("original_label")]
    table, records = build_hierarchy(args.taxonomy_table, rows,
                                     selected_targets=args.targets,
                                     taxonomy_version=args.taxonomy_version,
                                     allow_quarantine=args.allow_quarantine)
    gates = run_gates(records, table, selected_targets=args.targets)
    for name, g in gates["gates"].items():
        logger.info("G6.0 %-38s %s (%d issue(s))", name,
                    "PASS" if g["passed"] else "FAIL", g["n_issues"])
        for issue in g["issues"][:10]:
            logger.info("    - %s", issue)
    for w in gates["warnings"]:
        logger.warning("G6.0 WARN %s", w)
    if not gates["passed"]:
        logger.error("G6.0: gates did not pass; nothing was written")
        return 1
    art = freeze_hierarchy(table, records, gates,
                           selected_targets=args.targets)
    logger.info("G6.0: content_sha256 %s -- commit this artifact BEFORE any "
                "G6 training (gate G7 checks that git tracks it)",
                art["content_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
