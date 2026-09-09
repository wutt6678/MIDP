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
  G4b two distinct professions mapped to the SAME external occupation id are
      reported (warn by default): legitimate synonyms, but one transformation
      target, so a matrix counting them twice overstates its coverage;
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


def _find_repo_root(start=SCRIPT_DIR):
    """The git root, by walking up for ``.git``.

    Paths recorded in a COMMITTED artifact have to be portable: an absolute path
    bakes this machine's checkout location into the file, and then
    ``verify_hierarchy`` -- which re-reads the table the artifact names -- fails
    outright on any other clone.  Anchoring at the repo root is what makes the
    frozen hierarchy re-verifiable by someone who did not write it.
    """
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return start.parent


REPO_ROOT = _find_repo_root()
#: The dataset root this module belongs to; derived from the script's own
#: location rather than from the process cwd, so the CLI behaves the same
#: whether it is invoked from here or from the repository root.
DATASET_ROOT = SCRIPT_DIR.parent

import e2c_v3_granularity as gx

# e2c_v3_research_validity is deliberately NOT imported here.  It was imported
# only for sha256_file, and it pulls in torch at module scope -- which made
# --verify, a pure JSON-and-git audit that touches no model, depend on the whole
# deep-learning stack.  sha256_file is three lines; the dependency was not worth
# three lines.

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v3_mllmu_hier")


def sha256_file(path, chunk_bytes=1 << 20):
    """SHA-256 of a file's bytes, streamed.

    Local so that reading a hash never imports a model stack.  Chunked because
    the source dataset is megabytes and the verifier may run where memory is
    not generous.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def record_path(p):
    """Repo-root-relative when the file is in the repo, else absolute.

    This is the form that goes INTO a committed artifact.
    """
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)                   # outside the repo: nothing to anchor to


def resolve_path(p):
    """Absolute path for a recorded (possibly repo-relative) path.

    This is the form used to actually OPEN a file.  The repo root is tried
    first and the cwd second, so an artifact written elsewhere still resolves
    if the same relative layout is present.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    for base in (REPO_ROOT, Path.cwd()):
        if (base / p).exists():
            return base / p
    return REPO_ROOT / p                # canonical guess, even if missing


MANIFEST_DIR = DATASET_ROOT / "e2c_mllmu" / "manifests"
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
#: HARD GATE for confirmatory matrix generation, and scoped to the SELECTED
#: transformation targets rather than to the whole mapping table.  Two raw
#: professions sharing one SOC occupation is legitimate and useful -- the
#: lower-frequency duplicate is a same-leaf retention control, testing that
#: editing one identity does not disturb another that shares its external leaf.
#: It becomes invalid only when both are counted as distinct transformations.
G4B_IS_A_WARN = False

#: What a record is FOR.  Distinct from reviewer_decision, which says whether
#: the mapping was accepted at all: "excluded" conflated two different things,
#: so a label that is out of hierarchy scope (Student) and a label that is in
#: scope but must not be transformed (Biotechnologist, whose leaf is the
#: catch-all "Biological Scientists, All Other") now have separate roles.
MATRIX_ROLES = (
    "primary_target",                # transformed, counted as coverage
    "target_eligible",               # may be transformed; not selected
    "same_leaf_retention_control",   # shares a leaf with a target; never a
                                     # target, and NEVER called a sibling
    "retained_only",                 # in the hierarchy, preserved, not edited
    "excluded_out_of_scope",         # no hierarchy; out of the G6 matrix
)
#: Adjudication precedence.  A frozen manual decision outranks everything, and
#: no model-generated recommendation may override it -- which is also why
#: MAPPING_AUTHORITIES has no "model" member.
PRECEDENCE = ("manual_override", "exact_title", "automatic", "exclusion")
OVERRIDE_PATH = MANIFEST_DIR / "mllmu_manual_overrides.json"
#: The control-group contract in gx.sibling_controls does
#: ``_job, l1, l2 = hierarchy_of[iid]`` and unpacks EVERY entry, so an identity
#: with no chain raises rather than becoming an "unrelated" control.  Recorded
#: here because it decides the fate of the non-occupational labels.
CONTROLS_REQUIRE_HIERARCHY = True


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
        "path": record_path(path),
        # kept beside the portable form: the sha256 is the real anchor, and this
        # records where the bytes actually were when the freeze ran.
        "path_absolute_at_freeze": str(path.resolve()),
        "sha256": sha256_file(path),
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
def parents_from_table(code, table):
    """Resolve a code's parents from the TABLE's own level rows.

    Arithmetic on the numbering is not sufficient.  ``soc_levels("29-1221")``
    yields broad parent ``29-1220``, but the pinned release has no such row:
    "Pediatricians, General" sits directly under broad ``29-1210 Physicians``,
    which also holds 29-1211..29-1229.  Accepting the arithmetic parent would
    mean inventing its name -- exactly what G1 exists to prevent, and G1 did
    catch it, but the derivation was still wrong.

    Within one major group SOC codes sort in hierarchy order, so a parent is the
    greatest code at that level which is <= the child.  Returns None for a level
    the table does not carry, rather than guessing one.
    """
    lvl = table.get("level_of_code", {})
    if lvl.get(code) is None:
        return {}
    major = f"{str(code).split('-')[0]}-0000"
    out = {"major_group": major if lvl.get(major) else None}
    same_major = sorted(c for c in lvl if c.split("-")[0] == str(code).split("-")[0]
                        and "." not in c)
    for level, key in (("minor_group", "minor group"),
                       ("broad_occupation", "broad occupation")):
        at_level = [c for c in same_major if lvl.get(c) == key and c <= code]
        out[level] = at_level[-1] if at_level else None
    out["detailed_occupation"] = code if lvl.get(code) == "detailed occupation" else None
    arithmetic = soc_levels(code) or {}
    out["arithmetic_disagreements"] = sorted(
        k for k in ("minor_group", "broad_occupation")
        if arithmetic.get(k) and out.get(k) and arithmetic[k] != out[k])
    return out


def build_record(original_label, table, mapping_type, confidence,
                 reviewer_decision, exclusion_reason=None,
                 external_code=None, proposed_by_model=None,
                 reviewer=None, l2_level="broad_occupation",
                 taxonomy_version="2018", matrix_role=None,
                 output_label=None, adjudication=None):
    """One audited profession record, with parents taken from the table.

    ``external_code`` is the SOC detailed/broad occupation the reviewer
    accepted.  When it is None the code is looked up by title, and a title that
    matches zero or several rows is recorded as ambiguous rather than guessed.

    ``external_title`` is the official title VERBATIM from the pinned release,
    and ``output_label`` may be a shorter label for the frozen output space.  The
    two are kept apart on purpose: 19-2041 is officially "Environmental
    Scientists and Specialists, Including Health", and shortening the field that
    records the external authority would make the mapping unauditable, while
    forcing the long form into the model's output vocabulary would make the
    strict-accuracy gate measure transcription rather than the transformation.
    """
    norm = normalize_label(original_label)
    if matrix_role is not None and matrix_role not in MATRIX_ROLES:
        raise RuntimeError(f"{original_label!r}: matrix_role {matrix_role!r} is "
                           f"not one of {MATRIX_ROLES}")
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
        "matrix_role": matrix_role,
    }
    if adjudication is not None:
        # The frozen adjudication: chosen code, the candidates rejected and why,
        # and the rationale.  G8 fails a manual mapping that lacks it, so an
        # override cannot survive as an unexplained edit to a code.
        rec["adjudication"] = adjudication
    if output_label is not None:
        rec["output_label"] = output_label
        rec["output_label_is_shortened"] = True
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
    # the official leaf title, verbatim -- the external authority's own spelling
    rec["external_title"] = lookup_title(table, code)
    rec.setdefault("output_label", rec["external_title"])
    # Parents come from the TABLE, not from arithmetic: see parents_from_table.
    from_table = parents_from_table(code, table)
    l2_code = from_table.get(l2_level) or lv[l2_level]
    l1_code = from_table.get("major_group") or lv["major_group"]
    rec["parents_resolved_from"] = "source_table" if from_table else "arithmetic"
    if from_table.get("arithmetic_disagreements"):
        rec["parent_arithmetic_disagreements"] = {
            k: {"arithmetic": lv[k], "source_table": from_table.get(k)}
            for k in from_table["arithmetic_disagreements"]}
    rec["minor_group_parent"] = (
        {"code": from_table.get("minor_group"),
         "title": lookup_title(table, from_table.get("minor_group"))}
        if from_table.get("minor_group") else None)
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
# adjudication: manual overrides, precedence, representative selection
# ====================================================================== #
OVERRIDE_DECISIONS = ("manual_override", "same_mapping", "exclude")
REQUIRED_OVERRIDE_FIELDS = ("raw_label", "decision", "target_eligible",
                            "rationale", "decided_before_experiment")


def load_manual_overrides(path=OVERRIDE_PATH):
    """The frozen adjudication registry, keyed by raw label.

    Every override is recorded with the candidate it rejected and why, so an
    adjudication can be audited after the fact instead of being inferred from
    the outcome.  ``decided_before_experiment`` must be true: a decision made
    after seeing a result is not a frozen design choice.
    """
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["overrides"] if isinstance(data, dict) else data
    out = {}
    for e in entries:
        missing = [f for f in REQUIRED_OVERRIDE_FIELDS if f not in e]
        if missing:
            raise RuntimeError(f"override for {e.get('raw_label')!r} is missing "
                               f"{missing}; an adjudication must be explicit")
        if e["decision"] not in OVERRIDE_DECISIONS:
            raise RuntimeError(f"override for {e['raw_label']!r}: decision "
                               f"{e['decision']!r} is not one of "
                               f"{OVERRIDE_DECISIONS}")
        if not e["decided_before_experiment"]:
            raise RuntimeError(
                f"override for {e['raw_label']!r} is not marked "
                f"decided_before_experiment: a decision taken after seeing a "
                f"result cannot be part of a frozen design")
        if e["decision"] != "exclude" and not e.get("chosen_soc"):
            raise RuntimeError(f"override for {e['raw_label']!r}: decision "
                               f"{e['decision']!r} needs chosen_soc")
        out[e["raw_label"]] = e
    return out


def resolve_mapping(raw_label, overrides, proposal=None):
    """Apply the precedence: manual > exact title > automatic > exclusion.

    Returns ``(decision, precedence)``.  A manual override always wins, which is
    what stops an automatic ranker -- or a model proposal -- from quietly
    reversing a frozen adjudication.  ``proposal`` is the deterministic
    candidate list from ``proposal_candidates``; it is consulted only when no
    override exists.
    """
    ov = overrides.get(raw_label)
    if ov:
        return ov, "manual_override"
    p = proposal or {}
    kind = p.get("best_match_kind", "none")
    if kind in ("exact_title", "exact_head") and not p.get("ambiguous"):
        return {"raw_label": raw_label, "decision": "automatic",
                "chosen_soc": p["candidates"][0]["code"],
                "chosen_title": p["candidates"][0]["title"],
                "target_eligible": True}, (
                    "exact_title" if kind == "exact_title" else "automatic")
    if kind != "none" and not p.get("ambiguous") and p.get("best_recall", 0) >= 1.0:
        return {"raw_label": raw_label, "decision": "automatic",
                "chosen_soc": p["candidates"][0]["code"],
                "chosen_title": p["candidates"][0]["title"],
                "target_eligible": True}, "automatic"
    return {"raw_label": raw_label, "decision": "exclude",
            "target_eligible": False,
            "rationale": (f"no unambiguous mapping: best_match_kind={kind!r}, "
                          f"ambiguous={p.get('ambiguous')}")}, "exclusion"


def _semantic_cleanliness(label):
    """Deterministic proxy for "cleaner occupational semantics".  Lower is cleaner.

    Criterion 3 of the representative-selection rule.  It is deliberately
    conservative and only separates cases criteria 1 and 2 cannot: a residual
    catch-all leaf, a non-occupational qualifier, then fewer modifiers.  For the
    three duplicate pairs that actually occur it is never reached -- identity
    count decides all three -- which the tests pin rather than assume.
    """
    n = normalize_label(label)
    return (1 if "all other" in n else 0,
            1 if any(w in n for w in ("retired", "freelance", "student",
                                      "emeritus", "junior", "senior")) else 0,
            len(_tokens(label)),
            n)


def select_target_representatives(records, counts):
    """One target per SOC leaf code, by the frozen selection rule.

    Order: mapping confidence (higher), then identity frequency (higher), then
    cleaner occupational semantics, then deterministic lexical order as the
    FINAL tie-breaker only.  The losers are not discarded -- they become
    same-leaf retention controls, which is a stronger test than dropping them:
    editing one raw profession must not disturb another that shares its external
    leaf.

    Two DIFFERENT SOC occupations sharing a parent are not duplicates.  Coarsening
    sibling occupations to a common parent is a legitimate and important
    experiment, so only the leaf code and the complete chain are compared.
    """
    by_leaf = collections.defaultdict(list)
    for r in records:
        code = r.get("external_occupation_id")
        if code and r.get("reviewer_decision") == "retained":
            by_leaf[code].append(r)
    out = {}
    for code, group in sorted(by_leaf.items()):
        if len(group) == 1:
            out[code] = {"representative": group[0]["original_label"],
                         "same_leaf_controls": [], "n_raw_labels": 1,
                         "decided_by": "only one raw label maps here"}
            continue
        def rank(r):
            conf = r.get("confidence")
            return (-(conf if isinstance(conf, (int, float)) else 0.0),
                    -counts.get(r["original_label"], 0),
                    _semantic_cleanliness(r["original_label"]))
        ordered = sorted(group, key=rank)
        winner, rest = ordered[0], ordered[1:]
        # which criterion actually separated them, recorded not assumed
        by_conf = len({r.get("confidence") for r in group}) > 1
        by_freq = len({counts.get(r["original_label"], 0) for r in group}) > 1
        decided_by = ("mapping confidence" if by_conf and
                      rank(winner)[0] < max(rank(r)[0] for r in rest)
                      else "identity frequency" if by_freq and
                      rank(winner)[1] < min(rank(r)[1] for r in rest)
                      else "cleaner occupational semantics" if
                      len({_semantic_cleanliness(r["original_label"])[:3]
                            for r in group}) > 1
                      else "deterministic lexical order (final tie-break)")
        out[code] = {
            "representative": winner["original_label"],
            "same_leaf_controls": [r["original_label"] for r in rest],
            "n_raw_labels": len(group),
            "decided_by": decided_by,
            "identity_counts": {r["original_label"]:
                                counts.get(r["original_label"], 0)
                                for r in group},
            "selection_rule": (
                "mapping confidence, then identity frequency, then cleaner "
                "occupational semantics, then deterministic lexical order as "
                "the final tie-break only"),
        }
    return out


# ====================================================================== #
# the seven gates
# ====================================================================== #
def run_gates(records, table, selected_targets=None, retained_ids=None,
              identity_counts=None):
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

    # ---- G4b: no two confirmatory TARGETS collide --------------------- #
    # Scoped to the selected targets, NOT to the whole mapping table.  Two raw
    # professions sharing one SOC occupation is legitimate and useful: the
    # lower-frequency duplicate is a same-leaf retention control, testing that
    # editing one identity does not disturb another sharing its external leaf.
    # It is invalid only when both are counted as distinct transformations.
    #
    # Equally important is what this does NOT reject: two DIFFERENT SOC
    # occupations that share a parent.  Coarsening sibling occupations to a
    # common parent is a legitimate and important experiment, so duplication is
    # defined on the leaf code and on the complete chain -- never on a parent.
    by_code = collections.defaultdict(set)
    for r in records:
        if r["reviewer_decision"] == "retained" and r.get("external_occupation_id"):
            by_code[r["external_occupation_id"]].add(r["normalized_label"])
    shared_codes = {c: sorted(v) for c, v in sorted(by_code.items())
                    if len(v) > 1}

    chosen = set(selected_targets or [])
    target_recs = [r for r in records if r["original_label"] in chosen]
    g4b = []
    leaf_seen, chain_seen = {}, {}
    for r in target_recs:
        code = r.get("external_occupation_id")
        role = r.get("matrix_role")
        if role == "same_leaf_retention_control":
            g4b.append(f"{r['original_label']!r}: selected as a target but its "
                       f"matrix_role is same_leaf_retention_control -- a "
                       f"same-leaf duplicate is a control, never a target")
            continue
        if code is None:
            g4b.append(f"{r['original_label']!r}: selected as a target but has "
                       f"no external occupation id, so its leaf is undefined")
            continue
        if code in leaf_seen:
            g4b.append(f"target collision on SOC leaf {code}: "
                       f"{leaf_seen[code]!r} and {r['original_label']!r} are the "
                       f"same transformation, not two; one must become "
                       f"same_leaf_retention_control or be deselected")
        else:
            leaf_seen[code] = r["original_label"]
        chain = json.dumps([code,
                            (r.get("level2_parent") or {}).get("code"),
                            (r.get("level1_parent") or {}).get("code")],
                           sort_keys=True)
        if chain in chain_seen:
            g4b.append(f"target collision on the complete chain {chain}: "
                       f"{chain_seen[chain]!r} and {r['original_label']!r}")
        else:
            chain_seen[chain] = r["original_label"]
    # a parent shared by two different leaves is NOT a collision
    parent_only = collections.defaultdict(list)
    for r in target_recs:
        l2 = (r.get("level2_parent") or {}).get("code")
        if l2 and r.get("external_occupation_id"):
            parent_only[l2].append(r["external_occupation_id"])
    shared_parent_ok = {p: sorted(set(c)) for p, c in parent_only.items()
                        if len(set(c)) > 1}
    gates["G4b_no_duplicate_targets"] = {
        "passed": not g4b or G4B_IS_A_WARN, "n_issues": len(g4b),
        "issues": [] if G4B_IS_A_WARN else g4b,
        "warnings": [] if not G4B_IS_A_WARN else g4b,
        "warn_only": G4B_IS_A_WARN,
        "scope": "selected transformation targets only, not the whole table",
        "n_targets_selected": len(target_recs),
        "n_distinct_leaf_codes": len(leaf_seen),
        "duplicate_targets": shared_codes,
        "same_leaf_is_allowed_as": "same_leaf_retention_control",
        "different_leaves_sharing_a_parent": shared_parent_ok,
        "shared_parent_is_not_a_collision": (
            "two different SOC occupations may share a level-1 or level-2 "
            "parent; coarsening sibling occupations to a common parent is a "
            "legitimate experiment and must not be rejected"),
        "coverage_counts_distinct_leaves": (
            "a duplicate raw profession contributes ZERO additional "
            "hierarchy-coverage count; coverage is the number of distinct SOC "
            "leaf codes among targets, which must equal the number of targets"),
        "criterion": ("no two confirmatory targets share a SOC leaf code, and "
                      "no two have identical complete chains"),
        "consequence": (
            "Freezing fails until target collisions are resolved: counting one "
            "transformation twice overstates how many distinct transformations "
            "were tested."),
    }
    if not G4B_IS_A_WARN:
        warnings.extend(g4b)
    else:
        warnings.extend(f"{code}: {len(labels)} distinct professions map here "
                        f"({', '.join(repr(x) for x in labels)}), so they are "
                        f"ONE transformation target, not {len(labels)}"
                        for code, labels in shared_codes.items())

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
    # "Tracked" is NOT "committed".  git ls-files is satisfied by a file that
    # has been modified since the commit, so a gate built on it would certify an
    # artifact whose bytes nobody has committed -- and a downstream builder
    # would then train against bytes the repository does not contain.  Every
    # file the freeze depends on or produces is compared against both the index
    # and HEAD.
    states = committed_file_states(table)
    problems = _commitment_problems(states, records)
    gates["G7_committed_before_training"] = {
        "passed": not problems, "n_issues": len(problems),
        "issues": [f"{v['path']}: {v['reason']}.  Each file the freeze depends "
                   f"on or produces must be tracked and identical to both the "
                   f"index and HEAD before any G6 training or matrix build; a "
                   f"downstream builder refuses anything less."
                   for v in problems.values()],
        "files": states,
        "checked": sorted(states),
        "criterion": ("the hierarchy artifact, the target-selection manifest, "
                      "the source SOC table and the manual-override registry "
                      "are each tracked by git and identical to both the index "
                      "and HEAD"),
    }

    # ---- target selection: one representative per SOC leaf ------------ #
    selection = select_target_representatives(records, identity_counts or {})

    # ---- G8: the final pre-freeze checklist -------------------------- #
    # Nine things must hold before a hierarchy may become a committed input.
    # Each is reported separately so a failure names itself instead of hiding
    # inside a single boolean.
    g8, checks = [], {}
    chosen = list(selected_targets or [])
    trecs = [r for r in records if r["original_label"] in chosen]

    no_leaf = [r["original_label"] for r in trecs
               if not r.get("external_occupation_id")]
    checks["every_target_has_one_canonical_soc_leaf"] = not no_leaf
    if no_leaf:
        g8.append(f"targets with no canonical SOC leaf: {no_leaf}")

    bad_parents = [r["original_label"] for r in trecs
                   if not ((r.get("level1_parent") or {}).get("code")
                           and (r.get("level1_parent") or {}).get("title")
                           and (r.get("level2_parent") or {}).get("code")
                           and (r.get("level2_parent") or {}).get("title"))]
    checks["every_target_has_valid_l1_and_l2_parents"] = not bad_parents
    if bad_parents:
        g8.append(f"targets missing a valid L1/L2 parent code AND title: "
                  f"{bad_parents}")

    g4b = gates["G4b_no_duplicate_targets"]
    checks["g4b_reports_zero_selected_target_collisions"] = g4b["n_issues"] == 0
    if g4b["n_issues"]:
        g8.append(f"G4b found {g4b['n_issues']} selected-target collision(s)")
    checks["target_count_equals_distinct_leaf_count"] = (
        g4b["n_targets_selected"] == g4b["n_distinct_leaf_codes"])

    controls_selected = [r["original_label"] for r in records
                         if r.get("matrix_role") == "same_leaf_retention_control"
                         and r["original_label"] in chosen]
    checks["alternate_raw_labels_are_controls_not_targets"] = (
        not controls_selected)
    if controls_selected:
        g8.append(f"same-leaf duplicates selected as targets: "
                  f"{controls_selected}; they are retention controls")

    bad_title = [r["original_label"] for r in records
                 if r.get("external_occupation_id")
                 and r.get("external_title")
                 != table["by_code"].get(r["external_occupation_id"])]
    checks["official_spelling_preserved_in_external_title"] = not bad_title
    if bad_title:
        g8.append(f"external_title does not match the pinned release verbatim: "
                  f"{bad_title}")

    checks["source_labels_retain_dataset_spelling"] = all(
        r["original_label"] == r["original_label"].strip()
        and normalize_label(r["original_label"]) != ""
        for r in records)
    if not checks["source_labels_retain_dataset_spelling"]:
        g8.append("a source label was altered or blanked; the raw dataset "
                  "spelling must be preserved in original_label")

    missing_reg = [r["original_label"] for r in records
                   if r.get("mapping_type") == "manual"
                   and not r.get("adjudication")]
    checks["manual_overrides_and_rejected_alternatives_recorded"] = (
        not missing_reg)
    if missing_reg:
        g8.append(f"manual mappings with no recorded adjudication (chosen "
                  f"code plus rejected candidates and why): {missing_reg}")

    no_reason = [r["original_label"] for r in records
                 if r["reviewer_decision"] == "excluded"
                 and not r.get("exclusion_reason")]
    checks["all_excluded_labels_carry_explicit_reasons"] = not no_reason
    if no_reason:
        g8.append(f"excluded with no exclusion_reason: {no_reason}")

    # The same git states G7 used, so the two cannot disagree about what
    # "committed" means.
    src = states["source_soc_table"]
    checks["source_soc_file_is_committed"] = src["clean"]
    if not src["clean"]:
        g8.append(f"the source SOC table {src['path']} is {src['reason']}; the "
                  f"external authority must be committed before a hierarchy is "
                  f"frozen against it")
    reg = states["manual_override_registry"]
    checks["override_registry_is_committed"] = "manual_override_registry" \
        not in problems
    if not checks["override_registry_is_committed"]:
        g8.append(f"the manual-override registry {reg['path']} is "
                  f"{reg['reason']}; every adjudication must be committed, or "
                  f"the mapping cannot be audited against the decision that "
                  f"produced it")
    # A real boolean, not the explanatory string this used to be.  The two
    # OUTPUT files cannot be committed before the write that creates them, so a
    # first write -- where neither exists yet -- is not a failure here; G7 still
    # reports them and freeze_hierarchy still records g7_status pending_commit.
    outs = {k: states[k] for k in G7_OUTPUT_FILES}
    first_write = all(not v["exists"] for v in outs.values())
    checks["artifact_and_selection_manifest_commitment"] = (
        all(k not in problems for k in G7_OUTPUT_FILES) or first_write)
    if not checks["artifact_and_selection_manifest_commitment"]:
        g8.append("the hierarchy artifact and the target-selection manifest "
                  "must both be tracked and identical to the index and HEAD: "
                  + "; ".join(f"{outs[k]['path']} is {outs[k]['reason']}"
                              for k in G7_OUTPUT_FILES
                              if not outs[k]["clean"])
                  + ".  Commit them, or restore them with git checkout, before "
                    "re-freezing: a freeze will not silently overwrite "
                    "committed evidence that has since been edited, because "
                    "the edit and the regeneration are then "
                    "indistinguishable in the history.")
    checks["artifact_and_selection_manifest_commitment_note"] = (
        "enforced by G7 and re-checked by verify_hierarchy after the freeze; "
        "'first write' is exempt only because a file cannot be committed "
        "before it exists")

    gates["G8_pre_freeze_checklist"] = {
        "passed": not g8, "n_issues": len(g8), "issues": g8,
        "checks": checks,
        "criterion": ("the nine required pre-freeze conditions, with the "
                      "commitment condition split so that inputs (source table, "
                      "override registry) and outputs (hierarchy artifact, "
                      "target-selection manifest) are each reported as a "
                      "boolean rather than as prose"),
    }

    passed = all(g["passed"] for g in gates.values())
    return {"passed": passed, "gates": gates, "warnings": warnings,
            "failed_gates": sorted(n for n, g in gates.items()
                                   if not g["passed"]),
            "target_selection": selection,
            "pre_freeze_checks": checks,
            "hierarchy_of": hierarchy_of}


def _git(args, cwd):
    """Run git and return its exit code.  Never raises: an absent repo, a
    missing file and a dirty file are all ANSWERS, not errors."""
    import subprocess
    try:
        return subprocess.run(["git", *args], cwd=str(cwd), check=False,
                              capture_output=True, text=True).returncode
    except Exception:
        return 1


def _git_root_for(path):
    """The repository containing ``path``, or None.

    Derived per file rather than assumed to be REPO_ROOT, so the same check
    works on a throwaway repository in a test directory -- which is the only
    safe way to test a dirty-file refusal, since dirtying a real tracked
    artifact would fail CI's post-preflight cleanliness gate.
    """
    p = Path(path).resolve()
    for start in (p, *p.parents):
        if start.is_dir():
            break
    else:
        return None
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             cwd=str(start), check=False,
                             capture_output=True, text=True)
    except Exception:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return Path(out.stdout.strip())


def git_state(path):
    """Exact git state of one file: tracked AND identical to index AND HEAD.

    ``git ls-files`` alone answers "tracked", which is not the question G7 asks.
    A tracked-but-MODIFIED hierarchy passes that check while the bytes a
    downstream builder reads differ from the bytes the commit records, so the
    gate would certify an artifact nobody has actually committed.  The canonical
    comparisons are ``git diff`` (worktree vs index) and ``git diff --cached``
    (index vs HEAD); both are used instead of parsing porcelain codes because
    they respect .gitattributes filters and so compare what git would store.
    """
    p = Path(path).resolve()
    root = _git_root_for(p)
    rel = p.relative_to(root).as_posix() if root and root in p.parents else None
    out = {"path": rel or record_path(p), "absolute_path": str(p),
           "exists": p.exists(), "in_repository": root is not None,
           "repo_root": str(root) if root else None,
           "tracked": False, "matches_index": False, "matches_head": False,
           "clean": False, "reason": None}
    if not p.exists():
        out["reason"] = "file does not exist"
        return out
    if root is None or rel is None:
        out["reason"] = "not inside a git repository"
        return out
    out["tracked"] = _git(["ls-files", "--error-unmatch", rel], root) == 0
    if not out["tracked"]:
        out["reason"] = "not tracked by git"
        return out
    out["matches_index"] = _git(["diff", "--quiet", "--", rel], root) == 0
    out["matches_head"] = _git(["diff", "--cached", "--quiet", "--", rel], root) == 0
    out["clean"] = out["matches_index"] and out["matches_head"]
    if not out["clean"]:
        out["reason"] = ("tracked but modified in the working tree" if not
                         out["matches_index"] else
                         "tracked but staged and not identical to HEAD")
    return out


#: The four files whose exact committed state the freeze certifies.  Two are
#: INPUTS that must already be committed before anything is frozen, and two are
#: OUTPUTS that the freeze itself writes -- the distinction matters because an
#: output cannot be committed before the write that creates it.
G7_INPUT_FILES = ("source_soc_table", "manual_override_registry")
G7_OUTPUT_FILES = ("hierarchy_artifact", "target_selection_manifest")


def committed_file_states(table):
    """Exact git state of all four files, keyed by role."""
    return {
        "hierarchy_artifact": git_state(HIERARCHY_PATH),
        "target_selection_manifest": git_state(SELECTION_PATH),
        "source_soc_table": git_state(resolve_path(table["path"])),
        "manual_override_registry": git_state(OVERRIDE_PATH),
    }


def _commitment_problems(states, records):
    """Which files are not exactly committed, as one map G7 and G8 both use.

    Shared so the two gates cannot disagree about what "committed" means -- a
    G7 that passes while G8's own commitment boolean fails would be worse than
    either check alone, because each would look like it had covered the other.

    One exemption: a registry that does not exist is owed only if some record
    actually carries a manual mapping.  Requiring a file that need not exist
    would deadlock a first freeze, and excusing it unconditionally would let a
    manual mapping survive with no committed adjudication behind it.
    """
    n_manual = sum(1 for r in records if r.get("mapping_type") == "manual")
    problems = {}
    for role, st in states.items():
        if st["clean"]:
            continue
        if (role == "manual_override_registry" and not st["exists"]
                and n_manual == 0):
            continue
        problems[role] = st
    return problems


def _is_tracked(path):
    """Deprecated shim: tracked-ness alone.  Use ``git_state``.

    Kept only so an older caller cannot silently read "tracked" as "committed";
    G7 and G8 no longer use it.
    """
    return git_state(path)["tracked"]


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
                     identity_ids=None, identity_counts=None,
                     source_dataset=None, selection_manifest_path=None,
                     path=HIERARCHY_PATH):
    """Write the frozen artifact.  Refuses to write a hierarchy that fails."""
    if not gates["passed"]:
        failed = sorted(k for k, g in gates["gates"].items()
                        if not g["passed"])
        # G7 asks that the artifact be committed before TRAINING, and it cannot
        # be satisfied by the first write: the file does not exist yet, so it is
        # necessarily untracked.  Allow G7 alone to be outstanding, record that
        # it is, and let verify_hierarchy enforce it once the commit exists.
        # Every other gate still blocks the write absolutely.
        if failed != ["G7_committed_before_training"]:
            raise RuntimeError(
                f"refusing to freeze: gate(s) {failed} did not pass.  A "
                f"hierarchy that fails its own audit must not become a "
                f"committed input.")
        g7_pending = True
    else:
        g7_pending = False
    retained = [r for r in records if r["reviewer_decision"] == "retained"]
    artifact = {
        "kind": "mllmu_external_profession_hierarchy",
        "audited": True,
        # G7 is the one gate the first write cannot satisfy: the artifact is
        # untracked until the commit that follows this freeze.  Recorded rather
        # than silently waived, and verify_hierarchy enforces it afterwards.
        "g7_status": ("pending_commit: this artifact must be committed before "
                      "any G6 training or matrix build" if g7_pending
                      else "committed"),
        "taxonomy": TAXONOMY_NAME,
        "taxonomy_publisher": TAXONOMY_PUBLISHER,
        "taxonomy_url": TAXONOMY_URL,
        "source_table": {"path": table["path"], "sha256": table["sha256"],
                         # "path" is repo-root-relative so this artifact stays
                         # verifiable from any clone; the absolute form is kept
                         # as provenance only and is never used to open a file.
                         "path_relative_to": "repository root",
                         "path_absolute_at_freeze":
                             table.get("path_absolute_at_freeze"),
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
        # gates["gates"] -- the per-gate map, not the top-level result dict;
        # reading the wrong level silently recorded no duplicates at all.
        "duplicate_targets": (gates.get("gates", {})
                              .get("G4b_no_duplicate_targets", {})
                              .get("duplicate_targets", {})),
        "target_selection": gates.get("target_selection", {}),
        "pre_freeze_checks": gates.get("pre_freeze_checks", {}),
        "n_ambiguous_excluded": sum(
            1 for r in records if r["mapping_type"] == "ambiguous"),
        "retained_labels": sorted(r["original_label"] for r in retained),
        "selected_targets": sorted(selected_targets or []),
        "identity_ids": sorted(identity_ids or []),
        # The counts the representative selection was decided on.  Recorded so
        # verify_hierarchy can REPRODUCE that selection from committed state
        # alone: the source dataset lives outside this repository, so a
        # verification pass that re-read it could only ever run on the machine
        # that froze the artifact.
        "identity_counts": dict(sorted((identity_counts or {}).items())),
        # The counts are only evidence if they are bound to the bytes they were
        # counted from.  Without this the artifact asserts 117 Software
        # Engineers with nothing to check the claim against, and a later
        # recount could not distinguish a dataset change from a counting bug.
        "source_dataset": source_dataset,
        # Recorded so verify_hierarchy can find and validate the manifest that
        # selects targets from this hierarchy.  A path, not a digest: the
        # manifest already binds the hierarchy's digest, and binding each
        # other's would be circular and could never be written.
        "selection_manifest_path": (record_path(selection_manifest_path)
                                    if selection_manifest_path else None),
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


def manifest_digest(obj):
    """Digest of a manifest excluding its own digest field.

    A file cannot contain a hash of itself, so the digest is taken over
    everything else -- the same convention content_sha uses for the hierarchy.
    The previous code hashed the file AFTER writing and stored the result only
    in the returned dict, so the committed manifest carried no digest at all
    and nothing could check it.
    """
    return content_sha({k: v for k, v in obj.items()
                        if k != "selection_sha256"})


def _validate_selection_manifest(art, selection_path=None):
    """Cross-check the target-selection manifest against the hierarchy.

    The two files are written together and describe one design, so a manifest
    that has drifted -- edited by hand, regenerated against a different
    hierarchy, or left behind by a partial freeze -- must be caught rather than
    ignored.  Previously verify_hierarchy never opened it.
    """
    p = selection_path or art.get("selection_manifest_path")
    if not p:
        return {"expected": False, "present": False,
                "reason": ("the artifact names no target-selection manifest, so "
                           "there is nothing to cross-check; freeze_hierarchy "
                           "alone produces this, freeze_decisions does not")}
    p = resolve_path(p)
    out = {"expected": True, "path": record_path(p)}
    if not p.exists():
        raise RuntimeError(
            f"the hierarchy names a target-selection manifest at {out['path']} "
            f"but that file does not exist: the manifest is part of the frozen "
            f"design and must be committed beside the hierarchy")
    sel = json.loads(p.read_text(encoding="utf-8"))
    chosen = set(art.get("selected_targets") or [])
    leaves = {r["external_occupation_id"] for r in art["records"]
              if r["original_label"] in chosen}
    checks = {
        "hierarchy_digest_matches": (
            (sel.get("hierarchy_artifact") or {}).get("content_sha256")
            == art["content_sha256"]),
        "selected_targets_match": (
            sorted(sel.get("selected_targets") or []) == sorted(chosen)),
        "target_count_matches": sel.get("n_selected_targets") == len(chosen),
        "distinct_leaf_count_matches": (
            sel.get("n_distinct_leaf_codes") == len(leaves)),
        "identity_counts_match": (
            dict(sorted((sel.get("identity_counts") or {}).items()))
            == dict(sorted((art.get("identity_counts") or {}).items()))),
        "selection_digest_matches": sel.get("selection_sha256") == \
            manifest_digest(sel),
    }
    bad = sorted(k for k, v in checks.items() if v is not True)
    out["checks"] = checks
    out["ok"] = not bad
    if bad:
        detail = {
            "hierarchy_digest_matches":
                f"manifest records "
                f"{(sel.get('hierarchy_artifact') or {}).get('content_sha256')}, "
                f"hierarchy is {art['content_sha256']}",
            "target_count_matches":
                f"manifest records {sel.get('n_selected_targets')}, hierarchy "
                f"has {len(chosen)}",
            "distinct_leaf_count_matches":
                f"manifest records {sel.get('n_distinct_leaf_codes')}, the "
                f"selected targets span {len(leaves)} leaves",
            "selection_digest_matches":
                f"manifest records {sel.get('selection_sha256')}, its own "
                f"content hashes to {manifest_digest(sel)}",
        }
        raise RuntimeError(
            f"the target-selection manifest at {out['path']} disagrees with the "
            f"hierarchy: " + "; ".join(
                f"{k} ({detail[k]})" if k in detail else k for k in bad))
    return out


def _verify_git_state(artifact_path, art):
    """Exact committed state of the files this verification depends on.

    Re-running the gates proves the artifact is self-consistent; it does not
    prove the repository CONTAINS it.  Only files inside a repository are held
    to the standard -- a fixture frozen into a temporary directory is not
    committable, and failing on that would make every hermetic test depend on
    the real repository's state.
    """
    paths = {"hierarchy_artifact": Path(artifact_path),
             "source_soc_table": resolve_path(art["source_table"]["path"])}
    if art.get("selection_manifest_path"):
        paths["target_selection_manifest"] = resolve_path(
            art["selection_manifest_path"])
    out, dirty = {}, []
    for role, p in sorted(paths.items()):
        st = git_state(p)
        if st["in_repository"]:
            out[role] = st
            if not st["clean"]:
                dirty.append(f"{st['path']}: {st['reason']}")
        else:
            out[role] = dict(st, verdict=(
                "not applicable: outside any git repository"))
    if dirty:
        raise RuntimeError(
            "verification requires each file inside a repository to be "
            "committed exactly -- tracked and identical to both the index and "
            "HEAD: " + "; ".join(dirty))
    return out


def verify_hierarchy(path=HIERARCHY_PATH, selection_path=None,
                     recount_source=False):
    """Re-run every gate against a committed artifact, changing nothing.

    The artifact's own parents are re-checked against the table it names, so a
    table swapped underneath a frozen hierarchy is caught rather than trusted.

    Reads NOTHING outside the repository and the table the artifact names.  The
    identity counts come from the artifact itself: the MLLMU source dataset
    lives outside this checkout, so re-reading it made verification runnable
    only on the machine that froze the artifact -- and it was worse than
    useless, because the recomputed selection was then discarded, so the read
    cost a hard dependency and verified nothing.

    ``recount_source=True`` is the opt-in exception: it re-counts the dataset
    and compares, for the machine that has it.  Ordinary verification does not.
    """
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    # Committed state is checked FIRST.  A dirty artifact also fails G7 further
    # down, but "the hierarchy no longer passes: ['G7_...']" does not tell the
    # reader which file to commit, whereas this does -- and it is cheaper than
    # re-running every gate to discover the same thing.
    git_states = _verify_git_state(path, art)
    # The quarantine decision belongs to the artifact, not to the caller: a
    # verification pass must read the table exactly as the freeze did.
    table = load_taxonomy_table(
        resolve_path(art["source_table"]["path"]),
        allow_quarantine=bool(art["source_table"].get("allow_quarantine")))
    if table["sha256"] != art["source_table"]["sha256"]:
        raise RuntimeError(
            f"the source table at {table['path']} has sha256 "
            f"{table['sha256'][:16]}, but the frozen hierarchy records "
            f"{art['source_table']['sha256'][:16]}: the external authority "
            f"changed underneath a committed artifact")
    counts = art.get("identity_counts") or {}
    recount = _recount_source(counts, art) if recount_source else None
    gates = run_gates(art["records"], table,
                      selected_targets=art.get("selected_targets"),
                      identity_counts=counts)
    if not gates["passed"]:
        failed = sorted(k for k, g in gates["gates"].items()
                        if not g["passed"])
        raise RuntimeError(f"the committed hierarchy no longer passes: "
                           f"{failed}")
    # The representative selection is part of the frozen design, so it is
    # re-derived from the recorded counts and COMPARED rather than trusted: a
    # selection that cannot be reproduced from committed state was never frozen,
    # it was merely written down.
    if gates["target_selection"] != art.get("target_selection"):
        frozen = {k: (v or {}).get("representative")
                  for k, v in (art.get("target_selection") or {}).items()}
        recomputed = {k: (v or {}).get("representative")
                      for k, v in gates["target_selection"].items()}
        differing = sorted(k for k in set(frozen) | set(recomputed)
                           if frozen.get(k) != recomputed.get(k))
        raise RuntimeError(
            f"the representative selection recomputed from the artifact's own "
            f"identity_counts does not match the frozen target_selection for "
            f"SOC leaf/leaves {differing}: the selection is not reproducible "
            f"from committed state")
    if content_sha(art) != art["content_sha256"]:
        raise RuntimeError("content_sha256 does not match the artifact's own "
                           "content: the file was edited after freezing")
    manifest = _validate_selection_manifest(art, selection_path)
    return {"ok": True, "gates_passed": True,
            "content_sha256": art["content_sha256"],
            "n_records": art["n_records"],
            "selection_reproduced_from_artifact": True,
            "n_identity_counts": len(counts),
            "selection_manifest": manifest,
            "committed_state": git_states,
            "source_recount": recount,
            "warnings": gates["warnings"]}


def _recount_source(frozen_counts, art):
    """Re-count the source dataset and compare against the frozen counts.

    Opt-in because the dataset is not in the repository.  A drift is reported
    with both numbers rather than as a bare mismatch, so "the dataset changed"
    and "the counting was wrong" can be told apart.
    """
    prov = source_dataset_provenance()
    if prov is None:
        raise RuntimeError(
            f"--recount-source needs the MLLMU dataset at "
            f"{record_path(FULL_SET)}, which is not present.  Ordinary --verify "
            f"does not need it; this mode exists for the machine that staged "
            f"the data.")
    frozen_prov = art.get("source_dataset") or {}
    live = professions_in_source()
    drift = {k: {"frozen": frozen_counts.get(k), "recounted": live.get(k)}
             for k in set(frozen_counts) | set(live)
             if frozen_counts.get(k) != live.get(k)}
    out = {"source_path": prov["path"], "source_sha256": prov["sha256"],
           "frozen_source_sha256": frozen_prov.get("sha256"),
           "source_sha256_matches": prov["sha256"] == frozen_prov.get("sha256"),
           "n_drifted_professions": len(drift),
           "drift": dict(sorted(drift.items()))}
    if not out["source_sha256_matches"]:
        raise RuntimeError(
            f"the source dataset at {prov['path']} has sha256 "
            f"{prov['sha256'][:16]} but the frozen artifact records "
            f"{str(frozen_prov.get('sha256'))[:16]}: the counts were taken "
            f"from different bytes")
    if drift:
        raise RuntimeError(
            f"{len(drift)} profession(s) no longer match the frozen identity "
            f"counts even though the dataset hash is unchanged: "
            f"{dict(sorted(drift.items())[:5])}")
    return out


def professions_in_source(path=FULL_SET):
    """Every Employment value in MLLMU-Bench with its identity count.

    The frozen manifest took the six most frequent; the source has many more,
    and G6 needs branches, not just frequency.

    The dataset lives OUTSIDE this repository, so it is absent in a bare clone:
    freezing and proposing need it, verification does not.  A missing file is
    reported as that, rather than surfacing as a bare FileNotFoundError from
    deep inside a caller that had no idea the dataset was optional.
    """
    if not Path(path).exists():
        raise RuntimeError(
            f"the MLLMU source dataset is not present at {path}.  It is not "
            f"part of this repository, so --freeze-decisions, --decide and "
            f"--propose-mapping can only run where the dataset is staged.  "
            f"--verify does not need it: it reproduces the selection from the "
            f"identity_counts recorded in the frozen artifact.")
    counts = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts[json.loads(row["biography"])["Employment"]] += 1
    return dict(counts.most_common())


def source_dataset_provenance(path=FULL_SET):
    """Hash and shape of the dataset the identity counts were counted from.

    Recorded beside ``identity_counts`` so the counts are evidence rather than
    an assertion: with the hash bound, a later recount can distinguish "the
    dataset changed" from "the counting was wrong", which an unbound count
    cannot.  Deliberately carries NO timestamp -- a freeze must be a fixed
    point, and a field that differs on every run would change content_sha256
    even when nothing about the mapping did.
    """
    if not Path(path).exists():
        return None
    counts = professions_in_source(path)
    return {"path": record_path(path),
            "outside_repository": _git_root_for(Path(path).resolve()) is None,
            "sha256": sha256_file(path),
            "n_identities": sum(counts.values()),
            "n_distinct_professions": len(counts),
            "count_semantics": (
                "one identity per JSONL line; the profession is "
                "biography.Employment, a JSON string nested inside the row "
                "rather than a top-level key")}


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


#: Confidence assigned by the deterministic title index.  Deliberately the SAME
#: as a certain manual override, so that where both a manual and an automatic
#: mapping are equally sure the tie falls to identity frequency -- which is what
#: the adjudication rationale asks for ("choose Software Engineer, not Software
#: Developer, because it has substantially greater coverage").  Giving the manual
#: entry a higher number instead would let a 1-identity label outrank a 65-identity
#: one on a bookkeeping artifact.
AUTO_CONFIDENCE = 0.95


def build_decided_records(table_path, overrides_path=OVERRIDE_PATH):
    """Apply the frozen adjudications and return records plus the target list.

    Precedence per label: a manual override wins outright; otherwise an
    unambiguous match on a published title is accepted automatically; otherwise
    the label is excluded with a generated reason.  Nothing is dropped silently.

    After the records exist, the duplicate policy is applied: where several raw
    professions share one SOC leaf, ``select_target_representatives`` picks one by
    confidence, then frequency, then cleaner semantics, then lexical order, and
    every other label at that leaf is demoted to ``same_leaf_retention_control``.
    """
    table = load_taxonomy_table(table_path)
    overrides = load_manual_overrides(overrides_path)
    counts = professions_in_source()
    records, decisions = [], []
    for label in sorted(counts):
        prop = proposal_candidates(label, table)
        dec, precedence = resolve_mapping(label, overrides, prop)
        dec = dict(dec)
        dec["precedence"] = precedence
        dec["n_identities"] = counts[label]
        decisions.append(dec)
        if dec["decision"] == "exclude":
            reason = dec.get("exclusion_reason") or dec.get("rationale") or (
                "no unambiguous mapping in the pinned SOC release")
            records.append(build_record(
                label, table, "ambiguous", None, "excluded",
                exclusion_reason=reason,
                matrix_role=dec.get("matrix_role") or "excluded_out_of_scope",
                reviewer="frozen_adjudication_registry",
                adjudication=dec if label in overrides else None))
            continue
        mtype = {"manual_override": "manual", "same_mapping": "manual",
                 "exact_title": "exact"}.get(precedence, "synonym")
        conf = dec.get("confidence")
        if conf is None:
            conf = AUTO_CONFIDENCE
        records.append(build_record(
            label, table, mtype, conf, "retained",
            external_code=dec["chosen_soc"],
            reviewer="frozen_adjudication_registry",
            matrix_role=dec.get("matrix_role") or "target_eligible",
            output_label=dec.get("shorter_output_label"),
            adjudication=dec))

    # ---- apply the duplicate policy ---------------------------------- #
    selection = select_target_representatives(records, counts)
    by_label = {r["original_label"]: r for r in records}
    demotions = []
    for code, sel in selection.items():
        if sel["n_raw_labels"] < 2:
            continue
        for loser in sel["same_leaf_controls"]:
            r = by_label[loser]
            if r.get("matrix_role") in ("retained_only", "excluded_out_of_scope"):
                continue
            was = r.get("matrix_role")
            if was != "same_leaf_retention_control":
                demotions.append({
                    "raw_label": loser, "soc": code,
                    "was": was, "now": "same_leaf_retention_control",
                    "representative": sel["representative"],
                    "decided_by": sel["decided_by"],
                    "identity_counts": sel.get("identity_counts"),
                    "why": ("shares one SOC leaf with the representative, so it "
                            "is not an independent transformation target; it "
                            "stays in the hierarchy as a same-leaf control and "
                            "contributes zero additional coverage")})
                r["matrix_role"] = "same_leaf_retention_control"
                r["target_eligible"] = False
                r["demoted_by_duplicate_policy"] = True
    eligible = {"primary_target", "target_eligible"}
    eligible_targets = sorted(
        v["representative"] for v in selection.values()
        if by_label.get(v["representative"], {}).get("matrix_role") in eligible)
    confirmatory = select_confirmatory_targets(records, counts)
    targets = confirmatory["selected_targets"]
    primary = sorted(r["original_label"] for r in records
                     if r.get("matrix_role") == "primary_target")
    return {
        "table": table, "records": records, "counts": counts,
        "decisions": decisions, "selection": selection,
        "selected_targets": targets, "primary_targets": primary,
        "eligible_targets": eligible_targets,
        "confirmatory_selection": confirmatory,
        "duplicate_demotions": demotions,
        "overrides_applied": sorted(
            label for label in overrides if label in by_label),
        "overrides_not_found_in_source": sorted(
            label for label in overrides if label not in by_label),
    }


SELECTION_PATH = MANIFEST_DIR / "mllmu_target_selection.json"

#: Frozen confirmatory target-selection criteria, chosen before any MLLMU
#: experiment.  At most 3 targets per SOC major group so the matrix does not pile
#: into one branch, and at least 2 identities per target so a strict-accuracy
#: gate is never a single observation with no variance.  Professions that miss
#: the cut KEEP matrix_role=target_eligible: they are deferred, not rejected, and
#: can be promoted without re-adjudicating the mapping.
CONFIRMATORY_MAX_PER_MAJOR_GROUP = 3
CONFIRMATORY_MIN_IDENTITIES = 2


def select_confirmatory_targets(records, counts,
                                max_per_major_group=CONFIRMATORY_MAX_PER_MAJOR_GROUP,
                                min_identities=CONFIRMATORY_MIN_IDENTITIES):
    """The frozen confirmatory target list, deterministically.

    Order: the adjudicated primary targets first, then identity frequency
    descending, then label -- so the selection is reproducible from the records
    alone and does not depend on dict ordering.  Returns the chosen labels plus
    the reason each eligible profession missed the cut, because an unexplained
    omission is indistinguishable from an accidental one.
    """
    eligible = [r for r in records
                if r.get("matrix_role") in ("primary_target", "target_eligible")]
    ordered = sorted(eligible, key=lambda r: (
        0 if r["matrix_role"] == "primary_target" else 1,
        -counts.get(r["original_label"], 0), r["original_label"]))
    chosen, taken, skipped = [], collections.Counter(), []
    for r in ordered:
        label = r["original_label"]
        n = counts.get(label, 0)
        code = r.get("external_occupation_id")
        # A role is not a mapping: if an eligible record somehow carries no SOC
        # leaf, defer it with a reason rather than raising TypeError on a slice,
        # because a crash here loses the whole selection and the reason with it.
        if not code:
            skipped.append({"original_label": label, "soc": None,
                            "major_group": None, "n_identities": n,
                            "reason": ("role is target-eligible but the record "
                                       "has no canonical SOC leaf to select on"),
                            "still": "target_eligible, promotable"})
            continue
        group = code[:2]
        if n < min_identities:
            skipped.append({"original_label": label, "soc": code,
                            "major_group": group, "n_identities": n,
                            "reason": (f"fewer than {min_identities} "
                                       f"identities, so a strict-accuracy gate "
                                       f"would rest on one observation"),
                            "still": "target_eligible, promotable"})
            continue
        if taken[group] >= max_per_major_group:
            skipped.append({"original_label": label, "soc": code,
                            "major_group": group, "n_identities": n,
                            "reason": (f"major group {group} already has "
                                       f"{max_per_major_group} targets"),
                            "still": "target_eligible, promotable"})
            continue
        chosen.append(label)
        taken[group] += 1
    return {"selected_targets": sorted(chosen),
            "n_selected": len(chosen),
            "per_major_group": {g: taken[g] for g in sorted(taken)},
            "deferred": skipped,
            "criteria": {
                "max_per_major_group": max_per_major_group,
                "min_identities": min_identities,
                "order": ("adjudicated primary targets first, then identity "
                          "frequency descending, then label"),
                "deferred_are_not_rejected": (
                    "a deferred profession keeps matrix_role=target_eligible "
                    "and can be promoted without re-adjudicating its mapping")},
            }


def freeze_decisions(table_path, overrides_path=OVERRIDE_PATH,
                     hierarchy_path=HIERARCHY_PATH,
                     selection_path=SELECTION_PATH, quiet=False):
    """Apply the adjudications, run every gate, and write both artifacts.

    Writes the hierarchy AND a target-selection manifest, because the checklist
    requires the artifact, the source SOC file, its hash and the target-selection
    manifest to be committed cleanly together: the selection is a design choice
    and must be as frozen and as auditable as the hierarchy it selects from.

    Refuses to write anything unless every gate except G7 passes; G7 cannot be
    satisfied by the first write and is recorded as pending instead.
    """
    d = build_decided_records(table_path, overrides_path)
    gates = run_gates(d["records"], d["table"],
                      selected_targets=d["selected_targets"],
                      identity_counts=d["counts"])
    for name, g in gates["gates"].items():
        if not quiet:
            logger.info("G6.0 %-42s %s (%d issue(s))", name,
                        "PASS" if g["passed"] else "FAIL", g["n_issues"])
            for issue in g["issues"][:10]:
                logger.info("    - %s", issue)
    for w in gates["warnings"]:
        logger.warning("G6.0 WARN %s", w)
    outstanding = [n for n in gates["failed_gates"]
                   if n != "G7_committed_before_training"]
    if outstanding:
        raise RuntimeError(f"refusing to freeze: gate(s) {outstanding} did not "
                           f"pass; nothing was written")
    art = freeze_hierarchy(d["table"], d["records"], gates,
                           selected_targets=d["selected_targets"],
                           # recorded in the artifact so --verify can reproduce
                           # the selection without the out-of-repo dataset
                           identity_counts=d["counts"],
                           # ...and so those counts are bound to the bytes they
                           # were counted from, not merely asserted
                           source_dataset=source_dataset_provenance(),
                           selection_manifest_path=selection_path,
                           path=hierarchy_path)
    selection = {
        "kind": "mllmu_g6_target_selection",
        "frozen_before_training": True,
        "source_table": {"path": d["table"]["path"],
                         "sha256": d["table"]["sha256"]},
        "hierarchy_artifact": {"path": record_path(hierarchy_path),
                               "content_sha256": art["content_sha256"]},
        "override_registry": {
            "path": record_path(overrides_path),
            "sha256": (sha256_file(overrides_path)
                       if Path(overrides_path).exists() else None),
            "n_applied": len(d["overrides_applied"]),
            "applied": d["overrides_applied"],
            "not_found_in_source": d["overrides_not_found_in_source"],
            "precedence": list(PRECEDENCE)},
        "selected_targets": d["selected_targets"],
        "n_selected_targets": len(d["selected_targets"]),
        "eligible_targets": d["eligible_targets"],
        "n_eligible_targets": len(d["eligible_targets"]),
        "confirmatory_selection": d["confirmatory_selection"],
        "primary_targets": d["primary_targets"],
        "representative_per_soc_leaf": d["selection"],
        "duplicate_demotions": d["duplicate_demotions"],
        "duplicate_targets": gates["gates"][
            "G4b_no_duplicate_targets"]["duplicate_targets"],
        "different_leaves_sharing_a_parent": gates["gates"][
            "G4b_no_duplicate_targets"]["different_leaves_sharing_a_parent"],
        "shared_parent_is_not_a_collision": gates["gates"][
            "G4b_no_duplicate_targets"]["shared_parent_is_not_a_collision"],
        "coverage_counts_distinct_leaves": gates["gates"][
            "G4b_no_duplicate_targets"]["coverage_counts_distinct_leaves"],
        "n_distinct_leaf_codes": gates["gates"][
            "G4b_no_duplicate_targets"]["n_distinct_leaf_codes"],
        "pre_freeze_checks": gates["pre_freeze_checks"],
        "g4b_is_a_hard_gate": not G4B_IS_A_WARN,
        "controls_require_hierarchy": CONTROLS_REQUIRE_HIERARCHY,
        "non_occupational_labels_removed_from_the_matrix": (
            "gx.sibling_controls unpacks a 3-element chain for every identity, "
            "so a label with no hierarchy cannot serve even as an unrelated "
            "control; the non-occupational labels are therefore out of the G6 "
            "matrix entirely and are never described as occupational siblings "
            "or cousins"),
        "matrix_roles": {r["original_label"]: r.get("matrix_role")
                         for r in d["records"]},
        "identity_counts": d["counts"],
    }
    selection_path = Path(selection_path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    # Computed over the manifest's own content BEFORE writing, so the digest is
    # inside the committed file and verify_hierarchy can check it.  Hashing the
    # file afterwards -- what this used to do -- put the digest only in the
    # returned dict, so the manifest carried no checkable digest at all.
    selection["selection_sha256"] = manifest_digest(selection)
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True),
                              encoding="utf-8")
    return {"hierarchy": art, "selection": selection,
            "hierarchy_path": record_path(hierarchy_path),
            "selection_path": record_path(selection_path),
            "gates_passed": gates["passed"],
            "g7_status": art["g7_status"],
            "failed_gates": gates["failed_gates"],
            "n_warnings": len(gates["warnings"])}


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
    p.add_argument("--overrides", type=Path, default=OVERRIDE_PATH,
                   help="frozen manual-adjudication registry")
    p.add_argument("--decide", action="store_true",
                   help="dry run: apply the adjudications, run every gate and "
                        "report the targets, roles and duplicates.  Writes "
                        "nothing.")
    p.add_argument("--freeze-decisions", action="store_true",
                   help="apply the adjudications and write BOTH the hierarchy "
                        "artifact and the target-selection manifest")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--recount-source", action="store_true",
                   help="with --verify: re-count the MLLMU source dataset and "
                        "compare against the frozen identity_counts.  Needs the "
                        "dataset, which is not in this repository; ordinary "
                        "--verify does not and stays runnable in a bare clone")
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
    if args.decide or args.freeze_decisions:
        if not args.taxonomy_table:
            p.error("--decide / --freeze-decisions need --taxonomy-table")
        if args.decide:
            d = build_decided_records(args.taxonomy_table, args.overrides)
            g = run_gates(d["records"], d["table"],
                          selected_targets=d["selected_targets"],
                          identity_counts=d["counts"])
            print(json.dumps({
                "selected_targets": d["selected_targets"],
                "eligible_targets": d["eligible_targets"],
                "confirmatory_selection": d["confirmatory_selection"],
                "primary_targets": d["primary_targets"],
                "duplicate_demotions": d["duplicate_demotions"],
                "overrides_applied": d["overrides_applied"],
                "gates": {n: {"passed": x["passed"], "n_issues": x["n_issues"],
                              "issues": x["issues"]}
                          for n, x in g["gates"].items()},
                "failed_gates": g["failed_gates"],
                "pre_freeze_checks": g["pre_freeze_checks"],
                "wrote": "nothing",
            }, indent=2, default=str))
            return 0
        out = freeze_decisions(args.taxonomy_table, args.overrides)
        logger.info("G6.0: hierarchy %s", out["hierarchy"]["content_sha256"])
        logger.info("G6.0: %s", out["g7_status"])
        print(json.dumps({k: v for k, v in out.items() if k != "selection"},
                         indent=2, default=str))
        return 0
    if args.verify:
        out = verify_hierarchy(recount_source=args.recount_source)
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
