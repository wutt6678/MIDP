#!/usr/bin/env python3
"""E2C-v3 granularity phase (G0): schemas, validators, matrix builders, metrics.

Pure CPU logic for the granularity matrix runner.  Implements plan sections
2 (numeric task definition), 4 (declarative transformation schema), 5
(SALMU taxonomic matrix), 6 (CelebA numeric matrix), 11 (granularity /
numeric / preservation metrics) and 12 (frozen pass criteria).

Design rules enforced here
==========================
- All bin edges, rounding policy, category intervals and label formats are
  FROZEN in the numeric schema before any target selection or GPU run.
- Transformation targets must be genuine ancestors (taxonomic) or valid
  containments/roundings (numeric) of the source -- never chosen after
  observing model behaviour.
- Every non-target identity has an explicit retain expectation (its
  baseline association); siblings of transformed identities keep their OWN
  specific labels unless explicitly targeted.
- The candidate vocabulary contains every source and target label plus the
  refusal label; parsing-ambiguity checks run before any GPU execution.
"""
from __future__ import annotations

import itertools
import re

DELETED_LABEL = "Unknown"
SEEDS_DEFAULT = [17, 42, 123]
MATRIX_SEED = 17

# ----------------------------------------------------------------------
# Frozen pass criteria (plan section 12).  Frozen BEFORE any cell runs.
# ----------------------------------------------------------------------
PASS_CRITERIA = {
    "strict_expected_accuracy": 1.0,     # targets AND retained, exact
    "min_target_p_desired": 0.90,
    "max_target_p_source": 0.01,
    "min_candidate_mass": 0.99,
    "retained_strict_accuracy": 1.0,
    "sibling_strict_accuracy": 1.0,
    "max_wrong_branch_rate": 0.0,
    "max_unparseable_outputs": 0,
    "max_multi_label_outputs": 0,
    "e2e_policy": ("unconditional E2E is NOT a pass requirement where the "
                   "frozen route failed its held-out gate; association and "
                   "routing conclusions are gated separately"),
}

# Numeric resolutions ordered fine -> coarse.  "rounded" is a point value
# at depth 0 produced by the rounding operation (tracked separately).
RESOLUTION_DEPTH = {"exact": 0, "rounded": 0, "narrow": 1, "broad": 2,
                    "category": 3}


# ======================================================================
# Taxonomy (SALMU / audited MLLMU)
# ======================================================================
def build_label_dag(hierarchy_of):
    """Label DAG child->parents from per-identity [job, level1, level2].

    Raises on contradictions (a label that is both ancestor and descendant
    of another, or a repeated label inside one chain).
    """
    parents = {}
    for iid, chain in hierarchy_of.items():
        if len(set(chain)) != len(chain):
            raise ValueError(f"repeated label in chain of {iid}: {chain}")
        for child, parent in itertools.pairwise(chain):
            parents.setdefault(child, set()).add(parent)
    # acyclicity + contradiction check
    def anc(label, seen=None):
        seen = seen or set()
        out = set()
        for p in parents.get(label, ()):
            if p in seen:
                raise ValueError(f"cycle/contradiction at {label}->{p}")
            out.add(p)
            out |= anc(p, seen | {label})
        return out
    for label in list(parents):
        a = anc(label)
        if label in a:
            raise ValueError(f"label {label!r} is its own ancestor")
    return {k: sorted(v) for k, v in parents.items()}


def ancestors_of(dag, label):
    out, stack = set(), [label]
    while stack:
        cur = stack.pop()
        for p in dag.get(cur, ()):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


def descendants_of(dag, label):
    child_of = {}
    for child, ps in dag.items():
        for p in ps:
            child_of.setdefault(p, set()).add(child)
    out, stack = set(), [label]
    while stack:
        cur = stack.pop()
        for c in child_of.get(cur, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def classify_taxonomic(parsed, expected, dag):
    """Depth-error classification for ONE parsed label.

    correct | over_abstraction | under_abstraction | wrong_branch |
    refusal | unparseable
    """
    if parsed is None:
        return "unparseable"
    if parsed == DELETED_LABEL:
        return "refusal"
    if parsed == expected:
        return "correct"
    if parsed in ancestors_of(dag, expected):
        return "over_abstraction"
    if parsed in descendants_of(dag, expected):
        return "under_abstraction"
    return "wrong_branch"


def sibling_controls(iid, hierarchy_of, targets):
    """Retained sibling / cousin / unrelated controls for a target.

    sibling   = shares level1 (immediate parent), not itself targeted
    cousin    = shares level2 (upper group) but not level1, not targeted
    unrelated = different level2, not targeted
    """
    _job, l1, l2 = hierarchy_of[iid]
    sib, cou, unrel = [], [], []
    for other, (oj, ol1, ol2) in sorted(hierarchy_of.items()):
        if other == iid or other in targets:
            continue
        if ol1 == l1:
            sib.append(other)
        elif ol2 == l2:
            cou.append(other)
        else:
            unrel.append(other)
    return {"sibling": sib, "cousin": cou, "unrelated": unrel}


# ======================================================================
# Numeric schema (frozen bins; plan section 2)
# ======================================================================
def narrow_interval(field_schema, value):
    w, a = field_schema["narrow"]["width"], field_schema["narrow"]["anchor"]
    k = (value - a) // w
    return a + k * w, a + k * w + w - 1


def broad_interval(field_schema, value):
    w, a = field_schema["broad"]["width"], field_schema["broad"]["anchor"]
    k = (value - a) // w
    return a + k * w, a + k * w + w - 1


def round_value(field_schema, value):
    g = field_schema["rounding"]["to"]
    ties = field_schema["rounding"]["ties"]
    q, r = divmod(value, g)
    if r * 2 > g or (r * 2 == g and ties == "up"):
        q += 1
    return q * g


def category_of(field_schema, value):
    for name, (lo, hi) in field_schema["category"].items():
        if lo <= value <= hi:
            return name
    return None


def fmt_label(field_schema, kind, value=None, interval=None, category=None):
    u = field_schema["unit"]
    if kind == "exact" or kind == "rounded":
        return field_schema["label_format"]["exact"].format(v=value, unit=u)
    if kind in ("narrow", "broad"):
        a, b = interval
        return field_schema["label_format"]["bin"].format(a=a, b=b, unit=u)
    if kind == "category":
        return field_schema["label_format"]["category"].format(c=category)
    raise ValueError(kind)


_NUM_RE = re.compile(r"^(\d+)(?:\s*–\s*(\d+))?\s+(.+)$")


def parse_numeric_label(label, schema):
    """Parse a frozen-format numeric label.

    Returns dict {kind, field, value|(lo,hi)|category} or None.
    kinds: exact | bin | category.  (rounded values parse as exact-format
    point labels; the operation, not the surface form, distinguishes them.)
    """
    if not label or label == DELETED_LABEL:
        return None
    m = _NUM_RE.match(label.strip())
    if m:
        lo = int(m.group(1))
        unit = m.group(3).strip()
        field = None
        for fname, fs in schema.items():
            if fs["unit"] == unit:
                field = fname
                break
        if field is None:
            return None
        if m.group(2) is not None:
            hi = int(m.group(2))
            for kind in ("narrow", "broad"):
                w = schema[field][kind]["width"]
                if hi - lo + 1 == w:
                    return {"kind": "bin", "width_class": kind,
                            "field": field, "lo": lo, "hi": hi}
            return {"kind": "bin", "width_class": "nonstandard",
                    "field": field, "lo": lo, "hi": hi}
        return {"kind": "exact", "field": field, "value": lo}
    for fname, fs in schema.items():
        for cat in fs["category"]:
            if label.strip().lower() == cat.lower():
                return {"kind": "category", "field": fname, "category": cat}
    return None


def valid_numeric_transformation(op, field_schema, exact_value,
                                 source_label, target_label):
    """Frozen-policy validity of one numeric transformation."""
    fs = field_schema
    if op == "exact_to_narrow":
        want = fmt_label(fs, "narrow",
                         interval=narrow_interval(fs, exact_value))
    elif op == "exact_to_broad":
        want = fmt_label(fs, "broad",
                         interval=broad_interval(fs, exact_value))
    elif op == "exact_to_rounded":
        want = fmt_label(fs, "rounded", value=round_value(fs, exact_value))
    elif op == "narrow_to_broad":
        lo, hi = narrow_interval(fs, exact_value)
        if source_label != fmt_label(fs, "narrow", interval=(lo, hi)):
            return False, "source is not the narrow bin of the exact value"
        blo, bhi = broad_interval(fs, exact_value)
        if not (blo <= lo and hi <= bhi):
            return False, "narrow bin not contained in broad bin"
        want = fmt_label(fs, "broad", interval=(blo, bhi))
    else:
        return False, f"unknown numeric operation {op}"
    if target_label != want:
        return False, f"target {target_label!r} != frozen-policy {want!r}"
    if source_label == target_label:
        return False, "source == target"
    return True, "ok"


def classify_numeric(parsed_label, expected_label, field_schema,
                     exact_value, schema):
    """Depth/boundary-aware classification of one numeric output."""
    out = {"expected": expected_label, "parsed": parsed_label}
    if parsed_label is None:
        out["classification"] = "unparseable"
        return out
    if parsed_label == DELETED_LABEL:
        out["classification"] = "refusal"
        return out
    p = parse_numeric_label(parsed_label, schema)
    e = parse_numeric_label(expected_label, schema)
    if p is None or p["field"] != e["field"]:
        out["classification"] = "wrong_branch"
        return out
    if parsed_label == expected_label:
        out["classification"] = "correct"
    else:
        pk = "exact" if p["kind"] == "exact" else p.get("width_class",
                                                        p["kind"])
        ek = "exact" if e["kind"] == "exact" else e.get("width_class",
                                                        e["kind"])
        pd = RESOLUTION_DEPTH.get(pk if pk in RESOLUTION_DEPTH else
                                  "category", 3)
        ed = RESOLUTION_DEPTH.get(ek if ek in RESOLUTION_DEPTH else
                                  "category", 3)
        if pd > ed:
            out["classification"] = "over_abstraction"
        elif pd < ed:
            out["classification"] = "under_abstraction"
        else:
            out["classification"] = "wrong_branch"
    # secondary numeric metrics
    if p["kind"] == "bin":
        out["contains_exact_value"] = p["lo"] <= exact_value <= p["hi"]
        ew = parse_numeric_label(expected_label, schema)
        if ew and ew["kind"] == "bin":
            out["abstraction_width_error"] = ((p["hi"] - p["lo"])
                                              - (ew["hi"] - ew["lo"]))
            exp_w = ew["hi"] - ew["lo"] + 1
            same = (p["hi"] - p["lo"] + 1) == exp_w
            adjacent = same and (p["lo"] == ew["hi"] + 1
                                 or p["hi"] == ew["lo"] - 1)
            out["adjacent_bin_error"] = bool(adjacent)
    elif p["kind"] == "exact":
        out["rounding_error"] = abs(p["value"] - exact_value)
        out["contains_exact_value"] = (p["value"] == exact_value)
    return out


def boundary_tags(field_schema, value, all_values):
    """Frozen boundary-coverage tags for one exact value."""
    fs = field_schema
    nlo, nhi = narrow_interval(fs, value)
    blo, bhi = broad_interval(fs, value)
    tags = set()
    tags.add("narrow_lower" if value == nlo else
             "narrow_upper" if value == nhi else "narrow_interior")
    tags.add("broad_lower" if value == blo else
             "broad_upper" if value == bhi else "broad_interior")
    if value + 1 in all_values and broad_interval(fs, value + 1)[0] != blo:
        tags.add("adjacent_across_broad_boundary")
    if value - 1 in all_values and broad_interval(fs, value - 1)[0] != blo:
        tags.add("adjacent_across_broad_boundary")
    lo, hi = fs["domain"]
    span = hi - lo
    if value >= hi - span * 0.05 or value <= lo + span * 0.05:
        tags.add("sparse_region")
    return sorted(tags)


# ======================================================================
# Vocabulary + parsing-ambiguity validation (plan section 4)
# ======================================================================
def token_span(label):
    return tuple(label.lower().split())


def check_vocab_collisions(vocab):
    """Hard collisions and longest-match-wins nesting report.

    hard collision: two distinct labels with identical token spans.
    nesting: one label's span is a contiguous sub-span of another's --
    deterministic under longest-match-wins parsing, reported for audit.
    """
    hard, nested = [], []
    spans = {}
    for lab in vocab:
        s = token_span(lab)
        if s in spans:
            hard.append((spans[s], lab))
        spans[s] = lab
    labs = sorted(vocab)
    for a, b in itertools.combinations(labs, 2):
        sa, sb = token_span(a), token_span(b)
        for short, long_ in ((sa, sb), (sb, sa)):
            if len(short) < len(long_) and any(
                    tuple(long_[i:i + len(short)]) == short
                    for i in range(len(long_) - len(short) + 1)):
                nested.append((a, b))
                break
    return {"hard_collisions": hard, "nested_longest_match_wins": nested}


# ======================================================================
# Assignment-set validation (plan section 4 checklist)
# ======================================================================
def validate_set(entry, ctx):
    """ctx: {kind: 'taxonomic'|'numeric', hierarchy_of|profiles+schema,
    baseline_alias_of, vocab, dag?}.  Returns list of issue strings."""
    issues = []
    assignments = entry["assignments"]
    ids = ctx["identity_ids"]
    targets = set(assignments)
    # explicit retain expectations for every non-target
    retain = entry.get("retain_ids")
    if sorted(retain or []) != sorted(i for i in ids if i not in targets):
        issues.append("retain_ids must be exactly the non-target identities")
    for iid, a in assignments.items():
        if iid not in ids:
            issues.append(f"unknown identity {iid}")
            continue
        src = ctx["baseline_alias_of"][iid]
        if a["source"] != src:
            issues.append(f"{iid}: source {a['source']!r} != baseline {src!r}")
        if a["source"] == a["target"]:
            issues.append(f"{iid}: source == target")
        if a["target"] not in ctx["vocab"]:
            issues.append(f"{iid}: target {a['target']!r} not in vocab")
        if a["operation"] == "taxonomic":
            if a["target"] not in ancestors_of(ctx["dag"], a["source"]):
                issues.append(f"{iid}: target is not a genuine ancestor")
            if a["target_depth"] <= a["source_depth"]:
                issues.append(f"{iid}: target depth not coarser")
        elif a["operation"] == "refusal":
            if a["target"] != DELETED_LABEL:
                issues.append(f"{iid}: refusal target must be Unknown")
        else:
            fs = ctx["schema"][a["field"]]
            ok, why = valid_numeric_transformation(
                a["operation"], fs, a["exact_value"], a["source"],
                a["target"])
            if not ok:
                issues.append(f"{iid}: numeric validity: {why}")
    # sibling specificity: retained siblings keep their own labels
    if ctx["kind"] == "taxonomic":
        # sorted(): set iteration order varies with PYTHONHASHSEED and made
        # the committed gx0_validation.json bytes non-deterministic
        for iid in sorted(targets):
            if ctx["hierarchy_of"].get(iid) is None:
                continue
            ctl = sibling_controls(iid, ctx["hierarchy_of"], targets)
            entry.setdefault("controls", {})[iid] = ctl
            if not (ctl["sibling"] or ctl["cousin"] or ctl["unrelated"]):
                issues.append(f"{iid}: NO controls available at all")
            elif not ctl["sibling"]:
                # structural dataset fact (unique level-1 branch, or the
                # sibling is co-targeted): recorded, not a failure; the
                # sibling metric is reported as null for this identity,
                # never as a vacuous 1.0
                entry.setdefault("control_notes", {})[iid] = (
                    "no retained sibling (unique level1 branch or sibling "
                    "co-targeted); cousin/unrelated controls apply; "
                    "sibling metric null, not 1.0")
    # vocab completeness + ambiguity
    needed = {a["source"] for a in assignments.values()} | \
             {a["target"] for a in assignments.values()} | \
             set(ctx["baseline_alias_of"].values()) | {DELETED_LABEL}
    missing = needed - set(ctx["vocab"])
    if missing:
        issues.append(f"vocab missing labels: {sorted(missing)}")
    return issues


def validate_vocab(vocab):
    col = check_vocab_collisions(vocab)
    issues = [f"hard vocab collision: {pair}"
              for pair in col["hard_collisions"]]
    return issues, col


# ======================================================================
# FROZEN numeric schema (plan section 2).  Bin edges, rounding policy,
# category intervals and label formats are fixed HERE, before any target
# selection or GPU execution.  Only genuine numeric quantities are used;
# phone numbers, emails, identifiers and postal codes are excluded.
# ======================================================================
NUMERIC_SCHEMA = {
    "years_experience": {
        "unit": "years",
        "ordered": True,
        "domain": [0, 39],
        "narrow": {"width": 5, "anchor": 0},
        "broad": {"width": 10, "anchor": 0},
        "category": {"entry-level": [0, 19], "experienced": [20, 39]},
        "rounding": {"to": 5, "ties": "up"},
        "label_format": {"exact": "{v} {unit}", "bin": "{a}–{b} {unit}",
                         "category": "{c}"},
        "binning_policy": "fixed_before_experiments",
    },
    "activity_count": {
        "unit": "activities",
        "ordered": True,
        "domain": [0, 999],
        "narrow": {"width": 10, "anchor": 0},
        "broad": {"width": 100, "anchor": 0},
        "category": {"low": [0, 332], "moderate": [333, 665],
                     "high": [666, 999]},
        "rounding": {"to": 10, "ties": "up"},
        "label_format": {"exact": "{v} {unit}", "bin": "{a}–{b} {unit}",
                         "category": "{c}"},
        "binning_policy": "fixed_before_experiments",
    },
}

# FROZEN synthetic profiles: 24 identities (12 per field), 18 with an
# exact-value baseline association and 6 with a narrow-bin baseline (the
# narrow->broad sources).  Values chosen BEFORE the experiment to cover:
# bin interiors, narrow lower/upper boundaries, the adjacent pair across a
# broad boundary (19|20 years, 199|200 activities), sparse and dense
# regions, domain edges, and a rounding ties-up case (125 activities).
NUMERIC_PROFILES = [
    # (profile_id, field, exact_value, baseline_kind, rationale)
    ("Y01", "years_experience", 14, "exact", "dense-region interior"),
    ("Y02", "years_experience", 15, "exact", "narrow_lower of 15-19"),
    ("Y03", "years_experience", 19, "exact",
     "narrow_upper 15-19 AND broad_upper 10-19; adjacent to 20"),
    ("Y04", "years_experience", 20, "exact",
     "broad_lower 20-29; adjacent across boundary from 19"),
    ("Y05", "years_experience", 17, "exact", "interior of 15-19 / 10-19"),
    ("Y06", "years_experience", 38, "exact", "sparse region near domain top"),
    ("Y07", "years_experience", 0, "exact", "domain lower edge"),
    ("Y08", "years_experience", 12, "exact", "rounds down (12->10)"),
    ("Y09", "years_experience", 13, "exact", "rounds up (13->15)"),
    ("Y10", "years_experience", 34, "narrow",
     "narrow->broad source: 30-34 -> 30-39"),
    ("Y11", "years_experience", 7, "narrow",
     "narrow->broad source: 5-9 -> 0-9"),
    ("Y12", "years_experience", 39, "narrow",
     "narrow->broad source at domain top: 35-39 -> 30-39"),
    ("A01", "activity_count", 126, "exact", "dense interior 120-129"),
    ("A02", "activity_count", 120, "exact", "narrow_lower of 120-129"),
    ("A03", "activity_count", 129, "exact", "narrow_upper of 120-129"),
    ("A04", "activity_count", 199, "exact",
     "broad_upper 100-199; adjacent to 200"),
    ("A05", "activity_count", 200, "exact",
     "broad_lower 200-299; adjacent across boundary from 199"),
    ("A06", "activity_count", 977, "exact", "sparse region near domain top"),
    ("A07", "activity_count", 125, "exact", "rounding ties-up case (125->130)"),
    ("A08", "activity_count", 42, "exact", "interior 40-49 / 0-99"),
    ("A09", "activity_count", 348, "exact", "mid-domain interior 340-349"),
    ("A10", "activity_count", 264, "narrow",
     "narrow->broad source: 260-269 -> 200-299"),
    ("A11", "activity_count", 995, "narrow",
     "narrow->broad at domain top: 990-999 -> 900-999"),
    ("A12", "activity_count", 3, "narrow",
     "narrow->broad at domain bottom: 0-9 -> 0-99"),
]

# FROZEN cell selections (which profile serves which single-target cell).
# Fixed before any run; boundary coverage is validated, not tuned.
NUMERIC_SINGLE_SELECTIONS = {
    "exact_to_narrow": ["Y02", "Y03", "Y06", "Y01",
                        "A02", "A03", "A06", "A01"],
    "exact_to_broad": ["Y03", "Y04", "Y07", "Y05",
                       "A04", "A05", "A08", "A09"],
    "exact_to_rounded": ["Y08", "Y09", "Y03", "A07", "A01", "A09"],
}
NUMERIC_SIM_SAME_RESOLUTION = [   # all targets -> narrow bin
    ["Y01", "A08", "Y05"],
    ["Y06", "A09", "Y02"],
    ["A01", "Y03", "A03"],
]
NUMERIC_SIM_MIXED_RESOLUTION = [  # narrow + broad + narrow->broad per set
    [{"id": "Y02", "op": "exact_to_narrow"},
     {"id": "A04", "op": "exact_to_broad"},
     {"id": "Y10", "op": "narrow_to_broad"}],
    [{"id": "A02", "op": "exact_to_narrow"},
     {"id": "Y03", "op": "exact_to_broad"},
     {"id": "A10", "op": "narrow_to_broad"}],
    [{"id": "A03", "op": "exact_to_narrow"},
     {"id": "Y06", "op": "exact_to_broad"},
     {"id": "A12", "op": "narrow_to_broad"}],
]


def baseline_label(profile, schema):
    _pid, field, value, kind, _ = profile
    fs = schema[field]
    if kind == "exact":
        return fmt_label(fs, "exact", value=value)
    return fmt_label(fs, "narrow", interval=narrow_interval(fs, value))


def target_label(op, profile, schema):
    _pid, field, value, _kind, _ = profile
    fs = schema[field]
    if op == "exact_to_narrow":
        return fmt_label(fs, "narrow", interval=narrow_interval(fs, value))
    if op == "exact_to_broad":
        return fmt_label(fs, "broad", interval=broad_interval(fs, value))
    if op == "exact_to_rounded":
        return fmt_label(fs, "rounded", value=round_value(fs, value))
    if op == "narrow_to_broad":
        return fmt_label(fs, "broad", interval=broad_interval(fs, value))
    raise ValueError(op)


def build_numeric_manifest(schema=None):
    """Frozen numeric-profile manifest: 24 synthetic identities at the
    C->Y association level (codes GRN_00..GRN_23; no image router)."""
    schema = schema or NUMERIC_SCHEMA
    identities, alias_of, code_of, profiles = [], {}, {}, {}
    for i, prof in enumerate(NUMERIC_PROFILES):
        pid, field, value, kind, rationale = prof
        iid = f"{i:02d}"
        code = f"GRN_{i:02d}"
        lab = baseline_label(prof, schema)
        identities.append(iid)
        code_of[iid] = code
        alias_of[iid] = lab
        all_values = {p[2] for p in NUMERIC_PROFILES if p[1] == field}
        profiles[iid] = {
            "profile_id": pid, "field": field, "exact_value": value,
            "baseline_kind": kind, "rationale": rationale,
            "unit": schema[field]["unit"],
            "boundary_tags": boundary_tags(schema[field], value, all_values),
            "narrow_bin": list(narrow_interval(schema[field], value)),
            "broad_bin": list(broad_interval(schema[field], value)),
            "rounded": round_value(schema[field], value),
            "category": category_of(schema[field], value),
        }
    # uniqueness of baseline labels (validator requirement)
    if len(set(alias_of.values())) != len(alias_of):
        raise ValueError("baseline labels are not unique across identities")
    return {
        "dataset": "celeba_numeric_profiles",
        "level": "association (C->Y); no image router -- CelebA g redesign "
                 "is a separate future task",
        "schema": schema,
        "identity_ids": identities,
        "code_of": code_of,
        "alias_of": alias_of,
        "profiles": profiles,
        "deleted_label": DELETED_LABEL,
        "pass_criteria": PASS_CRITERIA,
    }


def build_numeric_matrix(manifest, seeds=None):
    """Frozen 28-set / 84-cell numeric matrix (plan section 6)."""
    seeds = seeds or SEEDS_DEFAULT
    schema = manifest["schema"]
    prof_by_pid = {p[0]: p for p in NUMERIC_PROFILES}
    iid_by_pid = {manifest["profiles"][i]["profile_id"]: i
                  for i in manifest["identity_ids"]}
    sets = []

    def assignment(iid, op):
        prof = manifest["profiles"][iid]
        src = manifest["alias_of"][iid]
        tgt = target_label(op, prof_by_pid[prof["profile_id"]], schema)
        return {"operation": op, "field": prof["field"],
                "source": src, "target": tgt,
                "exact_value": prof["exact_value"], "unit": prof["unit"],
                "source_resolution": prof["baseline_kind"],
                "target_resolution": op.split("_to_")[1],
                "boundary_tags": prof["boundary_tags"]}

    for op, pids in NUMERIC_SINGLE_SELECTIONS.items():
        for pid in pids:
            iid = iid_by_pid[pid]
            sets.append({
                "set_id": f"gx_num_s_{op}_{pid}",
                "mode": f"single_{op}",
                "assignments": {iid: assignment(iid, op)},
                "retain_ids": [i for i in manifest["identity_ids"]
                               if i != iid],
                "seeds": seeds,
            })
    for j, group in enumerate(NUMERIC_SIM_SAME_RESOLUTION):
        assigns = {iid_by_pid[pid]: assignment(iid_by_pid[pid],
                                               "exact_to_narrow")
                   for pid in group}
        sets.append({
            "set_id": f"gx_num_sim_narrow_{j}",
            "mode": "simultaneous_same_resolution",
            "assignments": assigns,
            "retain_ids": [i for i in manifest["identity_ids"]
                           if i not in assigns],
            "seeds": seeds,
        })
    for j, group in enumerate(NUMERIC_SIM_MIXED_RESOLUTION):
        assigns = {iid_by_pid[g["id"]]: assignment(iid_by_pid[g["id"]],
                                                   g["op"])
                   for g in group}
        sets.append({
            "set_id": f"gx_num_mix_{j}",
            "mode": "simultaneous_mixed_resolution",
            "assignments": assigns,
            "retain_ids": [i for i in manifest["identity_ids"]
                           if i not in assigns],
            "seeds": seeds,
        })
    return {"dataset": "celeba_numeric_profiles", "kind": "numeric",
            "edit_seeds": seeds, "sets": sets,
            "n_sets": len(sets),
            "n_cells": len(sets) * len(seeds),
            "selection_note": "single-target selections and simultaneous "
                              "groupings are FROZEN constants in "
                              "e2c_v3_granularity.py, chosen for boundary "
                              "coverage before any GPU run"}


def validate_numeric_boundary_coverage(manifest):
    """Boundary-coverage gate over the frozen single selections."""
    prof_by_pid = {p[0]: p for p in NUMERIC_PROFILES}
    tags_by_field = {"years_experience": set(), "activity_count": set()}
    pair_hits = {"years_experience": 0, "activity_count": 0}
    ties = False
    for op, pids in NUMERIC_SINGLE_SELECTIONS.items():
        for pid in pids:
            prof = prof_by_pid[pid]
            iid = next(i for i in manifest["identity_ids"]
                       if manifest["profiles"][i]["profile_id"] == pid)
            tags = set(manifest["profiles"][iid]["boundary_tags"])
            tags_by_field[prof[1]] |= tags
            if "adjacent_across_broad_boundary" in tags:
                pair_hits[prof[1]] += 1
            if (op == "exact_to_rounded" and prof[1] == "activity_count"
                    and prof[2] % 10 == 5):
                ties = True
    required = {"narrow_lower", "narrow_upper", "broad_lower",
                "broad_upper", "narrow_interior", "sparse_region"}
    problems = []
    for field, tags in tags_by_field.items():
        missing = required - tags
        if missing:
            problems.append(f"{field}: missing boundary tags {sorted(missing)}")
        if pair_hits[field] < 2:
            problems.append(f"{field}: adjacent-across-boundary pair not "
                            f"both selected ({pair_hits[field]}/2)")
    if not ties:
        problems.append("no rounding ties-up case selected")
    return problems


# ======================================================================
# SALMU taxonomic matrix (frozen deterministic construction, plan §5)
# ======================================================================
def salmu_groups(hierarchy_of):
    """Alternating balanced split into L1-target group A and L2-target
    group B: order identities by (level2 group size, level2, level1, id)
    and deal alternately so both groups span the same branches."""
    from collections import Counter
    l2_counts = Counter(ch[2] for ch in hierarchy_of.values())
    order = sorted(hierarchy_of,
                   key=lambda i: (l2_counts[hierarchy_of[i][2]],
                                  hierarchy_of[i][2],
                                  hierarchy_of[i][1], i))
    A, B = order[0::2], order[1::2]
    return A, B


def build_salmu_matrix(salmu_manifest, seeds=None):
    seeds = seeds or SEEDS_DEFAULT
    hierarchy_of = {i: list(lv) for i, lv in
                    salmu_manifest["job_levels"].items()}
    dag = build_label_dag(hierarchy_of)
    alias_of = salmu_manifest["alias_of"]
    ids = salmu_manifest["identity_ids"]
    A, B = salmu_groups(hierarchy_of)
    vocab = sorted(set(alias_of.values())
                   | {ch[1] for ch in hierarchy_of.values()}
                   | {ch[2] for ch in hierarchy_of.values()}
                   | {DELETED_LABEL})
    sets = []

    def tax_assignment(iid, depth):
        chain = hierarchy_of[iid]
        return {"operation": "taxonomic", "source": chain[0],
                "target": chain[depth], "source_depth": 0,
                "target_depth": depth}

    def make(set_id, mode, assigns):
        return {"set_id": set_id, "mode": mode, "assignments": assigns,
                "retain_ids": [i for i in ids if i not in assigns],
                "seeds": seeds}

    # 12 single-target sets: group A -> level1, group B -> level2
    for iid in A:
        sets.append(make(f"gx_sal_s_L1_{iid}", "single_level1",
                         {iid: tax_assignment(iid, 1)}))
    for iid in B:
        sets.append(make(f"gx_sal_s_L2_{iid}", "single_level2",
                         {iid: tax_assignment(iid, 2)}))
    # same-depth simultaneous: 3 L1 sets, 3 L2 sets
    for j, group in enumerate([A[0:3], A[3:6], [A[0], A[3], A[4]]]):
        sets.append(make(f"gx_sal_sim_L1_{j}", "simultaneous_same_depth_l1",
                         {i: tax_assignment(i, 1) for i in group}))
    for j, group in enumerate([B[0:3], B[3:6], [B[0], B[3], B[4]]]):
        sets.append(make(f"gx_sal_sim_L2_{j}", "simultaneous_same_depth_l2",
                         {i: tax_assignment(i, 2) for i in group}))
    # mixed-depth simultaneous: L1 + L2 + refusal control
    for j in range(3):
        assigns = {A[j]: tax_assignment(A[j], 1),
                   B[j]: tax_assignment(B[j], 2),
                   A[(j + 3) % 6]: {"operation": "refusal",
                                    "source": alias_of[A[(j + 3) % 6]],
                                    "target": DELETED_LABEL,
                                    "source_depth": 0,
                                    "target_depth": None}}
        sets.append(make(f"gx_sal_mix_{j}", "simultaneous_mixed_depth",
                         assigns))
    ctx = {"kind": "taxonomic", "identity_ids": ids,
           "baseline_alias_of": alias_of, "hierarchy_of": hierarchy_of,
           "dag": dag, "vocab": vocab}
    return {"dataset": "salmu", "kind": "taxonomic", "edit_seeds": seeds,
            "vocab": vocab, "dag": dag,
            "groups": {"A_level1": A, "B_level2": B},
            "sets": sets, "n_sets": len(sets),
            "n_cells": len(sets) * len(seeds)}, ctx
