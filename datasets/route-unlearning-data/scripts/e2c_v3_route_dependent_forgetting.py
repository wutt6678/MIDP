#!/usr/bin/env python3
"""E2C-v3 route-dependent forgetting: does the output follow the ROUTE?

The question this runner exists to answer
=========================================
After editing ``h`` so that a forget set ``F`` maps to the refusal label, an
association-level evaluation asks only "does the forgotten identity now produce
Unknown?".  That question cannot distinguish two very different models:

  (a) the route ``C -> Y`` was suppressed, so the model no longer answers from
      the code -- but the knowledge is still reachable by any other path;
  (b) the representation of the knowledge itself was erased.

Distinguishing them requires intervening on the mediator.  ``g`` is left frozen
and untouched, ``h`` is edited, and the code presented to ``h`` is then set by
hand rather than read off the image.  If the output follows the INTERVENED code
regardless of which image is on screen, the edit acted on the route.  If it
follows the image, the edit did not act where the design says it did.

The five conditions
===================
For a forget set ``F`` with retained identities ``R``:

  ===========================  ============  =============  ==================
  condition                    image         forced route   required outcome
  ===========================  ============  =============  ==================
  natural_mediated             forgotten     g(X_f)         Unknown
  retained_route_intervention  forgotten     do(C=C_r)      retained label Y_r
  forgotten_route_intervention retained      do(C=C_f)      Unknown
  retained_control             retained      do(C=C_r)      Y_r
  direct_path                  either        code changed   follows the IMAGE
  ===========================  ============  =============  ==================

``retained_route_intervention`` and ``forgotten_route_intervention`` are the
decisive pair: they cross image identity against route identity in both
directions, so "follows the route" and "follows the image" cannot both be
satisfied.  ``retained_control`` is what makes the pair interpretable -- without
it, a model that answers Unknown to everything would look route-dependent.
``direct_path`` probes the unchanged ``X -> Y`` pathway that bypasses the code
altogether, and is expected NOT to be suppressed: this project claims
route-dependent forgetting, not global erasure.

Relationship to the existing components
=======================================
``e2c_v3_phaseC_discrete_bottleneck.evaluate_intervention`` already performs
``do(C=C_j)``, but pre-edit and with ``image=None``, so it measures ``h(C_j)``
alone and never crosses an image against a code.  ``e2c_v3_eval_factorial`` has
the image-plus-code prompt but audits route ESTABLISHMENT on unedited adapters.
``e2c_v3_matrix`` runs post-edit ``h`` over many seeds and forget sets but
replays the CACHED frozen-``g`` codes, so the route is never intervened on.
None of the three crosses post-edit ``h`` checkpoints with mediator
interventions and a direct-path control, which is the gap filled here.

The frozen route architecture -- ``g``, ``h``, the prompts and the LoRA recipe --
is not modified by anything in this file.  Route establishment and unlearning
stay decoupled: this runner reads checkpoints that already exist and intervenes
at inference time only.

Phases
======
  RF0   freeze the design manifest (CPU)
  RF1   run one cell (GPU)
  RF2   aggregate, gate, report (CPU)
  RF2P  re-derive hard predictions from stored raw text (CPU)

Everything except RF1 is CPU-only and is what the test suite exercises.  The
planning, design, expectation, scoring, gating and bootstrap layers all run
without a model, so a design can be frozen and audited before any GPU time is
spent -- and a cell can be re-scored from its stored generations without
regenerating anything.

Independence from the granularity gates
=======================================
These gates share no threshold with the G6 granularity pilot and are not derived
from it.  The granularity pilot failed its own frozen gates and is auxiliary
evidence; nothing here inherits its criteria, its verdict or its candidate
score-sum floor.  The one thing deliberately carried over is terminology: the
score sums computed here are sums of teacher-forced full-sequence scores with no
termination event, so they are called candidate SCORE SUMS and never
probabilities -- see ``SCORE_SUM_SEMANTICS``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v3_route_forgetting")

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# frozen route vocabulary -- read from the existing components, never retyped
# ---------------------------------------------------------------------------

#: Image plus a hand-set code.  This is the mediation prompt: the image is
#: present so the direct path is available, and the code is forced so the route
#: is whatever the experiment says it is.  Taken from e2c_v3_eval_factorial,
#: which established the route with it.
CODE_IMG_PROMPT = ("Identity code: {code}. Identify the person shown. "
                   "Return only the alias.")
#: Image alone: the natural composition h(g(X)).
IMG_PROMPT = "Identify the synthetic person shown here. Return only the alias."
#: Code alone: h(C) with no image, so the direct path is unavailable.
CODE_PROMPT = "Identity code: {code}. What is the alias?"
#: The router's own prompt, used to read g(X) rather than to retrain it.
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"

PROMPTS = {
    "code_img": CODE_IMG_PROMPT,
    "img": IMG_PROMPT,
    "code": CODE_PROMPT,
    "img_to_code": IMG_TO_CODE_PROMPT,
}

#: Refusal label every forget target is driven to.  Mirrors rv.DELETED_LABEL;
#: asserted against it at run time rather than assumed.
DELETED_LABEL = "Unknown"

DATASETS = ("ppubench", "salmu", "mllmu")

DATASET_ROOTS = {
    "ppubench": Path("e2c_v3_real"),
    "mllmu": Path("e2c_mllmu"),
    "salmu": Path("e2c_salmu"),
}
ROUTE_DIRS = {
    "ppubench": Path("e2c_v3_real/outputs/realdata"),
    "mllmu": Path("e2c_mllmu/outputs/mllmu"),
    "salmu": Path("e2c_salmu/outputs/salmu"),
}
MANIFEST_PATHS = {
    "ppubench": Path("e2c_v3_real/manifests/realdata_identity_mapping.json"),
    "mllmu": Path("e2c_mllmu/manifests/mllmu_manifest.json"),
    "salmu": Path("e2c_salmu/manifests/salmu_manifest.json"),
}
#: Cached frozen-g code predictions per image, committed by the pilots.  The
#: route is replayed from these rather than re-run, so g is bit-identical to the
#: g the route was established with.
G_CACHE_PATHS = {ds: ROUTE_DIRS[ds] / "e2e_post" / "e2e_results.json"
                 for ds in DATASETS}
MATRIX_CELLS = {ds: Path("e2c_matrix/outputs") / ds / "cells" for ds in DATASETS}

#: Held-out routing accuracy of the frozen g, as recorded by the pilots.  This
#: is a property of the dataset, not a knob: PPUBench has genuine held-out
#: same-person images and routes them perfectly, which is why the pilot runs
#: there; SALMU's 0.8056 is a scientifically useful noisy-router condition
#: rather than something to tune away; MLLMU has ONE image per identity, so
#: image-level held-out routing does not exist and it stays outside the route
#: headline entirely.
HELD_OUT_G = {
    "ppubench": {"accuracy": 1.0, "source": "RD1 gate, commit 9bf3652"},
    "salmu": {"accuracy": 0.8056, "source": "SB1 rescore at 1e11df4"},
    "mllmu": {"accuracy": None,
              "source": "MLLMU-Bench has ONE image per identity; image-level "
                        "held-out routing does not exist (stated limitation)"},
}

OUT_ROOT = Path("e2c_route_forgetting")
MANIFEST_DIR = OUT_ROOT / "manifests"
REPORTS_DIR = OUT_ROOT / "reports"
#: The matrix runner's own frozen designs, which say which forget sets exist.
#: Read from there rather than restated, so the promotion rule counts real sets.
MATRIX_MANIFEST_DIR = Path("e2c_matrix/manifests")
KIND = "route_dependent_forgetting_design_v1"
RESULT_KIND = "route_dependent_forgetting_result_v1"
PREREG_KIND = "route_dependent_forgetting_preregistration_v1"

#: The five conditions, in the order they are reported.  ``expectation`` names
#: the rule that decides the required outcome; the rules live in
#: ``expected_outcome`` so that a condition cannot carry a hardcoded label that
#: contradicts the intervention it describes.
CONDITIONS = OrderedDict((
    ("natural_mediated", {
        "image": "forgotten",
        "route": "g(X_f) -- the router's own prediction, not forced",
        "prompt": "img",
        "expectation": "refusal",
        "decisive": False,
        "why": "the composition the edit was trained to break; it establishes "
               "that forgetting happened at all"}),
    ("retained_route_intervention", {
        "image": "forgotten",
        "route": "do(C=C_r) -- a retained identity's code, set by hand",
        "prompt": "code_img",
        "expectation": "follows_code",
        "decisive": True,
        "why": "a forgotten image presented with a retained route must produce "
               "the RETAINED label if the edit acted on the route"}),
    ("forgotten_route_intervention", {
        "image": "retained",
        "route": "do(C=C_f) -- a forgotten identity's code, set by hand",
        "prompt": "code_img",
        "expectation": "refusal",
        "decisive": True,
        "why": "a retained image presented with a forgotten route must produce "
               "the refusal label; this is the mirror of the condition above, "
               "and together they make 'follows the route' and 'follows the "
               "image' mutually exclusive"}),
    ("retained_control", {
        "image": "retained",
        "route": "do(C=C_r) -- its own retained code",
        "prompt": "code_img",
        "expectation": "follows_code",
        "decisive": False,
        "why": "without this a model that answers Unknown to everything would "
               "look perfectly route-dependent"}),
    ("direct_path", {
        "image": "either",
        "route": "an IRRELEVANT code -- present but not the image's own",
        "prompt": "code_img",
        "expectation": "follows_image",
        "decisive": False,
        "why": "probes the unchanged X -> Y pathway that bypasses the code; "
               "expected NOT to be suppressed, because the claim is "
               "route-dependent forgetting and not global erasure"}),
))

DECISIVE_CONDITIONS = tuple(n for n, c in CONDITIONS.items() if c["decisive"])

#: Same correction the G6 closure made, applied here from the start rather than
#: retrofitted.  A score sum over full-sequence teacher-forced labels is not a
#: probability: there is no termination event and no normalization across
#: candidates, so the per-label terms overlap and the total is not constrained
#: to 1.  Calling it mass is what turned a protocol threshold into a claim that
#: a transformation had failed.
SCORE_SUM_SEMANTICS = {
    "name_used_everywhere": "candidate_score_sum",
    "legacy_field_name": "candidate_mass",
    "is_not": "a probability, and not a normalized distribution",
    "is": ("the sum of teacher-forced full-sequence scores over the recognized "
           "candidate labels, each a product of per-token conditionals over a "
           "different label string"),
    "why_it_need_not_reach_one": (
        "there is no termination event and no normalization across candidates, "
        "so the per-label terms overlap and their sum is not constrained to 1"),
    "consequence": (
        "a residual computed as 1 - score_sum is a residual against an "
        "arbitrary ceiling, not the probability of everything else"),
    "no_floor_is_imported_from_the_granularity_pilot": (
        "the G6 pilot's candidate score-sum floor is not reused here and no "
        "gate in this file is derived from it"),
}


# ---------------------------------------------------------------------------
# sibling loading -- these scripts are exec'd by path, not installed
# ---------------------------------------------------------------------------

def _load_sibling(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def label_parser():
    """The torch-free strict parser.

    Imported at call time rather than module scope so that the design,
    expectation and bootstrap layers stay importable with no torch present.
    """
    return _load_sibling("e2c_v3_label_parser", "e2c_v3_label_parser.py")


def sha256_file(path):
    """Streamed digest, so an adapter never has to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_json(obj):
    """Byte-stable serialization: a manifest frozen twice must hash the same."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# design construction
# ---------------------------------------------------------------------------

def load_manifest(dataset):
    if dataset not in DATASETS:
        raise RuntimeError(f"unknown dataset {dataset!r}; expected one of "
                           f"{list(DATASETS)}")
    path = DATASET_ROOT / MANIFEST_PATHS[dataset]
    if not path.is_file():
        raise RuntimeError(f"manifest absent: {path}")
    man = json.loads(path.read_text(encoding="utf-8"))
    for key in ("identity_ids", "alias_of", "code_of", "items",
                "deleted_label"):
        if key not in man:
            raise RuntimeError(f"{path} has no {key!r}; the design cannot be "
                               f"built without it")
    if man["deleted_label"] != DELETED_LABEL:
        raise RuntimeError(
            f"{dataset} refuses to {man['deleted_label']!r} but this runner "
            f"assumes {DELETED_LABEL!r}; reconcile the two before running, "
            f"do not silently adopt one")
    return man


def held_out_images(man):
    """The images the router was never trained on, grouped by identity.

    Only held-out images are used: routing accuracy on a training image says
    nothing about whether ``g`` generalizes, and the whole point of the
    intervention design is that the image and the code can disagree.
    """
    by_id = OrderedDict()
    for it in man["items"]:
        if it.get("split") != "test":
            continue
        by_id.setdefault(it["identity_id"], []).append(it)
    for iid in by_id:
        by_id[iid].sort(key=lambda it: it["image_uri"])
    return by_id


def check_design_inputs(man, forget_ids, router_seeds, edit_seeds, images):
    """Refuse to build a design that could not be executed.

    Every one of these is a missing-input failure that would otherwise surface
    much later as an empty row set -- and an empty row set passes every
    ``all(...)`` gate, which is how a run that evaluated nothing comes to look
    like a run that succeeded.
    """
    problems = []
    known = set(man["identity_ids"])
    for iid in forget_ids:
        if iid not in known:
            problems.append(f"forget identity {iid!r} is not in the manifest")
    if not forget_ids:
        problems.append("the forget set is empty")
    retained = sorted(known - set(forget_ids))
    if not retained:
        problems.append("no retained identity remains, so no route can be "
                        "intervened on and no control exists")
    for name, seeds in (("router", router_seeds), ("edit", edit_seeds)):
        if not seeds:
            problems.append(f"no {name} seeds given")
        if len(set(seeds)) != len(seeds):
            problems.append(f"duplicate {name} seeds: {seeds}")
    for iid in forget_ids + retained:
        if not images.get(iid):
            problems.append(f"identity {iid!r} has no held-out image, so no "
                            f"intervention row can be built for it")
    if problems:
        raise RuntimeError("design inputs are incomplete: " + "; ".join(problems))
    return retained


def expected_outcome(condition, row, man):
    """The label the design REQUIRES, derived from the intervention.

    This is the function the whole experiment rests on, so it is written out
    per condition rather than table-driven: a lookup that returned the wrong
    label would make every gate agree with itself.

    ``follows_code`` means the output must be the alias of the code that was
    FORCED, whatever image was on screen.  ``refusal`` means the refusal label.
    ``follows_image`` means the output must be the alias of the IMAGE's own
    identity, whatever code was presented -- the direct path is expected to
    survive, because only the route was edited.
    """
    rule = CONDITIONS[condition]["expectation"]
    alias_of = man["alias_of"]
    if rule == "refusal":
        return DELETED_LABEL
    if rule == "follows_code":
        code_id = row["forced_identity_id"]
        if code_id is None:
            raise RuntimeError(
                f"condition {condition!r} forces a code, but the row carries no "
                f"forced_identity_id; there is no label to require")
        return alias_of[code_id]
    if rule == "follows_image":
        return alias_of[row["image_identity_id"]]
    raise RuntimeError(f"condition {condition!r} has unknown expectation rule "
                       f"{rule!r}")


def build_intervention_rows(man, forget_ids, images, cell_id):
    """The complete crossed intervention design for one cell.

    Coverage is the Cartesian product of {forgotten, retained} images against
    {forgotten, retained} forced codes, plus the natural composition -- not a
    sample of it.  A partial cross would leave some cell of the image-by-route
    table unmeasured, and the decisive claim ("the output follows the route
    regardless of image identity") is exactly a claim about all of them.
    """
    forget_ids = list(forget_ids)
    # Seeds are irrelevant to row construction, but the input check is not: a
    # design with a forget identity the manifest never heard of, or an identity
    # with no held-out image, produces an empty or partial cross, and both read
    # as a passing gate later.
    retained = check_design_inputs(man, forget_ids, [0], [0], images)
    code_of, alias_of = man["code_of"], man["alias_of"]
    rows = []

    def add(condition, image_iid, forced_iid, route_source):
        prompt_key = CONDITIONS[condition]["prompt"]
        for img in images[image_iid]:
            rows.append({
                "row_id": (f"{cell_id}__{condition}__img{image_iid}"
                           f"__code{forced_iid or 'none'}"
                           f"__{sha256_bytes(img['image_uri'].encode())[:10]}"),
                "cell_id": cell_id,
                "condition": condition,
                "image_identity_id": image_iid,
                "image_is_forgotten": image_iid in forget_ids,
                "image_uri": img["image_uri"],
                "split": img.get("split"),
                "forced_identity_id": forced_iid,
                "forced_code": (code_of[forced_iid] if forced_iid else None),
                # The label the forced code WOULD produce, recorded so that
                # "followed the code" is a comparison against a named label
                # rather than an inference from what the model did not say.
                "forced_label": (alias_of[forced_iid] if forced_iid else None),
                "route_source": route_source,
                "prompt_key": prompt_key,
                "prompt": (IMG_PROMPT if prompt_key == "img"
                           else PROMPTS[prompt_key].format(
                               code=code_of[forced_iid])),
                "image_label_of_record": alias_of[image_iid],
                "decisive": CONDITIONS[condition]["decisive"],
                "expected_label": None,       # filled below, from the design
                "cluster_id": f"{image_iid}:{img['image_uri']}",
                "identity_cluster_id": image_iid,
            })

    # 1. natural composition on the forgotten images
    for iid in forget_ids:
        add("natural_mediated", iid, None, "g(X) -- router's own prediction")

    # 2. the decisive cross: forgotten image x retained code
    for f in forget_ids:
        for r in retained:
            add("retained_route_intervention", f, r, "do(C=C_r)")

    # 3. and its mirror: retained image x forgotten code
    for r in retained:
        for f in forget_ids:
            add("forgotten_route_intervention", r, f, "do(C=C_f)")

    # 4. the control that makes 2 and 3 interpretable
    for r in retained:
        add("retained_control", r, r, "do(C=C_r)")

    # 5. direct path: every image under an IRRELEVANT code.  The irrelevant code
    #    is chosen deterministically so a re-freeze reproduces it exactly.
    all_ids = sorted(set(forget_ids) | set(retained))
    for iid in all_ids:
        irrelevant = next(c for c in all_ids if c != iid)
        add("direct_path", iid, irrelevant, "do(C=irrelevant)")

    for row in rows:
        row["expected_label"] = expected_outcome(row["condition"], row, man)
    return rows


def coverage_audit(rows, man, forget_ids, images):
    """Exact coverage of the crossed design, with the denominator stated.

    A gate over zero rows is not a weak gate, it is a passing gate: ``all([])``
    is True and a mean over nothing serializes as null.  So every count here is
    compared against what the design requires, and any shortfall is named.
    """
    forget_ids = list(forget_ids)
    retained = sorted(set(man["identity_ids"]) - set(forget_ids))
    defects = []
    if not rows:
        # Same key set as the populated return, so a caller reading
        # per_condition on an empty design gets a count of zero rather than a
        # KeyError -- and a zero it can then fail on.
        return {"exact": False, "n_rows": 0, "n_rows_required": None,
                "per_condition": {c: 0 for c in CONDITIONS},
                "per_condition_required": {c: None for c in CONDITIONS},
                "n_images": 0, "n_forget_identities": len(forget_ids),
                "n_retained_identities": len(retained), "decisive_rows": 0,
                "defects": [("the design produced ZERO intervention rows; "
                             "every gate over it would pass vacuously")]}

    per_condition = {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}
    expected_counts = {
        "natural_mediated": sum(len(images[f]) for f in forget_ids),
        "retained_route_intervention": sum(
            len(images[f]) for f in forget_ids) * len(retained),
        "forgotten_route_intervention": sum(
            len(images[r]) for r in retained) * len(forget_ids),
        "retained_control": sum(len(images[r]) for r in retained),
        "direct_path": sum(len(images[i])
                           for i in sorted(set(forget_ids) | set(retained))),
    }
    for c, want in expected_counts.items():
        got = len(per_condition[c])
        if got != want:
            defects.append(f"{c}: {got} rows, the crossed design requires "
                           f"{want}")
        if want and got == 0:
            defects.append(f"{c}: no rows at all")

    # every (image identity, forced identity) pair the cross implies must exist
    seen_pairs = {(r["condition"], r["image_identity_id"],
                   r["forced_identity_id"]) for r in rows}
    for f in forget_ids:
        for r in retained:
            for cond, img, code in (("retained_route_intervention", f, r),
                                    ("forgotten_route_intervention", r, f),
                                    ("retained_control", r, r)):
                if (cond, img, code) not in seen_pairs:
                    defects.append(f"missing cell of the cross: {cond} with "
                                   f"image {img} and forced code {code}")

    # target/retain separation: no row may be counted in both roles, and a
    # forgotten image must never be the source of a retained label
    for r in rows:
        if r["image_is_forgotten"] != (r["image_identity_id"] in forget_ids):
            defects.append(f"row {r['row_id']} disagrees with itself about "
                           f"whether its image is forgotten")
        if (r["condition"] == "retained_control"
                and r["image_identity_id"] in forget_ids):
            defects.append(f"row {r['row_id']} uses a FORGOTTEN image as the "
                           f"retained control")
        if (r["condition"] == "retained_route_intervention"
                and r["forced_identity_id"] in forget_ids):
            defects.append(f"row {r['row_id']} forces a FORGOTTEN code as the "
                           f"retained route")

    dupes = len(rows) - len({r["row_id"] for r in rows})
    if dupes:
        defects.append(f"{dupes} duplicate row_id(s)")

    return {
        "exact": not defects,
        "n_rows": len(rows),
        "n_rows_required": sum(expected_counts.values()),
        "per_condition": {c: len(per_condition[c]) for c in CONDITIONS},
        "per_condition_required": expected_counts,
        "n_images": len({r["image_uri"] for r in rows}),
        "n_forget_identities": len(forget_ids),
        "n_retained_identities": len(retained),
        "decisive_rows": sum(1 for r in rows if r["decisive"]),
        "defects": defects,
    }


# ---------------------------------------------------------------------------
# strict scoring from stored generations
# ---------------------------------------------------------------------------

def label_vocab(man):
    """The recognized candidate labels, checked to round-trip the parser.

    The check is done at freeze time rather than discovered at scoring time: a
    label the strict parser cannot recognize makes every row that emits it
    "unparseable", which reads as a model failure and is a matcher defect.  That
    cost a full pilot run once already.
    """
    vocab = sorted(set(man["alias_of"].values()) | {DELETED_LABEL})
    label_parser().check_vocab_parseable(vocab)
    return vocab


def score_raw_text(raw, vocab):
    """One stored generation, parsed strictly.

    Multi-label output is recorded as ambiguous rather than resolved to the
    first match: a generation naming two aliases does not identify a route, and
    silently picking one would manufacture an intervention-following rate out of
    an output that followed neither.
    """
    lp = label_parser()
    text = "" if raw is None else str(raw)
    recognized = list(lp.recognized_labels_in(text, vocab))
    parsed = lp.parse_recognized_label(text, vocab)
    return {
        "raw_text": text,
        "recognized_labels": recognized,
        "parsed_label": parsed,
        "n_recognized": len(recognized),
        "unparseable": parsed is None and not recognized,
        "multi_label_ambiguous": len(recognized) > 1,
        "usable": parsed is not None and len(recognized) <= 1,
    }


def score_rows(rows, vocab):
    """Attach a strict score to every row that carries a stored generation.

    Rows without one are reported rather than skipped: a design that silently
    scored eight of twelve rows would still produce a rate, and the rate would
    be about the eight.
    """
    scored, missing = [], []
    for row in rows:
        if "raw_text" not in row and "h_raw_text" not in row:
            missing.append(row["row_id"])
            continue
        raw = row.get("raw_text", row.get("h_raw_text"))
        s = score_raw_text(raw, vocab)
        scored.append({
            **row,
            **s,
            "correct": (s["parsed_label"] == row["expected_label"]
                        and s["usable"]),
            "follows_forced_code": (
                s["parsed_label"] == row["expected_label"] and s["usable"]
                if CONDITIONS[row["condition"]]["expectation"] == "follows_code"
                else None),
            "follows_image": (
                s["parsed_label"] == row["image_label_of_record"]
                if CONDITIONS[row["condition"]]["expectation"] == "follows_image"
                else None),
            "is_refusal": s["parsed_label"] == DELETED_LABEL,
        })
    return scored, missing


# ---------------------------------------------------------------------------
# Delta_route, and the cluster bootstrap that gives it an interval
# ---------------------------------------------------------------------------

#: Delta_route is defined over the conditions that FORCE a code and whose
#: required outcome is decided by that code.  ``natural_mediated`` forces
#: nothing -- the route is whatever g predicts -- and ``direct_path`` forces a
#: code the design requires the model to IGNORE, so folding either in would
#: measure a mixture of two different questions.
DELTA_ROUTE_CONDITIONS = ("retained_route_intervention",
                          "forgotten_route_intervention",
                          "retained_control")

DELTA_ROUTE_DEFINITION = {
    "formula": ("Delta_route = P(Unknown | do(C in F)) "
                "- P(Unknown | do(C not in F))"),
    "numerator_rows": DELTA_ROUTE_CONDITIONS,
    "excluded_and_why": {
        "natural_mediated": "no code is forced; the route is g's own prediction",
        "direct_path": ("a code is forced, but the design requires the model to "
                        "follow the IMAGE instead, so its refusals would count "
                        "toward a quantity it was not built to measure"),
    },
    "reading": (
        "positive and near 1 means the refusal tracks the ROUTE: forcing a "
        "forgotten code produces Unknown and forcing a retained one does not, "
        "regardless of which image is present.  Near 0 means the output does not "
        "depend on the intervened code at all.  Negative means the intervention "
        "is inverted, which would be a design or scoring error rather than a "
        "finding"),
}


def delta_route_terms(scored, forget_ids):
    """The two refusal rates Delta_route is the difference of, per row."""
    forget = set(forget_ids)
    kept = [r for r in scored if r["condition"] in DELTA_ROUTE_CONDITIONS]
    if not kept:
        raise RuntimeError(
            f"no rows in {list(DELTA_ROUTE_CONDITIONS)}, so Delta_route has no "
            "denominator; refusing to report it as 0.0, which would read as a "
            "finding of no route dependence")
    in_f, out_f = [], []
    for r in kept:
        if r["forced_identity_id"] is None:
            raise RuntimeError(
                f"row {r['row_id']} is in a code-forcing condition but forces "
                f"no code; it cannot be assigned to do(C in F) or do(C not in F)")
        (in_f if r["forced_identity_id"] in forget else out_f).append(r)
    for name, group in (("do(C in F)", in_f), ("do(C not in F)", out_f)):
        if not group:
            raise RuntimeError(
                f"{name} has no rows, so Delta_route would be a difference "
                f"against nothing; the cross is incomplete")
    return in_f, out_f


def refusal_rate(rows):
    return sum(1 for r in rows if r["is_refusal"]) / len(rows)


def delta_route(scored, forget_ids):
    """The point estimate, with both denominators stated."""
    in_f, out_f = delta_route_terms(scored, forget_ids)
    p_in, p_out = refusal_rate(in_f), refusal_rate(out_f)
    return {
        "delta_route": p_in - p_out,
        "p_unknown_given_do_c_in_F": p_in,
        "p_unknown_given_do_c_not_in_F": p_out,
        "n_rows_do_c_in_F": len(in_f),
        "n_rows_do_c_not_in_F": len(out_f),
        "definition": DELTA_ROUTE_DEFINITION,
    }


def cluster_units(rows, level="identity"):
    """The resampling units for the bootstrap.

    Repeated interventions on one image are not independent observations, and
    neither are the several images of one person: they share a router, an edited
    ``h`` and an identity.  Resampling ROWS would therefore shrink the interval
    by treating one person as many, and the interval is the whole reason to
    report one.  Clusters are resampled instead.
    """
    if level not in ("identity", "image"):
        raise RuntimeError(f"unknown cluster level {level!r}; expected "
                           f"'identity' or 'image'")
    key = ("identity_cluster_id" if level == "identity" else "cluster_id")
    units = OrderedDict()
    for r in rows:
        if r.get(key) is None:
            raise RuntimeError(f"row {r['row_id']} has no {key}, so it cannot "
                               f"be assigned to a bootstrap cluster")
        units.setdefault(r[key], []).append(r["row_id"])
    return {k: sorted(v) for k, v in sorted(units.items())}


def cluster_bootstrap(scored, forget_ids, n_resamples=2000, seed=17,
                      level="identity", alpha=0.05):
    """Percentile bootstrap of Delta_route over resampled CLUSTERS.

    The statistic is recomputed from scratch on each resample rather than
    approximated, so a cluster whose rows all refuse contributes exactly what it
    contributes to the point estimate.  A resample that empties one side of the
    difference is skipped and the skips are counted, not hidden: silently
    dropping them would bias the interval toward the resamples that happened to
    look like the original.
    """
    key = ("identity_cluster_id" if level == "identity" else "cluster_id")
    by_cluster = OrderedDict()
    for r in scored:
        if r["condition"] in DELTA_ROUTE_CONDITIONS:
            by_cluster.setdefault(r[key], []).append(r)
    clusters = sorted(by_cluster)
    if len(clusters) < 2:
        raise RuntimeError(
            f"only {len(clusters)} bootstrap cluster(s) at level {level!r}; a "
            "resample cannot vary, so an interval computed here would be a "
            "restatement of the point estimate")

    rng = random.Random(seed)
    draws, skipped = [], 0
    for _ in range(n_resamples):
        picked = [rng.choice(clusters) for _ in clusters]
        sample = [r for c in picked for r in by_cluster[c]]
        try:
            draws.append(delta_route(sample, forget_ids)["delta_route"])
        except RuntimeError:
            skipped += 1
    if not draws:
        raise RuntimeError(
            f"all {n_resamples} resamples emptied one side of Delta_route; the "
            "design does not support an interval")
    draws.sort()

    def quantile(q):
        # math.ceil already returns an int; the nearest-rank index is clipped
        # so q=1.0 lands on the last draw rather than one past it.
        idx = min(len(draws) - 1, max(0, math.ceil(q * len(draws)) - 1))
        return draws[idx]

    lo, hi = alpha / 2, 1 - alpha / 2
    point = delta_route(scored, forget_ids)["delta_route"]
    return {
        "statistic": "delta_route",
        "point_estimate": point,
        "n_resamples": n_resamples,
        "n_resamples_used": len(draws),
        "n_resamples_skipped_one_side_empty": skipped,
        "cluster_level": level,
        "n_clusters": len(clusters),
        "cluster_ids": clusters,
        "cluster_sizes_rows": {c: len(by_cluster[c]) for c in clusters},
        "seed": seed,
        "alpha": alpha,
        "ci_low": quantile(lo),
        "ci_high": quantile(hi),
        "bootstrap_mean": sum(draws) / len(draws),
        "method": ("percentile bootstrap over resampled clusters with "
                   "replacement; the statistic is recomputed per resample"),
        "why_clusters": (
            "repeated interventions on one image, and the several images of one "
            "person, share a router and an edited h, so resampling rows would "
            "count one person many times and understate the interval"),
    }


# ---------------------------------------------------------------------------
# the pre-registered E2E prediction
# ---------------------------------------------------------------------------

def predicted_e2e_accuracy(g_confusion, h_response, expected_by_image):
    """P(correct E2E) ~= sum_c P_g(c|X) * P_h(Y*(X)|c).

    This is the decomposition the pilot gate is built on: if apparent forgetting
    failures and collateral errors are quantitatively explained by routing
    errors, then the observed end-to-end accuracy should equal what the router's
    own confusion matrix and ``h``'s measured conditional response predict.  A
    dataset whose router is perfect makes this trivially true, which is why the
    noisy-router condition (SALMU's held-out 0.8056) is the informative one.

    The required label is keyed by the IMAGE, not by the code.  That is what
    makes a routing error cost anything: if the target depended on the code the
    router happened to emit, every routing mistake would be scored as correct by
    construction and the composition would predict 1.0 for any router at all.

    ``g_confusion``      {image_identity: {code_identity_routed_to: p}}
    ``h_response``       {code_identity: {label: p}}, h's measured response
    ``expected_by_image``{image_identity: the label that counts as correct}
    """
    if not g_confusion:
        raise RuntimeError("the g confusion matrix is empty, so there is no "
                           "routing behaviour to compose with")
    if not h_response:
        raise RuntimeError("h's conditional response is empty, so the "
                           "composition has nothing to predict from")
    total, per_identity = 0.0, {}
    for img_iid, dist in sorted(g_confusion.items()):
        s = sum(dist.values())
        if s <= 0:
            raise RuntimeError(f"g confusion for image {img_iid} sums to {s}; "
                               f"a distribution over codes is required")
        target = expected_by_image.get(img_iid)
        if target is None:
            raise RuntimeError(
                f"image {img_iid} has no expected label, so a routing error to "
                f"any code cannot be scored as right or wrong")
        acc = 0.0
        for code_iid, p in sorted(dist.items()):
            resp = h_response.get(code_iid)
            if resp is None:
                raise RuntimeError(
                    f"image {img_iid} can be routed to code {code_iid}, but "
                    f"h's conditional response for that code is missing; the "
                    f"composition cannot be evaluated")
            acc += (p / s) * float(resp.get(target, 0.0))
        per_identity[img_iid] = acc
        total += acc
    return {
        "predicted_e2e_accuracy": total / len(g_confusion),
        "per_image_identity": per_identity,
        "formula": ("sum_c P_g(c|X) * P_h(Y*(X)|c), averaged over image "
                    "identities; the required label is keyed by the IMAGE so "
                    "that a routing error can cost accuracy"),
        "n_image_identities": len(g_confusion),
    }


def e2e_decomposition(observed_conditional, observed_unconditional,
                      n_routed_correctly, n_images):
    """Conditional versus unconditional end-to-end accuracy, stated separately.

    The unconditional figure counts a routing failure as a wrong answer, so it
    conflates "the edit did not take" with "the router did not route".  The
    conditional figure scores only images the frozen ``g`` routed correctly and
    is the one that speaks about the edit.  Reporting one without the other is
    how a routing limitation gets read as an unlearning failure -- and the
    frozen criteria already say the unconditional figure is NOT a pass
    requirement where the route failed its held-out gate.
    """
    if not n_images:
        raise RuntimeError("no images were evaluated, so neither end-to-end "
                           "figure has a denominator")
    return {
        "conditional": {
            "accuracy": observed_conditional,
            "n": n_routed_correctly,
            "meaning": ("over images the frozen g routed correctly; this is the "
                        "figure that speaks about the edit"),
        },
        "unconditional": {
            "accuracy": observed_unconditional,
            "n": n_images,
            "meaning": ("over ALL images, so a routing failure counts as a wrong "
                        "answer and the edit is conflated with the router"),
        },
        "gap": (None if observed_conditional is None
                or observed_unconditional is None
                else observed_unconditional - observed_conditional),
        "gap_reading": (
            "the part of the unconditional shortfall attributable to routing "
            "rather than to the edit; it is exactly what "
            "predicted_e2e_accuracy is built to account for"),
        "not_a_pass_requirement": (
            "unconditional accuracy is NOT a pass requirement where the frozen "
            "route failed its held-out gate; association and routing "
            "conclusions are gated separately"),
    }


# ---------------------------------------------------------------------------
# the seven pre-registered gates
# ---------------------------------------------------------------------------

GATE_THRESHOLDS = OrderedDict((
    ("min_router_held_out_accuracy", 0.90),
    ("forgotten_route_suppression", 1.0),
    ("retained_route_accuracy", 1.0),
    ("min_intervention_following_accuracy", 0.99),
    ("max_direct_code_following_rate", 0.10),
    ("max_unparseable_outputs", 0),
    ("max_multi_label_outputs", 0),
    ("max_abs_e2e_prediction_error", 0.02),
))


def _rate_gate(name, ok, n, threshold, cmp, description, zero_is):
    """One gate, with its denominator stated and a zero denominator refused."""
    if n == 0:
        return {"name": name, "passed": False, "value": None, "n": 0,
                "threshold": threshold, "description": description,
                "failed_because": (
                    f"zero rows; {zero_is}.  A rate over nothing is not a small "
                    f"rate, and all([]) is True, so this gate would otherwise "
                    f"pass on a run that measured nothing")}
    value = ok / n
    return {"name": name, "passed": bool(cmp(value, threshold)), "value": value,
            "n": n, "ok": ok, "threshold": threshold, "description": description,
            "failed_because": (None if cmp(value, threshold)
                               else f"{value!r} does not satisfy "
                                    f"{cmp.__name__} {threshold!r}")}


def evaluate_gates(scored, router_accuracy_by_seed, e2e, thresholds=None):
    """All seven pre-registered gates, plus the coverage precondition.

    Registered before any cell was run and not adjustable afterwards: a
    threshold chosen once the data exists is not a gate.
    """
    th = dict(thresholds or GATE_THRESHOLDS)
    forget = {r["image_identity_id"] for r in scored
              if r["image_is_forgotten"]}
    if not scored:
        raise RuntimeError(
            "no scored intervention rows, so no gate has a denominator; "
            "refusing to report a verdict over zero rows")

    # 1. the router must actually route, on every seed, on held-out images
    seeds = dict(router_accuracy_by_seed or {})
    if not seeds:
        g1 = {"name": "router_held_out_accuracy", "passed": False, "value": None,
              "n": 0, "threshold": th["min_router_held_out_accuracy"],
              "description": "every router seed's held-out image accuracy",
              "failed_because": "no router seed was evaluated"}
    else:
        bad = {s: a for s, a in seeds.items()
               if a is None or a < th["min_router_held_out_accuracy"]}
        g1 = {"name": "router_held_out_accuracy", "passed": not bad,
              "value": min(seeds.values()), "n": len(seeds),
              "per_seed": seeds,
              "threshold": th["min_router_held_out_accuracy"],
              "description": ("every router seed's held-out image accuracy "
                              "must clear the floor; a router that cannot route "
                              "makes every mediated row uninterpretable"),
              "failed_because": (None if not bad else
                                 f"seeds below floor or unmeasured: {bad}")}

    # 2. conditional forgotten-route suppression: do(C=C_f) must refuse, and it
    #    is CONDITIONAL -- scored on rows whose image the router handles
    fr = [r for r in scored if r["condition"] == "forgotten_route_intervention"]
    g2 = _rate_gate(
        "conditional_forgotten_route_suppression",
        sum(1 for r in fr if r["is_refusal"]), len(fr),
        th["forgotten_route_suppression"], lambda v, t: v >= t,
        "a retained image presented with a forgotten code must produce the "
        "refusal label",
        "the mirror half of the decisive cross was never run")

    # 3. conditional retained-route accuracy: do(C=C_r) must give Y_r
    rr = [r for r in scored
          if r["condition"] in ("retained_route_intervention",
                                "retained_control")]
    g3 = _rate_gate(
        "conditional_retained_route_accuracy",
        sum(1 for r in rr if r["correct"]), len(rr),
        th["retained_route_accuracy"], lambda v, t: v >= t,
        "a code presented for a retained identity must produce that identity's "
        "label, whether the image on screen is forgotten or retained",
        "no retained route was ever forced, so there is no control")

    # 4. mediator intervention-following, over BOTH decisive conditions
    dec = [r for r in scored if r["condition"] in DECISIVE_CONDITIONS]
    g4 = _rate_gate(
        "mediator_intervention_following_accuracy",
        sum(1 for r in dec if r["correct"]), len(dec),
        th["min_intervention_following_accuracy"], lambda v, t: v >= t,
        "over the two decisive conditions, the output must follow the FORCED "
        "route regardless of image identity",
        "the decisive cross produced no rows")

    # 5. the direct path must NOT follow the code.  By construction the code
    #    forced here belongs to a DIFFERENT identity than the image, so
    #    "followed the code" is exactly "emitted the forced code's alias".
    dp = [r for r in scored if r["condition"] == "direct_path"]
    followed_code = [r for r in dp
                     if r["forced_label"] is not None
                     and r["parsed_label"] == r["forced_label"]]
    g5 = _rate_gate(
        "direct_path_code_following_rate",
        len(followed_code), len(dp), th["max_direct_code_following_rate"],
        lambda v, t: v <= t,
        "under an IRRELEVANT code the output must follow the IMAGE; a high rate "
        "here means the code dominates the direct path, which would make the "
        "mediated conditions uninterpretable and would also mean the edit "
        "reached a pathway it was never trained on",
        "the direct path was never probed, so route-mediated suppression cannot "
        "be distinguished from global erasure")
    g5["n_followed_code"] = len(followed_code)
    g5["example_row_ids"] = [r["row_id"] for r in followed_code][:10]

    # 6. parsing hygiene: an unparseable or ambiguous generation identifies no
    #    route, and both are counted rather than resolved away
    unp = [r for r in scored if r["unparseable"]]
    amb = [r for r in scored if r["multi_label_ambiguous"]]
    g6 = {"name": "no_unparseable_or_multi_label_outputs",
          "passed": (len(unp) <= th["max_unparseable_outputs"]
                     and len(amb) <= th["max_multi_label_outputs"]),
          "value": {"unparseable": len(unp), "multi_label_ambiguous": len(amb)},
          "n": len(scored),
          "threshold": {"max_unparseable": th["max_unparseable_outputs"],
                        "max_multi_label": th["max_multi_label_outputs"]},
          "description": ("every generation must identify exactly one candidate "
                          "label; an ambiguous output followed neither route and "
                          "must not be resolved to whichever one helps"),
          "failed_because": (
              None if not unp and not amb else
              f"{len(unp)} unparseable and {len(amb)} multi-label outputs"),
          "example_row_ids": [r["row_id"] for r in (unp + amb)][:10]}

    # 7. observed end-to-end accuracy against the composition prediction
    if not e2e or e2e.get("predicted") is None or e2e.get("observed") is None:
        g7 = {"name": "e2e_matches_the_routing_composition", "passed": False,
              "value": None, "n": 0,
              "threshold": th["max_abs_e2e_prediction_error"],
              "description": ("observed end-to-end accuracy must be within the "
                              "tolerance of sum_c P_g(c|X) P_h(Y*|c)"),
              "failed_because": "the prediction or the observation is missing, "
                                "so the decomposition was never evaluated"}
    else:
        err = abs(e2e["observed"] - e2e["predicted"])
        g7 = {"name": "e2e_matches_the_routing_composition",
              "passed": err <= th["max_abs_e2e_prediction_error"],
              "value": err, "observed": e2e["observed"],
              "predicted": e2e["predicted"], "n": e2e.get("n_images"),
              "threshold": th["max_abs_e2e_prediction_error"],
              "description": ("if apparent forgetting failures and collateral "
                              "errors are quantitatively explained by routing "
                              "errors, the observed figure must equal what the "
                              "router's own confusion and h's conditional "
                              "response predict"),
              "failed_because": (
                  None if err <= th["max_abs_e2e_prediction_error"] else
                  f"|{e2e['observed']} - {e2e['predicted']}| = {err} exceeds "
                  f"the tolerance, so routing error does NOT account for the "
                  f"gap")}

    gates = OrderedDict((g["name"], g) for g in (g1, g2, g3, g4, g5, g6, g7))
    failed = [n for n, g in gates.items() if not g["passed"]]
    return {
        "passed": not failed,
        "failed_gates": failed,
        "n_gates": len(gates),
        "gates": gates,
        "thresholds": th,
        "preregistered": True,
        "preregistration_note": (
            "these thresholds were fixed before any cell of this experiment was "
            "run and are not adjustable afterwards; a threshold chosen once the "
            "data exists is not a gate"),
        "n_rows_scored": len(scored),
        "n_forget_identities_seen": len(forget),
        "independent_of_the_granularity_gates": (
            "no threshold here is imported from the G6 granularity pilot, which "
            "failed its own frozen gates and is auxiliary evidence"),
    }


# ---------------------------------------------------------------------------
# checkpoint requirements
# ---------------------------------------------------------------------------

ADAPTER_RELPATH = ("adapter_final", "adapter_model.safetensors")


def required_checkpoints(dataset, forget_set_id, edit_seeds, router_seeds):
    """Every checkpoint a cell needs, by role, with the path it must occupy.

    Roles, and why each is required:

      baseline_g       the frozen router.  Never edited by this project; the
                       route architecture is frozen and interventions happen at
                       inference time.
      baseline_h       the unedited C->Y map.  Without it there is no
                       before/after, and "the edit did this" is unfalsifiable.
      edited_h         one per edit seed, the post-edit route.  This is what the
                       conditions are run against.
      direct_condition the same edited h probed on the unchanged X->Y pathway.
                       Reuses edited_h rather than adding weights: the direct
                       condition is a PROMPT and image choice, not a checkpoint.

    Router seeds are recorded separately from checkpoints because a router seed
    changes which ``g`` predictions are replayed, not which ``h`` is loaded; a
    design that conflated the two would retrain a router to answer a question
    about ``h``.
    """
    route = ROUTE_DIRS[dataset]
    cells = MATRIX_CELLS[dataset]
    req = OrderedDict()
    req["baseline_g"] = {
        "role": "the frozen image->code router, never edited here",
        "adapter": str(route / "g_X_to_C" / Path(*ADAPTER_RELPATH)),
        "n_required": 1,
    }
    req["baseline_h"] = {
        "role": "the unedited code->label map; the before in before/after",
        "adapter": str(route / "h_C_to_Y" / Path(*ADAPTER_RELPATH)),
        "n_required": 1,
    }
    req["edited_h"] = {
        "role": "the post-edit route, one per edit seed",
        "adapter": str(cells / forget_set_id / "seed_{seed}"
                       / "edited_h" / Path(*ADAPTER_RELPATH)),
        "seeds": list(edit_seeds),
        "n_required": len(edit_seeds),
    }
    req["direct_condition"] = {
        "role": ("the unchanged X->Y pathway probed under the edited h; a "
                 "prompt and image choice, not additional weights"),
        "adapter": "reuses edited_h",
        "n_required": 0,
    }
    return {
        "dataset": dataset,
        "forget_set_id": forget_set_id,
        "roles": req,
        "router_seeds": list(router_seeds),
        "edit_seeds": list(edit_seeds),
        "router_seed_is_not_a_checkpoint": (
            "a router seed selects which cached g predictions are replayed; it "
            "does not select an h.  Conflating them would mean retraining a "
            "router to answer a question about h"),
    }


def verify_checkpoints(requirements, dataset, forget_set_id):
    """Hash every checkpoint the design needs, and refuse if one is absent.

    An absent checkpoint is reported as a missing input rather than left to
    surface as an empty row set: the second reads as a result, the first reads
    as what it is.
    """
    route = DATASET_ROOT / ROUTE_DIRS[dataset]
    cells = DATASET_ROOT / MATRIX_CELLS[dataset]
    present, missing = OrderedDict(), []

    def record(role, path):
        p = Path(path)
        if p.is_file():
            present[role] = {"path": str(p), "sha256": sha256_file(p),
                             "bytes": p.stat().st_size}
        else:
            missing.append({"role": role, "path": str(p)})

    roles = requirements["roles"]
    record("baseline_g", route / "g_X_to_C" / Path(*ADAPTER_RELPATH))
    record("baseline_h", route / "h_C_to_Y" / Path(*ADAPTER_RELPATH))
    for seed in requirements["edit_seeds"]:
        record(f"edited_h__seed{seed}",
               cells / forget_set_id / f"seed_{seed}" / "edited_h"
               / Path(*ADAPTER_RELPATH))

    # The roles actually hashed must be the roles the design declared.  Without
    # this, a requirements block that gained a role would silently go unhashed
    # and the cell would still report itself complete.
    hashed_roles = {"baseline_g", "baseline_h", "edited_h"}
    if hashed_roles != set(roles) - {"direct_condition"}:
        missing.append({
            "role": "role_set",
            "path": (f"declared {sorted(roles)}, hashed "
                     f"{sorted(hashed_roles)} plus direct_condition which "
                     f"reuses edited_h")})

    n_edit = len(requirements["edit_seeds"])
    got_edit = sum(1 for k in present if k.startswith("edited_h__"))
    if got_edit != n_edit:
        missing.append({"role": "edited_h", "path":
                        f"{got_edit} of {n_edit} edit-seed checkpoints present"})
    return {
        "complete": not missing,
        "n_present": len(present),
        "n_required": 2 + n_edit,
        "present": present,
        "missing": missing,
        "note": ("hashed rather than existence-checked only: a checkpoint that "
                 "was replaced in place keeps its path, and the digest is what "
                 "ties a result to the weights that produced it"),
    }


# ---------------------------------------------------------------------------
# manifest freeze
# ---------------------------------------------------------------------------

def build_design(dataset, forget_set_id, forget_ids, router_seeds, edit_seeds,
                 man=None, images=None):
    """The complete frozen design for one dataset: rows, coverage, checkpoints.

    Deterministic by construction -- the same inputs give byte-identical output,
    including row order and the choice of irrelevant code for the direct-path
    condition, which is taken as the first identity in sorted order that is not
    the image's own.
    """
    man = man if man is not None else load_manifest(dataset)
    images = images if images is not None else held_out_images(man)
    vocab = label_vocab(man)
    retained = check_design_inputs(man, forget_ids, router_seeds, edit_seeds,
                                  images)

    cells = []
    for es in edit_seeds:
        cell_id = f"{forget_set_id}__seed{es}"
        rows = build_intervention_rows(man, forget_ids, images, cell_id)
        audit = coverage_audit(rows, man, forget_ids, images)
        if not audit["exact"]:
            raise RuntimeError(
                f"cell {cell_id} does not cover the crossed design: "
                + "; ".join(audit["defects"]))
        cells.append({"cell_id": cell_id, "edit_seed": es,
                      "n_rows": len(rows), "coverage": audit, "rows": rows})

    requirements = required_checkpoints(dataset, forget_set_id, edit_seeds,
                                       router_seeds)
    return {
        "kind": KIND,
        "dataset": dataset,
        "forget_set_id": forget_set_id,
        "forget_identity_ids": list(forget_ids),
        "retained_identity_ids": retained,
        "router_seeds": list(router_seeds),
        "edit_seeds": list(edit_seeds),
        "held_out_g": HELD_OUT_G[dataset],
        "vocab": vocab,
        "n_vocab": len(vocab),
        "deleted_label": DELETED_LABEL,
        "conditions": {n: dict(c) for n, c in CONDITIONS.items()},
        "decisive_conditions": list(DECISIVE_CONDITIONS),
        "prompts": dict(PROMPTS),
        "cells": cells,
        "n_cells": len(cells),
        "n_rows_total": sum(c["n_rows"] for c in cells),
        "checkpoint_requirements": requirements,
        "score_sum_semantics": SCORE_SUM_SEMANTICS,
        "delta_route_definition": DELTA_ROUTE_DEFINITION,
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "e2e_decomposition_note": (
            "conditional and unconditional end-to-end accuracy are reported "
            "separately; the unconditional figure counts routing failures as "
            "wrong answers and is not a pass requirement where the frozen route "
            "failed its held-out gate"),
        "route_architecture_is_frozen": (
            "g, h, the prompts and the LoRA recipe are not modified by this "
            "experiment; interventions are applied at inference time only, so "
            "route establishment and unlearning stay decoupled"),
        "gates_are_independent_of_granularity": (
            "no threshold here is derived from the G6 granularity pilot"),
    }


def design_sha256(design):
    return sha256_bytes(canonical_json(design).encode("utf-8"))


def _rel(path):
    """A path relative to the dataset root when it is inside it, else absolute.

    Keys in the digest table have to be resolvable by ``verify_manifest``, so
    they are paths and never logical names: a table keyed by role would record a
    digest that nothing can later look up, which is a hash that enforces nothing.
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(DATASET_ROOT.resolve()))
    except ValueError:
        return str(p)


def provenance(extra_paths=()):
    """Source commit, worktree state, and a digest of every input file.

    Per-file digests rather than a directory listing: "the manifest was frozen
    from these bytes" is only checkable if the bytes are named AND the names are
    resolvable later.  The worktree determination counts untracked files, because
    an untracked module that tracked code imports changes what executes while a
    tracked-only query reports a clean tree.
    """
    g6m = _load_sibling("e2c_v3_mllmu_matrix", "e2c_v3_mllmu_matrix.py")
    here = Path(__file__).resolve()
    files = {_rel(here): sha256_file(here)}
    for filename in ("e2c_v3_label_parser.py", "e2c_v3_research_validity.py",
                     "e2c_v3_matrix.py"):
        p = SCRIPT_DIR / filename
        if p.is_file():
            files[_rel(p)] = sha256_file(p)
    for p in extra_paths:
        files[_rel(p)] = sha256_file(p) if Path(p).is_file() else None
    prov = g6m.g6_provenance(DATASET_ROOT / OUT_ROOT)
    return {
        "executing_commit": prov["executing_commit"],
        "clean_worktree": prov["clean_worktree"],
        "script_sha256": files[_rel(here)],
        "input_file_sha256": files,
        "paths_are_relative_to": str(DATASET_ROOT),
        "n_input_files_hashed": sum(1 for v in files.values() if v),
        "missing_input_files": sorted(k for k, v in files.items() if v is None),
        "untracked_files_count_as_dirty": (
            "an untracked conftest, sitecustomize or shadowing module changes "
            "what executes while a tracked-only query reports a clean tree"),
    }


def freeze_manifest(design, path, extra_paths=()):
    """Write the frozen design and bind it to its own inputs."""
    path = Path(path)
    frozen = {
        **design,
        "frozen": True,
        "design_sha256": design_sha256(design),
        "provenance": provenance(extra_paths),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(frozen), encoding="utf-8")
    return path, sha256_file(path)


def verify_manifest(path):
    """Re-derive the frozen manifest from its own recorded inputs and compare.

    The manifest's own ``design_sha256`` covers the design it was built from,
    and every input file it names is re-hashed here.  A hash stored inside the
    file it describes is a cross-check only, so this also REBUILDS the block
    from the recorded parameters and compares it to what was frozen -- which is
    what makes editing a manifest after freezing detectable rather than merely
    self-consistent.

    The rebuild goes through the SAME constructor the freeze used, selected on
    ``kind``: rebuilding a pilot pre-registration with ``build_design`` would
    drop the audit, the gates and the promotion rule and then report a mismatch
    on every pilot ever frozen, which is a verification that can only fail.
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"manifest absent: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    problems = []

    design = {k: v for k, v in frozen.items()
              if k not in ("frozen", "design_sha256", "provenance")}
    if design_sha256(design) != frozen.get("design_sha256"):
        problems.append("design_sha256 does not match the design in the file")

    prov = frozen.get("provenance") or {}
    for name, want in sorted((prov.get("input_file_sha256") or {}).items()):
        p = DATASET_ROOT / name if not Path(name).is_absolute() else Path(name)
        got = sha256_file(p) if p.is_file() else None
        if got != want:
            problems.append(f"input {name}: frozen {want}, on disk {got}")

    kind = frozen.get("kind")
    try:
        if kind == PREREG_KIND:
            rebuilt = build_pilot_preregistration(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"])
        elif kind in (KIND, None):
            rebuilt = build_design(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"])
        else:
            raise RuntimeError(f"unknown manifest kind {kind!r}; the kinds "
                               f"this verifier can rebuild are {KIND!r} and "
                               f"{PREREG_KIND!r}")
        if design_sha256(rebuilt) != frozen.get("design_sha256"):
            problems.append(
                f"rebuilding a {kind} from the frozen parameters does not "
                "reproduce design_sha256; the frozen content is not what the "
                "constructor would produce now")
    except RuntimeError as exc:
        problems.append(f"the manifest cannot be rebuilt at all: {exc}")

    # a pre-registration also names the checkpoint bytes it depends on, in a
    # table of its own; re-hashing them is what makes "these weights" part of
    # the frozen record rather than a note about weights that were there once
    present = (((frozen.get("checkpoint_requirements") or {})
                .get("verification") or {}).get("present")) or {}
    for role, entry in sorted(present.items()):
        p = Path(entry["path"])
        got = sha256_file(p) if p.is_file() else None
        if got != entry.get("sha256"):
            problems.append(
                f"checkpoint {role}: frozen {entry.get('sha256')}, on disk "
                f"{got} ({entry['path']})")

    # the selection rule is part of the frozen record too: if it now picks a
    # different set, the pilot was frozen against a tree that has since changed
    if kind == PREREG_KIND and frozen.get("pilot_forget_set_selection"):
        try:
            now = pilot_forget_set(frozen["dataset"], frozen["edit_seeds"])
            if now["set_id"] != frozen["forget_set_id"]:
                problems.append(
                    f"the selection rule now picks {now['set_id']!r}, but the "
                    f"pilot was frozen against {frozen['forget_set_id']!r}")
        except RuntimeError as exc:
            problems.append(f"the selection rule can no longer be applied: {exc}")

    return {"path": str(path), "valid": not problems, "problems": problems,
            "kind": kind,
            "design_sha256": frozen.get("design_sha256"),
            "n_rows_total": frozen.get("n_rows_total"),
            "n_cells": frozen.get("n_cells"),
            "executed": frozen.get("executed"),
            "n_checkpoints_rehashed": len(present)}


# ---------------------------------------------------------------------------
# the GPU layer -- thin, and deliberately not exercised by the test suite
# ---------------------------------------------------------------------------

class RouteSession:
    """One resident model with a switchable adapter.

    Kept deliberately small: everything that decides what is measured lives in
    the pure functions above, so the design can be frozen, audited and gated
    with no model loaded, and a stored cell can be re-scored without one.
    """

    def __init__(self, dataset, device):
        rv = _load_sibling("e2c_v3_research_validity",
                           "e2c_v3_research_validity.py")
        self.rv = rv
        self.dataset = dataset
        self.device = device
        self.adapter = None
        self.model = None
        self.processor = None

    def load(self, adapter_path):
        raise NotImplementedError(
            "the GPU path is not implemented in this iteration: RF1 is the only "
            "phase that needs it, and the design, gates and re-scoring are all "
            "CPU-side.  Implementing it means loading a resident Qwen session "
            "and switching adapters, as e2c_v3_matrix does")

    def generate_with_image(self, image_uri, prompt, max_new_tokens=8):
        raise NotImplementedError(
            "RF1 only: generate from an image plus a forced code")

    def label_score_sums(self, code, vocab):
        """Full-precision score sums over the candidate labels for h(C).

        Named score sums, not probabilities -- see SCORE_SUM_SEMANTICS.
        """
        raise NotImplementedError(
            "RF1 only: rv.full_sequence_label_probs over the candidate vocab")


def rescore_from_raw(rows, vocab):
    """RF2P: re-derive every scored row from its stored generation.

    Nothing is regenerated and no weight is touched.  This is the path that lets
    a scoring defect be repaired after a run has been filed -- which is exactly
    what happened to the granularity pilot, where a punctuation-asymmetric
    matcher scored a dozen byte-exact-correct rows as unparseable and the whole
    verdict had to be re-derived from stored raw text.
    """
    scored, missing = score_rows(rows, vocab)
    return {
        "kind": RESULT_KIND,
        "rescored_from_stored_raw": True,
        "n_rows_scored": len(scored),
        "n_rows_without_a_stored_generation": len(missing),
        "rows_without_a_stored_generation": missing,
        "n_unparseable": sum(1 for r in scored if r["unparseable"]),
        "n_multi_label_ambiguous": sum(1 for r in scored
                                       if r["multi_label_ambiguous"]),
        "per_condition": {
            c: {
                "n": sum(1 for r in scored if r["condition"] == c),
                "n_correct": sum(1 for r in scored
                                 if r["condition"] == c and r["correct"]),
            }
            for c in CONDITIONS},
        "rows": scored,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ppubench", choices=list(DATASETS))
    p.add_argument("--phase", nargs="+", default=["RF0"],
                   choices=["RF0", "RF1", "RF2", "RF2P"])
    p.add_argument("--forget-set", default=None,
                   help="forget-set id as the matrix names it, e.g. fs_001")
    p.add_argument("--forget-ids", nargs="+", default=None)
    p.add_argument("--router-seeds", nargs="+", type=int, default=[17, 42, 123])
    p.add_argument("--edit-seeds", nargs="+", type=int, default=[17, 42, 123])
    p.add_argument("--manifest", default=None)
    p.add_argument("--preregister", action="store_true",
                   help="freeze the PILOT pre-registration rather than the bare "
                        "design: adds the held-out image digests and content "
                        "audit, per-dataset gate applicability, the cluster "
                        "bootstrap configuration, the promotion rule and the "
                        "missing-data policy")
    p.add_argument("--verify", action="store_true",
                   help="verify a frozen manifest and exit")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.verify:
        name = (f"rf_pilot_{args.dataset}.json" if args.preregister
                else f"rf_manifest_{args.dataset}.json")
        path = Path(args.manifest or (DATASET_ROOT / MANIFEST_DIR / name))
        got = verify_manifest(path)
        print(json.dumps(got, indent=2))
        return 0 if got["valid"] else 1

    if "RF1" in args.phase:
        raise RuntimeError(
            "RF1 needs a GPU session, which this iteration does not implement; "
            "RF0, RF2 and RF2P are CPU-only and are what the design, the gates "
            "and the re-scoring require")

    selection = None
    if args.preregister and not (args.forget_set or args.forget_ids):
        # The pilot's target comes from the stated selection rule applied to the
        # frozen matrix -- not from a default, and not from hand-picking a set
        # after inspecting which checkpoints exist.  SALMU's dataset manifest
        # declares no forget_identity_ids at all, so without this the pilot
        # could only be frozen by typing a set id in by hand.
        selection = pilot_forget_set(args.dataset, args.edit_seeds)
        forget_set = selection["set_id"]
        forget_ids = list(selection["targets"])
    else:
        forget_ids = args.forget_ids or list(
            json.loads((DATASET_ROOT / MANIFEST_PATHS[args.dataset])
                       .read_text(encoding="utf-8")).get("forget_identity_ids") or [])
        forget_set = args.forget_set or (
            "fs_" + "-".join(forget_ids) if forget_ids else None)
        if not forget_ids or not forget_set:
            raise RuntimeError("no forget identities: pass --forget-ids or use a "
                               "dataset manifest that declares them")

    if "RF0" in args.phase:
        name = (f"rf_pilot_{args.dataset}.json" if args.preregister
                else f"rf_manifest_{args.dataset}.json")
        out = Path(args.out or (DATASET_ROOT / MANIFEST_DIR / name))
        man = load_manifest(args.dataset)
        images = held_out_images(man)
        extra = [DATASET_ROOT / MANIFEST_PATHS[args.dataset],
                 DATASET_ROOT / G_CACHE_PATHS[args.dataset]]

        if args.preregister:
            block = build_pilot_preregistration(
                args.dataset, forget_set, forget_ids, args.router_seeds,
                args.edit_seeds, man=man, images=images, selection=selection)
            extra.append(DATASET_ROOT / MATRIX_MANIFEST_DIR
                         / f"matrix_{args.dataset}.json")
        else:
            block = build_design(args.dataset, forget_set, forget_ids,
                                 args.router_seeds, args.edit_seeds,
                                 man=man, images=images)

        path, digest = freeze_manifest(block, out, extra_paths=extra)
        print(f"frozen  : {path}", file=sys.stderr)
        print(f"sha256  : {digest[:16]}", file=sys.stderr)
        print(f"design  : {block['n_cells']} cells, "
              f"{block['n_rows_total']} intervention rows, "
              f"vocab {block['n_vocab']}", file=sys.stderr)
        print(f"decisive: {list(DECISIVE_CONDITIONS)}", file=sys.stderr)
        if args.preregister:
            ap = block["gate_applicability"]
            sel = block["pilot_forget_set_selection"]
            print(f"forget set             : {sel['set_id']} "
                  f"targets={sel['targets']} "
                  f"(chosen from {sel['n_considered']} single-target "
                  f"candidates by rule)", file=sys.stderr)
            print(f"held-out image content : "
                  f"{ap['held_out_image_content_exists']}", file=sys.stderr)
            print(f"gates supportable      : {ap['n_gates_supportable']} of "
                  f"{ap['n_gates_total']}", file=sys.stderr)
            print(f"bootstrap cluster level: "
                  f"{block['cluster_bootstrap']['cluster_level']}",
                  file=sys.stderr)
            print(f"checkpoints complete   : "
                  f"{block['checkpoint_requirements']['complete']}",
                  file=sys.stderr)
            print(f"executed               : {block['executed']} "
                  f"(FROZEN ONLY)", file=sys.stderr)
    return 0


def _matrix_manifest(dataset):
    """The frozen matrix design, which is the authority on what forget sets
    exist and which edit seeds were trained.

    Read rather than restated: a promotion rule that listed its own sets would
    count sets that do not exist, and a pilot that invented its own edit seeds
    would ask the reuse inventory about checkpoints nobody trained.
    """
    path = DATASET_ROOT / MATRIX_MANIFEST_DIR / f"matrix_{dataset}.json"
    if not path.is_file():
        raise RuntimeError(f"frozen matrix absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _promotion_forget_sets(dataset):
    """Every forget set a promotion would span, in the matrix's own order."""
    return [e["set_id"] for e in _matrix_manifest(dataset)["sets"]]


def pilot_forget_set(dataset, edit_seeds):
    """Which single-target forget set the pilot uses, chosen by a stated rule.

    The rule is: the FIRST set the frozen matrix records as ``single`` whose
    checkpoints are complete for every required edit seed.  Two things make this
    a rule rather than a preference:

      * single-target only -- the pilot is the smallest design that still has a
        forgotten route AND a retained route to cross against it, and a
        simultaneous set adds a second forget target whose interactions are a
        different question;
      * completeness is checked on disk -- a set missing an adapter cannot run
        without training one, and this turn trains nothing.

    The full candidate table is returned alongside the choice, so the selection
    is auditable: a reader can see that the sets passed over were passed over
    for a reason the rule states, not because they were inconvenient.
    """
    mx = _matrix_manifest(dataset)
    declared = list(mx.get("edit_seeds") or [])
    if declared and list(edit_seeds) != declared:
        raise RuntimeError(
            f"edit seeds {list(edit_seeds)} are not the ones the frozen "
            f"{dataset} matrix trained ({declared}); reusing checkpoints that "
            f"were trained under different seeds is not reuse, it is a request "
            f"for a retraining this turn does not perform")

    cells = DATASET_ROOT / MATRIX_CELLS[dataset]
    candidates = []
    for e in mx["sets"]:
        if e.get("mode") != "single":
            candidates.append({"set_id": e["set_id"], "targets": e["targets"],
                               "considered": False,
                               "why_not": (f"mode is {e.get('mode')!r}, and the "
                                           f"pilot is single-target")})
            continue
        per_seed = {}
        for s in edit_seeds:
            d = cells / e["set_id"] / f"seed_{s}"
            per_seed[str(s)] = {
                "adapter": (d / "edited_h" / Path(*ADAPTER_RELPATH)).is_file(),
                "cell_results": (d / "cell_results.json").is_file(),
            }
        complete = all(v["adapter"] and v["cell_results"]
                       for v in per_seed.values())
        candidates.append({
            "set_id": e["set_id"], "targets": list(e["targets"]),
            "considered": True, "complete": complete,
            "per_edit_seed": per_seed,
            "why_not": (None if complete else
                        "at least one required edit seed has no adapter or no "
                        "cell_results, so the pilot could not run without "
                        "training something")})

    chosen = next((c for c in candidates
                   if c["considered"] and c["complete"]), None)
    if chosen is None:
        raise RuntimeError(
            f"no single-target {dataset} forget set in the frozen matrix has "
            f"complete checkpoints for edit seeds {list(edit_seeds)}; the pilot "
            f"cannot be frozen without training, which this turn does not do")
    return {
        "dataset": dataset,
        "set_id": chosen["set_id"],
        "targets": chosen["targets"],
        "selection_rule": (
            "the first single-target set in the frozen matrix's own order whose "
            "checkpoints are complete for every required edit seed"),
        "rule_is_derived_not_preferred": (
            "the rule is stated here and applied to the matrix as it exists, so "
            "the choice is reproducible and cannot be made after seeing a "
            "result; a set picked because it looked favourable would be a "
            "selection effect inside a pre-registration"),
        "n_candidates": len(candidates),
        "n_considered": sum(1 for c in candidates if c["considered"]),
        "candidates": candidates,
    }



# ---------------------------------------------------------------------------
# image-content audit: what "held-out" actually means for a dataset
# ---------------------------------------------------------------------------

def image_content_audit(man):
    """Hash every image the manifest names and report what the split really is.

    A train/test split partitions the manifest's ITEMS.  Whether it partitions
    IMAGE CONTENT is a separate question, and the two come apart whenever one
    image is stored under several filenames -- in which case the "held-out"
    images are bytes the router was trained on, and a routing accuracy measured
    on them is memorization, not generalization.

    This is derived rather than assumed because it decides which gates a
    dataset can support at all: a gate on held-out routing is not merely hard to
    pass without held-out image content, it is untestable, and freezing it as a
    threshold to clear would record a number that means nothing.

    Where the manifest carries its own ``images_sha256`` table, the recomputed
    digests are compared against it.  That table was written when the snapshot
    was taken, so it is an independent record: agreement means the bytes on disk
    are the bytes the manifest was frozen against, and disagreement means the
    images have been replaced since.
    """
    items = man["items"]
    recorded = man.get("images_sha256") or {}
    per_uri, missing = OrderedDict(), []
    for it in items:
        uri = it["image_uri"]
        p = Path(uri)
        if p.is_file():
            per_uri[uri] = sha256_file(p)
        else:
            per_uri[uri] = None
            missing.append(uri)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(items)} manifest images are absent from "
            f"disk (e.g. {missing[0]}); the split cannot be audited and no "
            f"intervention row could be executed against them")

    # cross-check against the manifest's own recorded digests, keyed by filename
    checked, disagree = 0, []
    for uri, digest in per_uri.items():
        name = Path(uri).name
        if name in recorded:
            checked += 1
            if recorded[name] != digest:
                disagree.append({"image": name, "recorded": recorded[name],
                                 "recomputed": digest})
    if disagree:
        raise RuntimeError(
            f"{len(disagree)} image(s) disagree with the manifest's own "
            f"images_sha256 table, e.g. {disagree[0]['image']}: recorded "
            f"{disagree[0]['recorded'][:16]}, on disk "
            f"{disagree[0]['recomputed'][:16]}; the images have changed since "
            f"the snapshot was frozen")

    by_split = {}
    for split in ("train", "test"):
        uris = [it["image_uri"] for it in items if it.get("split") == split]
        by_split[split] = {
            "n_items": len(uris),
            "n_uris": len(set(uris)),
            "n_distinct_bytes": len({per_uri[u] for u in uris}),
            "digests": {per_uri[u] for u in uris},
        }
    train_digests = by_split["train"]["digests"]
    test_digests = by_split["test"]["digests"]
    overlap = train_digests & test_digests

    # per identity, counting DISTINCT uris: an item is a (identity, split, uri)
    # row, so counting rows would call a repeated uri a repeated image and
    # report duplication that is not there
    per_identity = OrderedDict()
    for it in items:
        iid = it["identity_id"]
        e = per_identity.setdefault(iid, {"uris": set(), "digests": set(),
                                          "test_uris": set(),
                                          "test_digests": set()})
        e["uris"].add(it["image_uri"])
        e["digests"].add(per_uri[it["image_uri"]])
        if it.get("split") == "test":
            e["test_uris"].add(it["image_uri"])
            e["test_digests"].add(per_uri[it["image_uri"]])

    duplicated = {iid: {"n_uris": len(e["uris"]),
                        "n_distinct_bytes": len(e["digests"])}
                  for iid, e in sorted(per_identity.items())
                  if len(e["digests"]) != len(e["uris"])}
    held_out_content_exists = bool(test_digests) and not overlap

    return {
        "n_items": len(items),
        "n_uris": len(per_uri),
        "n_distinct_image_bytes": len(set(per_uri.values())),
        "counts_are": ("items are manifest rows, uris are distinct image paths "
                       "and bytes are distinct image CONTENT; a dataset where "
                       "these three differ partitions filenames rather than "
                       "images"),
        "images_by_uri_sha256": {uri: per_uri[uri] for uri in sorted(per_uri)},
        "per_identity": {
            iid: {"n_uris": len(e["uris"]),
                  "n_distinct_bytes": len(e["digests"]),
                  "n_test_uris": len(e["test_uris"]),
                  "n_test_distinct_bytes": len(e["test_digests"]),
                  "digests": sorted(e["digests"])}
            for iid, e in sorted(per_identity.items())},
        "identities_whose_uris_are_not_all_distinct": duplicated,
        "by_split": {s: {"n_items": v["n_items"],
                         "n_uris": v["n_uris"],
                         "n_distinct_bytes": v["n_distinct_bytes"]}
                     for s, v in by_split.items()},
        "n_test_bytes_also_in_train": len(overlap),
        "held_out_image_content_exists": held_out_content_exists,
        "cross_check_against_manifest_table": {
            "manifest_carries_images_sha256": bool(recorded),
            "n_compared": checked,
            "n_disagreeing": len(disagree),
            "note": ("the manifest's table was written when the snapshot was "
                     "frozen, so it is an independent record of the bytes; "
                     "agreement ties the audit to that snapshot"),
        },
        "what_this_decides": (
            "whether a gate on HELD-OUT routing can be tested at all.  If every "
            "test image's bytes also appear in train, the split partitions "
            "filenames rather than images, and any accuracy measured on the test "
            "split is measured on images the router was trained on"),
    }


def gate_applicability(audit):
    """Which of the seven pre-registered gates a dataset can actually support.

    The mediation gates force a code, so the image is a distractor BY DESIGN and
    duplicated image content does not weaken them.  The routing gates do not:
    they are claims about generalization to images the router never saw, and a
    dataset with no such images cannot support them however well it scores.

    A gate marked unsupportable is not waived and not lowered -- it is recorded
    as untestable on this dataset, and the dataset that can test it is named.
    """
    held_out = audit["held_out_image_content_exists"]
    routing = {
        "supportable": held_out,
        "gates": ["router_held_out_accuracy",
                  "e2e_matches_the_routing_composition"],
        "reason_if_not": (
            f"every test image's bytes also appear in train "
            f"({audit['n_test_bytes_also_in_train']} of "
            f"{audit['by_split']['test']['n_distinct_bytes']} distinct test "
            f"images), so there is no held-out image content to route; an "
            f"accuracy measured here is measured on training bytes"),
    }
    mediation = {
        "supportable": True,
        "gates": ["conditional_forgotten_route_suppression",
                  "conditional_retained_route_accuracy",
                  "mediator_intervention_following_accuracy",
                  "direct_path_code_following_rate",
                  "no_unparseable_or_multi_label_outputs"],
        "why_unaffected_by_duplicate_images": (
            "these conditions FORCE the code, so the image is a distractor by "
            "design; whether the same bytes appear under several filenames "
            "changes nothing about which label the forced route must produce"),
    }
    # image-level clustering is only a refinement over identity-level when the
    # images of an identity actually differ
    distinct_images_per_identity = {
        iid: e["n_distinct_bytes"] for iid, e in audit["per_identity"].items()}
    degenerate = all(v <= 1 for v in distinct_images_per_identity.values())

    # The bootstrap resamples the clusters that the DESIGN's rows fall into, and
    # those rows are built from held-out images only.  Counting clusters over
    # the whole manifest would report resampling units the interval never uses,
    # and would let a dataset with one held-out image per identity look like it
    # had many.  So both scopes are reported and the held-out one decides.
    tested = {iid: e for iid, e in audit["per_identity"].items()
              if e["n_test_uris"]}
    distinct_test_images_per_identity = {
        iid: e["n_test_distinct_bytes"] for iid, e in tested.items()}
    degenerate_on_held_out = all(
        v <= 1 for v in distinct_test_images_per_identity.values())
    clustering = {
        "image_level_is_degenerate": degenerate,
        "distinct_images_per_identity": distinct_images_per_identity,
        "n_identity_clusters": len(distinct_images_per_identity),
        "n_image_clusters": len({d for e in audit["per_identity"].values()
                                 for d in e["digests"]}),
        "held_out_scope": {
            "n_identity_clusters": len(distinct_test_images_per_identity),
            "n_image_clusters": audit["by_split"]["test"]["n_distinct_bytes"],
            "distinct_test_images_per_identity":
                distinct_test_images_per_identity,
            "image_level_is_degenerate": degenerate_on_held_out,
            "note": ("the intervention rows are built from held-out images, so "
                     "these are the clusters the bootstrap actually resamples; "
                     "the whole-manifest counts above include training images "
                     "the design never touches"),
        },
        "level_to_use": "identity" if degenerate_on_held_out else "image",
        "clusters_at_the_level_to_use": (
            len(distinct_test_images_per_identity) if degenerate_on_held_out
            else audit["by_split"]["test"]["n_distinct_bytes"]),
        "reason": (
            "every identity has a single distinct held-out image, so image "
            "clusters collapse onto identity clusters and resampling them would "
            "only look like a finer unit"
            if degenerate_on_held_out else
            "identities carry genuinely distinct held-out images, so "
            "image-level clustering is a real refinement over identity-level"),
    }
    return {
        "held_out_image_content_exists": held_out,
        "routing_gates": routing,
        "mediation_gates": mediation,
        "clustering": clustering,
        "n_gates_supportable": (len(mediation["gates"])
                                + (len(routing["gates"]) if held_out else 0)),
        # counted from the gate NAMES, not from the threshold table: gate six
        # consumes two thresholds, so len(GATE_THRESHOLDS) is not the gate count
        # and would silently become wrong if a threshold were added.
        "n_gates_total": len(set(routing["gates"]) | set(mediation["gates"])),
        "policy": (
            "an unsupportable gate is recorded as untestable on this dataset; "
            "it is not waived, not lowered, and not counted as passed"),
    }


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------

#: Resampling units and reproducibility of the interval.  The count and the seed
#: are frozen here rather than chosen at analysis time: a bootstrap whose seed is
#: picked after seeing the interval is not a confidence interval.
BOOTSTRAP = {"n_resamples": 2000, "seed": 17, "alpha": 0.05,
             "method": "percentile bootstrap over resampled clusters"}

#: What a missing input does.  Every one of these is a refusal rather than a
#: default, because each of them would otherwise surface later as a smaller row
#: set -- and a smaller row set still produces a rate, so the run would report a
#: result about the rows that happened to be there.
MISSING_DATA_POLICY = OrderedDict((
    ("missing_router_seed", {
        "action": "refuse to run the cell",
        "why": ("the router seed selects which cached g predictions are replayed; "
                "running without it silently substitutes another router and the "
                "cell no longer tests the design it is labelled with")}),
    ("missing_edit_seed", {
        "action": "refuse to run the cell",
        "why": ("a cell without its edited h has no intervention to measure, and "
                "an absent checkpoint must not be read as a null result")}),
    ("missing_image", {
        "action": "refuse to freeze the design",
        "why": ("intervention rows are built per held-out image; a missing image "
                "means the cross is incomplete, and coverage_audit names the "
                "shortfall rather than reporting a smaller design as exact")}),
    ("missing_checkpoint", {
        "action": "refuse to run, and name every absent role",
        "why": ("verify_checkpoints hashes each role and reports the missing "
                "ones; existence is not identity, so a replaced checkpoint is "
                "also caught by digest")}),
    ("missing_intervention_row", {
        "action": "refuse to freeze, and refuse to gate",
        "why": ("a partial cross leaves a cell of the image-by-route table "
                "unmeasured, and the decisive claim is a claim about all of "
                "them")}),
    ("unparseable_output", {
        "action": "count it; gate six fails",
        "why": ("an output that names no candidate label identifies no route; "
                "resolving it to the nearest match would invent a result")}),
    ("multi_label_output", {
        "action": "count it as ambiguous; gate six fails",
        "why": ("an output naming two aliases followed neither route, and taking "
                "the first match would manufacture an intervention-following "
                "rate")}),
    ("partial_cell", {
        "action": "report no verdict for it",
        "why": ("a cell that scored some of its rows is not a weaker cell, it is "
                "a cell about a different design")}),
    ("gate_failure", {
        "action": "block promotion; thresholds are not adjustable afterwards",
        "why": ("a threshold chosen once the data exists is not a gate.  What may "
                "change is the measurement, and only with a new "
                "pre-registration")}),
))


def reuse_inventory(dataset, forget_set_ids, edit_seeds):
    """Which checkpoints already exist, so a promotion rule can say what it
    would actually have to train.

    Derived from the tree rather than declared: a promotion plan that assumes
    reuse which is not there costs a full retraining, and one that retrains what
    already exists wastes the GPU window that made the pilot possible.
    """
    cells = DATASET_ROOT / MATRIX_CELLS[dataset]
    route = DATASET_ROOT / ROUTE_DIRS[dataset]
    inv = OrderedDict()
    for sid in sorted(forget_set_ids):
        per_seed = {}
        for seed in edit_seeds:
            d = cells / sid / f"seed_{seed}"
            ad = d / "edited_h" / Path(*ADAPTER_RELPATH)
            res = d / "cell_results.json"
            per_seed[str(seed)] = {
                "adapter_present": ad.is_file(),
                "adapter_sha256": sha256_file(ad) if ad.is_file() else None,
                "cell_results_present": res.is_file(),
                "reusable": ad.is_file() and res.is_file(),
            }
        inv[sid] = {"per_edit_seed": per_seed,
                    "n_reusable": sum(1 for v in per_seed.values()
                                      if v["reusable"]),
                    "n_required": len(edit_seeds)}
    shared = {}
    for role, sub in (("baseline_g", "g_X_to_C"), ("baseline_h", "h_C_to_Y")):
        ad = route / sub / Path(*ADAPTER_RELPATH)
        shared[role] = {"present": ad.is_file(),
                        "sha256": sha256_file(ad) if ad.is_file() else None}
    return {
        "dataset": dataset,
        "n_forget_sets": len(inv),
        "sets": inv,
        "n_reusable_edited_h": sum(v["n_reusable"] for v in inv.values()),
        "n_edited_h_required": sum(v["n_required"] for v in inv.values()),
        "shared_route_checkpoints": shared,
        "complete": (all(v["n_reusable"] == v["n_required"] for v in inv.values())
                     and all(v["present"] for v in shared.values())),
    }


def build_pilot_preregistration(dataset, forget_set_id, forget_ids,
                                router_seeds, edit_seeds, promotion=None,
                                man=None, images=None, bootstrap=None,
                                selection=None):
    """The frozen pilot: design, images with digests, gates scoped to what the
    dataset can support, and the policies that decide what a failure means.

    Pre-registration means every number a later analysis could be tempted to
    choose -- thresholds, resample count, bootstrap seed, cluster level,
    promotion rule -- is fixed here, before any cell is run.

    ``promotion`` defaults to None, which DERIVES it: the rule has to know
    whether this dataset can support the routing gates before it can say what
    promotion is blocked by, and that comes from the image-content audit
    computed below.  This is also the single construction path -- ``main`` and
    ``verify_manifest`` both call it with ``promotion=None``, so the thing that
    is frozen and the thing that is rebuilt cannot be two implementations that
    happen to agree today.
    """
    man = man if man is not None else load_manifest(dataset)
    images = images if images is not None else held_out_images(man)
    bootstrap = dict(bootstrap or BOOTSTRAP)

    design = build_design(dataset, forget_set_id, forget_ids, router_seeds,
                          edit_seeds, man=man, images=images)
    audit = image_content_audit(man)
    applicability = gate_applicability(audit)

    inventory = None
    if promotion is None:
        inventory = reuse_inventory(dataset, _promotion_forget_sets(dataset),
                                    edit_seeds)
        promotion = promotion_rule(dataset, router_seeds, edit_seeds, inventory,
                                   applicability)

    # The forget set the pilot uses is the one the stated rule selects.  Asking
    # for a different one is refused rather than recorded, because a
    # pre-registration whose target was chosen by hand after the checkpoints
    # were inspected is not pre-registered.
    if selection is None:
        selection = pilot_forget_set(dataset, edit_seeds)
        if selection["set_id"] != forget_set_id:
            raise RuntimeError(
                f"the selection rule picks {selection['set_id']!r} for the "
                f"{dataset} pilot but {forget_set_id!r} was requested; pass "
                f"nothing and let the rule choose, or change the rule in a new "
                f"pre-registration -- not the request in this one")
        if sorted(selection["targets"]) != sorted(forget_ids):
            raise RuntimeError(
                f"{forget_set_id} targets {selection['targets']} in the frozen "
                f"matrix but {list(forget_ids)} was requested")

    held_out = []
    for iid in sorted(images):
        for it in images[iid]:
            uri = it["image_uri"]
            held_out.append({
                "identity_id": iid,
                "alias": man["alias_of"][iid],
                "code": man["code_of"][iid],
                "is_forgotten": iid in set(forget_ids),
                "image_id": Path(uri).name,
                "image_uri": uri,
                "sha256": audit["images_by_uri_sha256"][uri],
                "bytes": Path(uri).stat().st_size,
            })

    # The bootstrap level is taken from the audit, not chosen: where every
    # identity has one distinct held-out image, image-level clusters are
    # identity clusters wearing a different name.
    scope = applicability["clustering"]["held_out_scope"]
    level = applicability["clustering"]["level_to_use"]
    n_clusters = applicability["clustering"]["clusters_at_the_level_to_use"]
    if n_clusters < 2:
        raise RuntimeError(
            f"the {dataset} pilot has {n_clusters} {level} cluster(s) among its "
            f"held-out images; one cluster cannot produce an interval, so "
            f"Delta_route would have no confidence statement.  Freeze this "
            f"pilot on a dataset with at least two held-out clusters rather "
            f"than reporting a point estimate as if it were bootstrapped")
    # The audit counts clusters by image CONTENT and the design counts them by
    # the image each row names.  Those two have to agree at the level the
    # bootstrap will use, or the interval is computed over more units than
    # exist: a dataset whose held-out filenames outnumber its held-out bytes
    # would resample the same image several times and call them independent.
    rows = design["cells"][0]["rows"]
    n_row_clusters = len(cluster_units(rows, level))
    if n_row_clusters != n_clusters:
        raise RuntimeError(
            f"the audit counts {n_clusters} {level} cluster(s) among held-out "
            f"image CONTENT but the design's rows fall into {n_row_clusters} "
            f"{level} cluster(s); at this level the two must agree, because a "
            f"bootstrap over filename-clusters that are byte-identical counts "
            f"one image several times and understates the interval")

    bootstrap = {**bootstrap, "cluster_level": level,
                 "cluster_level_chosen_by": (
                     "derived from the image-content audit, not selected at "
                     "analysis time"),
                 "n_clusters_resampled": n_clusters,
                 "n_row_clusters_at_that_level": n_row_clusters,
                 "n_identity_clusters_held_out": scope["n_identity_clusters"],
                 "n_image_clusters_held_out": scope["n_image_clusters"],
                 "counts_are_held_out_only": (
                     "the rows the bootstrap resamples are built from held-out "
                     "images, so training images are not clusters of anything "
                     "here"),
                 "why_clusters_not_rows": (
                     "repeated interventions on one image share a router and an "
                     "edited h, and the several images of one person share an "
                     "identity; resampling rows would count one person many "
                     "times and understate the interval"),
                 "one_cluster_is_refused_not_reported": (
                     "a bootstrap over a single cluster restates the point "
                     "estimate as an interval of width zero")}

    requirements = design["checkpoint_requirements"]
    checkpoints = verify_checkpoints(requirements, dataset, forget_set_id)

    # the inventory goes INSIDE the block rather than being attached by the
    # caller afterwards: a mutation after freeze is not covered by
    # design_sha256, so the rebuilt block would not reproduce the frozen one
    rule_block = dict(promotion)
    if inventory is not None:
        rule_block["reuse_inventory"] = inventory

    return {
        **design,
        "kind": PREREG_KIND,
        "preregistered": True,
        "executed": False,
        "execution_policy": (
            "FROZEN AND NOT EXECUTED.  No cell of this pilot has been run, no "
            "router has been trained and no 63-cell matrix has been launched.  "
            "GPU work requires a separate decision"),

        "held_out_images": {
            "n": len(held_out),
            "images": held_out,
            "content_audit": audit,
            "warning": (None if audit["held_out_image_content_exists"] else
                        "THESE IMAGES ARE NOT HELD OUT IN CONTENT: every test "
                        "image's bytes also appear in the train split, so the "
                        "split partitions filenames rather than images.  The "
                        "URIs and digests below are exact, and the mediation "
                        "gates remain valid because they force the code, but no "
                        "held-out ROUTING claim can be made from them"),
        },

        "gate_applicability": applicability,
        "frozen_gates": {
            "thresholds": dict(GATE_THRESHOLDS),
            "n_gates": applicability["n_gates_total"],
            "supportable_on_this_dataset": applicability["n_gates_supportable"],
            "routing_gates": applicability["routing_gates"],
            "mediation_gates": applicability["mediation_gates"],
            "not_adjustable_afterwards": True,
            "independent_of_the_granularity_gates": (
                "no threshold is imported from the G6 granularity pilot"),
        },
        "delta_route": design["delta_route_definition"],
        "cluster_bootstrap": bootstrap,
        "e2e_decomposition": {
            "reported_separately": True,
            "conditional": ("accuracy over images the frozen g routed correctly; "
                            "this is the figure that speaks about the edit"),
            "unconditional": ("accuracy over ALL images, so a routing failure "
                              "counts as a wrong answer and the edit is "
                              "conflated with the router"),
            "prediction_formula": "sum_c P_g(c|X) * P_h(Y*(X)|c)",
            "required_label_is_keyed_by": (
                "the IMAGE, not the code -- keying it by the code would make "
                "every routing mistake correct by construction and predict 1.0 "
                "for any router whatsoever"),
            "unconditional_is_not_a_pass_requirement": (
                "where the frozen route failed its held-out gate, association "
                "and routing conclusions are gated separately"),
            "supportable_on_this_dataset":
                applicability["held_out_image_content_exists"],
        },
        "checkpoint_requirements": {
            **requirements,
            "verification": checkpoints,
            "complete": checkpoints["complete"],
        },
        "promotion_rule": rule_block,
        "pilot_forget_set_selection": selection,
        "missing_data_policy": MISSING_DATA_POLICY,
        "score_sum_semantics": design["score_sum_semantics"],
    }


def _promotion_common(dataset, router_seeds, edit_seeds, inventory):
    """What every promotion shares, with every count derived.

    "63 cells" is 3 router seeds x 7 forget sets x 3 edit seeds.  The product is
    computed here rather than written down, and the middle factor is read off the
    frozen matrix manifest, so a promotion rule cannot disagree with the matrix
    it promotes into.
    """
    n_router = len(router_seeds)
    n_sets = inventory["n_forget_sets"]
    n_edit = len(edit_seeds)
    reusable = inventory["n_reusable_edited_h"]
    return {
        "to": f"the full {dataset} route factorial",
        "n_router_seeds": n_router,
        "n_forget_sets": n_sets,
        "n_edit_seeds": n_edit,
        "n_e2e_cells": n_router * n_sets * n_edit,
        "counts_derived_from": (
            "the forget sets are read from the frozen matrix manifest by "
            "_promotion_forget_sets, and the reused checkpoints are hashed off "
            "disk by reuse_inventory; neither count is written down here"),
        "reuses_existing_edited_h": reusable,
        "edited_h_required": inventory["n_edited_h_required"],
        "reuse_inventory_is_complete": inventory["complete"],
        "retrained_per_cell": False,
        "reuse_statement": (
            f"{reusable} of {inventory['n_edited_h_required']} required "
            f"edited-h checkpoints already exist across {n_sets} forget sets x "
            f"{n_edit} edit seeds and are reused for the intervention matrix "
            f"rather than retrained; each edited h is probed under every router "
            f"seed"),
        "must_be_trained": {
            "routers": ("only the router seeds that do not exist yet; the "
                        "existing frozen g is one of them"),
            "edited_h": "none, if the reuse inventory above is complete"},
        "what_this_corrects": (
            "existing results vary ONLY the h-edit seed while holding g fixed, "
            "so reported seed-stability is stability to a single router"),
        "variance_to_be_reported_separately": [
            "router training seed", "edit training seed",
            "forget-set composition", "held-out image"],
    }


def promotion_rule_ppubench(router_seeds, edit_seeds, inventory, applicability):
    """PPUBench's factorial: 3 router seeds x the matrix's forget sets x 3 seeds.

    This corrects the standing multi-seed limitation -- every existing result
    varies only the h-edit seed while holding g fixed, so what looks like
    seed-stability is stability to one router -- and promotion therefore
    requires routers that do not exist yet.

    Whether the routing half of it is blocked is DERIVED from the image-content
    audit rather than asserted.  That PPUBench's test images are byte-identical
    to its training images is a measured property of the bytes on disk, so
    writing it down as prose would keep saying "untestable" if the images were
    ever replaced with genuinely held-out ones.
    """
    rule = _promotion_common("ppubench", router_seeds, edit_seeds, inventory)
    routing = applicability["routing_gates"]
    rule["held_out_routing_is_testable_here"] = routing["supportable"]
    if routing["supportable"]:
        rule["blocked_by"] = None
    else:
        rule["blocked_by"] = (
            f"{', '.join(routing['gates'])} are UNSUPPORTABLE on this dataset, "
            f"because {routing['reason_if_not']}.  Promoting would multiply a "
            f"routing claim the dataset cannot make across "
            f"{rule['n_e2e_cells']} cells, so the routing half of this "
            f"factorial has to run on a dataset with held-out image content")
    rule["conditions"] = [
        "the pilot's supportable gates all pass",
        "no gate threshold is altered between pilot and promotion",
        ("the reuse inventory is re-verified at promotion time, because a "
         "checkpoint replaced in place keeps its path")]
    return rule


def promotion_rule_salmu(router_seeds, edit_seeds, inventory, applicability):
    """SALMU's factorial, which is where the routing gates can be measured.

    SALMU is the dataset whose test images are genuinely distinct bytes from its
    training images, so the two gates that are untestable on PPUBench are
    measured here -- at the accuracy the frozen g actually achieves rather than
    at an accuracy the design would like it to achieve.

    Gate one's expected outcome is COMPUTED from the recorded held-out accuracy
    against the frozen threshold.  Stating it in prose would leave a number in
    the artifact that nothing checks, and it would survive a change to either
    input still describing the old one.
    """
    rule = _promotion_common("salmu", router_seeds, edit_seeds, inventory)
    routing = applicability["routing_gates"]
    rule["held_out_routing_is_testable_here"] = routing["supportable"]
    # always present, so "not blocked" is None rather than a missing key: a
    # reader comparing two datasets' rules cannot otherwise tell an absent
    # field from a field that was never computed
    rule["blocked_by"] = None if routing["supportable"] else (
        routing["reason_if_not"])

    recorded = HELD_OUT_G["salmu"]["accuracy"]
    floor = GATE_THRESHOLDS["min_router_held_out_accuracy"]
    if recorded is None:
        expected, reading = None, (
            "no held-out routing accuracy is recorded for the frozen g on this "
            "dataset, so gate one's outcome cannot be predicted")
    else:
        expected = bool(recorded >= floor)
        reading = (
            f"the frozen g scores {recorded} on SALMU's held-out images against "
            f"a pre-registered floor of {floor}, so gate one is expected to "
            f"{'PASS' if expected else 'FAIL'}.  "
            + ("that is a real measurement and it is not to be tuned away: the "
               "imperfect router is the scientifically useful condition, "
               "because it is the only one where the E2E composition has "
               "something to explain" if not expected else
               "the frozen router already clears the floor"))
    rule["expected_gate_one_outcome"] = {
        "recorded_held_out_accuracy": recorded,
        "recorded_source": HELD_OUT_G["salmu"]["source"],
        "frozen_threshold": floor,
        "expected_to_pass": expected,
        "reading": reading,
        "derived_not_asserted": ("computed from HELD_OUT_G against "
                                 "GATE_THRESHOLDS, not written down"),
    }
    rule["conditions"] = [
        "the mediation gates all pass",
        ("gate one's outcome is reported as a property of the frozen router, "
         "not repaired by retraining g -- the route architecture is frozen"),
        "no gate threshold is altered between pilot and promotion",
        ("the reuse inventory is re-verified at promotion time, because a "
         "checkpoint replaced in place keeps its path")]
    return rule


#: Which promotion rule applies where.  MLLMU is absent on purpose: it has ONE
#: image per identity, so held-out routing does not exist there and neither does
#: the factorial that multiplies it.  A default branch would quietly hand it
#: another dataset's prose.
PROMOTION_RULES = {"ppubench": promotion_rule_ppubench,
                   "salmu": promotion_rule_salmu}


def promotion_rule(dataset, router_seeds, edit_seeds, inventory, applicability):
    try:
        rule = PROMOTION_RULES[dataset]
    except KeyError:
        raise RuntimeError(
            f"no promotion rule for {dataset!r}; the rules that exist are "
            f"{sorted(PROMOTION_RULES)}.  MLLMU has ONE image per identity, so "
            f"image-level held-out routing does not exist there and it stays "
            f"outside the route headline rather than inheriting a rule written "
            f"for a dataset that has it") from None
    return rule(router_seeds, edit_seeds, inventory, applicability)



if __name__ == "__main__":
    sys.exit(main())
