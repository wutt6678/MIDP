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

The conditions
==============
For a forget set ``F`` with retained identities ``R``, ``h_base`` the unedited
mediator, ``h_e`` the edited one, ``g_s`` one router seed and ``D_s`` one direct
adapter:

  ===========================  =======  =====================  =================
  condition                    model    input                  required outcome
  ===========================  =======  =====================  =================
  baseline_forced_code         h_base   do(C=c), no image      the code's alias
  forgotten_route_intervention h_e      do(C=C_f), no image    Unknown
  retained_route_intervention  h_e      do(C=C_r), no image    Y_r
  retained_control             h_e      do(C=C_r), no image    Y_r
  natural_mediated             h_e      c = g_s(X), no image   Unknown if X in F
  direct_control               D_s      X, no code             the image's alias
  hybrid_conflict_probe        D_s      X and an irrelevant C  the image's alias
  ===========================  =======  =====================  =================

``baseline_forced_code`` is the BEFORE that makes the rest an after: a code the
edit was meant to break has to be shown producing its own alias first, or
"forgotten codes now produce Unknown" stays compatible with an ``h`` that never
answered them at all.  It is the intervention call with ``h_base`` in place of
``h_e`` and nothing else changed, so no part of the comparison depends on a
second prompt.

``retained_route_intervention`` and ``forgotten_route_intervention`` are the
decisive pair: they cross image identity against route identity in both
directions, so "follows the route" and "follows the image" cannot both be
satisfied.  ``retained_control`` is what makes the pair interpretable -- without
it, a model that answers Unknown to everything would look route-dependent.
``direct_control`` probes the unchanged ``X -> Y`` pathway that bypasses the code
altogether, and is expected NOT to be suppressed: this project claims
route-dependent forgetting, not global erasure.  ``hybrid_conflict_probe`` gives
that pathway a code to ignore, which is what bounds it from above -- an accuracy
floor alone would have passed a direct adapter that answered from the code and
was therefore not a direct model at all.

No image reaches ``h`` in any condition.  A row's image is metadata that says
which identity its code belongs to; putting it in ``h``'s prompt would make the
system under test f(X, C) rather than the frozen Y = h(C), and the mediation
claim would no longer be testable.

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
  RF0   verify the frozen manifest and every input it names (CPU)
  RFC   calibrate D_s's schedule on a development split, select one (GPU)
  RF1B  h_base(do(C=c)) for every code -- the BEFORE in before/after (GPU)
  RF1G  train and evaluate ONE router seed g_s (GPU)
  RF1D  train and evaluate ONE direct adapter D_s (GPU)
  RF1H  h_e(do(C=c)) for ONE edit seed (GPU)
  RF1E  h_e(g_s(X)) for ONE (router, edit) pair (GPU)
  RF2   aggregate, gate, bootstrap, report (CPU)
  RF2P  re-derive hard predictions from stored raw text (CPU)

RF1 no longer exists.  It was one stub that raised for every dataset, and a
phase that varies no factor cannot be resumed, re-run or reported per factor.

Everything except the RF1 phases and RFC is CPU-only and is what the test
suite exercises.  The planning, design, expectation, scoring, gating and bootstrap
layers all run without a model, so a design can be frozen and audited before any
GPU time is spent -- and a cell can be re-scored from its stored generations
without regenerating anything.

Design versions
===============
Five designs are reconstructible from this file, and all five have frozen
manifests committed: ``--design-version v1``, ``v2``, ``v3`` and ``v4`` are
SUPERSEDED and exist so the artifacts on disk can still be rebuilt and explained,
``v5`` is the current one and is the default.  Superseded means superseded in
place, never edited: each version's constructor reads its differences from a
``PilotSpec``, so the five share one construction path and a repair cannot reach
one version and not the others.  ``supersession_record`` states, per version, what
the previous one got wrong and what the current one does instead.

The three supersessions are not the same kind of correction, and the difference
matters to anyone deciding which results to cite.  ``v2`` was superseded for a
STRUCTURAL reason: its design hash covered which adapters were on disk, so it
could not survive the training it required.  ``v3`` is superseded for a SCOPE
reason: it declared ``hybrid_conflict_probe`` auxiliary and "reported separately
from every mediated gate", then counted that condition's rows toward a gate it
mapped to mediation.  ``v4`` changes that one denominator, reports the auxiliary
rows' hygiene as a diagnostic that feeds no verdict, and -- uniquely in this file
-- was amended AFTER the run whose verdicts prompted it.  Every v4 artifact
carries ``post_outcome_scope_amendment: true`` and the derived evidence from both
datasets, because results aggregated under v4 from v3's cells are a CORRECTIVE
analysis of a completed run; only a run executed under v4 from the start is
pre-registered under the corrected scope.

``v5`` is that run, and it supersedes v4 for a STATUS reason rather than a
scientific one.  v4's analysis of v3's cells is correct, stays committed and is
not re-derived here.  What v4 cannot be is confirmatory, by its own statement, so
v5 executes under the corrected scope from the start: five forget sets that no
earlier design consumed, chosen by a rule rather than listed; the router-seed x
edit-seed cross over all of them; v4's thresholds, gate map, conditions and
hygiene denominator held by OBJECT IDENTITY so "the scope is frozen" is a
property a test can check with ``is``; and no ancestor declared, which makes
``reused_from_the_ancestor_design`` unreachable and refuses a cell v1-v4 filed
rather than aggregating it.  It also calibrates the direct adapter's training
schedule on a development split disjoint from the held-out test images in image
CONTENT, and freezes the selected configuration together with every confirmatory
cell before any of them is produced.  The route itself is not calibrated and is
not touched: g and h are read from the frozen route scripts and the frozen
adapters, and a phase named RFC is not evidence that either was tuned.

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
import operator
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

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
#: is a property of the dataset, not a knob.  The accuracy alone does not say
#: what it is measured ON, and ``image_content_audit`` is what settles that:
#: PPUBench routes its twelve held-out filenames perfectly, but every one of
#: those images' BYTES also appears in its train split, so the split partitions
#: filenames rather than images and that 1.0 is repeated training content --
#: the pilot runs there for the mediation gates, which force the code and so do
#: not depend on the images being unseen, and it can make no held-out ROUTING
#: claim.  SALMU is the dataset whose held-out images are held out in content,
#: and its 0.8056 is a scientifically useful noisy-router condition rather than
#: something to tune away.  MLLMU has ONE image per identity, so image-level
#: held-out routing does not exist and it stays outside the route headline
#: entirely.
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


def resolve_recorded_path(raw):
    """Where a path recorded in a manifest or a dataset file actually is.

    Two sources need this and both fail the same way without it.

    Dataset manifests name images differently per dataset: PPUBench records
    absolute paths under the downloaded benchmark, SALMU records paths relative
    to this dataset's root.  Resolving a relative URI against the CURRENT
    WORKING DIRECTORY instead makes the content audit pass or fail depending on
    where the runner was invoked from, and a failure there reads as missing data
    rather than as what it is.

    Frozen manifests name the checkpoints they depend on.  v1's verification
    table recorded those as absolute paths, which ties the artifact to one
    filesystem root: rebuilt from any other clone, every checkpoint reads as
    absent even where the weights are on disk.  v2 records them with ``_rel``,
    and both shapes resolve here so the superseded v1 manifests still verify
    where they were frozen.

    Recorded strings are never rewritten, so a frozen manifest still reproduces
    byte for byte.
    """
    p = Path(raw)
    return p if p.is_absolute() else DATASET_ROOT / p


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
        "what_a_drifted_input_digest_means": (
            "this block is NOT covered by design_sha256, so nothing here can "
            "change what was pre-registered.  A digest below that no longer "
            "matches the file on disk means the world moved since freezing, and "
            "for the runner's own digest it means the phases would execute "
            "something other than the code this design was frozen against -- so "
            "verify_manifest reports it, returns valid=False, and load_prereg "
            "refuses.  Re-freezing is the repair and is provably not a new "
            "pre-registration: it re-binds these inputs and leaves "
            "design_sha256 byte-identical, which is the property that "
            "distinguishes re-binding a design from changing one.  What is NOT "
            "repaired by re-freezing is a design_sha256 that no longer "
            "reproduces, because that means the design itself moved"),
    }


def freeze_manifest(design, path, extra_paths=()):
    """Write the frozen design and bind it to its own inputs.

    Written atomically, for the same reason a cell result is: a manifest is the
    artifact every later phase reads, and a half-written one that a reader
    parsed as far as it got would look like a smaller design rather than a
    broken file.
    """
    frozen = {
        **design,
        "frozen": True,
        "design_sha256": design_sha256(design),
        "provenance": provenance(extra_paths),
    }
    return atomic_write_json(Path(path), frozen)


def _frozen_checkpoint_digests(table):
    """Every (label, path, digest) a manifest's checkpoint verification names.

    v1 recorded one path per role.  v2 records one path per FILE per role --
    adapter, held-out predictions, cell results -- because a router seed needs
    two files before it is a factor.  Both shapes are flattened here rather than
    branched on at every call site, so a third shape cannot slip past the
    re-hash and leave a digest that nothing checks.
    """
    out = []
    for role, entry in sorted((table or {}).items()):
        if not isinstance(entry, dict):
            continue
        if "path" in entry:
            out.append((role, entry["path"], entry.get("sha256")))
            continue
        for key in sorted(k for k, v in entry.items() if isinstance(v, dict)):
            sub = entry[key]
            if "path" in sub:
                out.append((f"{role}.{key}", sub["path"], sub.get("sha256")))
    return out


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

    Two different things are checked here and it matters which is which.  The
    DESIGN is rebuilt and its hash compared, so an edited artifact is caught.
    The STATUS of the work the design asks for is probed live and reported, so
    a pilot that is waiting for training keeps verifying while the training
    arrives.  v2 put the second inside the first: its constructor embedded the
    result of a filesystem probe in the hashed design, so training the one role
    a pilot was waiting for changed what the constructor produced, the manifest
    stopped reproducing at the moment it became runnable, and every phase that
    verifies before loading refused to load it.  A design may bind the bytes it
    is a statement about; it may not bind its own progress.
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
        p = resolve_recorded_path(name)
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
        elif kind == PREREG_V2_KIND:
            rebuilt = build_pilot_preregistration_v2(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind == PREREG_V3_KIND:
            rebuilt = build_pilot_preregistration_v3(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind == KIND_V3:
            rebuilt = build_design_v3(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind == PREREG_V4_KIND:
            rebuilt = build_pilot_preregistration_v4(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind == KIND_V4:
            rebuilt = build_design_v4(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind == CALIBRATION_PREREG_KIND_V5:
            # Rebuilt from the dataset alone: the split is derived from the
            # manifest the freeze bound as an input, so a rebuilt document that
            # disagrees is one whose split rule or whose images moved.
            rebuilt = build_calibration_preregistration(frozen["dataset"])
        elif kind == PREREG_V5_KIND:
            rebuilt = build_pilot_preregistration_v5(
                frozen["dataset"], frozen["forget_sets"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind == KIND_V5:
            rebuilt = build_design_v5(
                frozen["dataset"], frozen["forget_sets"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind == KIND_V2:
            rebuilt = build_design_v2(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"], frozen["direct_seeds"])
        elif kind in (KIND, None):
            rebuilt = build_design(
                frozen["dataset"], frozen["forget_set_id"],
                frozen["forget_identity_ids"], frozen["router_seeds"],
                frozen["edit_seeds"])
        else:
            raise RuntimeError(f"unknown manifest kind {kind!r}; the kinds "
                               f"this verifier can rebuild are {KIND!r}, "
                               f"{PREREG_KIND!r}, {KIND_V2!r}, "
                               f"{PREREG_V2_KIND!r}, {KIND_V3!r}, "
                               f"{PREREG_V3_KIND!r}, {KIND_V4!r}, "
                               f"{PREREG_V4_KIND!r}, {KIND_V5!r}, "
                               f"{PREREG_V5_KIND!r} and "
                               f"{CALIBRATION_PREREG_KIND_V5!r}")
        if design_sha256(rebuilt) != frozen.get("design_sha256"):
            problems.append(
                f"rebuilding a {kind} from the frozen parameters does not "
                "reproduce design_sha256; the frozen content is not what the "
                "constructor would produce now")
    except RuntimeError as exc:
        problems.append(f"the manifest cannot be rebuilt at all: {exc}")

    req = frozen.get("checkpoint_requirements") or {}
    # Live readiness, where the manifest declares roles in the shape the prober
    # reads.  v1's checkpoint table predates that shape, so a v1 manifest keeps
    # the frozen-block reading below and reports no live readiness rather than
    # raising inside a verifier whose job is to report what is wrong.
    readiness = (checkpoint_readiness(frozen)
                 if roles_are_v2_shaped(req) else None)

    frozen_verification = req.get("verification")
    rehashed = 0
    trained_since = []
    if isinstance(frozen_verification, dict):
        # v1 and v2 froze a status block INSIDE the design.  Re-hashing what it
        # named is what makes "these weights" part of the frozen record rather
        # than a note about weights that were there once -- and it is also why
        # those manifests stop reproducing once a pending role is trained, which
        # is the defect v3 exists for.
        for label, p_str, want in _frozen_checkpoint_digests(
                frozen_verification.get("present")):
            p = resolve_recorded_path(p_str)
            got = sha256_file(p) if p.is_file() else None
            rehashed += 1
            if got != want:
                problems.append(
                    f"checkpoint {label}: frozen {want}, on disk {got} "
                    f"({p_str})")

        # A role the manifest recorded as ABSENT is expected to appear later:
        # that is the training work it names.  What is worth reporting is which
        # ones have appeared, because the digests of those weights are bound by
        # the cell results that use them and not by this pre-registration.
        for label, p_str, want in _frozen_checkpoint_digests(
                frozen_verification.get("absent")):
            if want is not None:
                continue
            p = resolve_recorded_path(p_str)
            if p.is_file():
                trained_since.append({"role": label, "path": p_str,
                                      "resolved": str(p),
                                      "sha256": sha256_file(p)})
    elif readiness is not None:
        # v3 froze no status block, so there is nothing here that could go stale
        # and nothing to compare a live digest against.  What is reported is what
        # the live probe hashed on THIS disk: the roles that are present, whether
        # or not the design declared them as inputs or as work.  Nothing about
        # them is inside design_sha256, and no role file is listed in
        # provenance.input_file_sha256 either -- adapters are gitignored, so a
        # digest of one recorded at freeze time would make every fresh clone
        # report drift while telling the truth about its own disk.  The weights
        # are bound by the RESULTS that consume them, which RF2 re-checks.
        rehashed = readiness["n_files_present"]
        trained_since = readiness["trained_since_declared"]

    # the selection rule is part of the frozen record too: if it now picks a
    # different set, the pilot was frozen against a tree that has since changed
    if kind in (PREREG_KIND, PREREG_V2_KIND, PREREG_V3_KIND) \
            and frozen.get("pilot_forget_set_selection"):
        try:
            now = pilot_forget_set(frozen["dataset"], frozen["edit_seeds"])
            if now["set_id"] != frozen["forget_set_id"]:
                problems.append(
                    f"the selection rule now picks {now['set_id']!r}, but the "
                    f"pilot was frozen against {frozen['forget_set_id']!r}")
        except RuntimeError as exc:
            problems.append(f"the selection rule can no longer be applied: {exc}")

    # v5's rule picks SEVERAL sets and is re-applied for the same reason: a rule
    # that now returns a different list means the tree this pilot was frozen
    # against is not the tree being verified.  Kept beside the single-set check
    # rather than folded into it, because the two rules answer different
    # questions and a version is checked against the one it declared.
    if kind == PREREG_V5_KIND and frozen.get("pilot_forget_set_selection"):
        try:
            now = pilot_forget_sets_v5(frozen["dataset"], frozen["edit_seeds"])
            was = [s["set_id"] for s in
                   frozen["pilot_forget_set_selection"]["sets"]]
            is_now = [s["set_id"] for s in now["sets"]]
            if is_now != was:
                problems.append(
                    f"the selection rule now picks {is_now}, but the pilot was "
                    f"frozen against {was}")
        except RuntimeError as exc:
            problems.append(f"the selection rule can no longer be applied: {exc}")

    # Completeness and the outstanding training work are reported LIVE wherever
    # the shape allows, for every version: reading them from the frozen block is
    # what made RF0's runnable_now permanently False, since a block frozen
    # before the training existed cannot later say the training happened.
    live = readiness if readiness is not None else {}
    return {"path": str(path), "valid": not problems, "problems": problems,
            "kind": kind,
            "design_sha256": frozen.get("design_sha256"),
            "n_rows_total": frozen.get("n_rows_total"),
            "n_cells": frozen.get("n_cells"),
            "executed": frozen.get("executed"),
            "n_checkpoints_rehashed": rehashed,
            "checkpoints_complete": live.get("complete", req.get("complete")),
            "n_must_be_trained": live.get("n_must_be_trained",
                                          req.get("n_must_be_trained")),
            "must_be_trained": live.get(
                "must_be_trained",
                req.get("must_be_trained_before_this_pilot_can_run")),
            "unexpectedly_absent": live.get(
                "unexpectedly_absent", req.get("unexpectedly_absent")),
            "roles_trained_since_freeze": trained_since,
            "checkpoint_readiness": readiness,
            "readiness_is_computed_live": (
                "presence, digests, completeness and runnable_now are probed "
                "when this function runs and are not read out of the frozen "
                "design, so a manifest that is waiting for training keeps "
                "verifying while the training arrives")}


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

def _build_parser():
    """One parser for every design version.

    v1 and v2 are superseded but their four frozen manifests are committed, so
    the flags that produced them still have to work and still have to produce
    the same bytes; v2 adds the per-factor seeds, the device, the resume switch
    and the cell root; v3 adds RF1B.

    The phase list is the union over versions rather than the selected
    version's own, so asking for a phase the selected design does not have is
    refused by the dispatcher -- which can name the version it was asked for --
    rather than by argparse, which can only say the choice was invalid.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ppubench", choices=list(DATASETS))
    p.add_argument("--design-version", default=LATEST_PILOT_SPEC.version,
                   choices=["v1", *sorted(SPEC_BY_VERSION)],
                   help="v1 to v4 are SUPERSEDED and retained only so their "
                        "frozen manifests can still be rebuilt and verified; v5 "
                        "is the current design and is the default.  v2's pilots "
                        "hash their own live checkpoint status, so they stop "
                        "reproducing the moment the training they require "
                        "succeeds; v3 computes readiness outside the design but "
                        "counts auxiliary rows toward a mediated verdict it "
                        "declared them outside; v4 declares the hygiene gate's "
                        "row scope and says, at the top level of its own design, "
                        "that the scope was amended after an outcome; v5 is the "
                        "run executed under that scope from the start -- it "
                        "declares no ancestor, varies five unconsumed forget "
                        "sets, and calibrates the direct adapter's schedule "
                        "before it freezes its cells")
    p.add_argument("--phase", nargs="+", default=["RF0"],
                   choices=["RF1", *ALL_PHASES],
                   help="RF0 verify the frozen manifest and every input; "
                        "RF1B evaluate the UNEDITED h on every code (v3 only, "
                        "the before in before/after); RF1G train/evaluate one "
                        "router seed; RF1D train/"
                        "evaluate one direct-model seed; RF1H evaluate the "
                        "forced-code interventions for one edit seed; RF1E "
                        "evaluate the natural composition for one router/edit "
                        "pair; RF2 aggregate, gate and bootstrap; RF2P rescore "
                        "the stored raw outputs and reproduce RF2 with no GPU; "
                        "RFC calibrate the direct adapter's schedule on the "
                        "development split, or -- with no --candidate -- apply "
                        "the frozen selection rule to the filed measurements")
    p.add_argument("--forget-set", default=None,
                   help="forget-set id as the matrix names it, e.g. fs_001.  A "
                        "design that varies ONE forget set infers it; a design "
                        "that varies several needs it to say which set RF1B, "
                        "RF1H or RF1E fills, and refuses rather than guessing")
    p.add_argument("--candidate", default=None,
                   help="RFC only: the calibration candidate to train, e.g. "
                        "C0_incumbent.  Omitted, RFC applies the frozen "
                        "selection rule to the measurements already filed and "
                        "trains nothing")
    p.add_argument("--forget-ids", nargs="+", default=None)
    p.add_argument("--router-seeds", nargs="+", type=int, default=None,
                   help="the router seeds a DESIGN varies; v2 defaults to the "
                        "per-dataset seed policy")
    p.add_argument("--edit-seeds", nargs="+", type=int, default=None)
    p.add_argument("--direct-seeds", nargs="+", type=int, default=None,
                   help="v2 only: the D_s seeds a design varies")
    p.add_argument("--router-seed", type=int, default=None,
                   help="v2 only: the single router seed RF1G trains, or the "
                        "single one RF1E composes with")
    p.add_argument("--edit-seed", type=int, default=None,
                   help="v2 only: the single edited-h seed RF1H or RF1E uses")
    p.add_argument("--direct-seed", type=int, default=None,
                   help="v2 only: the single D_s seed RF1D trains")
    p.add_argument("--device", default="cuda",
                   help="v2 only: the device the RF1 phases load the model on")
    p.add_argument("--resume", action="store_true",
                   help="v2 only: reuse a filed cell only when every input "
                        "digest, both prompts and the row count still match; a "
                        "stale cell is re-run and the reason is logged")
    p.add_argument("--manifest", default=None)
    p.add_argument("--preregister", action="store_true",
                   help="freeze the PILOT pre-registration rather than the bare "
                        "design: adds the held-out image digests and content "
                        "audit, per-dataset gate applicability, the cluster "
                        "bootstrap configuration, the promotion rule and the "
                        "missing-data policy")
    p.add_argument("--preregister-calibration", action="store_true",
                   help="freeze the CALIBRATION pre-registration -- the "
                        "development split, the candidate grid, the seeds, the "
                        "metric, the tie-breaks and the floor -- rather than the "
                        "pilot.  Refused if any calibration output already "
                        "exists, because a grid frozen after the training it "
                        "constrains is a grid that agrees with its own outcome")
    p.add_argument("--verify", action="store_true",
                   help="verify a frozen manifest and exit")
    p.add_argument("--out", default=None,
                   help="v2/v3: the DIRECTORY this run writes into -- a report "
                        "for RF0/RF2/RF2P, or the frozen manifest for "
                        "--preregister, always under its canonical "
                        "version-suffixed name so a staged artifact cannot be "
                        "written under a name the phases will never look for "
                        "and a v3 report cannot overwrite a v2 one.  "
                        "v1: the manifest FILE to freeze")
    p.add_argument("--cells", default=None,
                   help="v2 only: the root the per-cell results are filed under; "
                        "trained adapters always go to the path the frozen "
                        "manifest declares, because a checkpoint anywhere else "
                        "is one the manifest cannot see")
    return p


def _run_v1(args):
    """The superseded v1 entry point, unchanged.

    Kept because the two v1 pilot manifests are committed and frozen: a runner
    that could no longer produce the artifact it has on disk could not explain
    it either.
    """
    if args.router_seeds is None:
        args.router_seeds = [17, 42, 123]
    if args.edit_seeds is None:
        args.edit_seeds = [17, 42, 123]

    if args.verify:
        name = (f"rf_pilot_{args.dataset}.json" if args.preregister
                else f"rf_manifest_{args.dataset}.json")
        path = Path(args.manifest or (DATASET_ROOT / MANIFEST_DIR / name))
        got = verify_manifest(path)
        print(json.dumps(got, indent=2))
        return 0 if got["valid"] else 1

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
        p = resolve_recorded_path(uri)
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
                "bytes": resolve_recorded_path(uri).stat().st_size,
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


# ===========================================================================
# PART II -- v2: the repaired design
# ===========================================================================
#
# PART I above is SUPERSEDED but deliberately retained, because the two v1
# pilot manifests are committed, frozen and unexecuted, and verify_manifest
# rebuilds a v1 manifest through the constructor that made it.  Deleting that
# constructor would leave a committed artifact that can no longer be verified,
# which is worse than an artifact that has been superseded.  Nothing in the CLI
# builds v1 any more; the ``supersession`` block of each v2 pilot records why it
# was replaced, entry by entry, so the reason lives in the artifact a reader
# lands on rather than in a separate file that can fall out of date.
#
# The repairs, and the defect each one answers:
#
#   1  v1 forced the code with an IMAGE ALSO PRESENT, so h was evaluated as
#      f(X, C) and not as h(C).  The frozen architecture is Y = h(C); once h
#      receives the image the mediator is no longer hard and the intervention
#      cannot speak about the route.  v1 also used the wrong code prompt -- a
#      paraphrase rather than the CODE_TO_ALIAS_PROMPT the route was trained
#      with -- so both the input and the prompt were off-protocol.
#   2  v1's direct condition "reused edited_h", which was trained on
#      code-to-label pairs and is not an image model at all.  v2 trains
#      separate direct adapters D_s: X -> Y and gates their image accuracy,
#      because bounding code-following from above alone is satisfied by a model
#      that outputs arbitrary wrong labels.
#   3  v1 declared three router seeds but had one g checkpoint and one cached
#      prediction file, so "router seed" was a label and not a factor.  v2
#      requires a checkpoint and a held-out prediction file PER router seed and
#      builds the real 3_g x 3_h natural cells, while evaluating forced-code
#      interventions once per edit seed -- they do not depend on g, and
#      repeating them across router seeds would count one measurement three
#      times.
#   4  v1 had one RF1 that raised for every dataset, so no phase could say which
#      factor it varied.  v2 splits it into RF1G/RF1D/RF1H/RF1E, one factor
#      each, with RF0 verifying every input first and RF2/RF2P refusing to
#      aggregate a cell that was never filed.
#   5  v1 exposed one omnibus ``passed``.  v2 reports three separate verdicts.
#   6  v1 chose image-level clustering on SALMU as "a real refinement".  It is
#      not: three images of one person share an identity, a router and an
#      edited h, so they are not independent.  Identity level is primary and
#      image level is a sensitivity analysis.
#   7  v1 fed candidate score sums into the factorization as if they were
#      probabilities.  v2 uses the empirical hard-response matrix
#      H_e(c, y) = 1{parse(h_e(c)) = y}.
#   8  v1's two manifests recorded all of the above.  They are preserved
#      byte-identical and marked superseded; v2 freezes new ones beside them.

KIND_V2 = "route_dependent_forgetting_design_v2"
RESULT_KIND_V2 = "route_dependent_forgetting_result_v2"
PREREG_V2_KIND = "route_dependent_forgetting_preregistration_v2"
SUPERSESSION_KIND = "route_dependent_forgetting_supersession_v1"

#: v3's kinds.  The supersession kind names what it SUPERSEDES, following v2's
#: convention, and v3 supersedes two versions rather than one -- so a reader can
#: tell from the kind alone how many frozen designs are behind this one.
KIND_V3 = "route_dependent_forgetting_design_v3"
RESULT_KIND_V3 = "route_dependent_forgetting_result_v3"
PREREG_V3_KIND = "route_dependent_forgetting_preregistration_v3"
SUPERSESSION_KIND_V3 = "route_dependent_forgetting_supersession_v1_and_v2"

#: The version one result kind belongs to, for the message that refuses a cell
#: filed by a different design.  Derived rather than parsed out of the kind
#: string, so renaming a kind cannot silently make the refusal name the wrong
#: version.
RESULT_KIND_VERSION = {RESULT_KIND: "v1", RESULT_KIND_V2: "v2",
                       RESULT_KIND_V3: "v3"}

#: Where v2 writes.  Kept apart from the v1 manifests so a superseded artifact
#: is never overwritten by its replacement.
OUT_ROOT_V2 = OUT_ROOT
CELLS_DIR_V2 = OUT_ROOT_V2 / "outputs"

#: The scripts that establish and replay the frozen route.  Prompts and the
#: training protocol are READ from these rather than retyped here: a runner that
#: paraphrases a prompt evaluates a different model than the one the route was
#: established with, and v1 did exactly that.
FROZEN_ROUTE_SCRIPTS = ("e2c_v3_realdata.py", "e2c_v3_salmu.py",
                        "e2c_v3_phaseC_discrete_bottleneck.py")
PROTOCOL_SCRIPTS = ("e2c_v3_matrix.py", "e2c_v3_salmu.py")

#: The auxiliary bypass/conflict probe presents an image AND a code, which is
#: not part of the frozen route at all -- the frozen route is g(X) then h(c),
#: and h never sees an image.  The bytes are v1's CODE_IMG_PROMPT, deliberately:
#: item 1 asks for the image-plus-code evaluation to be RENAMED and demoted to
#: an auxiliary probe, not for a new prompt to be invented.  Referencing the
#: same constant means the two cannot drift while claiming to be one probe.
HYBRID_PROBE_PROMPT = CODE_IMG_PROMPT


def _module_constants(filename, names):
    """Read module-level numeric/string constants out of a script by AST.

    The frozen route scripts import torch at module scope, so importing them
    would drag a GPU dependency into every CPU layer of this file -- and the
    design, the gates and the re-scoring are all meant to run without one.  AST
    extraction reads the bytes instead, which is also what makes the agreement
    check below a check on the SOURCE rather than on a value this file cached.
    """
    import ast
    path = SCRIPT_DIR / filename
    if not path.is_file():
        raise RuntimeError(f"frozen route script absent: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        try:
            found[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise RuntimeError(
                f"{filename}:{target.id} is not a literal constant, so the "
                f"frozen protocol cannot be read without importing the module "
                f"({exc})") from None
    missing = sorted(set(names) - set(found))
    if missing:
        raise RuntimeError(f"{filename} does not define {missing}; the frozen "
                           f"route protocol cannot be read from it")
    return found


def frozen_route_prompts():
    """The prompts the frozen route was established with, and proof they agree.

    Read from all three route scripts and required to be IDENTICAL across them.
    If they ever diverged, "the same prompt" would be an assumption rather than
    a fact, and an intervention run under one script's prompt would not be
    comparable to a route established under another's.
    """
    want = ("IMG_TO_CODE_PROMPT", "CODE_TO_ALIAS_PROMPT")
    per_script = {f: _module_constants(f, want) for f in FROZEN_ROUTE_SCRIPTS}
    for name in want:
        values = {f: per_script[f][name] for f in FROZEN_ROUTE_SCRIPTS}
        if len(set(values.values())) != 1:
            raise RuntimeError(
                f"the frozen route scripts disagree on {name}: {values}; this "
                f"runner will not pick one, because a prompt is part of the "
                f"frozen route architecture")
    img_prompt = _module_constants("e2c_v3_phaseC_discrete_bottleneck.py",
                                   ("IMG_PROMPT",))["IMG_PROMPT"]
    return {
        "g_image_to_code": per_script[FROZEN_ROUTE_SCRIPTS[0]]
                           ["IMG_TO_CODE_PROMPT"],
        "h_code_to_alias": per_script[FROZEN_ROUTE_SCRIPTS[0]]
                           ["CODE_TO_ALIAS_PROMPT"],
        "d_image_to_alias": img_prompt,
        "hybrid_image_and_code": HYBRID_PROBE_PROMPT,
        "hybrid_prompt_is_not_part_of_the_frozen_route": (
            "the frozen route is g(X) then h(c) and h never sees an image, so "
            "the image-plus-code prompt belongs to the AUXILIARY probe only; "
            "it is reported here so a reader can see exactly what the probe "
            "presents, and no mediated condition may reference this key"),
        "read_from": list(FROZEN_ROUTE_SCRIPTS),
        "agreement_checked": True,
        "why_read_not_retyped": (
            "v1 retyped a paraphrase of the h prompt and so evaluated a "
            "different input than the route was trained on; reading the "
            "constant from the frozen scripts makes that class of error "
            "impossible rather than merely unlikely"),
    }


def frozen_route_protocol():
    """The training protocol every route adapter was trained under.

    Read from the matrix and SALMU scripts and required to agree, because
    training g_42 and g_123 under a different recipe than g_17 would make
    "router seed" confound the seed with the protocol.
    """
    want = ("ROUTE_STEPS", "ROUTE_WARMUP", "ROUTE_LR", "ROUTE_REPEAT")
    per_script = {f: _module_constants(f, want) for f in PROTOCOL_SCRIPTS}
    for name in want:
        values = {f: per_script[f][name] for f in PROTOCOL_SCRIPTS}
        if len(set(values.values())) != 1:
            raise RuntimeError(
                f"the frozen route scripts disagree on {name}: {values}; a new "
                f"router seed must be trained under the SAME protocol as the "
                f"existing one or the seed is confounded with the recipe")
    src = per_script[PROTOCOL_SCRIPTS[0]]
    return {
        "steps": src["ROUTE_STEPS"], "warmup": src["ROUTE_WARMUP"],
        "lr": src["ROUTE_LR"], "repeat": src["ROUTE_REPEAT"],
        "read_from": list(PROTOCOL_SCRIPTS), "agreement_checked": True,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05,
                 "scope": "S0 -- q,k,v,o projections of the language layers",
                 "source": "rv.create_adapter_model / rv.attach_lora"},
        "model": {"model_id": "Qwen/Qwen3.5-9B",
                  "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                  "dtype": "bfloat16",
                  "source": "rv.create_adapter_model"},
        "fresh_lora_init": (
            "gxm.fresh_reinit_lora(named_params, seed): lora_A kaiming_uniform "
            "(a=sqrt5), lora_B zeros, under the router seed.  PEFT 0.20 has no "
            "reset_adapter API, so a new seed is a genuine fresh init rather "
            "than a re-load"),
    }


#: What each condition actually EXECUTES.  ``image_to_h`` is the field that
#: keeps the mediator hard: it is False for every mediated condition, and the
#: one condition where an image and a code are presented together is marked
#: auxiliary and is reported as a bypass/conflict probe rather than as the
#: causal intervention.
CONDITIONS_V2 = OrderedDict((
    ("natural_mediated", {
        "execution": "c = g_s(X) from that router seed's own held-out "
                     "prediction file, then h_e(c)",
        "prompt_key": "h_code_to_alias",
        "image_to_h": False,
        "depends_on": ("router_seed", "edit_seed"),
        "expectation": "refusal_if_forgotten_else_image_alias",
        "decisive": False,
        "role": "the composition the edit was trained to break; it is the only "
                "condition that involves X at all, and it is what the E2E "
                "factorization predicts"}),
    ("forgotten_route_intervention", {
        "execution": "h_e(do(C=C_f)) -- a forgotten identity's code, no image",
        "prompt_key": "h_code_to_alias",
        "image_to_h": False,
        "depends_on": ("edit_seed",),
        "expectation": "refusal",
        "decisive": True,
        "role": "the mediator is set by hand to a code in F and h must "
                "refuse"}),
    ("retained_route_intervention", {
        "execution": "h_e(do(C=C_r)) -- a retained identity's code, no image",
        "prompt_key": "h_code_to_alias",
        "image_to_h": False,
        "depends_on": ("edit_seed",),
        "expectation": "follows_code",
        "decisive": True,
        "role": "the mediator is set by hand to a code outside F and h must "
                "produce that code's alias"}),
    ("retained_control", {
        "execution": "h_e(do(C=C_r)) -- IDENTICAL to the condition above",
        "prompt_key": "h_code_to_alias",
        "image_to_h": False,
        "depends_on": ("edit_seed",),
        "expectation": "follows_code",
        "decisive": False,
        "same_execution_as": "retained_route_intervention",
        "role": "with a hard mediator the retained control and the "
                "retained-route intervention are the SAME call: h receives a "
                "code and nothing else, so no image distinguishes them.  v1 "
                "counted them as two rows because it also passed an image; v2 "
                "evaluates the code once and reports it under both names, "
                "because counting one measurement twice is not more "
                "evidence"}),
    ("direct_control", {
        "execution": "D_s(X) -- image only, no code, separate adapter",
        "prompt_key": "d_image_to_alias",
        "image_to_h": False,
        "depends_on": ("direct_seed",),
        "expectation": "follows_image",
        "decisive": False,
        "role": "the unchanged X -> Y pathway.  It is NOT edited_h: an adapter "
                "trained on code-to-label pairs is not an image model, so v1's "
                "'reuses edited_h' measured nothing.  Expected to survive, "
                "because the claim is route-dependent forgetting and not global "
                "erasure"}),
    ("hybrid_conflict_probe", {
        "execution": "D_s(X, C_irrelevant) -- image AND a code, auxiliary",
        "prompt_key": "hybrid_image_and_code",
        "image_to_h": False,
        "depends_on": ("direct_seed",),
        "expectation": "follows_image",
        "decisive": False,
        "auxiliary": True,
        "not_the_mediator_intervention": (
            "presenting an image and a code together makes the system "
            "f(X, C), which is not the frozen architecture Y = h(C).  This "
            "condition is retained because it is a useful bypass/conflict "
            "probe -- it asks whether a code can capture a pathway that was "
            "never trained on codes -- and it is reported separately from "
            "every mediated gate.  v1 reported it AS the causal intervention, "
            "which is the defect this repairs"),
        "role": "auxiliary bypass/conflict probe only"}),
))

DECISIVE_CONDITIONS_V2 = tuple(n for n, c in CONDITIONS_V2.items()
                               if c.get("decisive"))
AUXILIARY_CONDITIONS_V2 = tuple(n for n, c in CONDITIONS_V2.items()
                                if c.get("auxiliary"))
MEDIATED_CONDITIONS_V2 = tuple(
    n for n, c in CONDITIONS_V2.items()
    if not c.get("auxiliary") and c["prompt_key"] == "h_code_to_alias")

#: Conditions whose input to h is ENTIRELY a hand-set code: do(C=c).  These are
#: the rows that may not carry an image at all, because with a hard mediator
#: there is nothing an image could be doing there.  ``natural_mediated`` is
#: mediated but NOT forced -- its image goes to g, which is the composition
#: being measured -- so it is excluded by the ``router_seed`` test rather than
#: by name, and adding another g-dependent condition later keeps it excluded.
FORCED_CODE_CONDITIONS_V2 = tuple(
    n for n, c in CONDITIONS_V2.items()
    if c["prompt_key"] == "h_code_to_alias"
    and "router_seed" not in c["depends_on"])

#: The one condition v3 adds, named for what it EXECUTES rather than for the
#: adapter it runs on: the adapter is an input, the condition is a statement
#: about what is presented to it.
BASELINE_CONDITION_V3 = "baseline_forced_code"

#: A strict SUPERSET of the v2 table.  Every v2 condition is carried unchanged,
#: so a row builder written against the v2 table produces the same row under v3
#: and the difference between the two designs is one added measurement rather
#: than a rewording of an existing one.  The v2 table itself is not extended:
#: it is embedded verbatim in the frozen v2 manifests, so adding an entry to it
#: would change what those files reconstruct to.
CONDITIONS_V3 = OrderedDict([
    *CONDITIONS_V2.items(),
    (BASELINE_CONDITION_V3, {
        "execution": "h_base(do(C=c)) -- the UNEDITED h, a code and no image",
        "prompt_key": "h_code_to_alias",
        "image_to_h": False,
        "depends_on": (),
        "expectation": "follows_code",
        "decisive": False,
        "evaluated_once_because": (
            "it depends on no edit seed, no router seed and no direct seed: "
            "h_base is one adapter, so one cell evaluates it.  Repeating the "
            "cell per edit seed would file three copies of one measurement and "
            "report them as three observations"),
        "role": "the BEFORE in before/after.  Every code, a forgotten one "
                "included, must produce its own alias here; that is what makes "
                "h_e(do(C=C_f)) = Unknown afterwards a CHANGE rather than "
                "something this h did all along"}),
])

FORCED_CODE_CONDITIONS_V3 = tuple(
    n for n, c in CONDITIONS_V3.items()
    if c["prompt_key"] == "h_code_to_alias"
    and "router_seed" not in c["depends_on"])
MEDIATED_CONDITIONS_V3 = tuple(
    n for n, c in CONDITIONS_V3.items()
    if not c.get("auxiliary") and c["prompt_key"] == "h_code_to_alias")
AUXILIARY_CONDITIONS_V3 = tuple(n for n, c in CONDITIONS_V3.items()
                                if c.get("auxiliary"))
DECISIVE_CONDITIONS_V3 = tuple(n for n, c in CONDITIONS_V3.items()
                               if c.get("decisive"))

#: The frozen g was trained under this seed, and its held-out predictions are
#: already committed in the route's own e2e_post cache.  That seed is READ
#: rather than retrained, because retraining it would replace the router the
#: route was established with.  The seed is a label; what actually identifies
#: the checkpoint is its digest, which RF0 records and verifies.
EXISTING_ROUTER_SEED = 17


def route_codes(man):
    """Every code in the route vocabulary, in sorted identity order."""
    return [man["code_of"][i] for i in sorted(man["identity_ids"])]


def router_prediction_path(dataset, router_seed):
    """Where one router seed's held-out predictions live.

    One file PER router seed is what makes the seed a factor rather than a
    label.  v1 declared three seeds and replayed a single cache for all of
    them, so "router seed 42" and "router seed 123" described the same
    predictions three times.
    """
    if router_seed == EXISTING_ROUTER_SEED:
        return DATASET_ROOT / G_CACHE_PATHS[dataset]
    return (DATASET_ROOT / CELLS_DIR_V2 / dataset
            / f"g_seed_{router_seed}" / "held_out_predictions.json")


def router_checkpoint_path(dataset, router_seed):
    if router_seed == EXISTING_ROUTER_SEED:
        return (DATASET_ROOT / ROUTE_DIRS[dataset] / "g_X_to_C"
                / Path(*ADAPTER_RELPATH))
    return (DATASET_ROOT / CELLS_DIR_V2 / dataset / f"g_seed_{router_seed}"
            / "adapter_final" / "adapter_model.safetensors")


def direct_checkpoint_path(dataset, direct_seed, namespace=None):
    """D_s: the direct X -> Y adapter for one direct seed.

    Its own directory and its own digest, never edited_h's: the direct pathway
    is a different model trained on different pairs, and sharing a checkpoint
    would make the direct control a measurement of the route.

    ``namespace`` keeps one design's D_s out of another design's directory.
    Defaulted to None so that the path every filed cell bound, and every frozen
    role table recorded, is the path this function still returns: a design that
    trains a direct adapter under a different configuration has to write it
    somewhere else, because ``direct_seed_17/`` holds frozen weights whose
    digest is recorded in a committed cell result, and overwriting them would
    leave a report describing an adapter nobody can inspect.
    """
    name = f"direct_seed_{direct_seed}"
    if namespace:
        name = f"{name}__{namespace}"
    return (DATASET_ROOT / CELLS_DIR_V2 / dataset / name
            / "adapter_final" / "adapter_model.safetensors")


def held_out_uris(man):
    """Every held-out image URI, in the order the design uses.

    Passed around instead of the manifest wherever only the images are wanted:
    a phase that re-reads the manifest mid-run could disagree with the manifest
    the design was frozen from, and the frozen pilot already names these URIs.
    """
    images = held_out_images(man)
    return [str(it["image_uri"]) for iid in sorted(images)
            for it in images[iid]]


def load_router_predictions(dataset, router_seed, held_out):
    """Per-image held-out code predictions for one router seed.

    Keyed by image URI so a prediction can never be attached to the wrong
    image.  ``held_out`` is the list of URIs the design says must be covered --
    taken from the FROZEN pilot where there is one, so the aggregate cannot
    quietly describe a different set of images than the design was built over.

    Reads the route's own committed cache for the existing seed and RF1G's
    output for a trained one, and refuses if a held-out image has no prediction:
    an image silently missing from the confusion matrix would make the
    factorization agree with itself over the rows that happened to exist.
    """
    path = router_prediction_path(dataset, router_seed)
    if not path.is_file():
        raise RuntimeError(
            f"router seed {router_seed} has no held-out prediction file at "
            f"{path}; train it with --phase RF1G --router-seed {router_seed}.  "
            f"A router seed without its own predictions is a label, not a "
            f"factor, and composing with another seed's predictions would "
            f"report that other router's accuracy under this one's name")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc["rows"] if isinstance(doc, dict) and "rows" in doc else doc
    by_uri = {}
    for r in rows:
        uri = r.get("image_uri")
        if uri is None:
            raise RuntimeError(f"{path}: a prediction row has no image_uri, so "
                               f"it cannot be attached to an image")
        by_uri[str(uri)] = {"pred_code": r.get("pred_code"),
                            "g_raw_text": r.get("g_raw_text"),
                            "code_correct": r.get("code_correct"),
                            "identity_id": r.get("identity_id"),
                            "split": r.get("split")}
    missing = [u for u in held_out if str(u) not in by_uri]
    if missing:
        raise RuntimeError(
            f"router seed {router_seed}'s prediction file has no row for "
            f"{len(missing)} of the held-out images (e.g. {missing[0]}); the "
            f"confusion matrix would be built over a subset and the E2E "
            f"prediction would describe that subset rather than the design")
    return {"path": str(path), "sha256": sha256_file(path),
            "router_seed": router_seed, "by_uri": by_uri,
            "n_held_out": len(list(held_out))}


def _expected_label(rule, row, man, condition):
    """The label a row REQUIRES, from the rule its condition names.

    One implementation of the rules whichever condition table the row was built
    from.  Two copies would be two places for the same rule to come to mean
    something different, and the rule is what makes a row's ``correct`` field a
    statement about the intervention rather than about whichever code happens to
    be sitting on the row.
    """
    alias_of = man["alias_of"]
    if rule == "refusal":
        return DELETED_LABEL
    if rule == "follows_code":
        code_id = row["forced_identity_id"]
        if code_id is None:
            raise RuntimeError(
                f"condition {condition!r} forces a code but the row carries no "
                f"forced_identity_id; there is no label to require")
        return alias_of[code_id]
    if rule == "follows_image":
        return alias_of[row["image_identity_id"]]
    if rule == "refusal_if_forgotten_else_image_alias":
        return (DELETED_LABEL if row["image_is_forgotten"]
                else alias_of[row["image_identity_id"]])
    raise RuntimeError(f"condition {condition!r} has unknown expectation rule "
                       f"{rule!r}")


def expected_outcome_v2(condition, row, man, conditions=None):
    """The label the design REQUIRES, derived from what is actually executed.

    ``natural_mediated`` is the only condition whose expectation depends on the
    image, because it is the only one where the image reaches the system at all
    -- and it reaches g, not h.

    ``conditions`` defaults to the v2 table.  v3 passes its own, which carries
    one more entry: a condition's requirement is read from the table the design
    declares, so a row cannot end up requiring a label that some other version
    of the design would have asked for.
    """
    table = CONDITIONS_V2 if conditions is None else conditions
    return _expected_label(table[condition]["expectation"], row, man, condition)


def _forced_code_rows(man, forget_ids, cell_id, edit_seed, conditions,
                      condition_of):
    """One row per route code, for a cell whose input to h is entirely a code.

    Shared by the post-edit intervention cells and the pre-edit baseline cell:
    both hand h one code per identity and nothing else, so they differ only in
    WHICH h answers and in what each arm requires of it.  Two copies of this
    loop would be two places for the row shape to drift, and the row shape is
    what the coverage audit, the scorer and the hard-response matrix all read.

    ``condition_of(arm)`` maps a row's arm to the condition it is filed under.
    An intervention cell maps each arm to itself, so a retained row doubles as
    ``retained_control``; the baseline cell maps both arms to one condition,
    because the unedited h is not an arm of the edit and reporting its rows
    under the post-edit names would be a claim about an adapter nobody edited.
    """
    forget = set(forget_ids)
    images = held_out_images(man)
    rows = []
    for iid in sorted(man["identity_ids"]):
        code = man["code_of"][iid]
        arm = ("forgotten_route_intervention" if iid in forget
               else "retained_route_intervention")
        condition = condition_of(arm)
        spec = conditions[condition]
        row = {
            "row_id": f"{cell_id}::do_C_{iid}",
            "cell_id": cell_id,
            "condition": condition,
            "edit_seed": edit_seed,
            "execution": spec["execution"],
            "forced_identity_id": iid,
            "forced_code": code,
            "forced_label": man["alias_of"][iid],
            "arm": "forgotten" if iid in forget else "retained",
            "prompt_key": spec["prompt_key"],
            "image_to_h": False,
            "image_uri": None,
            "image_identity_id": None,
            "held_out_images_of_this_identity":
                [it["image_uri"] for it in images.get(iid, [])],
            "cluster_id": iid,
            "identity_cluster_id": iid,
            "image_cluster_id": None,
            "decisive": spec["decisive"],
        }
        # retained_control is the SAME measurement, reported under a second name
        # -- but only where the row IS the retained-route intervention.
        row["also_reported_as"] = (
            ("retained_control",)
            if condition == arm == "retained_route_intervention" else ())
        row["expected_label"] = expected_outcome_v2(condition, row, man,
                                                    conditions)
        rows.append(row)
    return rows


def build_intervention_rows_v2(man, forget_ids, cell_id, edit_seed):
    """One row per route code, evaluated ONCE per edited h.

    With a hard mediator h receives a code and nothing else, so a row is fully
    specified by the code: there is no image to vary and no second axis to
    cross.  That collapses what v1 counted as 42 rows per cell into one row per
    code, and the collapse is the point -- v1's extra rows were the same call
    repeated under different images that were never sent to h.

    The identity a code belongs to is recorded, because it decides which arm of
    Delta_route the row falls in, and so are the held-out images of that
    identity, so a reader can see exactly what v1 presented and why presenting
    it made the system f(X, C) instead of h(C).
    """
    return _forced_code_rows(man, forget_ids, cell_id, edit_seed, CONDITIONS_V2,
                             lambda arm: arm)


def build_baseline_rows_v3(man, forget_ids, cell_id):
    """h_base(do(C=c)) for every code: the BEFORE in before/after.

    v2 declared a ``baseline_h`` checkpoint role, described it in its own words
    as "the before in before/after", and never evaluated it.  So the runner
    measured ``h_e(do(C=C_f)) = Unknown`` without ever establishing that the
    same code produced its own alias before the edit, and an h that answered
    Unknown to every code -- edited or not, forgotten or retained -- would have
    satisfied every suppression gate v2 computed.

    Every code is required to produce ITS OWN alias here, the forgotten ones
    included: that is the whole content of the measurement, and it is why the
    expectation rule is ``follows_code`` rather than the refusal rule the
    post-edit forgotten arm uses.

    ``edit_seed`` is None because there is no edit, and the cell is evaluated
    once rather than once per edit seed for the reason an intervention is never
    repeated per router seed: it depends on neither factor, so three copies
    would be one measurement reported three times.
    """
    return _forced_code_rows(man, forget_ids, cell_id, None, CONDITIONS_V3,
                             lambda arm: BASELINE_CONDITION_V3)


def _image_identity(uri, item, image_sha_by_uri):
    """(content digest, image-level cluster key) for one held-out image.

    The image-level cluster is the image's BYTES, not its filename.  PPUBench
    has twelve held-out filenames over four distinct byte-values, so clustering
    by name would resample one picture three times as three independent units
    and understate every interval -- which is exactly the defect the
    image-content audit exists to expose.  Where no digest is available the URI
    stands in and the pre-registration's cluster-count check refuses the design,
    because a cluster key that silently means two different things cannot be
    checked against the audit.
    """
    sha = (image_sha_by_uri or {}).get(str(uri)) or item.get("image_sha256")
    return sha, (sha or str(uri))


def build_natural_rows_v2(man, forget_ids, preds, cell_id, router_seed,
                          edit_seed, image_sha_by_uri=None):
    """c = g_s(X), then h_e(c).  One row per held-out image.

    ``preds`` may be None, which is how the design is FROZEN: which images are
    run and what each one requires are fixed before any router exists, while the
    code the router actually emits is an observation that RF1E fills in from
    that seed's own prediction file.  Freezing the observation too would mean
    pre-registering a result.

    An image whose code does not parse is marked unroutable rather than dropped:
    it is a routing failure and the factorization has to account for it, because
    dropping it would make the composition look better than the router is.
    """
    forget = set(forget_ids)
    images = held_out_images(man)
    rows = []
    for iid in sorted(images):
        for it in images[iid]:
            uri = str(it["image_uri"])
            got = preds["by_uri"][uri] if preds is not None else {}
            sha, img_cluster = _image_identity(uri, it, image_sha_by_uri)
            row = {
                "row_id": f"{cell_id}::nat::{iid}::{uri.rsplit('/', 1)[-1]}",
                "cell_id": cell_id,
                "condition": "natural_mediated",
                "router_seed": router_seed,
                "edit_seed": edit_seed,
                "execution": CONDITIONS_V2["natural_mediated"]["execution"],
                "image_identity_id": iid,
                "image_is_forgotten": iid in forget,
                "image_uri": uri,
                "image_sha256": sha,
                "split": it.get("split", "test"),
                "routed_code": got.get("pred_code"),
                "routed_code_raw": got.get("g_raw_text"),
                "routed_code_correct": got.get("code_correct"),
                "routable": (None if preds is None
                             else got.get("pred_code") is not None),
                "observation_pending": preds is None,
                "filled_by": (None if preds is not None else
                              f"RF1E, from router seed {router_seed}'s own "
                              f"held-out prediction file"),
                "code_of_image_identity": man["code_of"][iid],
                "prompt_key": CONDITIONS_V2["natural_mediated"]["prompt_key"],
                "image_to_h": False,
                "forced_identity_id": None,
                "cluster_id": iid,
                "identity_cluster_id": iid,
                "image_cluster_id": img_cluster,
                "decisive": False,
                "also_reported_as": (),
            }
            row["expected_label"] = expected_outcome_v2("natural_mediated",
                                                        row, man)
            rows.append(row)
    return rows


def build_direct_rows_v2(man, cell_id, direct_seed, hybrid=False,
                         image_sha_by_uri=None, images=None):
    """D_s(X), image only -- and the auxiliary hybrid probe.

    The direct control presents NO code, so it cannot follow one; the hybrid
    probe presents an irrelevant code and asks whether the image still wins.
    They are separate conditions with separate gates, because the first measures
    whether the direct pathway survived and the second measures whether a code
    can capture a pathway that was never trained on codes.
    """
    condition = "hybrid_conflict_probe" if hybrid else "direct_control"
    spec = CONDITIONS_V2[condition]
    # Defaulted to the held-out images, which is the only set v1-v4 ever asked
    # for, so their rows are the rows they have always been.  A caller that
    # passes images is the calibration, which scores the SAME condition over the
    # development split so that its metric is the confirmatory gate's metric
    # rather than a second statistic that happens to share a name.
    images = images if images is not None else held_out_images(man)
    ids = sorted(man["identity_ids"])
    rows = []
    for iid in sorted(images):
        for it in images[iid]:
            uri = str(it["image_uri"])
            irrelevant = min(j for j in ids if j != iid) if hybrid else None
            sha, img_cluster = _image_identity(uri, it, image_sha_by_uri)
            row = {
                # The identity is part of the row_id because a FILENAME is not
                # unique across identities: two people can each have a
                # ``test_0.png``, and a row_id collision would silently merge
                # two observations into one.
                "row_id": (f"{cell_id}::{'hyb' if hybrid else 'dir'}::{iid}::"
                           f"{uri.rsplit('/', 1)[-1]}"),
                "cell_id": cell_id,
                "condition": condition,
                "direct_seed": direct_seed,
                "execution": spec["execution"],
                "auxiliary": bool(spec.get("auxiliary", False)),
                "image_identity_id": iid,
                "image_is_forgotten": False,
                "image_uri": uri,
                "image_sha256": sha,
                "split": it.get("split", "test"),
                "forced_identity_id": irrelevant,
                "forced_code": (man["code_of"][irrelevant] if irrelevant
                                else None),
                "forced_label": (man["alias_of"][irrelevant] if irrelevant
                                 else None),
                "prompt_key": spec["prompt_key"],
                "image_to_h": False,
                "routed_code": None,
                "cluster_id": iid,
                "identity_cluster_id": iid,
                "image_cluster_id": img_cluster,
                "decisive": False,
                "also_reported_as": (),
            }
            row["image_is_forgotten"] = iid in set(
                man.get("forget_identity_ids") or [])
            row["expected_label"] = expected_outcome_v2(condition, row, man)
            rows.append(row)
    return rows


#: What an h response that named no candidate label counts as in H_e.  It is a
#: key of its own rather than a zero row: an unparseable response means the
#: composition produces NO label, which is different from producing a wrong one,
#: and folding it into the wrong-label columns would hide a parse failure inside
#: an accuracy figure.
UNPARSEABLE = "<unparseable>"

#: Which denominator a cell kind must meet.  ``baseline`` joins ``intervention``
#: because it is the same shape of measurement -- one row per route code, h
#: given a code and nothing else -- and differs only in WHICH h answers.  A kind
#: that fell through to the image branch would be audited against the wrong
#: denominator and could report exact coverage over rows it never checked.
CODE_CELL_KINDS = ("intervention", "baseline")
IMAGE_CELL_KINDS = ("natural", "direct")


def coverage_audit_v2(rows, man, forget_ids, kind, images=None,
                      forced=FORCED_CODE_CONDITIONS_V2):
    """Exact denominators for one cell, plus the invariant that keeps the
    mediator hard.

    A ``CODE_CELL_KINDS`` cell carries one row per route code; an
    ``IMAGE_CELL_KINDS`` cell carries one row per held-out image.  A zero-row
    cell returns the full key set with defects rather than an empty dict, which
    would serialize as universal compliance.

    ``forced`` is the set of conditions whose input to h is entirely a hand-set
    code and which therefore may not carry an image at all.  It is a parameter
    because v3 adds one such condition: a design must be audited against the
    table it was built from, or the audit checks a rule the design never stated
    and passes cells the design would have refused.
    """
    images = images if images is not None else held_out_images(man)
    ids = sorted(man["identity_ids"])
    forget = set(forget_ids)
    retained = sorted(set(ids) - forget)
    n_held_out = sum(len(v) for v in images.values())
    defects = []

    if kind in CODE_CELL_KINDS:
        required = {"n_rows": len(ids), "n_forgotten_codes": len(forget),
                    "n_retained_codes": len(retained)}
    elif kind in IMAGE_CELL_KINDS:
        required = {"n_rows": n_held_out, "n_identities": len(ids)}
    else:
        raise RuntimeError(f"unknown cell kind {kind!r}; expected one of "
                           f"{CODE_CELL_KINDS + IMAGE_CELL_KINDS}")

    if not rows:
        return {"exact": False, "kind": kind, "n_rows": 0,
                "required": required,
                "defects": [(f"ZERO rows for a {kind} cell, so no gate has a "
                             f"denominator; required {required}")]}

    got = {"n_rows": len(rows)}
    if kind in CODE_CELL_KINDS:
        got["n_forgotten_codes"] = sum(1 for r in rows
                                       if r["arm"] == "forgotten")
        got["n_retained_codes"] = sum(1 for r in rows if r["arm"] == "retained")
        codes = [r["forced_code"] for r in rows]
        if sorted(codes) != sorted(route_codes(man)):
            defects.append(
                f"the codes intervened on are not the route vocabulary: got "
                f"{len(set(codes))} distinct of {len(codes)} rows, expected "
                f"{len(ids)} distinct codes")
        if len(set(codes)) != len(codes):
            defects.append(
                f"{len(codes) - len(set(codes))} code(s) evaluated more than "
                f"once in one cell; with a hard mediator h(c) is the same call "
                f"every time, so a repeat is a duplicated measurement and not "
                f"additional evidence")
    else:
        got["n_identities"] = len({r["image_identity_id"] for r in rows})
        uris = [r["image_uri"] for r in rows]
        if len(set(uris)) != len(uris):
            defects.append(f"{len(uris) - len(set(uris))} image(s) appear more "
                           f"than once in one {kind} cell")

    for key, want in required.items():
        if got.get(key) != want:
            defects.append(f"{kind} cell has {got.get(key)} {key}, the design "
                           f"requires {want}")

    # THE invariant item 1 exists to protect: no row may send an image to h, and
    # a row whose h input is entirely a hand-set code must not carry an image at
    # all -- there is nothing an image could legitimately be doing there.
    # ``natural_mediated`` is deliberately excluded from the second check: its
    # image goes to g, which IS the composition being measured, so the row is
    # properly keyed by that image.  Applying the check to it would forbid the
    # only condition where X reaches the system at all.
    for r in rows:
        if r.get("image_to_h"):
            defects.append(
                f"row {r['row_id']} sends an image to h, which makes the system "
                f"f(X, C) rather than the frozen Y = h(C)")
            break
    for r in rows:
        if r["condition"] in forced and r.get("image_uri"):
            defects.append(
                f"row {r['row_id']} forces a code but carries an image_uri; "
                f"h receives the code and nothing else, so the image could only "
                f"be there to be presented, and presenting it makes the system "
                f"f(X, C).  Record the identity's held-out images as metadata "
                f"instead")
            break

    ids_seen = {r["row_id"] for r in rows}
    if len(ids_seen) != len(rows):
        defects.append(f"{len(rows) - len(ids_seen)} duplicate row_id(s)")

    return {"exact": not defects, "kind": kind, "n_rows": len(rows),
            "required": required, "got": got, "defects": defects,
            "mediator_is_hard": not any(r.get("image_to_h") for r in rows)}


def hard_response_matrix(rows, vocab, conditions=FORCED_CODE_CONDITIONS_V3):
    """H(c, y) = 1{parse(h(c)) = y} -- the empirical hard response.

    This replaces candidate score sums as P_h in the factorization.  A score sum
    over teacher-forced full sequences has no termination event and no
    normalization across candidates, so its terms overlap and it is not the
    probability of anything; putting one into
    sum_c P_g(c|X) P_h(Y*(X)|c) makes the prediction a function of a quantity
    that was never a probability.  The hard response is what the system actually
    outputs, and the factorization is a claim about what the system does.

    ``conditions`` says which rows are a forced-code measurement of an h.  It
    defaults to the widest set this runner declares, so the post-edit
    intervention cells and the pre-edit baseline cell build their matrices
    through the same function and the two are comparable cell for cell by
    construction.  Filtering on a hardcoded pair of names instead would silently
    drop every baseline row and then report an empty matrix as a finding.
    """
    if not rows:
        raise RuntimeError("H cannot be built from zero rows; an empty matrix "
                           "would predict 0 for every image and read as a "
                           "finding about the edit")
    H = {}
    for r in rows:
        if r["condition"] not in conditions:
            continue
        code = r["forced_code"]
        if code in H:
            raise RuntimeError(
                f"code {code} appears twice in the forced-code rows; h(c) is "
                f"one call, so a duplicate is a contradiction rather than a "
                f"replicate")
        row = {y: 0 for y in vocab}
        if r.get("unparseable") or not r.get("usable", True):
            row[UNPARSEABLE] = 1
        else:
            label = r.get("parsed_label")
            if label not in row:
                raise RuntimeError(
                    f"h({code}) parsed to {label!r}, which is not in the "
                    f"candidate vocabulary {sorted(vocab)}; the parse and the "
                    f"vocab must agree before a hard response can be recorded")
            row[label] = 1
        H[code] = row
    if not H:
        raise RuntimeError(
            f"no row in these {len(rows)} rows is one of the forced-code "
            f"conditions {list(conditions)}, so there is nothing to build H "
            f"from; a matrix built from the wrong rows would be a matrix of the "
            f"wrong map")
    return {"matrix": H, "n_codes": len(H), "vocab": sorted(vocab),
            "unparseable_key": UNPARSEABLE,
            "definition": "H_e(c, y) = 1{parse(h_e(c)) = y}",
            "why_not_score_sums": (
                "candidate score sums are sums of teacher-forced full-sequence "
                "scores with no termination event and no cross-candidate "
                "normalization; their terms overlap and they are not "
                "probabilities, so they cannot be P_h in a factorization")}


def router_confusion(natural_rows):
    """P_g(c|X) as an empirical HARD distribution over held-out images.

    Greedy decoding gives one code per image, so each row is a 0/1 vector.  An
    image whose code did not parse contributes to no column and is counted
    separately: folding it into a wrong code would invent a routing decision the
    router never made, and dropping it silently would make the confusion matrix
    describe a subset.
    """
    if not natural_rows:
        raise RuntimeError("the router confusion matrix cannot be built from "
                           "zero natural rows")
    per_image = {}
    n_unroutable = 0
    for r in natural_rows:
        uri = r["image_uri"]
        if uri in per_image:
            raise RuntimeError(f"image {uri} appears twice in the natural rows")
        code = r.get("routed_code")
        if code is None:
            n_unroutable += 1
            per_image[uri] = None
        else:
            per_image[uri] = code
    codes = sorted({c for c in per_image.values() if c is not None})
    return {"per_image_code": per_image, "codes": codes,
            "n_images": len(per_image), "n_unroutable": n_unroutable,
            "unroutable_counted_not_dropped": (
                "an image whose code did not parse contributes to no column of "
                "the confusion matrix and is counted here; it is a routing "
                "failure and the factorization must account for it"),
            "hard_not_smoothed": (
                "greedy decoding yields one code per image, so P_g(c|X) is 0/1 "
                "rather than a distribution; a smoothed version would need "
                "repeated generations under a preregistered sampling "
                "configuration, which this design does not use")}


def predicted_e2e_from_empirical(natural_rows, H, confusion):
    """sum_c P_g(c|X) H_e(c, Y*(X)) over the held-out images.

    With a hard router and a hard response this reduces to H_e(g_s(X), Y*(X))
    per image, which is exactly the composition the design runs.  Under greedy
    decoding the two therefore coincide WHENEVER h_e is a function of the code
    alone -- so this is not a tautology but a test of the hard mediator itself:

      * if h also saw the image, its output would depend on X as well as on c,
        and the same code reaching it from two different images would give two
        different answers, so the composition would NOT factorize;
      * if the prediction file RF1E replayed is not the one the confusion matrix
        was built from, the two sides describe different routers;
      * if the parse is inconsistent between the per-code and the composed call,
        one side moves and the other does not.

    A disagreement is therefore evidence about the architecture, which is the
    thing item 1 asks to be preserved, and agreement is evidence that h really
    did receive only a code.

    The required label is keyed by the IMAGE.  Keying it by the code would make
    every routing mistake correct by construction and predict 1.0 for any router
    whatsoever, which is the quantity the decomposition exists to explain.
    """
    if not natural_rows:
        raise RuntimeError("the E2E prediction has no denominator: zero natural "
                           "rows")
    per_image = []
    n_correct = 0
    for r in natural_rows:
        code = r.get("routed_code")
        want = r["expected_label"]
        if code is None:
            pred = 0
            why = "the router produced no parseable code, so the composition " \
                  "yields no label"
        elif code not in H["matrix"]:
            raise RuntimeError(
                f"the router sent image {r['image_uri']} to code {code!r}, "
                f"which H_e has no row for; H_e must cover every code the "
                f"router can emit or the prediction silently omits it")
        else:
            pred = H["matrix"][code].get(want, 0)
            why = f"H_e({code}, {want})"
        n_correct += pred
        per_image.append({"image_uri": r["image_uri"],
                          "identity_id": r["image_identity_id"],
                          "routed_code": code, "required_label": want,
                          "predicted_correct": pred, "why": why})
    n = len(natural_rows)
    return {"predicted": n_correct / n, "n_images": n,
            "n_predicted_correct": n_correct,
            "n_unroutable": confusion["n_unroutable"],
            "formula": "sum_c P_g(c|X) * H_e(c, Y*(X))",
            "h_is_the_empirical_hard_response": True,
            "required_label_is_keyed_by": "the IMAGE, not the code",
            "per_image": per_image}


DELTA_ROUTE_V2 = {
    "formula": ("Delta_route = P(Unknown | do(C in F)) - "
                "P(Unknown | do(C not in F))"),
    "unit_of_analysis": (
        "the CODE, not the image.  With a hard mediator the intervention is "
        "h_e(do(C=c)) and nothing else varies, so one code is one observation "
        "and there is no image axis to average over"),
    "numerator_arms": ("forgotten_route_intervention",
                       "retained_route_intervention"),
    "excluded_and_why": {
        "natural_mediated": ("no code is forced; the route is g's own "
                             "prediction, so it belongs to the composition and "
                             "not to the intervention"),
        "direct_control": ("presents no code at all -- that is what makes it a "
                           "control on the X -> Y pathway"),
        "hybrid_conflict_probe": ("presents an image AND a code, so it is "
                                  "f(X, C) and not the mediator; it is "
                                  "auxiliary and is reported separately"),
        "retained_control": ("the same rows as retained_route_intervention, so "
                             "including it would count one measurement twice"),
    },
}


def delta_route_v2(scored_intervention_rows, forget_ids):
    """Delta_route over the code-level intervention rows.

    Reports each arm's denominator, because with a single-target forget set the
    forgotten arm has exactly ONE code and no interval can be built on it.  That
    is a property of the design and is stated up front rather than discovered
    when the bootstrap refuses.
    """
    rows = [r for r in scored_intervention_rows
            if r["condition"] in DELTA_ROUTE_V2["numerator_arms"]]
    if not rows:
        raise RuntimeError("Delta_route has no denominator: zero forced-code "
                           "intervention rows")
    arms = {}
    for arm, cond in (("forgotten", "forgotten_route_intervention"),
                      ("retained", "retained_route_intervention")):
        sub = [r for r in rows if r["condition"] == cond]
        refusals = sum(1 for r in sub
                       if r.get("parsed_label") == DELETED_LABEL)
        arms[arm] = {"condition": cond, "n_codes": len(sub),
                     "n_refusals": refusals,
                     "rate": (refusals / len(sub)) if sub else None,
                     "codes": sorted(r["forced_code"] for r in sub)}
    if arms["forgotten"]["n_codes"] == 0 or arms["retained"]["n_codes"] == 0:
        raise RuntimeError(
            f"Delta_route needs codes on both arms; got "
            f"{arms['forgotten']['n_codes']} forgotten and "
            f"{arms['retained']['n_codes']} retained")
    delta = arms["forgotten"]["rate"] - arms["retained"]["rate"]
    return {
        **DELTA_ROUTE_V2,
        "delta_route": delta,
        "arms": arms,
        "interval_possible_on_forgotten_arm":
            arms["forgotten"]["n_codes"] >= 2,
        "interval_possible_on_retained_arm":
            arms["retained"]["n_codes"] >= 2,
        "single_target_limitation": (
            None if arms["forgotten"]["n_codes"] >= 2 else
            f"a single-target forget set puts exactly "
            f"{arms['forgotten']['n_codes']} code on the forgotten arm, so "
            f"P(Unknown | do(C in F)) is one observation and no cluster "
            f"bootstrap can produce an interval on it.  Delta_route is reported "
            f"as a point estimate with its denominators stated; an interval on "
            f"the forgotten arm needs a simultaneous forget set with at least "
            f"two target codes"),
        "rows_are_codes_not_images": (
            "v1 bootstrapped Delta_route over image-clustered rows, but a "
            "forced-code row has no image: h receives the code and nothing else, "
            "so the resampling unit is the code"),
    }


#: Item 6.  Identity level is PRIMARY and image level is a SENSITIVITY analysis.
#:
#: v1 selected image level on SALMU as "a real refinement over identity-level".
#: It is not a refinement: three images of one person share an identity, share
#: the router that was trained on that person, and share the edited h, so they
#: are not independent observations and resampling them as if they were produces
#: a narrower interval than the data supports.
BOOTSTRAP_PLAN_V2 = {
    "primary": {
        "level": "identity",
        "why": ("images of one person share an identity, the router's training "
                "and the edited h, so they are not independent; identity is the "
                "unit the design actually randomizes over"),
    },
    "sensitivity": {
        "level": "image",
        "why": ("reported alongside the primary interval, never instead of it: "
                "if the two disagree the identity-level interval is the one to "
                "believe, because it is the one whose units are independent"),
    },
    "repeated_intervention_rows_are_never_independent": (
        "every row of one cell shares a router and an edited h, and every row of "
        "one code shares the same h(c) call, so no bootstrap resamples rows"),
    "n_resamples": 2000,
    "seed": 17,
    "alpha": 0.05,
    "method": "percentile bootstrap over resampled clusters",
    "one_cluster_is_refused_not_reported": (
        "a bootstrap over a single cluster restates the point estimate as an "
        "interval of width zero"),
}


def bootstrap_plan_for(audit, dataset):
    """The plan with this dataset's cluster counts filled in, held-out scoped.

    Training images are not clusters of anything the bootstrap resamples, so the
    counts come from the held-out scope only.
    """
    scope = gate_applicability(audit)["clustering"]["held_out_scope"]
    counts = {"identity": scope["n_identity_clusters"],
              "image": scope["n_image_clusters"]}
    # Derived from the content audit rather than from the dataset's name:
    # "exploratory" is a statement about whether the held-out images are held
    # out in CONTENT, and a dataset whose test split stopped repeating its
    # training bytes stops being exploratory without anyone editing this
    # function.  Naming the dataset here would leave a branch that keeps saying
    # "exploratory" after the fact that made it exploratory had gone.
    held_out_content = bool(audit["held_out_image_content_exists"])
    plan = {
        **BOOTSTRAP_PLAN_V2,
        "dataset": dataset,
        "cluster_counts_held_out": counts,
        "primary_n_clusters": counts["identity"],
        "sensitivity_n_clusters": counts["image"],
        "image_level_is_degenerate": scope["image_level_is_degenerate"],
        "exploratory": not held_out_content,
    }
    if not held_out_content:
        plan["exploratory_because"] = (
            f"{counts['identity']} identity clusters, and every held-out image's "
            f"bytes also appear in the train split, so any interval computed "
            f"here is exploratory rather than a held-out claim")
    if scope["image_level_is_degenerate"]:
        plan["sensitivity_is_not_a_refinement_here"] = (
            "every identity has a single distinct held-out image, so image "
            "clusters ARE identity clusters and the sensitivity analysis "
            "restates the primary one")
    for name in ("primary", "sensitivity"):
        if plan[f"{name}_n_clusters"] < 2:
            raise RuntimeError(
                f"the {dataset} pilot has {plan[f'{name}_n_clusters']} "
                f"{name} cluster(s) at the {plan[name]['level']} level; one "
                f"cluster cannot produce an interval, so this dataset cannot "
                f"support a {name} bootstrap and must not be frozen as if it "
                f"could")
    return plan


# ---------------------------------------------------------------------------
# v2 gates and the three separate verdicts
# ---------------------------------------------------------------------------

GATE_THRESHOLDS_V2 = OrderedDict((
    ("min_router_held_out_accuracy", 0.90),
    ("forgotten_route_suppression", 1.0),
    ("retained_route_accuracy", 1.0),
    ("min_intervention_following_accuracy", 0.99),
    ("min_direct_image_accuracy", 0.90),
    ("max_direct_code_following_rate", 0.10),
    ("max_unparseable_outputs", 0),
    ("max_multi_label_outputs", 0),
    ("max_abs_e2e_prediction_error", 0.02),
))

#: Which verdict each gate feeds.  There is deliberately no gate that feeds all
#: three, and no aggregate that combines them: item 5 exists because one
#: omnibus boolean let a routing failure cancel a mediation success.
GATE_TO_VERDICT = OrderedDict((
    ("router_held_out_accuracy", "routing_reliability"),
    ("conditional_forgotten_route_suppression", "mediation"),
    ("conditional_retained_route_accuracy", "mediation"),
    ("mediator_intervention_following_accuracy", "mediation"),
    ("direct_image_accuracy", "mediation"),
    ("direct_code_following_rate", "mediation"),
    ("no_unparseable_or_multi_label_outputs", "mediation"),
    ("e2e_matches_the_routing_composition", "routing_factorization"),
))

VERDICT_NAMES = ("mediation", "routing_reliability", "routing_factorization")

NOT_ESTABLISHED = "not_established"

#: v3 adds one threshold, for the one gate v2 had no way to compute: that the
#: forgotten codes produced their own aliases BEFORE the edit.  The floor is 1.0
#: to match ``forgotten_route_suppression`` exactly, because the two are the
#: paired halves of one claim -- every forgotten code gave its alias before, and
#: gives Unknown after -- and a floor looser on the before than on the after
#: would let a partial pre-edit failure count as an edit-induced change.
GATE_THRESHOLDS_V3 = OrderedDict((
    *GATE_THRESHOLDS_V2.items(),
    ("min_baseline_forgotten_route_following", 1.0),
))

GATE_TO_VERDICT_V3 = OrderedDict((
    *GATE_TO_VERDICT.items(),
    ("baseline_forgotten_route_following", "mediation"),
))


#: The two denominators a hygiene gate can have, named so a spec picks one
#: rather than a reader inferring it from the aggregation code.  Defined here,
#: above the spec table, because a spec field's value has to exist before the
#: spec does.  See PART III for the version that uses the second one.
HYGIENE_SCOPE_ALL_ROWS = "all_rows"
HYGIENE_SCOPE_NON_AUXILIARY = "non_auxiliary_rows"


class PilotSpec(NamedTuple):
    """Every way one pilot version differs from another, named in one place.

    Both versions share ONE construction path and differ only through the
    fields below.  A difference written as ``if version == "v3"`` inside six
    functions is a difference nobody can enumerate and nobody can test as a
    whole; a difference written as a field is one a reader can list and a test
    can pin.  It also keeps the two from drifting apart silently, which is what
    happened to the GPU and CPU scoring paths before they were given one scorer.

    ``embed_live_checkpoint_status`` is the field the v2/v3 split exists for.
    v2 answered True, so ``design_sha256`` covered WHICH ADAPTERS WERE ON DISK:
    training the one role a pilot was waiting for changed what the constructor
    produced, the pre-registration stopped reproducing at the exact moment it
    became runnable, and every later phase that verifies before loading refused
    to load it.  A design may bind the bytes it is a statement ABOUT; it may not
    bind the state of work it has not done yet.
    """

    version: str
    kind: str
    prereg_kind: str
    result_kind: str
    supersession_kind: str
    conditions: object
    forced_code_conditions: tuple
    mediated_conditions: tuple
    auxiliary_conditions: tuple
    decisive_conditions: tuple
    thresholds: object
    gate_to_verdict: object
    baseline_cell: bool
    embed_live_checkpoint_status: bool
    direct_gate_aggregation: str
    declared_gate_aggregation: object
    #: Which rows the verdict-bearing hygiene count covers.  A denominator is
    #: half a rule, so it is a field of the design rather than a choice the
    #: aggregation code makes once the rows already exist.
    hygiene_gate_row_scope: str
    #: What the design SAYS about that scope, merged into ``frozen_gates``
    #: beside the declared aggregation.  Empty for a version whose bytes are
    #: frozen and which therefore cannot gain a field.
    declared_hygiene_scope: object
    #: What the design says about consuming cells another version filed.  Empty
    #: for a version that never reads another design's cells.
    declared_cell_compatibility: object
    superseded_filenames: tuple
    #: Whether this version calibrates anything before it confirms it.  A
    #: trailing default so that the specs already frozen -- whose bytes are the
    #: reason ``design_sha256`` still reproduces -- construct exactly as they
    #: always did and answer False.  The alternative is a ``version == "v5"``
    #: test inside ``_phases_for``, which is the thing this NamedTuple's
    #: docstring exists to argue against: a difference written as a field is one
    #: a reader can list and a test can pin.
    calibration_phase: bool = False


#: What a design says about HOW its rate gates aggregate, as opposed to what
#: number they compare against.  The threshold and the aggregation are two halves
#: of one rule: "a 0.90 floor" and "a 0.90 floor on every direct seed
#: separately" are different analyses of the same rows and reach different
#: verdicts, so a pre-registration that recorded only the number had not recorded
#: the rule it was frozen with.
#:
#: Empty for v2, whose bytes are frozen and cannot gain a field; its pooled rule
#: is recoverable from ``PILOT_SPEC_V2.direct_gate_aggregation`` and from the
#: supersession record every v3 design carries.
PER_SEED_GATE_AGGREGATION = OrderedDict((
    ("rule", "every_seed"),
    ("gates", ("direct_image_accuracy", "direct_code_following_rate")),
    ("applies_to_every_level_of", "direct_seed"),
    ("level_field_on_a_row", "direct_seed"),
    ("why", ("a factor with three levels is three measurements, each on its own "
             "separately trained and separately hashed adapter; pooling them "
             "lets one adapter that clears the floor carry two that do not, and "
             "the pooled rate still reads as a measurement of the pathway "
             "rather than of the seed that worked")),
    ("witness", ("ten of twelve on one seed and twelve of twelve on the other "
                 "two is 0.833 against a 0.90 floor and 0.944 pooled; three of "
                 "twelve captured on one seed is 0.25 against a 0.10 ceiling "
                 "and 0.083 pooled")),
    ("a_row_with_no_level_is",
     ("refused, not pooled: pooling is the only way such a row could be counted "
      "at all, and it would be counted toward a seed it does not belong to")),
    ("levels_that_failed_is_a_field",
     ("naming the level is the content of the rule, so the report carries it as "
      "a field and not only inside a sentence")),
    ("the_router_gate_has_always_worked_this_way",
     ("router_held_out_accuracy is required of every router seed; the direct "
      "gates now use the same shape rather than a looser one")),
))


PILOT_SPEC_V2 = PilotSpec(
    version="v2", kind=KIND_V2, prereg_kind=PREREG_V2_KIND,
    result_kind=RESULT_KIND_V2, supersession_kind=SUPERSESSION_KIND,
    conditions=CONDITIONS_V2,
    forced_code_conditions=FORCED_CODE_CONDITIONS_V2,
    mediated_conditions=MEDIATED_CONDITIONS_V2,
    auxiliary_conditions=AUXILIARY_CONDITIONS_V2,
    decisive_conditions=DECISIVE_CONDITIONS_V2,
    thresholds=GATE_THRESHOLDS_V2, gate_to_verdict=GATE_TO_VERDICT,
    baseline_cell=False, embed_live_checkpoint_status=True,
    direct_gate_aggregation="pooled_over_seeds",
    declared_gate_aggregation={},
    hygiene_gate_row_scope=HYGIENE_SCOPE_ALL_ROWS,
    declared_hygiene_scope={}, declared_cell_compatibility={},
    superseded_filenames=("rf_pilot_ppubench.json", "rf_pilot_salmu.json"))

PILOT_SPEC_V3 = PilotSpec(
    version="v3", kind=KIND_V3, prereg_kind=PREREG_V3_KIND,
    result_kind=RESULT_KIND_V3, supersession_kind=SUPERSESSION_KIND_V3,
    conditions=CONDITIONS_V3,
    forced_code_conditions=FORCED_CODE_CONDITIONS_V3,
    mediated_conditions=MEDIATED_CONDITIONS_V3,
    auxiliary_conditions=AUXILIARY_CONDITIONS_V3,
    decisive_conditions=DECISIVE_CONDITIONS_V3,
    thresholds=GATE_THRESHOLDS_V3, gate_to_verdict=GATE_TO_VERDICT_V3,
    baseline_cell=True, embed_live_checkpoint_status=False,
    direct_gate_aggregation="every_seed",
    declared_gate_aggregation={
        "direct_gate_aggregation": PER_SEED_GATE_AGGREGATION},
    hygiene_gate_row_scope=HYGIENE_SCOPE_ALL_ROWS,
    declared_hygiene_scope={}, declared_cell_compatibility={},
    superseded_filenames=("rf_pilot_ppubench.json", "rf_pilot_salmu.json",
                          "rf_pilot_ppubench_v2.json",
                          "rf_pilot_salmu_v2.json"))

#: The version the phases run against.  One constant rather than a default
#: repeated per phase, so repointing the runner is a single edit that cannot
#: leave one phase reading an older design than its siblings.
LATEST_PILOT_SPEC = PILOT_SPEC_V3
SPEC_BY_VERSION = {s.version: s for s in (PILOT_SPEC_V2, PILOT_SPEC_V3)}
SPEC_BY_PREREG_KIND = {s.prereg_kind: s for s in (PILOT_SPEC_V2, PILOT_SPEC_V3)}

#: Three tables a later version fills in, defined empty here so the shared
#: builders can consult them without knowing which versions exist.  Keyed on the
#: version rather than tested for inside a builder, because a builder that had to
#: name a version is a builder the next version has to be edited into.
#:
#: ``SCOPE_AMENDMENT_EVIDENCE_BY_VERSION`` names the function that derives the
#: evidence a post-outcome scope amendment has to carry.
#: ``EXTRA_FREEZE_INPUTS_BY_VERSION`` names the files one version's freeze binds
#: beyond the inputs every version binds.
#: ``ANCESTOR_MANIFEST_BY_VERSION`` names the frozen manifest a version's cells
#: may legitimately have been produced under, if not its own.
#: ``FORGET_SET_SELECTION_BY_VERSION`` names the rule that chooses the forget
#: set(s) a version's pilot covers.
#: ``FREEZE_BY_VERSION`` names the freeze entry point a version uses, where it is
#: not the shared one.
SCOPE_AMENDMENT_EVIDENCE_BY_VERSION = {}
EXTRA_FREEZE_INPUTS_BY_VERSION = {}
ANCESTOR_MANIFEST_BY_VERSION = {}
FREEZE_BY_VERSION = {}

#: ``FORGET_SET_SELECTION_BY_VERSION`` names the rule that picks the forget
#: set(s) one version's pilot is built over.  Absent means the rule every
#: version so far used, ``pilot_forget_set``, so the lookup default IS the
#: existing behaviour rather than a branch that reproduces it.
FORGET_SET_SELECTION_BY_VERSION = {}

#: Every phase, in the order a complete run executes them.  Each version's own
#: list is a FILTER of this one rather than a separately written tuple, so a
#: version cannot reorder the phases it does have, and the parser's choices
#: cannot disagree with what a dispatcher will accept.
ALL_PHASES = ("RF0", "RFC", "RF1B", "RF1G", "RF1D", "RF1H", "RF1E", "RF2",
              "RF2P")


def _phases_for(spec):
    """The phases one pilot version has, in the order a run executes them.

    Derived from the spec rather than retyped per version: a design that
    declares a baseline cell is a design that has a phase to fill it, and a
    table written out twice is a place for the two to disagree.

    RF1B comes first among the RF1 phases because the baseline needs no trained
    adapter -- it is the route's own h, declared as already present -- so it is
    the one measurement obtainable before any of the work the design names, and
    taking it early is taking it before anything has moved.

    RFC comes after RF0 and before every RF1 phase, because a calibration is
    what SELECTS the configuration the confirmatory adapters are trained under:
    run it after RF1D and the choice it makes has already been made by something
    else.  It is filtered on ``spec.calibration_phase`` rather than offered to
    every version, so a version that declared no calibration cannot be asked to
    run one and a dispatcher that accepted the request would be inventing a
    phase the design never froze.
    """
    return tuple(p for p in ALL_PHASES
                 if (p != "RF1B" or spec.baseline_cell)
                 and (p != "RFC" or spec.calibration_phase))


PHASES_BY_VERSION = {s.version: _phases_for(s) for s in SPEC_BY_VERSION.values()}


def _threshold_resolution(threshold, n):
    """What a floor actually requires at this denominator.

    With one row per code a hard mediator leaves few rows, so a 0.99 floor over
    12 codes is not "one failure allowed" -- 11/12 is 0.917.  Stating the
    resolution stops a threshold reading as more permissive than it is.
    """
    if not n:
        return None
    need = math.ceil(threshold * n - 1e-9)
    return {"n": n, "smallest_step": 1 / n,
            "minimum_count_that_clears": need,
            "note": (f"at n={n} the achievable rates are multiples of "
                     f"{1 / n:.4f}, so a floor of {threshold} requires at "
                     f"least {need} of {n}")}


def _per_seed_rate_gate(name, rows, ok_when, threshold, cmp_, description,
                        zero_is, seed_field):
    """One rate gate applied to EVERY level of a factor, not to the pool.

    The router gate has always worked this way and the direct gates did not: v2
    pooled its direct seeds into one denominator, so adapters that failed the
    floor could be carried by one that cleared it comfortably and the gate would
    still report a rate that looked like a measurement.  A factor with three
    levels is three measurements, and a floor on the factor applies to each level
    -- the level that fails is precisely the one a pooled number hides.

    A row carrying no seed is refused rather than pooled: pooling is the only
    way such a row could be counted at all, and it would be counted toward a
    level it does not belong to.
    """
    by_seed = OrderedDict()
    for r in rows:
        seed = r.get(seed_field)
        if seed is None:
            raise RuntimeError(
                f"{name}: a row carries no {seed_field}, so it cannot be "
                f"attributed to a level of the factor; pooling it would count "
                f"it toward a seed it does not belong to")
        ok, n = by_seed.get(seed, (0, 0))
        by_seed[seed] = (ok + (1 if ok_when(r) else 0), n + 1)
    aggregation = f"every {seed_field}, not the pooled mean"
    if not by_seed:
        return {"name": name, "passed": False, "value": None, "n": 0,
                "n_seeds": 0, "threshold": threshold,
                "description": description, "aggregation": aggregation,
                # Null rather than an empty list, matching ``value``: nothing
                # was measured, so there is no level to name as having failed.
                "levels_that_failed": None, "rate_by_failed_level": None,
                "failed_because": (
                    f"zero rows; {zero_is}.  A rate over nothing is not a small "
                    f"rate, and all([]) is True, so this gate would otherwise "
                    f"pass on a run that measured nothing")}
    ordered = OrderedDict(sorted(by_seed.items()))
    rates = OrderedDict((s, ok / n) for s, (ok, n) in ordered.items())
    bad = OrderedDict((s, v) for s, v in rates.items()
                      if not cmp_(v, threshold))
    return {"name": name, "passed": not bad, "value": rates,
            "n": sum(n for _, n in ordered.values()), "n_seeds": len(rates),
            "per_seed_n": OrderedDict((s, n) for s, (_, n) in ordered.items()),
            "ok": OrderedDict((s, ok) for s, (ok, _) in ordered.items()),
            "threshold": threshold, "description": description,
            "aggregation": aggregation,
            # The levels that failed, as a field and not only as prose: naming
            # the level is the whole content of this repair, so a reader or a
            # downstream tool must be able to get it without parsing a sentence.
            "levels_that_failed": list(bad),
            "rate_by_failed_level": dict(bad),
            "failed_because": (
                None if not bad else
                f"{seed_field}(s) {list(bad)} do not satisfy {cmp_.__name__} "
                f"{threshold!r}: {dict(bad)}")}


def evaluate_gates_for(spec, intervention_rows, natural_rows, direct_rows,
                       hybrid_rows, router_accuracy_by_seed, e2e,
                       thresholds=None, baseline_rows=()):
    """The gates of one pilot version, each with its denominator, feeding three
    separate verdicts.

    Every rate refuses a zero denominator rather than returning its neutral
    value: ``all([])`` is True and a mean over nothing serializes as null, which
    is how a run that evaluated nothing once reported green gates.

    ``spec`` selects the threshold table, the direct-gate aggregation and
    whether the baseline gate exists at all, so v2 computes exactly the gates it
    was frozen with and v3 adds the one it needs -- without either version
    carrying a second copy of the other's arithmetic.
    """
    th = dict(thresholds or spec.thresholds)
    missing = sorted(set(spec.thresholds) - set(th))
    if missing:
        raise RuntimeError(f"gates called without thresholds {missing}; a gate "
                           f"whose threshold was not supplied is not a gate")

    inter = [r for r in intervention_rows
             if r["condition"] in ("forgotten_route_intervention",
                                   "retained_route_intervention")]
    gates = OrderedDict()

    # 1 -- routing reliability: the floor applies to EVERY router seed, not the
    # mean, because a seed that routes badly is a real router and averaging it
    # away is how a factor stops being a factor
    accs = router_accuracy_by_seed or {}
    if not accs:
        gates["router_held_out_accuracy"] = {
            "name": "router_held_out_accuracy", "passed": False, "value": None,
            "n": 0, "threshold": th["min_router_held_out_accuracy"],
            "levels_that_failed": None, "rate_by_failed_level": None,
            "failed_because": "no router seed reported a held-out accuracy, so "
                              "the gate has no denominator"}
    else:
        below = {s: a for s, a in sorted(accs.items())
                 if a is None or a < th["min_router_held_out_accuracy"]}
        gates["router_held_out_accuracy"] = {
            "name": "router_held_out_accuracy",
            "passed": not below, "value": dict(sorted(accs.items())),
            "n": len(accs), "n_seeds": len(accs),
            "threshold": th["min_router_held_out_accuracy"],
            "aggregation": "every seed, not the mean",
            # Same two fields the direct gates carry, so every per-seed gate in
            # the report names its failing levels the same way
            "levels_that_failed": sorted(below),
            "rate_by_failed_level": dict(below),
            "failed_because": (None if not below else
                               f"router seed(s) {sorted(below)} are below the "
                               f"floor: {below}")}

    def arm_gate(name, cond, want, cmp_, floor, describe):
        sub = [r for r in inter if r["condition"] == cond]
        ok = sum(1 for r in sub if want(r))
        g = _rate_gate(name, ok, len(sub), floor, cmp_, describe,
                       "this arm of the intervention was never run")
        g["resolution"] = _threshold_resolution(floor, len(sub))
        return g

    gates["conditional_forgotten_route_suppression"] = arm_gate(
        "conditional_forgotten_route_suppression",
        "forgotten_route_intervention",
        lambda r: r.get("parsed_label") == DELETED_LABEL,
        operator.ge, th["forgotten_route_suppression"],
        "h_e(do(C=C_f)) must produce the refusal label for every forgotten code")
    gates["conditional_retained_route_accuracy"] = arm_gate(
        "conditional_retained_route_accuracy",
        "retained_route_intervention",
        lambda r: bool(r.get("correct")), operator.ge,
        th["retained_route_accuracy"],
        "h_e(do(C=C_r)) must produce that code's own alias for every retained "
        "code")
    gates["mediator_intervention_following_accuracy"] = _rate_gate(
        "mediator_intervention_following_accuracy",
        sum(1 for r in inter if bool(r.get("correct"))), len(inter),
        th["min_intervention_following_accuracy"], operator.ge,
        "over every forced-code row on both arms: the output must be what the "
        "FORCED code requires, which is the mediation claim itself",
        "no forced-code intervention was run at all")
    gates["mediator_intervention_following_accuracy"]["resolution"] = \
        _threshold_resolution(th["min_intervention_following_accuracy"],
                              len(inter))

    if spec.baseline_cell:
        # 4b -- the BEFORE.  v2 declared a baseline_h role, described it in its
        # own words as "the before in before/after", and never evaluated it, so
        # the suppression gates above were satisfiable by an h that produced the
        # refusal label for every code whatever.  Gated on the FORGOTTEN arm
        # because that is the arm the edit claims to change; the retained arm is
        # reported beside it, since a baseline that also fails on retained codes
        # means the route was never clean, and that is worth knowing even though
        # no verdict turns on it.
        base_forgotten = [r for r in baseline_rows if r["arm"] == "forgotten"]
        base_retained = [r for r in baseline_rows if r["arm"] == "retained"]
        gates["baseline_forgotten_route_following"] = _rate_gate(
            "baseline_forgotten_route_following",
            sum(1 for r in base_forgotten if bool(r.get("correct"))),
            len(base_forgotten),
            th["min_baseline_forgotten_route_following"], operator.ge,
            "h_base(do(C=C_f)) BEFORE any edit: a forgotten code must produce "
            "its own alias.  This is the other half of "
            "conditional_forgotten_route_suppression -- its own alias before, "
            "Unknown after -- and without it the suppression gate is satisfied "
            "by an adapter that never produced those aliases at all",
            "the unedited h was never evaluated on the forgotten codes")
        gates["baseline_forgotten_route_following"]["resolution"] = \
            _threshold_resolution(
                th["min_baseline_forgotten_route_following"],
                len(base_forgotten))
        gates["baseline_forgotten_route_following"]["paired_with"] = \
            "conditional_forgotten_route_suppression"
        gates["baseline_forgotten_route_following"][
            "retained_arm_reported_not_gated"] = {
                "accuracy": (
                    sum(1 for r in base_retained if bool(r.get("correct")))
                    / len(base_retained)) if base_retained else None,
                "n": len(base_retained),
                "why_not_gated": (
                    "the edit makes no claim about a retained code's behaviour, "
                    "so the retained arm's pre-edit accuracy is context for the "
                    "change claim rather than a condition on it")}

    # 5/6 -- the direct pathway.  Item 2: bounding code-following from above is
    # satisfied by a model that emits arbitrary wrong labels, so the accuracy
    # floor is the gate that carries the weight and the code-following bound is
    # the one that says the pathway was not captured by a code.
    acc_describe = (
        "D_s(X), image only and no code presented: the unchanged X -> Y pathway "
        "must still identify the image.  This is the gate v1 lacked -- it reused "
        "edited_h, which was trained on code-to-label pairs and is not an image "
        "model, so a model emitting arbitrary wrong labels passed")
    hyb_describe = (
        "AUXILIARY hybrid probe D_s(X, C_irrelevant): the rate at which an "
        "irrelevant code captures a pathway never trained on codes.  Reported "
        "beside the accuracy gate, never instead of it")
    acc_zero = "the direct model was never evaluated on an image"
    hyb_zero = "the hybrid probe was never run"

    def _followed_irrelevant_code(r):
        return bool(r.get("forced_label")) and r.get("parsed_label") == r["forced_label"]

    if spec.direct_gate_aggregation == "every_seed":
        # Required of EVERY direct seed, exactly as router accuracy is required
        # of every router seed.  Pooling the seeds let one good adapter carry two
        # that failed the floor, and the pooled rate still read as a measurement
        # of "the direct pathway" rather than of the one seed that worked.
        gates["direct_image_accuracy"] = _per_seed_rate_gate(
            "direct_image_accuracy", direct_rows,
            lambda r: bool(r.get("correct")),
            th["min_direct_image_accuracy"], operator.ge,
            acc_describe + ", for EVERY direct seed", acc_zero, "direct_seed")
        gates["direct_image_accuracy"]["resolution"] = {
            s: _threshold_resolution(th["min_direct_image_accuracy"], n)
            for s, n in gates["direct_image_accuracy"]["per_seed_n"].items()}
        gates["direct_image_accuracy"]["resolution_is_per_seed"] = (
            "a floor is resolved at the denominator it is applied to, and here "
            "that is one seed's held-out images rather than all seeds pooled")
        gates["direct_image_accuracy"]["measured_on"] = (
            "D_s, a separately trained and separately hashed adapter, one gate "
            "value per direct seed")
        gates["direct_code_following_rate"] = _per_seed_rate_gate(
            "direct_code_following_rate", hybrid_rows,
            _followed_irrelevant_code,
            th["max_direct_code_following_rate"], operator.le,
            hyb_describe + "  Measured for EVERY direct seed", hyb_zero,
            "direct_seed")
    else:
        dir_ok = sum(1 for r in direct_rows if bool(r.get("correct")))
        gates["direct_image_accuracy"] = _rate_gate(
            "direct_image_accuracy", dir_ok, len(direct_rows),
            th["min_direct_image_accuracy"], operator.ge, acc_describe, acc_zero)
        gates["direct_image_accuracy"]["resolution"] = _threshold_resolution(
            th["min_direct_image_accuracy"], len(direct_rows))
        gates["direct_image_accuracy"]["measured_on"] = (
            "D_s, a separately trained and separately hashed adapter")
        hyb_follow = sum(1 for r in hybrid_rows if _followed_irrelevant_code(r))
        gates["direct_code_following_rate"] = _rate_gate(
            "direct_code_following_rate", hyb_follow, len(hybrid_rows),
            th["max_direct_code_following_rate"], operator.le, hyb_describe,
            hyb_zero)

    # Every row this design produced, whichever cell it came from.  An output
    # naming no candidate label is a defect of the run and not of one condition,
    # so the count is over the whole run rather than per gate -- and that sentence
    # stays true of the RUN.  What it cannot also be is the denominator of a
    # verdict, because the same design declares its auxiliary condition "reported
    # separately from every mediated gate" and then maps this count to mediation.
    #
    # Which rows the verdict counts is therefore a field of the spec rather than
    # a branch on the version, so a reader can list it and a test can pin it.  The
    # whole-run figure is still reported, on the gate, beside the one the verdict
    # reads.
    every_row = (inter + list(baseline_rows) + natural_rows + direct_rows
                 + hybrid_rows)
    aux_rows = [r for r in every_row
                if r["condition"] in spec.auxiliary_conditions]
    if spec.hygiene_gate_row_scope == HYGIENE_SCOPE_ALL_ROWS:
        verdict_rows = every_row
    elif spec.hygiene_gate_row_scope == HYGIENE_SCOPE_NON_AUXILIARY:
        verdict_rows = [r for r in every_row
                        if r["condition"] not in spec.auxiliary_conditions]
    else:
        raise RuntimeError(
            f"spec {spec.version!r} names hygiene_gate_row_scope "
            f"{spec.hygiene_gate_row_scope!r}; the scopes this gate can count "
            f"are {HYGIENE_SCOPE_ALL_ROWS!r} and "
            f"{HYGIENE_SCOPE_NON_AUXILIARY!r}.  A denominator no design "
            f"declared is a verdict no design pre-registered")
    n_unp = sum(1 for r in verdict_rows if r.get("unparseable"))
    n_amb = sum(1 for r in verdict_rows if r.get("multi_label_ambiguous"))
    gates["no_unparseable_or_multi_label_outputs"] = {
        "name": "no_unparseable_or_multi_label_outputs",
        "passed": (n_unp <= th["max_unparseable_outputs"]
                   and n_amb <= th["max_multi_label_outputs"]),
        "value": {"unparseable": n_unp, "multi_label_ambiguous": n_amb},
        "n": len(verdict_rows),
        "threshold": {"max_unparseable_outputs":
                      th["max_unparseable_outputs"],
                      "max_multi_label_outputs": th["max_multi_label_outputs"]},
        "description": ("an output naming no candidate label identifies no "
                        "route, and one naming two followed neither; resolving "
                        "either would invent a result"),
        "failed_because": (
            None if (n_unp <= th["max_unparseable_outputs"]
                     and n_amb <= th["max_multi_label_outputs"])
            else f"{n_unp} unparseable and {n_amb} multi-label outputs"),
        # Added only for a version that declared a scope, so the gates of a
        # version whose bytes are frozen stay exactly the keys they were.
        **({"row_scope": spec.hygiene_gate_row_scope,
            "conditions_excluded_from_the_count":
                list(spec.auxiliary_conditions),
            "n_rows_excluded": len(aux_rows),
            "n_rows_in_the_whole_run": len(every_row),
            "both_denominators_are_named": (
                "n is the one the verdict reads; the whole-run figure is here "
                "beside it so what was excluded is a stated fact rather than "
                "something a reader has to reconstruct")}
           if spec.declared_hygiene_scope else {})}

    if spec.declared_hygiene_scope:
        # Reported, not dropped.  Excluding rows from a verdict is a statement
        # about what the verdict rests on; it is not a statement that nothing
        # happened to those rows, and a diagnostic nobody reports is a
        # measurement nobody made.  ``is_a_gate`` is False and it appears in no
        # gate_to_verdict map, so no verdict reads it and ``passed`` stays None
        # rather than inventing a pass or a fail.
        aux_unp = [r for r in aux_rows if r.get("unparseable")]
        aux_amb = [r for r in aux_rows if r.get("multi_label_ambiguous")]
        gates["auxiliary_output_hygiene"] = {
            "name": "auxiliary_output_hygiene",
            "is_a_gate": False,
            "feeds_no_verdict": True,
            "absent_from_gate_to_verdict": (
                "auxiliary_output_hygiene" not in spec.gate_to_verdict),
            "passed": None,
            "conditions": list(spec.auxiliary_conditions),
            "n": len(aux_rows),
            "value": {"unparseable": len(aux_unp),
                      "multi_label_ambiguous": len(aux_amb)},
            "rate_unparseable": (round(len(aux_unp) / len(aux_rows), 6)
                                 if aux_rows else None),
            "example_row_ids": [r.get("row_id") for r in aux_unp[:8]],
            "example_raw_outputs": [r.get("raw_text") for r in aux_unp[:8]],
            "distinct_expected_labels_missed": sorted(
                {str(r.get("expected_label")) for r in aux_unp}),
            "n_examples_shown": min(len(aux_unp), 8),
            "why_it_is_reported_at_all": (
                "these rows are outside every mediated hygiene count, which is "
                "a statement about the verdict and not about the rows.  What "
                "they did is still a finding: an auxiliary probe that derails "
                "out of the candidate vocabulary is telling us the direct "
                "pathway ignores a code it was never trained on by answering "
                "something else entirely"),
        }

    tol = th["max_abs_e2e_prediction_error"]
    if not e2e or e2e.get("predicted") is None or e2e.get("observed") is None:
        gates["e2e_matches_the_routing_composition"] = {
            "name": "e2e_matches_the_routing_composition", "passed": False,
            "value": None, "n": (e2e or {}).get("n_images"),
            "threshold": tol,
            "description": ("observed E2E must equal sum_c P_g(c|X) H_e(c, "
                            "Y*(X)) within tolerance"),
            "failed_because": "the prediction or the observation is missing, so "
                              "the factorization was never evaluated"}
    else:
        err = abs(e2e["observed"] - e2e["predicted"])
        gates["e2e_matches_the_routing_composition"] = {
            "name": "e2e_matches_the_routing_composition",
            "passed": err <= tol, "value": err,
            "observed": e2e["observed"], "predicted": e2e["predicted"],
            "n": e2e.get("n_images"), "threshold": tol,
            "description": ("if apparent forgetting failures and collateral "
                            "errors are quantitatively explained by routing "
                            "errors, the observed figure must equal what the "
                            "router's own confusion and h's empirical hard "
                            "response predict"),
            "failed_because": (
                None if err <= tol else
                f"|{e2e['observed']} - {e2e['predicted']}| = {err:.6f} exceeds "
                f"the tolerance, so routing error does NOT account for the gap")}

    return gates


def evaluate_gates_v2(intervention_rows, natural_rows, direct_rows,
                      hybrid_rows, router_accuracy_by_seed, e2e,
                      thresholds=None):
    """The v2 gates.  Thin: the one implementation is ``evaluate_gates_for``."""
    return evaluate_gates_for(PILOT_SPEC_V2, intervention_rows, natural_rows,
                              direct_rows, hybrid_rows,
                              router_accuracy_by_seed, e2e, thresholds)


def evaluate_gates_v3(intervention_rows, natural_rows, direct_rows,
                      hybrid_rows, router_accuracy_by_seed, e2e,
                      thresholds=None, baseline_rows=()):
    """The v3 gates: v2's, plus the baseline gate and a floor on EVERY direct
    seed rather than on the three pooled."""
    return evaluate_gates_for(PILOT_SPEC_V3, intervention_rows, natural_rows,
                              direct_rows, hybrid_rows,
                              router_accuracy_by_seed, e2e, thresholds,
                              baseline_rows)


def verdicts_from_gates(gates, applicability, thresholds=None,
                        spec=PILOT_SPEC_V2):
    """Three separate conclusions, and no omnibus boolean anywhere.

    Each verdict is a three-state value: ``pass``, ``fail``, or
    ``not_established``.  The third is not a soft fail -- it means the dataset
    cannot support the question, so no measurement could have answered it, and
    coercing it to False would report a failure that was never observed while
    coercing it to True would report a success that was never measured.

    ``spec`` supplies the gate-to-verdict map and the threshold table, so a
    verdict is computed over the gates the design that produced it actually
    declared.  Reading the map from a module constant instead would let a
    version add a gate and have no verdict ever notice it was missing -- the
    ``absent`` refusal below only fires for gates the map names.
    """
    th = dict(thresholds or spec.thresholds)
    out = OrderedDict()
    for name in VERDICT_NAMES:
        mine = [g for g, v in spec.gate_to_verdict.items() if v == name]
        present = [gates[g] for g in mine if g in gates]
        absent = [g for g in mine if g not in gates]
        if absent:
            raise RuntimeError(f"verdict {name!r} is missing gates {absent}; a "
                               f"verdict computed over part of its gates is not "
                               f"a verdict")
        supportable = applicability["verdict_support"].get(name, {})
        failed = [g["name"] for g in present if not g["passed"]]
        if not supportable.get("supportable", True):
            out[f"{name}_pass"] = None
            out[name] = {
                "state": NOT_ESTABLISHED, "passed": None,
                "gates": [g["name"] for g in present],
                "failed_gates": failed,
                "reason": supportable.get("reason"),
                "thresholds_unchanged": {k: th[k] for k in th},
                "policy": ("recorded as not established on this dataset; the "
                           "threshold is neither waived nor lowered, and the "
                           "gates are still reported with their values")}
            continue
        out[f"{name}_pass"] = not failed
        out[name] = {
            "state": "pass" if not failed else "fail",
            "passed": not failed,
            "gates": [g["name"] for g in present],
            "failed_gates": failed,
            "reason": (None if not failed else
                       f"{len(failed)} of {len(present)} gates failed"),
            "thresholds_unchanged": {k: th[k] for k in th}}
    out["no_omnibus_verdict"] = (
        "there is deliberately no combined 'passed' field: a routing failure "
        "and a mediation success answer different questions, and one boolean "
        "would let either cancel the other")
    out["per_dataset_expectation"] = applicability.get("expected_verdicts")
    return out


def gate_applicability_v2(audit, dataset, spec=PILOT_SPEC_V2):
    """Which verdicts a dataset can support, and what each measurement is
    actually measured on.

    The mediation gates force the code, so duplicated image content does not
    weaken them.  The routing verdicts are claims about images the router never
    saw, and a dataset with no such images cannot support them however well it
    scores.  The direct gate sits in between and is labelled accordingly: on a
    dataset with duplicated content it bounds ERASURE on the training bytes
    rather than establishing generalization.

    ``spec`` supplies the gate map the counts are taken over: a version that
    adds a gate changes how many verdicts this dataset can support, and a count
    read from a module constant would keep reporting the old version's total.
    """
    v1 = gate_applicability(audit)
    held_out = audit["held_out_image_content_exists"]
    support = {
        "mediation": {
            "supportable": True,
            "reason": ("every mediated condition forces the code and sends h "
                       "nothing else, so the image is not an input and "
                       "duplicated image content cannot affect it"),
            "direct_gate_scope": (
                "held-out images" if held_out else
                "images whose bytes also appear in the train split, so "
                "direct_image_accuracy bounds ERASURE of the X -> Y pathway "
                "rather than establishing that it generalizes"),
        },
        "routing_reliability": {
            "supportable": held_out,
            "reason": (None if held_out else
                       f"every test image's bytes also appear in train "
                       f"({audit['n_test_bytes_also_in_train']} of "
                       f"{audit['by_split']['test']['n_distinct_bytes']} "
                       f"distinct test images), so an accuracy measured here is "
                       f"measured on training bytes and is not a held-out "
                       f"routing accuracy"),
        },
        "routing_factorization": {
            "supportable": held_out,
            "reason": (None if held_out else
                       "the factorization explains the gap between conditional "
                       "and unconditional E2E accuracy; with no held-out image "
                       "content there is no such gap to explain, so the formula "
                       "would be filed against a quantity this dataset cannot "
                       "produce"),
        },
    }
    expected = {
        "mediation": "supported",
        "routing_reliability": ("measured" if held_out else NOT_ESTABLISHED),
        "routing_factorization": (
            "measured" if held_out else f"{NOT_ESTABLISHED} on unseen content"),
    }
    # Derived from the recorded held-out accuracy against the frozen floor for
    # ANY dataset, rather than asserted for one by name: a dataset whose router
    # clears the floor gets "measured", one whose router does not gets the
    # expected failure, and one with no recorded accuracy is left alone.  Naming
    # SALMU here would leave a number in the artifact that nothing checks and
    # would survive a change to either input still describing the old one.
    recorded = HELD_OUT_G[dataset]["accuracy"]
    floor = spec.thresholds["min_router_held_out_accuracy"]
    if recorded is not None and recorded < floor:
        expected["routing_reliability"] = (
            f"measured, expected to FAIL at {recorded} against {floor}")
        expected["routing_factorization"] = (
            "supported and scientifically informative: an imperfect router is "
            "the only condition where the factorization has something to "
            "explain")
    return {
        "dataset": dataset,
        "held_out_image_content_exists": held_out,
        "verdict_support": support,
        "expected_verdicts": expected,
        "expected_verdicts_are_derived": (
            "from the image-content audit and from HELD_OUT_G against the "
            "frozen floor; no dataset is named in a branch here, so a dataset "
            "whose images were replaced would get a different expectation "
            "rather than a stale one"),
        "n_gates_total": len(spec.gate_to_verdict),
        "n_gates_supportable": sum(
            1 for g, v in spec.gate_to_verdict.items()
            if support[v]["supportable"]),
        "clustering": v1["clustering"],
        "policy": ("a verdict that is not established is recorded as such; it "
                   "is not waived, not lowered, and not counted as passed or "
                   "as failed"),
    }


# ---------------------------------------------------------------------------
# v2 checkpoints, design and atomic per-cell writing
# ---------------------------------------------------------------------------

def atomic_write_json(path, obj):
    """Write a cell result so a reader never sees a half-written file.

    Written to a sibling temporary file, flushed and fsynced, then renamed over
    the destination: ``os.replace`` is atomic within a filesystem, so a run that
    is interrupted leaves either the previous complete result or none, never a
    truncated one.  A truncated JSON that a later ``--resume`` read as "already
    done" would silently drop a cell from the aggregate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    payload = canonical_json(obj)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path, sha256_file(path)


#: The file keys a checkpoint role can declare, in the order they are recorded.
#: One list, used by the requirement builder, by the verifier and by the count
#: of what a role needs.
#:
#: That per-role count is named ``n_files_required`` and not ``n_required``
#: because the verification block already uses ``n_required`` for the number of
#: ROLES: one key meaning files in one place and roles in its own parent is a
#: count that reads correctly and adds incorrectly.
CHECKPOINT_FILE_KEYS_V2 = ("adapter", "held_out_predictions", "cell_results")


def required_checkpoints_v2(dataset, forget_set_id, router_seeds, edit_seeds,
                            direct_seeds):
    """Every checkpoint and prediction file a v2 pilot needs, by role.

      router_g__seed{S}   one g adapter AND one held-out prediction file per
                          router seed -- two files, because a checkpoint without
                          its own predictions cannot compose and predictions
                          without a checkpoint cannot be attributed to a
                          training run.  This pair per seed is what makes the
                          router seed a factor instead of a label.
      baseline_h          the unedited C->Y map: the before in before/after.
      edited_h__seed{E}   one per edit seed, reused from the frozen matrix.
      direct_d__seed{S}   one per direct seed: a SEPARATE X->Y adapter.

    v1 listed a ``direct_condition`` role whose adapter was "reuses edited_h".
    That is not an unchanged X->Y pathway -- edited_h was trained on
    code-to-label pairs, so it is not an image model at all, and evaluating it
    on an image measures whatever a code-trained adapter happens to do with
    pixels.  There is no such role here.
    """
    cells = MATRIX_CELLS[dataset]
    route = ROUTE_DIRS[dataset]
    req = OrderedDict()
    for seed in router_seeds:
        existing = seed == EXISTING_ROUTER_SEED
        req[f"router_g__seed{seed}"] = {
            "role": ("the frozen router, read not retrained" if existing else
                     "a router trained under the identical frozen protocol with "
                     "a fresh LoRA init at this seed"),
            "adapter": _rel(router_checkpoint_path(dataset, seed)),
            "held_out_predictions": _rel(router_prediction_path(dataset, seed)),
            "exists_already": existing,
        }
    req["baseline_h"] = {
        "role": "the unedited code->label map; the before in before/after",
        "adapter": _rel(DATASET_ROOT / route / "h_C_to_Y"
                        / Path(*ADAPTER_RELPATH)),
        "exists_already": True}
    for seed in edit_seeds:
        req[f"edited_h__seed{seed}"] = {
            "role": "the post-edit route, reused from the frozen matrix",
            "adapter": _rel(DATASET_ROOT / cells / forget_set_id
                            / f"seed_{seed}" / "edited_h"
                            / Path(*ADAPTER_RELPATH)),
            "cell_results": _rel(DATASET_ROOT / cells / forget_set_id
                                 / f"seed_{seed}" / "cell_results.json"),
            "exists_already": True}
    for seed in direct_seeds:
        req[f"direct_d__seed{seed}"] = {
            "role": ("a separately trained X -> Y adapter on the same training "
                     "associations, image-only, never edited when h is edited"),
            "adapter": _rel(direct_checkpoint_path(dataset, seed)),
            "exists_already": False}
    # The file count is DERIVED from the files each role actually declares
    # rather than written next to them: an edited_h role names an adapter AND
    # the matrix cell results it is reused from, and a literal 1 beside two
    # hashed files is a count nothing checks.
    for label, spec in req.items():
        spec["n_files_required"] = sum(1 for k in CHECKPOINT_FILE_KEYS_V2
                                       if spec.get(k))
        if not spec["n_files_required"]:
            raise RuntimeError(
                f"checkpoint role {label!r} declares no file in "
                f"{list(CHECKPOINT_FILE_KEYS_V2)}, so there is nothing to hash "
                f"and nothing that could be reported missing")
    return {
        "dataset": dataset,
        "forget_set_id": forget_set_id,
        "roles": req,
        "router_seeds": list(router_seeds),
        "edit_seeds": list(edit_seeds),
        "direct_seeds": list(direct_seeds),
        "no_direct_condition_reuses_edited_h": (
            "v1's direct condition reused edited_h and so measured a "
            "code-trained adapter on an image; v2 requires an independent D_s "
            "with its own digest"),
        "router_seed_is_a_real_factor": (
            "one checkpoint and one held-out prediction file per router seed; "
            "forced-code interventions are evaluated once per EDIT seed because "
            "they do not depend on g, so they are not repeated across router "
            "seeds and counted as additional evidence"),
        "paths_are_recorded_relative_not_absolute": (
            "v1's verification table recorded the checkpoint paths it hashed as "
            "ABSOLUTE (its requirement table recorded the same paths relative), "
            "so a frozen v1 manifest reproduces its design_sha256 only on a "
            "checkout rooted where it was frozen; from another clone every role "
            "reads as absent.  v2 records both tables relative to the dataset "
            "root -- the convention provenance.paths_are_relative_to already "
            "states -- and verify_manifest resolves a recorded path through "
            "resolve_recorded_path, which still accepts the absolute shape so "
            "the superseded v1 manifests keep verifying where they were frozen"),
    }


def verify_checkpoints_v2(requirements):
    """Hash every v2 input and separate what exists from what must be trained.

    Existence is not identity and identity is not freshness: a checkpoint
    replaced in place keeps its path, so the digest is what ties a result to the
    weights that produced it, and a role that does not exist yet is reported as
    training work rather than as a missing file to be worked around.
    """
    present, absent = OrderedDict(), OrderedDict()
    for role, spec in requirements["roles"].items():
        entry = {"role": role, "declared_exists_already":
                 spec.get("exists_already", False)}
        paths = {k: spec.get(k) for k in CHECKPOINT_FILE_KEYS_V2}
        ok = True
        for key, raw in sorted(paths.items()):
            if not raw or raw == "reuses edited_h":
                continue
            p = resolve_recorded_path(raw)
            if p.is_file():
                entry[key] = {"path": _rel(p), "sha256": sha256_file(p),
                              "bytes": p.stat().st_size}
            else:
                entry[key] = {"path": _rel(p), "sha256": None, "bytes": None,
                              "absent": True}
                ok = False
        (present if ok else absent)[role] = entry
    must_train = sorted(r for r, e in absent.items()
                        if not e["declared_exists_already"])
    unexpectedly_absent = sorted(r for r, e in absent.items()
                                 if e["declared_exists_already"])
    return {
        "complete": not absent,
        "n_present": len(present),
        "n_required": len(requirements["roles"]),
        "n_files_required": sum(s["n_files_required"]
                                for s in requirements["roles"].values()),
        "present": present,
        "absent": absent,
        "must_be_trained": must_train,
        "n_must_be_trained": len(must_train),
        "unexpectedly_absent": unexpectedly_absent,
        "note": ("must_be_trained is the GPU work this pilot requires and does "
                 "not have; unexpectedly_absent is a role this design assumed "
                 "was already on disk and is not, which is a broken assumption "
                 "rather than training work"),
        "hashed_not_just_existence_checked": (
            "a checkpoint replaced in place keeps its path, so the digest is "
            "what ties a result to the weights that produced it"),
    }


def roles_are_v2_shaped(req):
    """Whether a role table is the shape the live prober reads.

    v1 counted each role's files as ``n_required`` and declared no
    ``exists_already``, so probing a v1 table as a v2 one raises a KeyError
    inside a verifier whose whole job is to report what is wrong rather than to
    break on it.  v1 also recorded its verification paths as ABSOLUTE, so its
    presence answers are not portable between checkouts anyway: a v1 manifest
    keeps the frozen-block reading and reports no live readiness, which is the
    honest answer for a table this prober cannot read.
    """
    roles = (req or {}).get("roles")
    return isinstance(roles, dict) and all(
        isinstance(s, dict) and "n_files_required" in s for s in roles.values())


def checkpoint_readiness(frozen):
    """Live checkpoint status for a frozen pilot: presence, digests, readiness.

    Computed here and never inside a constructor, so it is never covered by
    ``design_sha256``.  The distinction is the one v2 got wrong.  A design may
    bind the bytes it is a statement ABOUT -- the images, the dataset manifest,
    the prompts, the thresholds -- because those are inputs, and a change in
    them means the design now describes something else.  Which of the adapters a
    design asks to be trained have arrived is not an input, it is PROGRESS, and
    hashing progress makes a pre-registration invalidate itself as its own work
    is completed: v2's pilots stopped reproducing the moment the first role they
    were waiting for appeared, so the training that made a pilot runnable was
    the training that made it unverifiable and unloadable.

    Reads the role DECLARATIONS out of the frozen document and probes the tree,
    so it answers for any version: a v2 manifest carries a frozen status block
    beside the same declarations, and this ignores it in favour of what is true
    now, reporting where the two have diverged.
    """
    req = (frozen or {}).get("checkpoint_requirements") or {}
    live = verify_checkpoints_v2(req)
    frozen_block = req.get("verification")

    n_files_present = sum(
        1 for entry in live["present"].values() for v in entry.values()
        if isinstance(v, dict) and "path" in v and v.get("sha256"))

    # A role the design declared as training work that is now on disk.  Named
    # from the DECLARATION rather than from a frozen status snapshot, so it
    # works for a manifest that never recorded one.
    trained = []
    for role, entry in live["present"].items():
        if entry.get("declared_exists_already"):
            continue
        trained.append({
            "role": role,
            "files": {k: {"path": v["path"], "sha256": v["sha256"],
                          "bytes": v["bytes"]}
                      for k, v in entry.items()
                      if isinstance(v, dict) and "path" in v}})

    changed_since_freeze = []
    if isinstance(frozen_block, dict):
        changed_since_freeze = sorted(
            set(frozen_block.get("present") or {}) ^ set(live["present"]))

    return {
        **live,
        "n_files_present": n_files_present,
        "computed_live_not_read_from_the_frozen_design": True,
        "frozen_status_block_present_in_this_manifest":
            frozen_block is not None,
        "roles_whose_presence_changed_since_freeze": changed_since_freeze,
        "trained_since_declared": trained,
        "runnable_now": bool(live["complete"]
                             and not live["unexpectedly_absent"]),
        "runnable_now_is_about_inputs_not_about_results": (
            "True means every weight, prediction file and image this design "
            "names is present and hashed; it says nothing about whether any "
            "gate would pass"),
        "why_this_is_not_in_the_design": (
            "because it changes when training finishes, and a frozen design "
            "that changes when training finishes is one that cannot be "
            "verified by the run that satisfies it"),
    }


CELL_KIND_PHASE = {"intervention": "RF1H", "natural": "RF1E",
                   "direct": "RF1D", "hybrid": "RF1D", "baseline": "RF1B"}


def build_design_for(spec, dataset, forget_set_id, forget_ids, router_seeds,
                     edit_seeds, direct_seeds, man=None, images=None,
                     image_sha_by_uri=None):
    """The design of one pilot version: what is executed, by which phase, and
    how many times.

    Cell counts follow from what each measurement actually depends on:

      intervention  one per EDIT seed      -- h_e(do(C=c)) does not involve g
      natural       one per (router, edit) -- the composition involves both
      direct        one per DIRECT seed    -- D_s(X) involves neither
      hybrid        one per DIRECT seed    -- auxiliary probe on D_s
      baseline      exactly one, v3 only   -- h_base(do(C=c)) involves no factor

    Repeating an intervention cell across router seeds would run the identical
    call three times and report it as three observations, which is the defect
    item 3 names.  The baseline cell obeys the same rule from the other
    direction: it depends on no factor at all, so there is one of it however
    many seeds the design varies.

    ``spec`` is the only thing that differs between versions in here.  Two
    constructors would be two places for the cell structure to drift, and a
    drift in one of them surfaces as a design no verifier can rebuild rather
    than as a difference anybody intended.
    """
    man = man if man is not None else load_manifest(dataset)
    images = images if images is not None else held_out_images(man)
    vocab = label_vocab(man)
    retained = check_design_inputs(man, forget_ids, router_seeds, edit_seeds,
                                   images)
    if not direct_seeds:
        raise RuntimeError("no direct seeds given: the direct control needs its "
                           "own adapter per seed, and without one there is no "
                           "X -> Y pathway to compare the route against")
    if len(set(direct_seeds)) != len(direct_seeds):
        raise RuntimeError(f"duplicate direct seeds: {direct_seeds}")

    prompts = frozen_route_prompts()
    protocol = frozen_route_protocol()
    cells = []

    def add(cell_id, kind, rows, **meta):
        audit = coverage_audit_v2(rows, man, forget_ids,
                                  "direct" if kind == "hybrid" else kind,
                                  images=images,
                                  forced=spec.forced_code_conditions)
        if not audit["exact"]:
            raise RuntimeError(f"cell {cell_id} does not cover its design: "
                               + "; ".join(audit["defects"]))
        cells.append({"cell_id": cell_id, "kind": kind,
                      "phase": CELL_KIND_PHASE[kind], "n_rows": len(rows),
                      "coverage": audit, "rows": rows, **meta})

    if spec.baseline_cell:
        # The BEFORE, evaluated once.  Filed first because it is the
        # measurement every suppression cell is a difference from, and because
        # it needs no trained adapter: h_base is the route's own h, so this cell
        # can run before any of the training work the design names.
        cid = "baseline__h_base"
        add(cid, "baseline", build_baseline_rows_v3(man, forget_ids, cid))
    for es in edit_seeds:
        cid = f"intervention__h{es}"
        add(cid, "intervention",
            build_intervention_rows_v2(man, forget_ids, cid, es),
            edit_seed=es)
    for rs in router_seeds:
        for es in edit_seeds:
            cid = f"natural__g{rs}__h{es}"
            add(cid, "natural",
                build_natural_rows_v2(man, forget_ids, None, cid, rs, es,
                                      image_sha_by_uri),
                router_seed=rs, edit_seed=es)
    for ds in direct_seeds:
        # Through direct_cell_id with no namespace, which returns exactly the id
        # this constructor has always produced.  The call is here so that v5's
        # namespaced id and this one are one rule rather than two spellings.
        cid = direct_cell_id("direct", ds)
        add(cid, "direct",
            build_direct_rows_v2(man, cid, ds,
                                 image_sha_by_uri=image_sha_by_uri),
            direct_seed=ds)
        cid = direct_cell_id("hybrid", ds)
        add(cid, "hybrid",
            build_direct_rows_v2(man, cid, ds, hybrid=True,
                                 image_sha_by_uri=image_sha_by_uri),
            direct_seed=ds, auxiliary=True)

    by_kind = {}
    for c in cells:
        by_kind.setdefault(c["kind"], []).append(c["cell_id"])
    requirements = required_checkpoints_v2(dataset, forget_set_id,
                                           router_seeds, edit_seeds,
                                           direct_seeds)
    return {
        "kind": spec.kind,
        "dataset": dataset,
        "forget_set_id": forget_set_id,
        "forget_identity_ids": list(forget_ids),
        "retained_identity_ids": retained,
        "router_seeds": list(router_seeds),
        "edit_seeds": list(edit_seeds),
        "direct_seeds": list(direct_seeds),
        "existing_router_seed": EXISTING_ROUTER_SEED,
        "held_out_g": HELD_OUT_G[dataset],
        "vocab": vocab, "n_vocab": len(vocab),
        "deleted_label": DELETED_LABEL,
        "mediator_is_hard": {
            "statement": "Y = h(C): h receives a code and never an image",
            "image_to_h_in_any_condition": any(
                c["image_to_h"] for c in spec.conditions.values()),
            "prompt_used_for_h": prompts["h_code_to_alias"],
            "prompt_read_from": prompts["read_from"],
            "auxiliary_conditions_reported_separately":
                list(spec.auxiliary_conditions),
        },
        "conditions": {n: dict(c) for n, c in spec.conditions.items()},
        "decisive_conditions": list(spec.decisive_conditions),
        "prompts": prompts,
        "route_protocol": protocol,
        "cells": cells,
        "n_cells": len(cells),
        "cells_by_kind": by_kind,
        "n_rows_total": sum(c["n_rows"] for c in cells),
        "rows_are_not_duplicated_across_router_seeds": (
            "forced-code interventions are one cell per EDIT seed only; the "
            "router seed varies the natural cells, which are the only ones that "
            "involve g"),
        "checkpoint_requirements": requirements,
        "score_sum_semantics": SCORE_SUM_SEMANTICS,
        "p_h_is_not_a_score_sum": (
            "the factorization uses the empirical hard-response matrix "
            "H_e(c, y) = 1{parse(h_e(c)) = y}; candidate score sums are still "
            "reported where a phase records them, but never as P_h"),
        "delta_route_definition": DELTA_ROUTE_V2,
        "gate_thresholds": dict(spec.thresholds),
        "gate_to_verdict": dict(spec.gate_to_verdict),
        "verdict_names": list(VERDICT_NAMES),
        "bootstrap_plan": BOOTSTRAP_PLAN_V2,
        "image_cluster_key": {
            "is": "the image's BYTES, not its filename",
            "digest_table_supplied": image_sha_by_uri is not None,
            "why": ("a dataset whose held-out filenames outnumber its held-out "
                    "byte-values would resample one picture several times and "
                    "call the copies independent, which understates every "
                    "interval; the pre-registration checks the row clusters "
                    "against the image-content audit at BOTH levels so the two "
                    "cannot disagree"),
        },
        "route_architecture_is_frozen": (
            "g, h, the prompts and the LoRA recipe are read from the frozen "
            "route scripts and never modified; the new adapters this design "
            "requires (g_42, g_123, D_s) are trained UNDER that same protocol, "
            "which is what makes them comparable to the existing route rather "
            "than a change to it"),
        "gates_are_independent_of_granularity": (
            "no threshold here is derived from the G6 granularity pilot"),
    }


def build_design_v2(dataset, forget_set_id, forget_ids, router_seeds,
                    edit_seeds, direct_seeds, man=None, images=None,
                    image_sha_by_uri=None):
    """The v2 design.  Thin: the one implementation is ``build_design_for``."""
    return build_design_for(PILOT_SPEC_V2, dataset, forget_set_id, forget_ids,
                            router_seeds, edit_seeds, direct_seeds, man=man,
                            images=images, image_sha_by_uri=image_sha_by_uri)


def build_design_v3(dataset, forget_set_id, forget_ids, router_seeds,
                    edit_seeds, direct_seeds, man=None, images=None,
                    image_sha_by_uri=None):
    """The v3 design: v2's cells plus the one baseline cell, and v3's gates.

    Every count in it still follows from the factors rather than being written
    down, so adding the baseline cell moves ``n_cells`` and ``n_rows_total``
    through the same arithmetic that produced them before rather than past a
    number somebody updated by hand.
    """
    return build_design_for(PILOT_SPEC_V3, dataset, forget_set_id, forget_ids,
                            router_seeds, edit_seeds, direct_seeds, man=man,
                            images=images, image_sha_by_uri=image_sha_by_uri)


def _design_constructor(spec):
    """The public design constructor for one spec, looked up by name at call time.

    The pilot constructor calls the SAME named function ``verify_manifest``
    calls, so freeze and verify reach the design through one door.  Two call
    sites that both happen to end at ``build_design_for`` agree today and are
    still two places for a later edit to reach only one of -- and a freeze that
    built a design verify cannot rebuild is the one failure this whole file
    exists to make impossible.

    Looked up rather than bound into the spec: the spec table is built before
    these functions exist, and binding the object at import time would also
    freeze out a caller that replaced the wrapper -- including the tests that
    check the pilot constructor refuses a design whose cell counts do not follow
    from its factors, and which can only do that by replacing it.
    """
    name = f"build_design_{spec.version}"
    fn = globals().get(name)
    if fn is None:
        raise RuntimeError(
            f"spec {spec.version!r} has no design constructor {name!r}; a "
            f"version whose pilot cannot be built is a version that cannot be "
            f"frozen or verified")
    return fn


#: The repairs v2 makes, each tied to the v1 field it contradicts.  Recorded in
#: the supersession file rather than only in a commit message, so a reader who
#: finds a v1 manifest in the tree can see why it is not the design to run.
SUPERSESSION_ITEMS = OrderedDict((
    ("hard_mediator", {
        "v1": "forced-code rows used CODE_IMG_PROMPT, so h received the image "
              "and the system was f(X, C) rather than the frozen Y = h(C)",
        "v2": "h receives a code and nothing else; the image-plus-code "
              "condition is retained as hybrid_conflict_probe and reported as "
              "an auxiliary bypass/conflict test, never as the intervention",
        "also": "v1 used the prompt 'Identity code: {code}. What is the "
                "alias?', which is not the CODE_TO_ALIAS_PROMPT the route was "
                "trained with; v2 reads the prompt from the frozen scripts and "
                "requires all three to agree"}),
    ("direct_control", {
        "v1": "checkpoint_requirements.roles.direct_condition.adapter was "
              "'reuses edited_h'",
        "v2": "separate D_s: X -> Y adapters, independently trained and hashed, "
              "image-only, with a pre-edit held-out accuracy and a "
              "direct_image_accuracy >= 0.90 gate, because bounding "
              "code-following from above alone is satisfied by a model that "
              "emits arbitrary wrong labels"}),
    ("router_seed_is_a_factor", {
        "v1": "three router seeds declared over one baseline_g checkpoint and "
              "one cached prediction file, with three cells indexed by EDIT "
              "seed only, so the router seed was a label rather than an "
              "experimental factor",
        "v2": "one checkpoint and one held-out prediction file per router seed, "
              "each hashed; the real 3_g x 3_h natural cells on SALMU; "
              "forced-code interventions once per EDIT seed, because they do "
              "not depend on g and repeating them would count one measurement "
              "three times"}),
    ("phases", {
        "v1": "RF0/RF1/RF2/RF2P, where RF1 raised NotImplementedError",
        "v2": "RF0, RF1G, RF1D, RF1H, RF1E, RF2, RF2P, with --router-seed, "
              "--edit-seed, --direct-seed, --device, --resume, --out and "
              "atomic per-cell writes; RF2 and RF2P fail on absent inputs"}),
    ("verdicts", {
        "v1": "one omnibus 'passed' boolean over seven gates",
        "v2": "mediation_pass, routing_reliability_pass and "
              "routing_factorization_pass, each three-state so that a question "
              "a dataset cannot answer is recorded as not_established rather "
              "than coerced to True or False"}),
    ("inference_unit", {
        "v1": "image-level bootstrap selected on SALMU as 'a real refinement "
              "over identity-level', 36 clusters",
        "v2": "identity-level PRIMARY (12 clusters on SALMU) with image-level "
              "as a SENSITIVITY analysis: three images of one person share an "
              "identity, a router and an edited h, so they are not independent"}),
    ("p_h", {
        "v1": "the factorization took P_h from candidate score sums",
        "v2": "P_h is the empirical hard-response matrix H_e(c, y) = "
              "1{parse(h_e(c)) = y}, composed with the empirical hard router "
              "confusion; score sums are reported only as auxiliary and never "
              "as probabilities"}),
    ("manifests", {
        "v1": "rf_pilot_ppubench.json and rf_pilot_salmu.json",
        "v2": "rf_pilot_ppubench_v2.json and rf_pilot_salmu_v2.json; the v1 "
              "files are preserved byte-identical and their designs are still "
              "reconstructible through the v1 constructor, which is retained "
              "for exactly this reason"}),
))


#: The repairs v3 makes over v2.  v2's own repairs are still carried in a v3
#: record -- a reader who finds a v1 manifest in the tree needs both sets to
#: understand what replaced it -- and each item names the two versions it moves
#: between, so the progression reads v1 -> v2 -> v3 rather than as one flat list
#: of complaints about v1.
SUPERSESSION_ITEMS_V3 = OrderedDict((
    ("preregistration_survives_its_own_training", {
        "v2": "build_pilot_preregistration_v2 called verify_checkpoints_v2 and "
              "embedded the result inside the design that design_sha256 covers, "
              "so the hash included which adapters were on disk.  Training the "
              "one role a pilot was waiting for changed what the constructor "
              "produced, the manifest stopped reproducing at the moment it "
              "became runnable, RF0 reported 'the manifest does not verify' "
              "instead of runnable_now, and every phase that verifies before "
              "loading refused to load it.  A v2 pilot could never be run",
        "v3": "the frozen design carries the immutable role DECLARATIONS only; "
              "presence, digests, completeness and readiness are computed live "
              "by checkpoint_readiness, outside design_sha256, so a manifest "
              "still reproduces after any subset of its training work lands.  A "
              "design may bind the bytes it is a statement about; it may not "
              "bind the state of work it has not done yet"}),
    ("training_image_paths", {
        "v2": "evaluation resolved image URIs through resolve_recorded_path but "
              "RouteSessionV2.train called Image.open(uri) directly, and SALMU "
              "records its URIs relative to the dataset root, so RF1G and RF1D "
              "failed unless launched from exactly that directory -- failing as "
              "a missing file, which reads as broken data",
        "v3": "training opens images through the same _open_image the "
              "evaluation phases use, so one rule decides where every recorded "
              "path is, and a test changes CWD before training a mocked SALMU "
              "example to keep it that way"}),
    ("rf1g_runs_against_the_preregistration", {
        "v2": "RF1G was the only phase that never called load_prereg, and "
              "check_seed_budget permits any seed where held-out content exists, "
              "so --router-seed 999 could consume a full GPU training run that "
              "no verdict could use and no manifest named",
        "v3": "RF1G verifies the manifest, requires the seed to appear in the "
              "preregistered router_seeds, requires the matching checkpoint role "
              "to exist, and records both the design SHA and the manifest SHA in "
              "the result it files"}),
    ("run_provenance", {
        "v2": "a filed cell bound its prompts and the weights it selected, but "
              "not the commit that executed it, whether the worktree was dirty, "
              "the SHA of the script that ran, the SHA of the pre-registration "
              "it ran against, the CLI that launched it, or the sibling modules "
              "it loaded -- so a cell could not be attributed to a state of the "
              "code",
        "v3": "start_of_run_provenance captures all of it once per run and every "
              "cell carries it; a result is a record of an execution and so "
              "should say what executed, which is exactly what a frozen design "
              "must not do"}),
    ("rf2_hashes_its_inputs", {
        "v2": "RF2 read the filed cells and trusted them: replacing an adapter "
              "after a cell was filed left an apparently valid aggregate, "
              "because nothing re-hashed the cell files or compared the weights "
              "on disk with the weights each cell said it used",
        "v3": "RF2 hashes every cell file it consumes and re-hashes every "
              "checkpoint and prediction file behind them, refusing where a "
              "digest no longer matches the one the cell recorded"}),
    ("baseline_h_is_evaluated", {
        "v2": "checkpoint_requirements described baseline_h as 'the before in "
              "before/after' and no cell ever ran it, so the runner measured "
              "h_e(do(C=C_f)) = Unknown without establishing that the same code "
              "produced its own alias before the edit; an h answering Unknown to "
              "every code would have satisfied every suppression gate",
        "v3": "one baseline cell, h_base(do(C=c)) for every code, gated at 1.0 "
              "on the forgotten arm to match forgotten_route_suppression "
              "exactly, and evaluated once because it depends on no factor"}),
    ("direct_gates_are_per_seed", {
        "v2": "direct_image_accuracy and direct_code_following_rate pooled "
              "every direct seed into one denominator, so adapters that failed "
              "the floor could be carried by one that cleared it comfortably",
        "v3": "both are required of EVERY direct seed, in the shape the router "
              "gate has always used, because a factor with three levels is three "
              "measurements and the level that fails is the one a pooled number "
              "hides"}),
))

#: Which repairs a version's record carries: its own and every earlier one.
REPAIRS_BY_VERSION = {
    "v2": SUPERSESSION_ITEMS,
    "v3": OrderedDict((*SUPERSESSION_ITEMS.items(),
                       *SUPERSESSION_ITEMS_V3.items())),
}

#: What is true of a superseded artifact OF A GIVEN KIND, keyed on the kind it
#: records rather than on the version replacing it: a v1 manifest binds absolute
#: checkpoint paths whether v2 or v3 supersedes it, and restating that once per
#: replacing version would be several copies of one fact free to disagree.
SUPERSEDED_NOTES = {
    PREREG_KIND: {
        "design_is_still_reconstructible": (
            "verify_manifest dispatches on kind and rebuilds a v1 pilot "
            "through build_pilot_preregistration, which is retained for "
            "exactly this reason, so the frozen design_sha256 still "
            "reproduces and the checkpoint digests still match"),
        "input_digests_will_report_drift": (
            "a v1 manifest binds the bytes of this script as it was when "
            "that manifest was frozen, and this script has since gained "
            "v2, so verify_manifest names the script digest as drifted and "
            "returns valid=False.  That is the expected state of a "
            "superseded artifact and not a defect: it records what was "
            "pre-registered at the commit above, and it is not re-frozen "
            "because a pre-registration changed after freezing was never a "
            "pre-registration"),
        "verification_is_bound_to_the_root_it_was_frozen_at": (
            "the v1 verification table recorded the checkpoint paths it "
            "hashed as ABSOLUTE, so a v1 manifest rebuilds to its own "
            "design_sha256 only on a checkout rooted where it was frozen; "
            "from another clone every role reads as absent and the rebuild "
            "does not reproduce.  v2 records the same paths relative to the "
            "dataset root and resolve_recorded_path still accepts the "
            "absolute shape, so this is reported as a property of the "
            "superseded artifacts rather than retro-fitted into bytes that "
            "are frozen"),
    },
    PREREG_V2_KIND: {
        "design_is_still_reconstructible": (
            "verify_manifest dispatches on kind and rebuilds a v2 pilot "
            "through build_pilot_preregistration_v2, so the frozen "
            "design_sha256 still reproduces -- for as long as every role the "
            "pilot was waiting for is still absent, which is the defect below"),
        "input_digests_will_report_drift": (
            "a v2 manifest binds the bytes of this script as it was when that "
            "manifest was frozen, and this script has since gained v3, so "
            "verify_manifest names the script digest as drifted and returns "
            "valid=False.  That is the expected state of a superseded artifact "
            "and not a defect, and it is not re-frozen because a "
            "pre-registration changed after freezing was never a "
            "pre-registration"),
        "design_sha256_covered_live_checkpoint_status": (
            "THE DEFECT v3 EXISTS FOR.  The v2 constructor embedded the result "
            "of verify_checkpoints_v2 -- which adapters were on disk, their "
            "digests, whether the pilot was complete -- inside the design that "
            "design_sha256 covers, so training the one role a pilot was waiting "
            "for invalidated the pre-registration that named it.  Reproduced "
            "before v3 was written: valid=True with one role outstanding, then "
            "valid=False with 'does not reproduce design_sha256' the moment that "
            "role appeared, RF0 reporting 'the manifest does not verify' instead "
            "of runnable_now, and load_prereg refusing so that RF1H, RF1E, RF2 "
            "and RF2P could not run either.  A v2 pilot was therefore unrunnable "
            "by construction and is superseded for a structural reason rather "
            "than a scientific one"),
        "paths_are_relative_so_they_still_resolve": (
            "v2 records its checkpoint paths relative to the dataset root, so "
            "unlike v1 it is not bound to one filesystem root; that repair is "
            "carried forward unchanged"),
    },
}

SUPERSESSION_POLICY = {
    "v2": ("the v1 manifests are preserved unmodified and their designs are "
           "still reconstructible; they are marked superseded here rather than "
           "edited, because a pre-registration changed after freezing was never "
           "a pre-registration"),
    "v3": ("the v1 and v2 manifests are preserved unmodified and their designs "
           "are still reconstructible; they are marked superseded here rather "
           "than edited, because a pre-registration changed after freezing was "
           "never a pre-registration.  v2 is superseded for a structural reason "
           "rather than a scientific one: its design hash covered which adapters "
           "were on disk, so it could not survive the training it required"),
}


def supersession_record(spec, superseded_paths=()):
    """Mark the manifests this version replaces as superseded, without touching
    their bytes.

    They are legitimately frozen and unexecuted, so they stay exactly as they
    are: deleting or rewriting them would destroy the record of what was
    pre-registered and why it was replaced.  What changes is that a reader can
    see, from the tree, which design to run.

    Only STABLE facts are recorded here -- each superseded file's digest and the
    commit it was frozen at, both read out of its own frozen bytes.  A live
    verification verdict is deliberately absent: this block sits inside the
    design and so is covered by ``design_sha256``, and embedding a result that
    changes as the tree changes would make the manifest irreproducible from its
    own recorded inputs.  That is the same mistake v2 made with its checkpoint
    table, and it is why the rule is stated here rather than left implicit.
    """
    defaults = [DATASET_ROOT / MANIFEST_DIR / n
                for n in spec.superseded_filenames]
    entries = []
    for p in (list(superseded_paths) or defaults):
        p = Path(p)
        frozen = (json.loads(p.read_text(encoding="utf-8"))
                  if p.is_file() else {})
        notes = SUPERSEDED_NOTES.get(frozen.get("kind"))
        if frozen.get("kind") is not None and notes is None:
            raise RuntimeError(
                f"{_rel(p)} records kind {frozen.get('kind')!r}, which has no "
                f"supersession notes; the kinds this record can describe are "
                f"{sorted(SUPERSEDED_NOTES)}.  Describing an artifact with "
                f"another kind's limitations would state something false about "
                f"it, and stating nothing would hide what a reader needs")
        entries.append({
            "path": _rel(p), "present": p.is_file(),
            "sha256": sha256_file(p) if p.is_file() else None,
            "bytes": p.stat().st_size if p.is_file() else None,
            "kind": frozen.get("kind"),
            "design_sha256": frozen.get("design_sha256"),
            "frozen_at_commit": (frozen.get("provenance") or {})
            .get("executing_commit"),
            "status": f"superseded_by_{spec.version}",
            "bytes_preserved": True,
            "executed": frozen.get("executed"),
            **(notes or {}),
        })
    repairs = REPAIRS_BY_VERSION[spec.version]
    return {
        "kind": spec.supersession_kind,
        "authoritative_design": spec.version,
        "superseded": entries,
        "repairs": repairs,
        "n_repairs": len(repairs),
        "policy": SUPERSESSION_POLICY[spec.version],
    }


def pilot_cell_plan_v2(dataset, router_seeds, edit_seeds, direct_seeds,
                       n_forget_sets):
    """How many cells a promoted factorial would run, by kind.

    Counted from what each measurement depends on rather than from one product,
    because the kinds have different factors: an intervention cell does not
    involve g, and a direct cell involves neither g nor h nor the forget set.
    """
    n_g, n_h, n_d = len(router_seeds), len(edit_seeds), len(direct_seeds)
    return {
        "dataset": dataset,
        "n_router_seeds": n_g, "n_edit_seeds": n_h, "n_direct_seeds": n_d,
        "n_forget_sets": n_forget_sets,
        "intervention_cells": n_forget_sets * n_h,
        "natural_cells": n_g * n_forget_sets * n_h,
        "direct_cells": n_d,
        "hybrid_cells": n_d,
        "direct_cells_do_not_vary_by_forget_set": (
            "D_s is trained on the same associations whatever is forgotten and "
            "is not modified when h is edited, so it is one adapter set per "
            "dataset rather than one per cell"),
        "total_cells": (n_forget_sets * n_h + n_g * n_forget_sets * n_h
                        + 2 * n_d),
        "counts_derived_not_written_down": True,
    }


def promotion_rule_v2(dataset, router_seeds, edit_seeds, direct_seeds,
                      inventory, applicability):
    """What a promoted v2 factorial would run, and what it must train first.

    Built on the v1 common block so the reuse inventory and the forget-set count
    stay derived from the frozen matrix, then corrected for the fact that the
    cell kinds have DIFFERENT factors: v1's single ``n_e2e_cells`` product
    implied every measurement varies with the router seed, and item 3 is exactly
    the observation that forced-code interventions do not.
    """
    rule = _promotion_common(dataset, router_seeds, edit_seeds, inventory)
    n_sets = inventory["n_forget_sets"]
    rule["cell_plan"] = pilot_cell_plan_v2(dataset, router_seeds, edit_seeds,
                                           direct_seeds, n_sets)
    rule["n_e2e_cells"] = rule["cell_plan"]["natural_cells"]
    rule["n_cells_total"] = rule["cell_plan"]["total_cells"]
    rule["v1_product_was_wrong_because"] = (
        "v1 reported one product, 3 x sets x 3, as the cell count; that is the "
        "NATURAL cell count only.  Intervention cells do not involve g and "
        "direct cells involve neither g nor the forget set, so multiplying them "
        "by the router seeds would run the identical call three times and file "
        "it as three observations")

    support = applicability["verdict_support"]
    unsupported = sorted(n for n in VERDICT_NAMES
                         if not support[n]["supportable"])
    rule["blocked_by"] = (None if not unsupported else
                          f"the {unsupported} verdicts are not supportable on "
                          f"this dataset: "
                          + "; ".join(support[n]["reason"] for n in unsupported))
    rule["conditions"] = [
        "the mediation verdict passes on the pilot",
        ("no gate threshold is altered between pilot and promotion; a threshold "
         "chosen once the data exists is not a gate"),
        ("every router seed in the promotion has its OWN checkpoint and its own "
         "held-out prediction file, hashed at promotion time, because a "
         "checkpoint replaced in place keeps its path"),
        ("every direct seed has its own D_s checkpoint, and no D_s is edited "
         "when an h is edited"),
        ("a verdict recorded as not_established on the pilot stays "
         "not_established on the promotion unless the dataset changes"),
    ]
    # Where the router-training budget goes follows from the audit, not from the
    # dataset's name: item 3 spends it where an additional router can actually
    # establish held-out routing, and nowhere else.
    if applicability["held_out_image_content_exists"]:
        rule["router_budget_goes_here"] = (
            "three real routers, so the natural cells are a genuine "
            "router-seed x edit-seed factorial over images this dataset never "
            "trained on")
    else:
        rule["router_budget_is_not_spent_here"] = (
            "this pilot stays at three edited-h intervention cells and replays "
            "the one frozen router it has: an additional router seed on a "
            "dataset whose test bytes are its train bytes could only measure "
            "seed stability on repeated content and could never establish "
            "held-out routing, so the router-training budget goes to a dataset "
            "that can use it")
    return rule


def pilot_seed_policy_v2(dataset, man=None):
    """Which seeds the v2 pilot varies, and why the budgets differ.

    Item 3: the router-training budget goes to SALMU only.  PPUBench's test
    images are the same bytes as its training images, so an additional router
    seed there could only measure seed stability on repeated content and could
    never establish held-out routing -- training one would spend GPU time on a
    question the dataset cannot answer.  PPUBench therefore keeps three
    EDITED-H intervention cells, which is what its mediation verdict needs, and
    replays the one frozen router it already has.

    Derived from the image-content audit rather than written down per dataset,
    so replacing PPUBench's images with genuinely held-out ones would make the
    routing seeds available instead of leaving a stale exemption in the code.
    """
    man = man if man is not None else load_manifest(dataset)
    held_out = image_content_audit(man)["held_out_image_content_exists"]
    all_seeds = (17, 42, 123)
    if all_seeds[0] != EXISTING_ROUTER_SEED:
        raise RuntimeError(
            f"the seed triple {all_seeds} no longer starts with the router the "
            f"frozen route was established under ({EXISTING_ROUTER_SEED}), so "
            f"'replay the existing one' would replay a different seed")
    single = (all_seeds[0],)
    router_seeds = all_seeds if held_out else single
    direct_seeds = all_seeds if held_out else single
    return {
        "dataset": dataset,
        "router_seeds": list(router_seeds),
        "edit_seeds": list(all_seeds),
        "direct_seeds": list(direct_seeds),
        "held_out_image_content_exists": held_out,
        "router_seed_budget": (
            "spent here: three real routers, so the 3_g x 3_h natural cells are "
            "a genuine factorial" if held_out else
            f"not spent here: only the existing frozen g (seed "
            f"{EXISTING_ROUTER_SEED}) is replayed, because an additional router "
            f"seed on a dataset whose test bytes are its train bytes measures "
            f"seed stability on repeated content and cannot establish held-out "
            f"routing"),
        "edited_h_seeds_are_always_three": (
            "the mediation verdict is what every dataset can support, and "
            "edit-seed stability is what makes it a measurement rather than a "
            "single lucky training run"),
    }


def check_seed_budget(dataset, router_seeds, direct_seeds, man=None):
    """Refuse to train routers or direct models a dataset cannot use.

    A refusal rather than a note: the cost of ignoring it is two 3000-step
    trainings whose results cannot enter any verdict, and the artifact would
    still record three router seeds as if the seed were a factor.
    """
    policy = pilot_seed_policy_v2(dataset, man=man)
    if policy["held_out_image_content_exists"]:
        return policy
    extra_g = [s for s in router_seeds if s not in policy["router_seeds"]]
    extra_d = [s for s in direct_seeds if s not in policy["direct_seeds"]]
    if extra_g:
        raise RuntimeError(
            f"{dataset} was asked for router seeds {extra_g} beyond the frozen "
            f"g, but this dataset has no held-out image content, so those "
            f"routers could only measure seed stability on repeated training "
            f"bytes and no routing verdict would use them.  Item 3 spends the "
            f"router-training budget on SALMU; freeze the {dataset} pilot with "
            f"router_seeds={policy['router_seeds']}")
    if extra_d:
        raise RuntimeError(
            f"{dataset} was asked for direct seeds {extra_d}; with no held-out "
            f"image content a second D_s would measure training-seed stability "
            f"on repeated bytes.  Freeze this pilot with "
            f"direct_seeds={policy['direct_seeds']}")
    return policy


def _check_selection(spec, dataset, selection, forget_set_id, forget_ids):
    """That the target requested is the target the stated rule chose.

    One function for both shapes a selection can have, because the claim is the
    same in both: the pilot covers what the RULE picks, not what was asked for.
    A rule a request can override is a preference recorded after the fact, which
    is the thing ``pilot_forget_set``'s docstring calls a selection effect
    inside a pre-registration.
    """
    sets = selection.get("sets")
    if sets is None:
        if selection["set_id"] != forget_set_id:
            raise RuntimeError(
                f"the selection rule picks {selection['set_id']!r} for the "
                f"{dataset} {spec.version} pilot but {forget_set_id!r} was "
                f"requested; pass nothing and let the rule choose, or change "
                f"the rule in a new pre-registration -- not the request in "
                f"this one")
        if sorted(selection["targets"]) != sorted(forget_ids):
            raise RuntimeError(
                f"{forget_set_id} targets {selection['targets']} in the frozen "
                f"matrix but {list(forget_ids)} was requested")
        return
    chosen = [s["set_id"] for s in sets]
    if forget_set_id is not None:
        raise RuntimeError(
            f"the {spec.version} rule picks {len(chosen)} forget sets {chosen}, "
            f"so the design names them in forget_sets and records no single "
            f"forget_set_id; {forget_set_id!r} was requested")
    union = sorted(i for s in sets for i in s["targets"])
    if union != sorted(forget_ids):
        raise RuntimeError(
            f"the sets the rule picks {chosen} target {union} in the frozen "
            f"matrix but {sorted(forget_ids)} was requested")


def build_pilot_preregistration_for(spec, dataset, forget_set_id, forget_ids,
                                    router_seeds=None, edit_seeds=None,
                                    direct_seeds=None, man=None, images=None,
                                    selection=None, superseded_paths=(),
                                    forget_sets=None):
    """The frozen pilot of one version: corrected architecture, real checkpoint
    requirements, actual cell structure, identity-level inference, separate
    verdicts.

    This is the single construction path -- ``main --preregister`` and
    ``verify_manifest`` both call it -- so the artifact that is frozen and the
    artifact that is later rebuilt cannot be two implementations that happen to
    agree today.  Every count in it is multiplied out from the inputs rather
    than written down, and every expectation is computed from a recorded
    measurement against a frozen threshold.

    One builder for both versions rather than two: the differences are the
    fields of ``spec``, so a repair applied to one version cannot silently fail
    to reach the other, and a reader can enumerate every difference by reading
    the two spec literals.
    """
    man = man if man is not None else load_manifest(dataset)
    images = images if images is not None else held_out_images(man)

    # Seeds default to the pilot policy rather than to a fixed triple, so the
    # artifact records what THIS dataset can use.  A caller may still pass them
    # explicitly -- which is what verify_manifest does, replaying the frozen
    # values -- and check_seed_budget below refuses a request that would train
    # adapters no verdict could use.
    if router_seeds is None or edit_seeds is None or direct_seeds is None:
        default = pilot_seed_policy_v2(dataset, man=man)
        router_seeds = default["router_seeds"] if router_seeds is None \
            else list(router_seeds)
        edit_seeds = default["edit_seeds"] if edit_seeds is None \
            else list(edit_seeds)
        direct_seeds = default["direct_seeds"] if direct_seeds is None \
            else list(direct_seeds)

    # The audit comes FIRST because the design needs it: the image-level
    # bootstrap cluster is the image's content digest, and only the audit has
    # hashed the bytes.
    audit = image_content_audit(man)
    # A version whose rule picks several forget sets passes the LIST in this
    # position; every version so far passes None for forget_sets and so reaches
    # its constructor with the same single id it always did.
    design = _design_constructor(spec)(
        dataset, forget_sets if forget_sets is not None else forget_set_id,
        forget_ids, router_seeds, edit_seeds,
        direct_seeds, man=man, images=images,
        image_sha_by_uri=audit["images_by_uri_sha256"])
    applicability = gate_applicability_v2(audit, dataset, spec)
    plan = bootstrap_plan_for(audit, dataset)
    policy = check_seed_budget(dataset, router_seeds, direct_seeds, man=man)

    # The pilot's target comes from the stated selection rule applied to the
    # frozen matrix, not from a default and not from hand-picking a set after
    # inspecting which checkpoints exist.
    if selection is None:
        chooser = FORGET_SET_SELECTION_BY_VERSION.get(spec.version,
                                                      pilot_forget_set)
        selection = chooser(dataset, edit_seeds)
        _check_selection(spec, dataset, selection, forget_set_id, forget_ids)

    inventory = reuse_inventory(dataset, _promotion_forget_sets(dataset),
                                edit_seeds)
    promotion = promotion_rule_v2(dataset, router_seeds, edit_seeds,
                                  direct_seeds, inventory, applicability)

    # --- the cell structure is checked against what each kind depends on -----
    by_kind = design["cells_by_kind"]
    # How many forget sets this design varies.  One where the design declares no
    # forget_sets key at all -- which is every design frozen so far -- so the
    # multiplications below are by 1 and these are the counts v2, v3 and v4 have
    # always been checked against.
    n_sets = len(design.get("forget_sets") or (None,))
    expected_kind_counts = {
        "intervention": len(edit_seeds) * n_sets,
        "natural": len(router_seeds) * len(edit_seeds) * n_sets,
        "direct": len(direct_seeds),
        "hybrid": len(direct_seeds),
    }
    if spec.baseline_cell:
        # One per FORGET SET and never one per edit seed.  The baseline depends
        # on neither factor, so three copies per seed would be one measurement
        # reported three times -- but which codes are supposed to return their
        # own alias is what two forget sets differ in, so five sets have five
        # BEFOREs and collapsing them would compare an edit against a before
        # that was not its own.
        expected_kind_counts["baseline"] = n_sets
    # Cross-checked against the module's own derivation instead of a second
    # arithmetic written beside it: pilot_cell_plan_v2 already counts what a
    # factorial over several forget sets runs, and already states which kinds do
    # NOT vary by forget set.  Two counts of the same cells that agree only
    # until one of them is edited are not a check.
    _plan = pilot_cell_plan_v2(dataset, router_seeds, edit_seeds, direct_seeds,
                               n_sets)
    for _kind, _key in (("intervention", "intervention_cells"),
                        ("natural", "natural_cells"),
                        ("direct", "direct_cells"),
                        ("hybrid", "hybrid_cells")):
        if expected_kind_counts[_kind] != _plan[_key]:
            raise RuntimeError(
                f"the {spec.version} design implies {expected_kind_counts[_kind]} "
                f"{_kind} cells but pilot_cell_plan_v2 derives {_plan[_key]} "
                f"over {n_sets} forget set(s); one of the two is counting a "
                f"measurement more times than it depends on a factor")
    for kind, want in expected_kind_counts.items():
        got = len(by_kind.get(kind, []))
        if got != want:
            raise RuntimeError(
                f"the {spec.version} design has {got} {kind} cell(s) but the "
                f"factors imply {want}; a cell count that does not follow from "
                f"what the measurement depends on is the defect item 3 names")
    if len(by_kind.get("intervention", [])) != len(edit_seeds):
        raise RuntimeError("intervention cells were multiplied by router seed")

    n_held_out = sum(len(v) for v in images.values())
    for cell in design["cells"]:
        if cell["kind"] in IMAGE_CELL_KINDS + ("hybrid",) and \
                cell["n_rows"] != n_held_out:
            raise RuntimeError(
                f"cell {cell['cell_id']} has {cell['n_rows']} rows but there "
                f"are {n_held_out} held-out images")
        if cell["kind"] in CODE_CELL_KINDS and \
                cell["n_rows"] != len(man["identity_ids"]):
            raise RuntimeError(
                f"cell {cell['cell_id']} has {cell['n_rows']} rows but the "
                f"route has {len(man['identity_ids'])} codes, and a hard "
                f"mediator makes one call per code")

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
                "bytes": resolve_recorded_path(uri).stat().st_size,
            })

    # The row clusters at BOTH levels must equal the counts the plan resamples,
    # or an interval is computed over more units than exist.  This is the check
    # that catches a dataset whose held-out filenames outnumber its held-out
    # bytes: at the image level such a dataset would resample one picture
    # several times and call the copies independent, which narrows every
    # interval and makes a noisy result look precise.
    for level, want in (("primary", plan["primary_n_clusters"]),
                        ("sensitivity", plan["sensitivity_n_clusters"])):
        for kind in ("natural", "direct"):
            cell = next((c for c in design["cells"] if c["kind"] == kind), None)
            if cell is None:
                continue
            got = len(cluster_units_v2(cell["rows"], plan[level]["level"]))
            if got != want:
                raise RuntimeError(
                    f"the audit counts {want} "
                    f"{plan[level]['level']} cluster(s) among held-out image "
                    f"CONTENT but the rows of a {kind} cell fall into {got}; "
                    f"the {level} bootstrap resamples the latter, so at this "
                    f"level the two must agree")

    requirements = design["checkpoint_requirements"]
    # The live checkpoint probe runs ONLY where the version embeds its result.
    # v2 did, and that is the defect v3 exists for: design_sha256 then covered
    # which adapters were on disk, so the pre-registration stopped reproducing
    # at the exact moment the training it required succeeded.  v3 freezes the
    # immutable role declarations alone and computes presence, digests,
    # completeness and readiness in ``checkpoint_readiness``, outside the hash,
    # whenever somebody asks.
    checkpoints = (verify_checkpoints_v2(requirements)
                   if spec.embed_live_checkpoint_status else None)
    frozen_checkpoint_block = (
        {**requirements,
         "verification": checkpoints,
         "complete": checkpoints["complete"],
         "must_be_trained_before_this_pilot_can_run":
             checkpoints["must_be_trained"],
         "n_must_be_trained": checkpoints["n_must_be_trained"],
         "unexpectedly_absent": checkpoints["unexpectedly_absent"]}
        if checkpoints is not None else
        {**requirements,
         "readiness_is_computed_live_not_frozen": (
             "this table declares the ROLES this pilot needs, the file keys "
             "each role has and the relative path each file must occupy.  It "
             "does not record whether they are there.  Presence, digests, "
             "completeness and readiness are computed by checkpoint_readiness "
             "when somebody asks, outside design_sha256.  v2 recorded them "
             "inside the frozen design, which made design_sha256 a function of "
             "which adapters had been trained: the manifest stopped reproducing "
             "the moment training succeeded, RF0 reported 'the manifest does "
             "not verify' instead of runnable_now, and every phase that "
             "verifies before loading refused to load it"),
         "what_the_design_does_bind": (
             "the role names, the file keys each role declares, the relative "
             "path each file must occupy, and the seed lists the roles are "
             "multiplied out from -- all of them statements about the design, "
             "none of them changed by a training run finishing"),
         "where_an_input_that_already_existed_is_bound": (
             "a role declared exists_already is an input this design was built "
             "over rather than work it is waiting for, and its digest is bound "
             "in two places, neither of them inside design_sha256: "
             "checkpoint_readiness re-hashes every role file it finds whenever "
             "RF0 or verify_manifest runs, and the cell result that consumed the "
             "weights records the digest it used, which RF2 re-checks -- a "
             "mismatch is refused and an absence is reported.  It is "
             "deliberately NOT listed in provenance.input_file_sha256: adapters "
             "are gitignored, so a digest of one recorded at freeze time would "
             "make verify_manifest report drift in every fresh clone, where the "
             "clone is telling the truth about its own disk and the artifact "
             "would be unreadable because of it.  The design binds the design; "
             "the result binds the weights")})

    execution_policy = (
        "FROZEN AND NOT EXECUTED.  No cell of this pilot has been run, no "
        "router has been trained, no direct adapter has been trained and no "
        "matrix has been launched.  "
        + ("The checkpoint table below names exactly what must be trained "
           "first, and RF2 refuses to aggregate cells that do not exist"
           if spec.embed_live_checkpoint_status else
           "The checkpoint table below declares every role this pilot needs "
           "and where each one's files must live; which of them exist is "
           "computed live by RF0 rather than frozen here, and RF2 refuses to "
           "aggregate cells that do not exist"))

    # Derived rather than written down: what this version adds is whatever its
    # gate map and threshold table contain that the v2 ones do not, so the
    # statement cannot go stale when a gate is added or removed.
    new_gates = [g for g in spec.gate_to_verdict if g not in GATE_TO_VERDICT]
    new_thresholds = [k for k in spec.thresholds
                      if k not in GATE_THRESHOLDS_V2]

    # Both blocks below come from the spec's own declarations and are absent for
    # a version that made none, so the frozen v2 and v3 bytes do not move.  The
    # evidence function is looked up in a table keyed on the version rather than
    # named here: a builder that had to say "v4" is a builder v5 has to be
    # edited into.
    evidence_for = SCOPE_AMENDMENT_EVIDENCE_BY_VERSION.get(spec.version)
    scope_amendment = evidence_for(dataset, spec) if evidence_for else None
    lineage_declared = (
        OrderedDict((
            *spec.declared_cell_compatibility.items(),
            ("ancestor", ancestor_design_sha256(spec, dataset))))
        if spec.declared_cell_compatibility else None)

    return {
        **design,
        "kind": spec.prereg_kind,
        "preregistered": True,
        "executed": False,
        "execution_policy": execution_policy,
        "corrected_architecture": {
            "statement": "Y = h(C) with C = g(X); h receives a code and never "
                         "an image",
            "conditions_as_executed": {
                n: {"execution": c["execution"],
                    "image_to_h": c["image_to_h"],
                    "prompt_key": c["prompt_key"],
                    "auxiliary": bool(c.get("auxiliary", False))}
                for n, c in spec.conditions.items()},
            "natural_mediated": "c = g_s(X), then h_e(c)",
            "retained_route_intervention": "h_e(do(C=C_r))",
            "forgotten_route_intervention": "h_e(do(C=C_f))",
            "retained_control": "h_e(do(C=C_r)) -- the SAME call as the "
                                "retained-route intervention, reported under a "
                                "second name rather than run twice",
            "direct_control": "D_s(X), image only, separate adapter",
            "image_identity_on_forced_code_rows": (
                "retained as experimental metadata "
                "(held_out_images_of_this_identity) and never sent to h"),
            "auxiliary_bypass_probe": (
                "hybrid_conflict_probe presents an image and a code together, "
                "which makes the system f(X, C); it is reported as a "
                "bypass/conflict test and is not the causal mediator "
                "intervention"),
            "why_v1_was_wrong": SUPERSESSION_ITEMS["hard_mediator"],
            **({"baseline_forced_code": "h_base(do(C=c)) -- the unedited h, "
                                        "one code and no image",
                "why_the_baseline_is_part_of_the_architecture": (
                    "the four conditions above are all post-edit.  Without the "
                    "unedited h evaluated on the same codes, a suppression gate "
                    "is satisfied by any adapter that produces the refusal "
                    "label, edited or not, and the run measures post-edit "
                    "behaviour without establishing that the edit changed "
                    "anything")}
               if spec.baseline_cell else {}),
        },
        "seed_policy": policy,
        "held_out_images": {
            "n": len(held_out),
            "images": held_out,
            "content_audit": audit,
            "warning": (None if audit["held_out_image_content_exists"] else
                        "THESE IMAGES ARE NOT HELD OUT IN CONTENT: every test "
                        "image's bytes also appear in the train split, so the "
                        "split partitions filenames rather than images.  The "
                        "URIs and digests below are exact and the mediation "
                        "gates remain valid because they force the code, but no "
                        "held-out ROUTING claim can be made from them"),
        },
        "gate_applicability": applicability,
        # Top level, and not only inside the scope block: a reader deciding
        # whether an artifact was pre-registered should not have to open the
        # rationale to find out that it was amended after its own outcome.
        **({"post_outcome_scope_amendment":
                bool(spec.declared_hygiene_scope
                     .get("post_outcome_scope_amendment"))}
           if spec.declared_hygiene_scope else {}),
        **({"scope_amendment": scope_amendment} if scope_amendment else {}),
        **({"cell_design_lineage": lineage_declared}
           if lineage_declared else {}),
        "frozen_gates": {
            "thresholds": dict(spec.thresholds),
            "gate_to_verdict": dict(spec.gate_to_verdict),
            "n_gates": len(spec.gate_to_verdict),
            "supportable_on_this_dataset": applicability["n_gates_supportable"],
            "not_adjustable_afterwards": True,
            # How the rate gates AGGREGATE, which is the other half of the rule a
            # threshold states.  Comes from the spec rather than from a version
            # test here, so the difference between two versions stays a field a
            # reader can list; it is empty for a version whose bytes are frozen.
            **spec.declared_gate_aggregation,
            # Which rows the hygiene gate counts, which is the other half of
            # that gate's rule for the same reason the aggregation is the other
            # half of a rate gate's.
            **spec.declared_hygiene_scope,
            **({"added_in_this_version": {
                    "gates": new_gates, "thresholds": new_thresholds,
                    "why": ("a gate added after a design was frozen is not a "
                            "gate; these were in this pre-registration before "
                            "any cell was run")}}
                if new_gates or new_thresholds else {}),
            "independent_of_the_granularity_gates": (
                "no threshold is imported from the G6 granularity pilot"),
            "new_in_v2": ("direct_image_accuracy, because bounding "
                          "code-following from above alone is satisfied by a "
                          "model that emits arbitrary wrong labels"),
        },
        "verdicts_to_be_reported": {
            "names": list(VERDICT_NAMES),
            "three_state": ("pass / fail / not_established; a question this "
                            "dataset cannot answer is recorded as "
                            "not_established rather than coerced to True "
                            "(unmeasured success) or False (unobserved failure)"),
            "expected_on_this_dataset": applicability["expected_verdicts"],
            "no_omnibus_boolean": (
                "there is deliberately no combined 'passed' field: a routing "
                "failure and a mediation success answer different questions, "
                "and one boolean would let either cancel the other"),
            "expected_routing_reliability_false_because": (
                None if not applicability["expected_verdicts"][
                    "routing_reliability"].startswith("measured, expected")
                else
                f"the frozen g scores {HELD_OUT_G[dataset]['accuracy']} on "
                f"this dataset's held-out images against a floor of "
                f"{spec.thresholds['min_router_held_out_accuracy']}, so "
                f"routing_reliability_pass is expected to be False.  That does "
                f"not prevent the noisy-router factorization analysis, which is "
                f"the only condition where the factorization has something to "
                f"explain"),
        },
        "cluster_bootstrap": plan,
        "delta_route": DELTA_ROUTE_V2,
        "e2e_decomposition": {
            "prediction_formula": "sum_c P_g(c|X) * H_e(c, Y*(X))",
            "p_h_is": ("the empirical hard-response matrix H_e(c, y) = "
                       "1{parse(h_e(c)) = y}, read off h's own generations"),
            "p_h_is_not": ("a candidate score sum.  A score sum over "
                           "teacher-forced full-sequence labels has no "
                           "termination event and no normalization across "
                           "candidates, so it is not a probability and cannot "
                           "be substituted for one"),
            "p_g_is": ("the empirical hard router confusion: one code per "
                       "image, from that router seed's own held-out prediction "
                       "file, with an unparseable code counted as unroutable "
                       "rather than dropped"),
            "required_label_is_keyed_by": (
                "the IMAGE, not the code -- keying it by the code would make "
                "every routing mistake correct by construction and predict 1.0 "
                "for any router whatsoever"),
            "if_stochastic_decoding_is_ever_introduced": (
                "P_h must then be estimated from repeated generations under a "
                "preregistered sampling configuration recorded in a NEW "
                "pre-registration; greedy decoding needs no such estimate "
                "because h_e(c) is one call with one answer"),
            "what_a_disagreement_would_mean": (
                "under greedy decoding the composed system and this prediction "
                "coincide exactly whenever h is a function of the code alone, so "
                "a disagreement is evidence that the mediator is NOT hard -- that "
                "h also responded to the image, or that the router predictions "
                "replayed are not the ones the confusion matrix was built from, "
                "or that the parse differs between the per-code and the composed "
                "call.  The gate is therefore a test of the architecture item 1 "
                "asks to preserve, not a restatement of the observation"),
            "supportable_on_this_dataset":
                applicability["verdict_support"]
                ["routing_factorization"]["supportable"],
        },
        "checkpoint_requirements": frozen_checkpoint_block,
        "promotion_rule": {**promotion, "reuse_inventory": inventory},
        "pilot_forget_set_selection": selection,
        "missing_data_policy": MISSING_DATA_POLICY,
        "score_sum_semantics": SCORE_SUM_SEMANTICS,
        "supersession": supersession_record(spec, superseded_paths),
    }


def build_pilot_preregistration_v2(dataset, forget_set_id, forget_ids,
                                   router_seeds=None, edit_seeds=None,
                                   direct_seeds=None, man=None, images=None,
                                   selection=None, v1_paths=()):
    """The frozen v2 pilot.  Thin: one implementation is
    ``build_pilot_preregistration_for``.

    Retained because two v2 manifests are frozen and committed: a runner that
    could no longer produce the artifact it has on disk could not explain it
    either.  ``v1_paths`` keeps its v2 name so the manifests still rebuild from
    the same call.
    """
    return build_pilot_preregistration_for(
        PILOT_SPEC_V2, dataset, forget_set_id, forget_ids, router_seeds,
        edit_seeds, direct_seeds, man=man, images=images, selection=selection,
        superseded_paths=v1_paths)


def build_pilot_preregistration_v3(dataset, forget_set_id, forget_ids,
                                   router_seeds=None, edit_seeds=None,
                                   direct_seeds=None, man=None, images=None,
                                   selection=None, superseded_paths=()):
    """The frozen v3 pilot.

    Same construction path as v2 with a different spec, so every check v2 ran
    -- the cell counts against the factors, the row counts against the held-out
    images, the cluster counts against the content audit, the selection rule
    against the frozen matrix -- runs here too.  A second constructor would
    have to be given all of them again, and the ones it was not given would be
    the ones nobody noticed missing.
    """
    return build_pilot_preregistration_for(
        PILOT_SPEC_V3, dataset, forget_set_id, forget_ids, router_seeds,
        edit_seeds, direct_seeds, man=man, images=images, selection=selection,
        superseded_paths=superseded_paths)


# ---------------------------------------------------------------------------
# v2 scoring -- ONE implementation, shared by the GPU phases and by RF2P
# ---------------------------------------------------------------------------

#: Which stored field holds the generation for a condition, keyed by the prompt
#: that produced it.  h's output and D_s's output are different measurements of
#: different models, so they are stored under different names and a scorer that
#: read the wrong one would score a direct answer as a mediated one.
GENERATION_FIELD_V2 = {"h_code_to_alias": "h_raw_text",
                       "d_image_to_alias": "d_raw_text",
                       "hybrid_image_and_code": "d_raw_text"}


def generation_field(condition, conditions=None):
    """Which stored field holds this condition's generation.

    Defaults to the widest condition table this runner builds designs from, so a
    condition a later version added resolves here too.  Reading the v2 table
    alone made the baseline condition -- whose prompt is the same
    ``h_code_to_alias`` every other mediated condition uses, and whose output
    therefore belongs in the same field -- raise as unknown instead of resolving,
    which would have failed at scoring time rather than at design time.

    A condition the runner genuinely does not know still raises, and names the
    table it was looked up in so the failure says which design was assumed.
    """
    table = CONDITIONS_V3 if conditions is None else conditions
    try:
        return GENERATION_FIELD_V2[table[condition]["prompt_key"]]
    except KeyError:
        raise RuntimeError(
            f"condition {condition!r} has no generation field; the conditions "
            f"this runner knows are {list(table)}") from None


def score_rows_v2(rows, vocab, conditions=None):
    """Attach a strict score to every row that carries a stored generation.

    This is the ONLY scoring implementation.  RF1B, RF1H, RF1D and RF1E call it
    as they write a cell; RF2 reads what they stored; RF2P calls it again on the
    stored raw text and compares.  Two scorers would be two verdicts, and the
    granularity pilot already showed what a scoring defect costs once the raw
    outputs are all that survive: a punctuation-asymmetric matcher scored a
    dozen byte-exact-correct rows as unparseable and the whole verdict had to be
    re-derived from stored text.

    ``conditions`` defaults to the WIDEST table this runner declares rather than
    to v2's, because one scorer has to serve every version and a version-aware
    scorer would be two scorers with a switch between them.  v3's table is v2's
    items plus one, so a v2 row reads the identical entry it always did and a
    baseline row reads the entry its own design declared.  Reading v2's table
    here raised a KeyError on the baseline condition, which would have failed at
    scoring time -- after a GPU run had already produced the generations.

    Rows without a generation are reported rather than skipped, because a design
    that silently scored eight of twelve rows would still produce a rate and the
    rate would be about the eight.
    """
    table = CONDITIONS_V3 if conditions is None else conditions
    scored, missing = [], []
    for row in rows:
        field = generation_field(row["condition"], table)
        if field not in row or row[field] is None:
            # An unroutable natural row has no h call to score: g emitted no
            # code, so the composition produced nothing.  That is recorded as a
            # routing failure by the confusion matrix, not as a missing row.
            # ``is False`` rather than a falsy test on purpose: a row frozen
            # before any router existed carries routable=None, which means the
            # observation is PENDING and must be reported missing, not scored as
            # an end-to-end failure.
            if row["condition"] == "natural_mediated" and \
                    row.get("routable") is False:
                scored.append({
                    **row, "raw_text": None, "recognized_labels": [],
                    "parsed_label": None, "n_recognized": 0,
                    "unparseable": False, "multi_label_ambiguous": False,
                    "usable": False, "correct": False,
                    "follows_forced_code": None, "follows_image": None,
                    "names_forced_code_label": None,
                    "is_refusal": False,
                    "unscored_because": ("g emitted no recognizable code, so "
                                         "h was never called; counted as an "
                                         "end-to-end failure"),
                })
                continue
            missing.append(row["row_id"])
            continue
        s = score_raw_text(row[field], vocab)
        rule = table[row["condition"]]["expectation"]
        forced_label = row.get("forced_label")
        scored.append({
            **row, **s,
            "generation_field": field,
            "correct": s["parsed_label"] == row["expected_label"] and s["usable"],
            "follows_forced_code": (
                s["parsed_label"] == row["expected_label"] and s["usable"]
                if rule == "follows_code" else None),
            # For the image-following conditions the required label IS the
            # image's own alias, so this is the same comparison as ``correct``
            # and is recorded separately only to make the row self-describing.
            "follows_image": (
                s["parsed_label"] == row["expected_label"] and s["usable"]
                if rule == "follows_image" else None),
            # The auxiliary probe's actual question: did a code that has nothing
            # to do with the image capture a pathway never trained on codes?
            "names_forced_code_label": (
                s["parsed_label"] == forced_label and s["usable"]
                if forced_label else None),
            "is_refusal": s["parsed_label"] == DELETED_LABEL,
            "unscored_because": None,
        })
    return scored, missing


#: The fields a rescore is allowed to change.  Anything outside this set means
#: the stored row and the frozen design disagree about the EXPERIMENT rather
#: than about the parse, and RF2P refuses rather than reporting a difference.
RESCORABLE_FIELDS_V2 = ("parsed_label", "recognized_labels", "n_recognized",
                        "unparseable", "multi_label_ambiguous", "usable",
                        "correct", "follows_forced_code", "follows_image",
                        "names_forced_code_label", "is_refusal")

#: The fields that must NOT change between a stored row and the design row it
#: claims to come from.  A cell whose expectation was edited after the run would
#: otherwise rescore cleanly against a question nobody pre-registered.
DESIGN_BOUND_FIELDS_V2 = ("row_id", "condition", "expected_label", "forced_code",
                          "image_uri", "edit_seed", "image_to_h")


def check_rows_match_design(stored, design_rows):
    """Refuse a cell whose stored rows are not the design's rows.

    Compared on the fields that define the experiment, not on the fields a
    rescore legitimately recomputes.  Without this, RF2P would happily re-score
    a cell that had been pointed at a different expectation, and both RF2 and
    RF2P would agree on a result the pre-registration does not describe.
    """
    by_id = {r["row_id"]: r for r in design_rows}
    problems = []
    if len(stored) != len(design_rows):
        problems.append(f"the cell stores {len(stored)} rows but the frozen "
                        f"design has {len(design_rows)}")
    seen = set()
    for row in stored:
        rid = row.get("row_id")
        seen.add(rid)
        want = by_id.get(rid)
        if want is None:
            problems.append(f"row {rid!r} is not in the frozen design")
            continue
        for f in DESIGN_BOUND_FIELDS_V2:
            if f in want and row.get(f) != want.get(f):
                problems.append(
                    f"row {rid}: {f} is {row.get(f)!r} in the stored cell but "
                    f"{want.get(f)!r} in the frozen design")
    absent = sorted(set(by_id) - seen)
    if absent:
        problems.append(f"{len(absent)} design row(s) are absent from the cell, "
                        f"e.g. {absent[0]}")
    if problems:
        raise RuntimeError("the stored cell does not match the frozen design: "
                           + "; ".join(problems[:8])
                           + ("; ..." if len(problems) > 8 else ""))
    return True


# ---------------------------------------------------------------------------
# v2 cell results: where they live, and how a resume decides they are current
# ---------------------------------------------------------------------------

def cell_dir(dataset, cell_id, out=None):
    root = Path(out) if out else (DATASET_ROOT / CELLS_DIR_V2 / dataset)
    return root / cell_id


def cell_result_path(dataset, cell_id, out=None):
    return cell_dir(dataset, cell_id, out) / "cell_results.json"


#: The report file each phase writes, by phase name.  One table so ``--out``
#: cannot mean a directory in one phase and a file in another: RF0 used to take
#: it as a file while RF2 and RF2P took it as a directory, and passing a
#: directory to RF0 failed inside ``os.replace`` with an IsADirectoryError
#: rather than saying what the flag means.
REPORT_FILENAMES = {
    "RF0": "rf0_{dataset}_{version}.json",
    "RF2": "rf_report_{dataset}_{version}.json",
    "RF2P": "rf_report_{dataset}_{version}_rescored.json",
}


def report_path(dataset, kind, out=None, spec=None):
    """Where a phase files its report.

    ``out`` is a DIRECTORY, the same reading ``--cells`` gives it.

    The names differ per phase on purpose: RF2P reproduces RF2 from stored raw
    text, and a reproduction that overwrote the original would leave no way to
    see whether the two agreed.  They differ per pilot VERSION for the same
    reason: a v3 report written over a v2 one would be a report whose name said
    it was the older design's, and the two designs do not have the same cells.
    """
    if kind not in REPORT_FILENAMES:
        raise RuntimeError(
            f"phase {kind!r} files no report; the phases that do are "
            f"{sorted(REPORT_FILENAMES)}")
    spec = spec or LATEST_PILOT_SPEC
    root = Path(out) if out else (DATASET_ROOT / REPORTS_DIR)
    return root / REPORT_FILENAMES[kind].format(dataset=dataset,
                                                version=spec.version)


def write_cell_result(dataset, cell_id, obj, out=None):
    """Atomically file one cell, and return the digest of what was written."""
    return atomic_write_json(cell_result_path(dataset, cell_id, out), obj)


def load_cell_result(dataset, cell_id, out=None, result_kind=RESULT_KIND_V2):
    """Read one filed cell, refusing if it is absent or unreadable.

    Absent is an error and not an empty result: a cell that was never run and a
    cell that measured nothing serialize differently but would otherwise
    aggregate the same way, and the aggregate is what the verdicts are read off.

    ``result_kind`` is the kind of the design being aggregated, which the caller
    reads off the manifest rather than assuming: a cell filed by a v3 phase is a
    v3 cell, and refusing it as "not a v2 cell result" would send the operator to
    look for a v2 problem in a v3 run.
    """
    path = cell_result_path(dataset, cell_id, out)
    if not path.is_file():
        raise RuntimeError(
            f"cell {cell_id!r} has no result at {path}; run --phase "
            f"{CELL_KIND_PHASE.get(cell_id.split('__')[0], 'RF1')} for it.  An "
            f"absent cell is a missing input, not a null result")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"cell {cell_id!r} at {path} is not readable JSON ({exc}); a "
            f"half-written file must not be resumed over as if it were "
            f"complete -- delete it and re-run the cell") from None
    if doc.get("kind") != result_kind:
        raise RuntimeError(
            f"cell {cell_id!r} at {path} has kind {doc.get('kind')!r}, expected "
            f"{result_kind!r}; this is not a "
            f"{RESULT_KIND_VERSION.get(result_kind, result_kind)} cell result "
            f"and aggregating it would mix designs")
    return doc


#: What a cell result must bind, so a resume can tell "already done" from
#: "done against different inputs".  Existence is not currency: a checkpoint
#: replaced in place keeps its path, and a prompt edited in the source keeps its
#: name, so a resume keyed on filenames alone reuses a result that no longer
#: describes the experiment.
RESUME_BINDING_FIELDS_V2 = ("cell_id", "kind", "phase", "router_seed",
                            "edit_seed", "direct_seed", "input_sha256",
                            "prompts", "n_rows")


def resume_decision(stored, want):
    """Whether a filed cell may be reused, and if not exactly why.

    Returns a decision rather than raising, because ``--resume`` skipping a cell
    is normal and skipping the WRONG cell is the failure; the caller reports the
    reason so a reused cell is a visible choice.
    """
    reasons = []
    for f in RESUME_BINDING_FIELDS_V2:
        if f not in want:
            continue
        if stored.get(f) != want[f]:
            reasons.append(f"{f}: filed {stored.get(f)!r}, now {want[f]!r}")
    n_stored = len(stored.get("rows") or [])
    if n_stored != want.get("n_rows"):
        reasons.append(f"the filed cell has {n_stored} rows but the design has "
                       f"{want.get('n_rows')}")
    return {"reuse": not reasons, "reasons": reasons,
            "policy": ("a filed cell is reused only if every input digest, both "
                       "prompts and the row count still match; otherwise it is "
                       "re-run, because a result produced by different weights "
                       "or a different prompt is a result about a different "
                       "experiment")}


# ---------------------------------------------------------------------------
# v2 aggregation: the pure layer both RF2 and RF2P run
# ---------------------------------------------------------------------------

def router_held_out_accuracy(dataset, router_seed, held_out):
    """One router seed's held-out accuracy, from its OWN prediction file.

    Computed rather than read off a recorded number: the gate, the confusion
    matrix and the natural cells all have to describe the same router, and the
    only way to guarantee that is to derive all three from one file.  A recorded
    accuracy next to a prediction file is two claims that can disagree.
    """
    preds = load_router_predictions(dataset, router_seed, held_out)
    n = ok = 0
    unknown = []
    for uri in held_out:
        got = preds["by_uri"][str(uri)].get("code_correct")
        if got is None:
            unknown.append(uri)
            continue
        n += 1
        ok += bool(got)
    if unknown:
        raise RuntimeError(
            f"router seed {router_seed}'s prediction file does not say whether "
            f"{len(unknown)} held-out image(s) were routed correctly (e.g. "
            f"{unknown[0]}); an accuracy over the rows that happen to carry the "
            f"field would describe a subset of the design")
    if n == 0:
        raise RuntimeError(f"router seed {router_seed} has no held-out images "
                           f"to score, so it has no held-out accuracy")
    return {"router_seed": router_seed, "accuracy": ok / n, "n": n,
            "n_correct": ok, "source": preds["path"],
            "source_sha256": preds["sha256"],
            "computed_from": ("that seed's own held-out prediction file, not "
                              "from a number recorded elsewhere")}


def cluster_units_v2(rows, level):
    """The resampling units a v2 bootstrap would use, by level.

    Separate from v1's ``cluster_units`` because the two key the image level
    differently: v1 rows carried one ``cluster_id``, while v2 rows carry an
    ``identity_cluster_id`` and an ``image_cluster_id`` that is the image's
    content digest.  Sharing one function would silently resample v2 rows by
    identity at the image level and report a sensitivity analysis that was never
    sensitive to anything.
    """
    if level not in ("identity", "image"):
        raise RuntimeError(f"unknown cluster level {level!r}; expected "
                           f"'identity' or 'image'")
    key = "identity_cluster_id" if level == "identity" else "image_cluster_id"
    units = OrderedDict()
    for r in rows:
        if r.get(key) is None:
            raise RuntimeError(f"row {r['row_id']} has no {key}, so it cannot "
                               f"be assigned to a {level} cluster")
        units.setdefault(r[key], []).append(r["row_id"])
    return {str(k): sorted(v) for k, v in sorted(units.items(), key=str)}


def cluster_bootstrap_v2(rows, statistic, level, name, n_resamples=2000,
                         seed=17, alpha=0.05):
    """Percentile bootstrap of any per-row statistic over resampled clusters.

    The statistic is recomputed from scratch on each resample rather than
    approximated from a variance formula, so a cluster whose rows all fail
    contributes exactly what it contributes to the point estimate.  Resamples
    the statistic refuses are counted rather than hidden: dropping them silently
    would bias the interval toward the resamples that happened to look like the
    original.
    """
    key = "identity_cluster_id" if level == "identity" else "image_cluster_id"
    by_cluster = OrderedDict()
    for r in rows:
        if r.get(key) is None:
            raise RuntimeError(
                f"row {r['row_id']} has no {key}, so it cannot be assigned to a "
                f"{level} bootstrap cluster; a row that belongs to no cluster "
                f"is silently dropped from every resample and narrows the "
                f"interval")
        by_cluster.setdefault(r[key], []).append(r)
    clusters = sorted(by_cluster, key=str)
    if len(clusters) < 2:
        raise RuntimeError(
            f"only {len(clusters)} {level} cluster(s) for {name}; a resample "
            f"cannot vary, so an interval computed here would restate the point "
            f"estimate as an interval of width zero")

    rng = random.Random(seed)
    draws, skipped = [], 0
    for _ in range(n_resamples):
        picked = [rng.choice(clusters) for _ in clusters]
        sample = [r for c in picked for r in by_cluster[c]]
        try:
            draws.append(statistic(sample))
        except (RuntimeError, ZeroDivisionError):
            skipped += 1
    if not draws:
        raise RuntimeError(
            f"all {n_resamples} resamples of {name} were refused by the "
            f"statistic; the design does not support an interval at the "
            f"{level} level")
    draws.sort()

    def quantile(q):
        idx = min(len(draws) - 1, max(0, math.ceil(q * len(draws)) - 1))
        return draws[idx]

    return {
        "statistic": name, "level": level, "n_clusters": len(clusters),
        "cluster_ids": [str(c) for c in clusters],
        "cluster_sizes_rows": {str(c): len(by_cluster[c]) for c in clusters},
        "point_estimate": statistic(rows),
        "n_resamples": n_resamples, "n_resamples_used": len(draws),
        "n_resamples_skipped": skipped, "seed": seed, "alpha": alpha,
        "ci_low": quantile(alpha / 2), "ci_high": quantile(1 - alpha / 2),
        "bootstrap_mean": sum(draws) / len(draws),
        "method": ("percentile bootstrap over resampled clusters with "
                   "replacement; the statistic is recomputed per resample"),
    }


def bootstrap_both_levels(rows, statistic, name, plan):
    """The primary identity-level interval and the image-level sensitivity one.

    Item 6.  Both are always reported: the sensitivity interval is not a
    refinement of the primary one, because images of one person share an
    identity, the router's training and the edited h, so they are not
    independent.  Where the two disagree, the identity-level interval is the one
    to believe.
    """
    out = {"primary": cluster_bootstrap_v2(
               rows, statistic, plan["primary"]["level"], name,
               plan["n_resamples"], plan["seed"], plan["alpha"]),
           "sensitivity": cluster_bootstrap_v2(
               rows, statistic, plan["sensitivity"]["level"], name,
               plan["n_resamples"], plan["seed"], plan["alpha"]),
           "which_to_believe": ("the primary identity-level interval; the "
                                "image-level one is a sensitivity analysis "
                                "whose units are not independent"),
           "exploratory": plan.get("exploratory", False)}
    if plan.get("image_level_is_degenerate"):
        out["sensitivity_is_degenerate"] = (
            "every identity has a single distinct held-out image, so the "
            "sensitivity interval restates the primary one")
    return out


def _mean_correct(rows):
    if not rows:
        raise RuntimeError("a rate over zero rows is not a small rate")
    return sum(1 for r in rows if r.get("correct")) / len(rows)


def _cell_scored(doc, design_cell, vocab, rescore):
    """One filed cell's scored rows, bound to the frozen design.

    ``rescore`` is the only difference between RF2 and RF2P: RF2 reads the
    scores the phase stored, RF2P recomputes them from the stored raw text and
    reports every field that moved.  Both run this function, so the two cannot
    drift into being two analyses.
    """
    check_rows_match_design(doc["rows"], design_cell["rows"])
    if not rescore:
        scored = doc.get("scored")
        if not scored:
            raise RuntimeError(
                f"cell {design_cell['cell_id']} filed no scored rows; RF2 does "
                f"not reconstruct them from raw text because that is RF2P's "
                f"job, and a cell with no scores has no contribution to make")
        missing = doc.get("rows_without_a_stored_generation") or []
        if len(scored) + len(missing) != len(doc["rows"]):
            raise RuntimeError(
                f"cell {design_cell['cell_id']} filed {len(scored)} scored rows "
                f"and {len(missing)} ungenerated ones for "
                f"{len(doc['rows'])} design rows; the cell is partial and a "
                f"partial cell is a cell about a different design")
        return scored, missing, []

    scored, missing = score_rows_v2(doc["rows"], vocab)
    by_id = {r["row_id"]: r for r in (doc.get("scored") or [])}
    changed = []
    for r in scored:
        old = by_id.get(r["row_id"])
        if old is None:
            changed.append({"row_id": r["row_id"], "field": None,
                            "stored": None, "rescored": "row absent from the "
                                                        "stored scored block"})
            continue
        for f in RESCORABLE_FIELDS_V2:
            if old.get(f) != r.get(f):
                changed.append({"row_id": r["row_id"], "field": f,
                                "stored": old.get(f), "rescored": r.get(f)})
    return scored, missing, changed


#: Which declared role file each ``input_sha256`` key names.  RF2 re-hashes the
#: file a cell says it used, and to do that it has to know which of the design's
#: roles the key refers to and which of that role's files it is -- the same role
#: table the phase resolved the path from, so a cell is checked against the
#: declaration it was filed under rather than against a path guessed later.
CELL_INPUT_BINDINGS = {
    "baseline_h": ("baseline_h", "adapter"),
    "edited_h": ("edited_h__seed{edit_seed}", "adapter"),
    "direct_adapter": ("direct_d__seed{direct_seed}", "adapter"),
    "router_predictions": ("router_g__seed{router_seed}",
                           "held_out_predictions"),
}

#: The same binding for a design that varies MORE THAN ONE forget set.  Five
#: forget sets have five edited-h adapters per edit seed, so a role name that
#: does not say which set it belongs to resolves five files to one role -- and
#: the cell would be re-hashed against whichever of the five the table named
#: first, which is a check that passes against weights it did not use.
EDITED_H_ROLE_MULTI = "edited_h__{forget_set}__seed{edit_seed}"


def _binding_table(prereg):
    """Which role-name templates one design's cells bind to.

    Read off the DESIGN rather than off the version, because the question is
    whether this pre-registration varies one forget set or several and not
    which version froze it.  A design that does not declare several gets the
    table every earlier design used, the same object rather than a copy, so
    nothing about a v1-v4 cell resolves any differently than it did.
    """
    if not prereg.get("forget_sets"):
        return CELL_INPUT_BINDINGS
    return {**CELL_INPUT_BINDINGS, "edited_h": (EDITED_H_ROLE_MULTI, "adapter")}


def edited_h_role_name(prereg, edit_seed, forget_set=None):
    """The declared role one edited h occupies, in this design's own naming.

    One function rather than an f-string at each of the three call sites, so a
    phase and ``verify_cell_inputs`` cannot name the same adapter differently --
    which is the disagreement ``CELL_INPUT_BINDINGS`` exists to prevent, and
    which five forget sets per edit seed make reachable for the first time.
    """
    template = _binding_table(prereg)["edited_h"][0]
    if "{forget_set}" in template and not forget_set:
        raise RuntimeError(
            f"this design varies {len(prereg.get('forget_sets') or ())} forget "
            f"sets, so an edited-h role has to say which one it belongs to; got "
            f"{forget_set!r} for edit seed {edit_seed}")
    return template.format(edit_seed=edit_seed, forget_set=forget_set)


#: Which of those keys one cell kind binds.  The phase that FILES a cell and RF2
#: that RE-CHECKS it both read this table, so they cannot disagree about what a
#: cell was supposed to consume: a filer that hashed one weight and a checker
#: that expected another reports a mismatch against weights nobody used.
CELL_KIND_INPUT_KEYS = {
    "baseline": ("baseline_h",),
    "intervention": ("edited_h",),
    "natural": ("edited_h", "router_predictions"),
    "direct": ("direct_adapter",),
    "hybrid": ("direct_adapter",),
}


def _role_for_binding(key, cell, table=CELL_INPUT_BINDINGS):
    """The declared role and file one recorded input key names for this cell.

    The role name is a template over the cell's own seeds, because the seeds are
    what distinguishes one level of a factor from another: resolving ``edited_h``
    without the cell's edit seed would check a natural cell against whichever
    edited h happened to be named first in the table.

    ``table`` is the binding table of the DESIGN the cell was filed under, for
    the same reason and one level up: a design with several forget sets has
    several edited-h adapters per edit seed, and the cell's own forget set is
    what tells them apart.
    """
    role_name, file_key = table[key]
    # ``forget_set`` is passed to every template and used only by the ones that
    # name it: str.format ignores a keyword its template does not contain, so
    # the single-forget-set templates resolve exactly as they always did.
    return role_name.format(edit_seed=cell.get("edit_seed"),
                            direct_seed=cell.get("direct_seed"),
                            router_seed=cell.get("router_seed"),
                            forget_set=cell.get("forget_set_id")), file_key


def cell_input_digests(prereg, cell):
    """Hash every role file one cell consumes, keyed as the cell will record it.

    Called by the phase BEFORE it generates anything, so the digest in a filed
    cell is of the weights the cell actually ran against rather than of whatever
    occupies the path when RF2 comes to look.  A file that is absent is refused
    here rather than recorded as a null: a phase cannot produce rows from weights
    it does not have, and filing a cell with a hole in its bindings would leave
    RF2 reporting the binding as not re-checkable when the real problem is that
    the phase should never have run.
    """
    kind = cell.get("kind") or cell.get("kind_of_cell")
    if kind not in CELL_KIND_INPUT_KEYS:
        raise RuntimeError(
            f"cell {cell.get('cell_id')!r} has kind {kind!r}, which binds no "
            f"declared role file; the kinds that do are "
            f"{sorted(CELL_KIND_INPUT_KEYS)}")
    roles = (prereg.get("checkpoint_requirements") or {}).get("roles") or {}
    table = _binding_table(prereg)
    out = OrderedDict()
    for key in CELL_KIND_INPUT_KEYS[kind]:
        role_name, file_key = _role_for_binding(key, cell, table)
        raw = (roles.get(role_name) or {}).get(file_key)
        if not raw:
            raise RuntimeError(
                f"cell {cell['cell_id']} consumes {key}, which the design binds "
                f"to {role_name}.{file_key}, and the frozen design declares no "
                f"such file -- so either the role table or the binding is wrong, "
                f"and a cell filed against the wrong one is checked against "
                f"nothing")
        p = resolve_recorded_path(raw)
        if not p.is_file():
            raise RuntimeError(
                f"cell {cell['cell_id']} consumes {role_name}.{file_key}, which "
                f"is absent at {_rel(p)}; a cell filed without hashing the "
                f"weights that produced it cannot be tied back to them, and RF2 "
                f"would report the binding as uncheckable instead of reporting "
                f"that this phase had no weights to run on")
        out[key] = sha256_file(p)
    return out


def verify_cell_design_lineage(spec, prereg, cells):
    """Which design produced each cell this aggregate is about to consume.

    Returns None for a version that declared no cell compatibility policy, so the
    reports of versions that never read another design's cells stay exactly the
    bytes they were.

    v4 reuses v3's result kind, and that is what lets it read the 28 filed cells
    instead of repeating 26 GPU phase invocations to write identical bytes back.
    Reusing a kind removes the only check that stood between an aggregate and a
    cell some other design produced, so the check moves to the thing that
    actually identifies a design: the design hash each cell recorded for the
    pre-registration it ran against.

    A cell matching neither this design nor the declared ancestor is REFUSED
    rather than annotated.  An aggregate that reported verdicts over rows
    produced under a design nobody named would be a result with no
    pre-registration behind it, which is the one thing this file exists to make
    impossible.
    """
    declared = spec.declared_cell_compatibility
    if not declared:
        return None
    block = prereg.get("cell_design_lineage") or {}
    ancestor = block.get("ancestor") or {}
    this_design = prereg.get("design_sha256")
    ancestor_design = ancestor.get("design_sha256")
    # A version that DECLARES an ancestor and has none is broken: its cells
    # could have come from two designs and only one is named.  A version that
    # declares none is not -- it consumes nothing but its own cells, which is
    # what makes an absent ancestor the strictest policy available rather than
    # a missing one.  Keyed on the version's own declaration so this stays a
    # property of the spec and not a guess from the shape of the document.
    if not ancestor_design and ANCESTOR_MANIFEST_BY_VERSION.get(spec.version):
        raise RuntimeError(
            f"{spec.version} declared a cell compatibility policy but the "
            f"pre-registration it was read from names no ancestor design; "
            f"without one, a cell filed under the ancestor cannot be told from "
            f"a cell filed under a design nobody declared")
    accepted = {this_design: "produced_under_this_design"}
    if ancestor_design:
        accepted[ancestor_design] = "reused_from_the_ancestor_design"
    per_cell, counts, unrecognized = OrderedDict(), {}, []
    for cell_id, doc in cells.items():
        prov = doc.get("run_provenance") or {}
        got = (doc.get("preregistration_design_sha256")
               or prov.get("preregistration_design_sha256"))
        kind = doc.get("kind")
        if kind != spec.result_kind:
            raise RuntimeError(
                f"cell {cell_id} has kind {kind!r} and {spec.version} consumes "
                f"{spec.result_kind!r}; a cell of another result kind is a cell "
                f"whose rows another design specified")
        state = accepted.get(got, "unrecognized_design")
        if state == "unrecognized_design":
            unrecognized.append(f"{cell_id} records design {got}")
        per_cell[cell_id] = OrderedDict((
            ("state", state),
            ("recorded_preregistration_design_sha256", got),
            ("recorded_pilot_version", prov.get("pilot_version")),
            ("recorded_preregistration_kind", prov.get("preregistration_kind")),
            ("recorded_executing_commit", prov.get("executing_commit")),
            ("result_kind", kind),
        ))
        counts[state] = counts.get(state, 0) + 1
    if unrecognized:
        raise RuntimeError(
            "cannot aggregate: " + "; ".join(unrecognized[:4])
            + (f" (and {len(unrecognized) - 4} more)"
               if len(unrecognized) > 4 else "")
            + f".  {spec.version} consumes cells produced under "
              + (f"its own design ({this_design}) or under the declared "
                 f"{ancestor.get('ancestor_version')} ancestor "
                 f"({ancestor_design})"
                 if ancestor_design else
                 f"its own design ({this_design}) alone")
              + "; a cell produced under any other design was produced by one "
              "this pre-registration does not name, and aggregating it would "
              "report a verdict no frozen design stands behind")
    return OrderedDict((
        ("rule", declared.get("rule")),
        ("this_design_sha256", this_design),
        ("ancestor", ancestor),
        ("per_cell", per_cell),
        ("n_cells", len(per_cell)),
        ("n_cells_by_state", OrderedDict(sorted(counts.items()))),
        ("states", list(declared.get("states") or ())),
        ("the_cells_were_not_relabelled",
         ("a reused cell keeps its ancestor's kind string, its ancestor's "
          "run_provenance and its own file bytes.  This block is the "
          "disclosure, not a rewrite")),
        ("why_this_check_exists",
         declared.get("what_a_v4_report_must_disclose")),
        ("policy", declared.get("what_is_checked_before_a_cell_is_consumed")),
    ))


def verify_cell_inputs(prereg, cells):
    """Re-hash the weights behind every filed cell and refuse on a mismatch.

    A cell binds the digest of the weights that produced its rows, and nothing
    checked that binding at aggregation time.  So an adapter replaced AFTER a
    cell was filed left an aggregate that looked valid while describing weights
    that no longer existed -- and the aggregate is the artifact the verdicts are
    read off, so the swap would have been invisible in every number reported.

    A MISMATCH is refused.  An ABSENT file is reported and is not a refusal,
    which is not leniency: RF2P exists to reproduce a run from stored raw
    generations with no GPU, and in a fresh clone the adapters are gitignored and
    legitimately not there.  Refusing on absence would break the one property
    that makes a filed run re-checkable later, and the rows a cell needs are
    stored in the cell.  What absence does mean is that the binding could not be
    re-checked, and saying so is the honest report.
    """
    roles = (prereg.get("checkpoint_requirements") or {}).get("roles") or {}
    # The same table the filing phase resolved through, read off the same
    # design: a checker that named roles differently from the filer would
    # compare a cell against an adapter it never used.
    table = _binding_table(prereg)
    per_cell, mismatched, unverifiable = OrderedDict(), [], []
    not_a_role_file = OrderedDict()
    n_compared = 0
    for cell_id, doc in cells.items():
        recorded = doc.get("input_sha256") or {}
        row = OrderedDict()
        # A cell that recorded FEWER bindings than its kind consumes is a cell
        # that was filed by a phase which did not know what it used.  Checked
        # here rather than only at filing time, because the filed cells are what
        # RF2 reads and a cell on disk may predate the check.
        kind = doc.get("kind_of_cell")
        for key in CELL_KIND_INPUT_KEYS.get(kind, ()):
            if key not in recorded:
                mismatched.append(
                    f"cell {cell_id} is a {kind} cell and recorded no {key} "
                    f"digest, so the weights that produced its rows were never "
                    f"bound and cannot be re-checked")
        for key, want in sorted(recorded.items()):
            binding = table.get(key)
            if binding is None:
                # A protocol digest or a design hash: bound to bytes that are
                # not one of the design's declared role files, so there is
                # nothing on disk to re-hash it against.  Named rather than
                # dropped, because a binding nobody can check is worth seeing.
                not_a_role_file[key] = cell_id
                row[key] = {"recorded": want, "bound_to_a_role_file": False}
                continue
            role_name, file_key = _role_for_binding(key, doc, table)
            raw = (roles.get(role_name) or {}).get(file_key)
            entry = {"role": role_name, "file_key": file_key, "recorded": want,
                     "bound_to_a_role_file": True}
            if not raw:
                entry["current"] = None
                entry["state"] = "the design declares no such file"
                mismatched.append(f"cell {cell_id} binds {key} to "
                                  f"{role_name}.{file_key}, which the design "
                                  f"does not declare")
            else:
                p = resolve_recorded_path(raw)
                entry["path"] = _rel(p)
                got = sha256_file(p) if p.is_file() else None
                entry["current"] = got
                if not p.is_file():
                    entry["state"] = "absent, so the binding was not re-checked"
                    unverifiable.append(f"{cell_id}:{key}")
                elif got == want:
                    entry["state"] = "matches"
                    n_compared += 1
                else:
                    entry["state"] = "MISMATCH"
                    mismatched.append(
                        f"cell {cell_id} was produced by {key} {want} but "
                        f"{role_name}.{file_key} is now {got} at {_rel(p)}")
            row[key] = entry
        per_cell[cell_id] = row
    if mismatched:
        raise RuntimeError(
            "cannot aggregate: " + "; ".join(mismatched[:6])
            + (f" (and {len(mismatched) - 6} more)"
               if len(mismatched) > 6 else "")
            + ".  A cell binds the weights that produced its rows, so weights "
              "that have since been replaced mean the filed rows describe an "
              "adapter nobody can inspect, and the aggregate would report "
              "verdicts about it")
    return {"per_cell": per_cell,
            "n_digests_compared": n_compared,
            "n_bindings_not_recheckable": len(unverifiable),
            "bindings_not_recheckable": unverifiable,
            "digests_that_are_not_role_files": dict(not_a_role_file),
            "policy": ("every digest a cell records against a declared role "
                       "file is re-hashed and compared; a mismatch refuses the "
                       "aggregate rather than annotating it, and an absent file "
                       "is reported as a binding that could not be re-checked "
                       "because RF2P must still reproduce a run from stored "
                       "generations where no adapter is present")}


def _cell_forget_ids(prereg, design_cell):
    """The forget identities one cell's rows were built against.

    A design that varies a single forget set gives every cell the same answer,
    and that answer is the design's own, so this returns it untouched and a
    v2-v4 report keeps the bytes it has always had.

    A design that varies several has to ask the CELL, because which identities
    are forgotten is exactly what two forget sets differ in -- and
    ``delta_route_v2`` splits its rows on that.  Computed against the wrong set
    every row lands in the wrong arm, and the statistic is still a number, which
    is what makes this worth a function rather than an index.
    """
    own = design_cell.get("forget_identity_ids")
    return list(own) if own else list(prereg["forget_identity_ids"])


def _response_key(design_cell):
    """The key one hard-response matrix is filed under.

    The edit seed alone where the design varies one forget set, which is every
    design frozen so far and what keeps their reports byte-identical.

    Where the design varies several, the seed alone is not a key.  Five forget
    sets times three edit seeds is fifteen intervention cells and a dict keyed
    on the seed holds three, each overwritten by whichever cell came last -- so
    a natural cell would be composed against an H_e that was trained to suppress
    a DIFFERENT identity, the factorization gate would still return a number,
    and nothing downstream would report that the number described the wrong
    adapter.
    """
    fs = design_cell.get("forget_set_id")
    seed = design_cell.get("edit_seed")
    return f"{fs}__seed{seed}" if fs else seed


def _baseline_matrix_for(baseline_response, forget_set):
    """The BEFORE that belongs to this edit's own forget set.

    ``forget_set`` is None exactly where the design varies one forget set, and
    there the container IS the matrix.  The discriminator is therefore a
    property of the cell being compared and not a guess read off the shape of a
    dict, which two different things can share.
    """
    if forget_set is None:
        return baseline_response
    if forget_set not in baseline_response:
        raise RuntimeError(
            f"the edit under forget set {forget_set!r} has no baseline matrix "
            f"to be the BEFORE of; this design filed baselines for "
            f"{sorted(baseline_response)}.  Comparing it against another set's "
            f"BEFORE would report as an edit what two forget sets differ in")
    return baseline_response[forget_set]


def aggregate_cells_for(spec, prereg, cells, rescore=False, cell_paths=None,
                        run_provenance=None):
    """RF2 and RF2P: gates, three verdicts and intervals, from filed cells.

    Nothing here generates anything and nothing here needs a GPU.  The inputs are
    the frozen pre-registration, the cell results the RF1 phases filed, and each
    router seed's prediction file -- so the same call produces the same report on
    a machine with no model loaded, which is what makes RF2P a reproduction
    rather than a re-run.

    It does READ the weights where they are present, to re-hash them against the
    digest each cell recorded.  That is a check on the inputs and not a use of
    them: an aggregate that never looked at the files behind its cells would
    report verdicts about an adapter somebody had since replaced.  Where the
    weights are absent -- a fresh clone, since adapters are gitignored -- the
    binding is reported as one that could not be re-checked and the aggregate
    still runs, because reproducing a filed run without the weights is the whole
    reason the raw generations are stored.

    The held-out images are taken from the FROZEN pilot rather than by re-reading
    the dataset manifest: re-reading would let the aggregate describe a set of
    images the design was never built over.

    A required input that is absent is an ERROR.  Returning success over the
    cells that happen to exist would report a verdict about a smaller design
    than the one that was frozen, and every rate in it would still look like a
    measurement.
    """
    dataset = prereg["dataset"]
    vocab = prereg["vocab"]
    # No design-level forget list is read in this function.  A design that
    # varies several forget sets records their UNION at the top level and no
    # cell was built over the union, so the identities a statistic splits on
    # come from the cell -- see _cell_forget_ids, which falls back to the
    # design's own list for a cell that does not carry one.
    thresholds = prereg.get("gate_thresholds") or dict(spec.thresholds)
    applicability = prereg["gate_applicability"]
    plan = prereg["cluster_bootstrap"]
    held_out = [e["image_uri"] for e in prereg["held_out_images"]["images"]]
    phase = "RF2P" if rescore else "RF2"
    if not held_out:
        raise RuntimeError(f"{phase}: the frozen pilot names no held-out "
                           f"images, so there is nothing to compose over")

    design_cells = OrderedDict((c["cell_id"], c) for c in prereg["cells"])
    absent = sorted(set(design_cells) - set(cells))
    if absent:
        raise RuntimeError(
            f"{phase} cannot aggregate: {len(absent)} of {len(design_cells)} "
            f"cells have no filed result, e.g. {absent[0]}.  {phase} refuses "
            f"rather than reporting success over the cells that do exist, "
            f"because the verdicts would then describe a smaller design than "
            f"the one that was frozen")
    extra = sorted(set(cells) - set(design_cells))
    if extra:
        raise RuntimeError(
            f"{phase} was given result(s) for cells the frozen design does not "
            f"contain: {extra}; aggregating them would let a cell nobody "
            f"pre-registered move a gate")

    per_cell = OrderedDict()
    changed_total = []
    # Every kind this design can contain, including one it may contain none of:
    # a kind missing from this dict would surface as a KeyError inside the gate
    # arithmetic rather than as a gate with no rows, and a gate with no rows is
    # the thing this module refuses loudest.
    by_kind = {k: [] for k in CODE_CELL_KINDS + IMAGE_CELL_KINDS + ("hybrid",)}
    for cell_id, design_cell in design_cells.items():
        doc = cells[cell_id]
        scored, missing, changed = _cell_scored(doc, design_cell, vocab,
                                                rescore)
        if missing:
            raise RuntimeError(
                f"cell {cell_id} has {len(missing)} row(s) with no stored "
                f"generation (e.g. {missing[0]}); a partial cell is a cell "
                f"about a different design and {phase} will not aggregate it")
        changed_total.extend({**c, "cell_id": cell_id} for c in changed)
        by_kind[design_cell["kind"]].extend(scored)
        block = {
            "cell_id": cell_id, "kind": design_cell["kind"],
            "phase_that_produced_it": doc.get("phase"),
            "edit_seed": design_cell.get("edit_seed"),
            "router_seed": design_cell.get("router_seed"),
            "direct_seed": design_cell.get("direct_seed"),
            **({"forget_set_id": design_cell["forget_set_id"]}
               if design_cell.get("forget_set_id") else {}),
            "n_rows": len(scored),
            "input_sha256": doc.get("input_sha256"),
            "n_unparseable": sum(1 for r in scored if r.get("unparseable")),
            "n_multi_label_ambiguous": sum(1 for r in scored
                                           if r.get("multi_label_ambiguous")),
            "n_unroutable": sum(1 for r in scored
                                if r.get("routable") is False),
        }
        if design_cell["kind"] in IMAGE_CELL_KINDS + ("hybrid",):
            block["accuracy"] = _mean_correct(scored)
            block["bootstrap"] = bootstrap_both_levels(
                scored, _mean_correct,
                f"{design_cell['kind']}_accuracy", plan)
        per_cell[cell_id] = block

    # --- H_e per EDIT seed, from that seed's own intervention cell -----------
    # Keyed by _response_key rather than by the seed: see its docstring for what
    # a seed-only key does to a design with several forget sets.
    hard_response = OrderedDict()
    for cell_id, design_cell in design_cells.items():
        if design_cell["kind"] != "intervention":
            continue
        es = design_cell["edit_seed"]
        key = _response_key(design_cell)
        scored = [r for r in by_kind["intervention"] if r["cell_id"] == cell_id]
        hard_response[key] = hard_response_matrix(scored, vocab)
        hard_response[key]["edit_seed"] = es
        hard_response[key]["cell_id"] = cell_id
        if design_cell.get("forget_set_id"):
            hard_response[key]["forget_set_id"] = design_cell["forget_set_id"]

    # --- H_base, the same matrix for the UNEDITED h --------------------------
    # Reported beside H_e because the two are the before and after of one map:
    # a reader can see which cells of C x Y the edit moved, rather than being
    # told only that a rate went down.  Built by the same function from the same
    # row shape, so the two matrices are comparable cell for cell by
    # construction and not by coincidence.
    baseline_matrices = OrderedDict()
    for cell_id, design_cell in design_cells.items():
        if design_cell["kind"] != "baseline":
            continue
        scored = [r for r in by_kind["baseline"] if r["cell_id"] == cell_id]
        m = hard_response_matrix(scored, vocab, (BASELINE_CONDITION_V3,))
        m["cell_id"] = cell_id
        m["adapter"] = "baseline_h, the route's own unedited h"
        fs = design_cell.get("forget_set_id")
        if fs:
            m["forget_set_id"] = fs
        baseline_matrices[fs] = m
    if not baseline_matrices:
        baseline_response = None
    elif list(baseline_matrices) == [None]:
        # One forget set, so one BEFORE, filed as the single matrix every design
        # so far filed.  A loop that simply kept reassigning would also produce
        # this for one cell and would silently keep the LAST of several.
        baseline_response = baseline_matrices[None]
    else:
        # Several forget sets, so several BEFOREs: which codes are supposed to
        # come back Unknown is what the sets differ in, and one matrix would
        # compare five sets' edits against one set's before.
        baseline_response = baseline_matrices
    what_the_edit_moved = None
    if baseline_response is not None and hard_response:
        # Which codes changed their response, per edit seed.  The suppression
        # gate is a rate over codes; this is the list behind the rate, and it is
        # what makes "the edit moved the forgotten codes and left the retained
        # ones alone" a thing a reader can check rather than accept.
        what_the_edit_moved = OrderedDict()
        for key, h_e in hard_response.items():
            before = _baseline_matrix_for(
                baseline_response, h_e.get("forget_set_id"))["matrix"]
            after = h_e["matrix"]
            diffs = []
            for code in sorted(set(before) | set(after)):
                if before.get(code) != after.get(code):
                    diffs.append({"code": code, "h_base": before.get(code),
                                  "h_edited": after.get(code)})
            what_the_edit_moved[key] = {
                "n_codes": len(set(before) | set(after)),
                "n_codes_whose_response_moved": len(diffs),
                "codes": diffs}

    # --- router accuracy per SEED, from that seed's own prediction file ------
    router_accuracy = OrderedDict()
    for seed in prereg["router_seeds"]:
        got = router_held_out_accuracy(dataset, seed, held_out)
        router_accuracy[seed] = got

    # --- the factorization, per (router seed, edit seed) ---------------------
    e2e = OrderedDict()
    for cell_id, design_cell in design_cells.items():
        if design_cell["kind"] != "natural":
            continue
        rs, es = design_cell["router_seed"], design_cell["edit_seed"]
        scored = [r for r in by_kind["natural"] if r["cell_id"] == cell_id]
        confusion = router_confusion(scored)
        predicted = predicted_e2e_from_empirical(
            scored, hard_response[_response_key(design_cell)], confusion)
        observed = _mean_correct(scored)
        e2e[cell_id] = {
            "cell_id": cell_id, "router_seed": rs, "edit_seed": es,
            "observed": observed, "predicted": predicted["predicted"],
            "abs_error": abs(observed - predicted["predicted"]),
            "n_images": predicted["n_images"],
            "n_unroutable": confusion["n_unroutable"],
            "formula": predicted["formula"],
            "p_h_is_the_empirical_hard_response": True,
            "confusion": {k: v for k, v in confusion.items()
                          if k != "per_image_code"},
            "per_image": predicted["per_image"],
        }
    if not e2e:
        raise RuntimeError(f"{phase} found no natural cell, so the "
                           f"factorization was never evaluated")
    worst = max(e2e.values(), key=lambda b: b["abs_error"])

    gates = evaluate_gates_for(
        spec, by_kind["intervention"], by_kind["natural"], by_kind["direct"],
        by_kind["hybrid"], {s: v["accuracy"] for s, v in router_accuracy.items()},
        {"observed": worst["observed"], "predicted": worst["predicted"],
         "n_images": worst["n_images"],
         "cell": worst["cell_id"],
         "n_cells_composed": len(e2e),
         "aggregation": ("the worst of the composed cells, not the mean: a "
                         "factorization that holds on eight cells and fails on "
                         "the ninth has failed, and averaging the errors would "
                         "hide it")},
        thresholds, baseline_rows=by_kind["baseline"])
    verdicts = verdicts_from_gates(gates, applicability, thresholds, spec)

    delta = OrderedDict()
    for cell_id, design_cell in design_cells.items():
        if design_cell["kind"] != "intervention":
            continue
        scored = [r for r in by_kind["intervention"] if r["cell_id"] == cell_id]
        delta[cell_id] = delta_route_v2(
            scored, _cell_forget_ids(prereg, design_cell))

    # Every weight behind every cell, re-hashed against the digest that cell
    # recorded -- and it refuses on a mismatch rather than reporting one.  Then
    # every cell file this aggregate consumed, hashed, so the report names its
    # own inputs by content instead of by a directory somebody can edit.
    input_check = verify_cell_inputs(prereg, cells)
    # Beside the input check rather than inside it: that one re-hashes weights,
    # this one names the design behind each cell, and a version that declared no
    # compatibility policy gets None and reports nothing it did not do.
    lineage = verify_cell_design_lineage(spec, prereg, cells)
    consumed = OrderedDict()
    for cid, p in sorted((cell_paths or {}).items()):
        p = Path(p)
        consumed[_rel(p)] = sha256_file(p) if p.is_file() else None

    out = {
        "kind": spec.result_kind,
        "produced_by": phase,
        "rescored_from_stored_raw": bool(rescore),
        "dataset": dataset,
        "forget_set_id": prereg["forget_set_id"],
        **({"forget_sets": prereg["forget_sets"]}
           if prereg.get("forget_sets") else {}),
        "preregistration_design_sha256": prereg.get("design_sha256"),
        "n_cells": len(per_cell),
        "n_rows": sum(b["n_rows"] for b in per_cell.values()),
        "cells": per_cell,
        "hard_response_matrices": hard_response,
        **({"baseline_hard_response_matrix": baseline_response,
            "what_the_edit_moved": what_the_edit_moved,
            "baseline_is_the_before_of_the_same_map": (
                "H_base and each H_e are built by one function from one row "
                "shape over the same code vocabulary, so the two matrices are "
                "comparable cell for cell by construction and the difference "
                "between them is what the edit did")}
           if baseline_response is not None else {}),
        "router_held_out_accuracy": router_accuracy,
        "e2e_factorization": {
            "per_cell": e2e,
            "gate_is_applied_to": worst["cell_id"],
            "aggregation_is_stated_on_the_gate": (
                "see gates.e2e_matches_the_routing_composition.aggregation; "
                "the rationale lives in one place so the two cannot drift"),
            "max_abs_error": worst["abs_error"],
            "tolerance": thresholds["max_abs_e2e_prediction_error"],
        },
        "delta_route": delta,
        "gates": gates,
        # The gates dict can hold an entry that is not a gate, so both counts are
        # stated rather than leaving a reader to reconcile them.
        **({"n_entries_in_gates": len(gates),
            "n_gates_the_verdicts_read": len(spec.gate_to_verdict),
            "entries_in_gates_that_are_not_gates": sorted(
                n for n, g in gates.items() if not g.get("is_a_gate", True)),
            "why_the_two_counts_differ": (
                "a reported diagnostic sits beside the gates rather than in a "
                "separate block nobody reads, and is marked is_a_gate=False so "
                "no verdict and no summary can mistake it for one")}
           if spec.declared_hygiene_scope else {}),
        "verdicts": verdicts,
        "conditions_that_feed_the_mediation_verdict":
            list(spec.mediated_conditions),
        "auxiliary_conditions_reported_separately":
            list(spec.auxiliary_conditions),
        "input_verification": input_check,
        **({"cell_design_lineage": lineage} if lineage is not None else {}),
        "consumed_cell_files": consumed,
        "n_consumed_cell_files": len(consumed),
        **({"run_provenance": run_provenance} if run_provenance else {}),
    }
    # The three conclusions are top-level fields, each three-state, and there is
    # deliberately no combined boolean: item 5 exists because one omnibus
    # 'passed' let a routing failure cancel a mediation success.
    for name in VERDICT_NAMES:
        out[f"{name}_pass"] = verdicts[f"{name}_pass"]
    if rescore:
        out["rescore_agreement"] = {
            "identical_to_the_filed_scores": not changed_total,
            "n_fields_that_moved": len(changed_total),
            "fields_that_moved": changed_total[:50],
            "why_this_phase_exists": (
                "a scoring defect is repairable after a run has been filed only "
                "if the raw generations were stored, so RF2P re-derives every "
                "score from them and reports exactly what moved.  The "
                "granularity pilot needed this: a punctuation-asymmetric "
                "matcher had scored a dozen byte-exact-correct rows as "
                "unparseable"),
        }
    return out


def aggregate_cells_v2(prereg, cells, rescore=False, cell_paths=None,
                       run_provenance=None):
    """The v2 aggregate.  Thin: one implementation is ``aggregate_cells_for``."""
    return aggregate_cells_for(PILOT_SPEC_V2, prereg, cells, rescore,
                               cell_paths, run_provenance)


def aggregate_cells_v3(prereg, cells, rescore=False, cell_paths=None,
                       run_provenance=None):
    """The v3 aggregate: v2's, plus the baseline matrix, the per-direct-seed
    gates and the re-hashing of every input the report is built from."""
    return aggregate_cells_for(PILOT_SPEC_V3, prereg, cells, rescore,
                               cell_paths, run_provenance)


# ---------------------------------------------------------------------------
# the v2 GPU layer -- thin, lazily imported, and never touched by the tests
# ---------------------------------------------------------------------------
#
# Everything that decides WHAT is measured lives above and runs without torch.
# This layer only loads weights, calls generate, and files what came back, so a
# design can be frozen, audited, gated and re-scored on a machine with no GPU.

def _gpu_siblings():
    """Import the model siblings only when a phase actually needs them.

    Imported inside the function rather than at module scope: these modules pull
    in torch, and the CPU layers above must stay importable without it.
    """
    rv = _load_sibling("e2c_v3_research_validity",
                       "e2c_v3_research_validity.py")
    mx = _load_sibling("e2c_v3_matrix", "e2c_v3_matrix.py")
    gxm = _load_sibling("e2c_v3_granularity_matrix",
                        "e2c_v3_granularity_matrix.py")
    return rv, mx, gxm


class RouteSessionV2:
    """One resident model with adapters switched in place.

    Resident rather than per-cell: loading Qwen3.5-9B costs more than every
    generation in a cell, and switching adapters in one live session is what
    ``e2c_v3_matrix`` already does.  ``reset_to`` is fail-closed in
    ``rv.load_trained_weights``, so a checkpoint that does not fully place
    raises rather than leaving a partially loaded adapter to be measured.
    """

    def __init__(self, device, adapter_name, seed):
        rv, mx, gxm = _gpu_siblings()
        self.rv, self.mx, self.gxm = rv, mx, gxm
        self.device = device
        self.seed = seed
        self.args = argparse.Namespace(device=device, seed=seed)
        self.session = mx.ModelSession(self.args, adapter_name)
        self._backend = None

    def reset_fresh(self, seed):
        """A genuine fresh LoRA init under ``seed``.

        PEFT 0.20 exposes no reset_adapter API, so a new seed has to be
        reproduced from init_lora_weights=True by hand: lora_A kaiming_uniform
        (a=sqrt5), lora_B zeros.  Re-loading an existing checkpoint instead
        would make "router seed" a label for the same weights.
        """
        self.gxm.fresh_reinit_lora(list(self.session.model.named_parameters()),
                                   seed)
        self._backend = None

    def reset_to(self, checkpoint):
        self.session.reset_to(checkpoint)
        self._backend = None

    def backend(self):
        if self._backend is None:
            self._backend = self.session.backend()
            self.session.model.eval()
        return self._backend

    def generate(self, image, prompt, max_new_tokens=8):
        """One generation.  ``image=None`` is the hard mediator: h gets a code
        and nothing else, which is the property item 1 exists to protect."""
        import torch
        with torch.no_grad():
            return self.backend().generate(image, prompt,
                                           max_new_tokens=max_new_tokens).text.strip()

    def train(self, condition, pairs, output_dir, protocol):
        """Supervised descent on (image, prompt, answer) triples.

        The protocol comes from ``frozen_route_protocol()``, i.e. from the
        frozen route scripts, so a new router or direct model is trained under
        the SAME recipe as the existing route rather than a comparable one.
        """
        adapter = self.session.adapter
        processor = self.session.processor
        sup = []
        for uri, prompt, answer in pairs:
            # Through resolve_recorded_path, like every other image this runner
            # opens.  SALMU records its image URIs RELATIVE to the dataset root,
            # so opening one against the current working directory makes
            # training succeed or fail on where the process was launched from --
            # and it fails as a missing file, which reads as broken data rather
            # than as a launch directory.  Evaluation already resolved these;
            # training is the path that did not.
            image = _open_image(uri)
            ex = adapter.build_supervised_example(
                processor, image=image, prompt=prompt, answer_text=answer)
            sup.extend([ex] * protocol["repeat"])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.mx.seed_everything(self.seed)
        self.rv.train_supervised(
            condition, adapter, self.session.model, processor, sup, output_dir,
            self.device, steps=protocol["steps"], warmup=protocol["warmup"],
            lr=protocol["lr"])
        self._backend = None
        return output_dir

    def release(self):
        self._backend = None
        self.session.release()


def _open_image(uri):
    """Open a recorded image URI, resolving it the way every recorded path is.

    SALMU records its image URIs RELATIVE to the dataset root, so ``Image.open``
    on the recorded string works only when the process happens to be launched
    from there -- and fails as a missing file everywhere else, which reads as
    broken data rather than as a launch directory.  Evaluation already resolved
    through ``resolve_recorded_path``; training did not, which is the asymmetry
    this closes.

    Opened in a context manager because ``convert`` returns an independent image
    and the handle on the source would otherwise stay open until collection, and
    a training loop over a split opens hundreds of them.
    """
    from PIL import Image
    with Image.open(resolve_recorded_path(uri)) as source:
        return source.convert("RGB")


#: Every sibling script a phase reads at run time, whether it imports it or
#: parses it by AST.  ``provenance`` already hashes this script plus the label
#: parser, the research-validity module and the matrix module; these are the
#: rest, so a run's provenance names every module whose code took part in
#: producing the result.
#:
#: The frozen route and protocol scripts are DERIVED from the tables that name
#: them rather than retyped here, because their bytes are part of what executed:
#: ``frozen_route_prompts`` and ``frozen_route_protocol`` read the prompt and the
#: recipe out of those files at run time, so a script this runner starts reading
#: has to be hashed by the same edit that started reading it.
RUNTIME_SIBLING_SCRIPTS = tuple(dict.fromkeys((
    "e2c_v3_granularity_matrix.py",
    "e2c_v3_mllmu_matrix.py",
    *FROZEN_ROUTE_SCRIPTS,
    *PROTOCOL_SCRIPTS,
)))

#: Process-level cache.  The git queries are subprocesses and the module hashes
#: are file reads; doing either per cell would repeat identical work for every
#: row of every cell, and could straddle a state change so that one run's cells
#: recorded two different commits.
_RUN_PROVENANCE = {}

#: The configuration this process was launched with, noted once by ``main``.
#: Phases take explicit arguments rather than the parsed namespace, so a cell
#: that is supposed to record the COMPLETE launch configuration needs it kept
#: somewhere the filing code can reach.
_RUN_CLI = {"cli": None}


def cli_configuration(args):
    """The complete configuration a run was launched with.

    Every flag, not only the ones the phase being run happens to read: a cell
    that recorded its own phase's arguments alone could not be distinguished
    from one produced under a different device, a different cell root, or a
    ``--resume`` that reused a stale file.
    """
    return {k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in sorted(vars(args).items())}


def note_cli(args):
    """Record this process's launch configuration once, at the start."""
    _RUN_CLI["cli"] = cli_configuration(args)
    return _RUN_CLI["cli"]


def executing_code_identity():
    """Commit, worktree state, and a digest of this script and every sibling a
    phase can load.

    Cached per process for the reasons ``_RUN_PROVENANCE`` gives.  The worktree
    determination counts untracked files, because an untracked conftest,
    sitecustomize or shadowing module changes what executes while a tracked-only
    query reports a clean tree -- and a result attributed to a commit that could
    not have produced it is worse than a result with no attribution.
    """
    if "code" not in _RUN_PROVENANCE:
        base = provenance(tuple(SCRIPT_DIR / n
                                for n in RUNTIME_SIBLING_SCRIPTS))
        _RUN_PROVENANCE["code"] = {
            "executing_commit": base["executing_commit"],
            "clean_worktree": base["clean_worktree"],
            "executing_script_sha256": base["script_sha256"],
            "runtime_module_sha256": base["input_file_sha256"],
            "n_runtime_modules_hashed": base["n_input_files_hashed"],
            "missing_runtime_modules": base["missing_input_files"],
            "paths_are_relative_to": base["paths_are_relative_to"],
        }
    return _RUN_PROVENANCE["code"]


def start_of_run_provenance(prereg=None, prereg_file=None, spec=None, cli=None):
    """What executed, captured at the START of a run and carried by every cell.

    A filed cell already binds its prompts and the weights it selected, which
    says WHAT was measured.  This says what measured it: the commit, whether the
    worktree was dirty, the digest of this script and of every sibling module a
    phase loads, the digest and design hash of the pre-registration the run read,
    and the complete CLI that launched it.

    Without it a cell is a result that cannot be attributed to a state of the
    code, so a defect found later cannot be scoped to the runs it affected, and
    a cell reused by ``--resume`` cannot be told from one produced fresh.

    Recorded in RESULTS and never in a frozen design.  A design binds the bytes
    it is a statement about; a result records the execution that produced it.
    Putting this in a design would make the design differ on every run, which is
    the same category of mistake as v2 hashing which adapters had been trained.
    """
    spec = spec or LATEST_PILOT_SPEC
    out = OrderedDict()
    out["pilot_version"] = spec.version
    out["captured_at_the_start_of_the_run"] = True
    out.update(executing_code_identity())
    if prereg_file is not None:
        p = Path(prereg_file)
        out["preregistration_path"] = _rel(p)
        out["preregistration_file_sha256"] = (sha256_file(p) if p.is_file()
                                              else None)
    if prereg is not None:
        out["preregistration_design_sha256"] = prereg.get("design_sha256")
        out["preregistration_kind"] = prereg.get("kind")
    out["cli"] = _RUN_CLI["cli"] if cli is None else cli_configuration(cli)
    if out["cli"] is None:
        # Stated rather than left as a bare null: null reads as "no flags were
        # given", which is a different claim and a false one.
        out["cli_is_absent_because"] = (
            "this phase was called as a library function rather than through "
            "main(), which is where the launch configuration is captured; a run "
            "from the command line always records it")
    return out


def _input_sha256(prereg, **weights):
    """The digests a result is bound to: the weights it used, and the design it
    was run against.

    Assembled in one place because ``resume_decision`` compares this dict against
    the one a filed result recorded, and the two are built at different points
    in a phase.  A field present in one and absent from the other makes every
    ``--resume`` refuse and re-run work that nothing had changed -- which looks
    like a resume that never resumes rather than like a mismatched key.

    The DESIGN hash belongs here and the manifest's FILE hash does not: a design
    that changed means a different experiment, while a manifest re-frozen from an
    identical design at a new commit is the same experiment recorded in new
    bytes.
    """
    return {**weights, "preregistration_design": prereg.get("design_sha256")}


def _cell_skeleton(design_cell, phase, extra_inputs, result_kind=RESULT_KIND_V2):
    """The fields every filed cell carries, so a resume can judge currency.

    ``input_sha256`` names the weights and prediction files that produced the
    rows.  It is recorded rather than inferred from paths because a checkpoint
    replaced in place keeps its path, and a result bound only to a path would be
    reused after the weights behind it changed.

    ``run_provenance`` is REQUIRED of every caller rather than defaulted: a
    default would let a new phase file a cell with no attribution and nothing
    would ever notice, whereas refusing here fails the phase before it files.
    """
    if "run_provenance" not in extra_inputs:
        raise RuntimeError(
            f"cell {design_cell['cell_id']} was filed with no run_provenance; a "
            f"result that does not name the commit, the script digest and the "
            f"pre-registration that produced it cannot be attributed to a state "
            f"of the code, so a defect found in it later cannot be scoped to "
            f"the runs it affected")
    return {
        "kind": result_kind,
        "cell_id": design_cell["cell_id"],
        "kind_of_cell": design_cell["kind"],
        "phase": phase,
        "edit_seed": design_cell.get("edit_seed"),
        "router_seed": design_cell.get("router_seed"),
        "direct_seed": design_cell.get("direct_seed"),
        # Present only where the design cell carries one.  A design that varies
        # a single forget set gives every cell the same answer, so recording it
        # per cell would add a field that says nothing and would move the bytes
        # of every cell v2, v3 and v4 file.
        **({"forget_set_id": design_cell["forget_set_id"]}
           if design_cell.get("forget_set_id") else {}),
        "n_rows": len(design_cell["rows"]),
        "prompts": dict(extra_inputs.pop("prompts")),
        "input_sha256": dict(extra_inputs.pop("input_sha256")),
        "image_to_h": False,
        "mediator_is_hard": ("no generation in this cell was produced with an "
                             "image and a code presented to h together"),
        **extra_inputs,
    }


def _file_cell(dataset, design_cell, phase, rows, scored, missing, extra,
               cells_out=None, result_kind=RESULT_KIND_V2):
    """Score with the ONE scorer, then file atomically."""
    doc = _cell_skeleton(design_cell, phase, extra, result_kind)
    doc["rows"] = rows
    doc["scored"] = scored
    doc["rows_without_a_stored_generation"] = missing
    doc["n_scored"] = len(scored)
    if missing:
        raise RuntimeError(
            f"cell {design_cell['cell_id']} generated {len(scored)} of "
            f"{len(rows)} rows ({len(missing)} missing, e.g. {missing[0]}); a "
            f"partial cell is not filed, because RF2 would refuse it and a "
            f"--resume that found the file would treat the cell as done")
    path, digest = write_cell_result(dataset, design_cell["cell_id"], doc,
                                     cells_out)
    logger.info("%s: filed %s (%d rows) %s", phase, path.name, len(rows),
                digest[:16])
    return doc


def _resume_or_run(dataset, cell_id, want, resume, run, path=None):
    """Reuse a filed result only if every binding still matches.

    ``path`` overrides where the record lives: a design cell files under its
    cell id, while a training phase files beside the adapter it produced.
    """
    path = Path(path) if path else cell_result_path(dataset, cell_id)
    if not (resume and path.is_file()):
        return run(), {"resumed": False}
    stored = json.loads(path.read_text(encoding="utf-8"))
    decision = resume_decision(stored, want)
    if not decision["reuse"]:
        logger.info("%s: re-running, the filed result is stale: %s", cell_id,
                    "; ".join(decision["reasons"]))
        return run(), {"resumed": False, "stale_because": decision["reasons"]}
    logger.info("%s: reusing the filed result, every input digest matches",
                cell_id)
    return stored, {"resumed": True, "reason": decision["policy"]}


# ---------------------------------------------------------------------------
# the phases
# ---------------------------------------------------------------------------

def prereg_path_for(spec, dataset, path=None):
    return Path(path) if path else (DATASET_ROOT / MANIFEST_DIR
                                    / f"rf_pilot_{dataset}_{spec.version}.json")


def prereg_path(dataset, path=None, spec=None):
    """Where the pilot the phases run against lives.

    Defaults to the LATEST version rather than to a name written into each
    phase: one constant decides which design the runner executes, so repointing
    it cannot leave one phase reading an older pre-registration than its
    siblings, and a phase cannot be silently aimed at a superseded artifact by
    somebody editing its default and not the others.
    """
    return prereg_path_for(spec or LATEST_PILOT_SPEC, dataset, path)


def prereg_path_v2(dataset, path=None):
    """The superseded v2 pilot's path, retained so its frozen files stay
    readable and verifiable by name."""
    return prereg_path_for(PILOT_SPEC_V2, dataset, path)


def prereg_path_v3(dataset, path=None):
    return prereg_path_for(PILOT_SPEC_V3, dataset, path)


def load_prereg_for(spec, dataset, path=None, verify=True):
    """The frozen pilot of one version, verified before anything runs on it.

    Verified by default: a phase that reads a manifest nobody checked would
    happily execute a design that had been edited after freezing, and the edit
    would be invisible in the results.

    Verification here means the DESIGN reproduces.  Whether the training the
    design asks for has arrived is a separate live question that
    ``checkpoint_readiness`` answers, and conflating the two is what made a v2
    pilot unloadable the moment its first adapter was trained.
    """
    p = prereg_path_for(spec, dataset, path)
    if not p.is_file():
        raise RuntimeError(
            f"no frozen {spec.version} pilot at {p}; freeze it first with "
            f"--preregister --dataset {dataset}.  The phases run against a "
            f"pre-registered design or they do not run")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("kind") != spec.prereg_kind:
        raise RuntimeError(
            f"{p} has kind {doc.get('kind')!r}, expected "
            f"{spec.prereg_kind!r}; this is not a {spec.version} pilot and the "
            f"{spec.version} phases will not run against it")
    if verify:
        got = verify_manifest(p)
        if not got["valid"]:
            raise RuntimeError(
                f"the frozen {spec.version} pilot at {p} does not verify: "
                + "; ".join(got["problems"]))
    return doc


def load_prereg(dataset, path=None, verify=True, spec=None):
    """The pilot the phases run against: the latest version by default."""
    return load_prereg_for(spec or LATEST_PILOT_SPEC, dataset, path, verify)


def load_prereg_v2(dataset, path=None, verify=True):
    return load_prereg_for(PILOT_SPEC_V2, dataset, path, verify)


def load_prereg_v3(dataset, path=None, verify=True):
    return load_prereg_for(PILOT_SPEC_V3, dataset, path, verify)


def _phase_prereg(dataset, prereg=None, spec=None, prereg_file=None):
    """The pre-registration a phase runs against, its run provenance, and the
    spec both came from.

    Returned together because they are obtained together: the provenance names
    the pre-registration's file digest and design hash, so a phase that loaded a
    design without recording it filed a cell that cannot be tied back to the
    design it claims to implement.

    Where a caller supplies a pre-registration directly -- which is how the tests
    run a phase without freezing anything -- the spec is read off that
    document's own kind, so a phase files its results in the kind of the design
    it was actually given rather than in whatever the module default says.

    ``spec`` selects which manifest to look for when none was supplied.  It
    comes from ``--design-version`` and is passed by the dispatcher rather than
    left to the default, because a phase that defaults to the newest design
    while the operator asked for an older one runs a different experiment from
    the one requested and files it under the newest version's name.  The
    document still wins over it: a spec cannot make a v2 manifest describe a v3
    design.
    """
    if prereg is not None:
        spec = SPEC_BY_PREREG_KIND.get(prereg.get("kind"),
                                       spec or LATEST_PILOT_SPEC)
        return (prereg,
                start_of_run_provenance(prereg, prereg_file, spec), spec)
    spec = spec or LATEST_PILOT_SPEC
    p = Path(prereg_file) if prereg_file else prereg_path_for(spec, dataset)
    loaded = load_prereg_for(spec, dataset, p)
    return loaded, start_of_run_provenance(loaded, p, spec), spec


def phase_rf0(dataset, path=None, out=None, cells=None):
    """RF0: verify the frozen manifest and every input it depends on.

    Reports what is present, what must still be trained and what has already
    been filed, and refuses to call the pilot runnable while anything is absent.
    Verification is not advisory here: RF2 re-reads the same manifest, and an
    artifact that does not reproduce from its own recorded inputs is not a
    pre-registration.

    Verification and readiness are reported as two things, because they are two
    things.  ``manifest_valid`` asks whether the DESIGN still reproduces from
    its own recorded inputs; ``runnable_now`` asks whether the work the design
    names has arrived.  v2 collapsed them: it read completeness out of the
    frozen document, so ``runnable_now`` could never become True however much
    training was done, and worse, the training that would have made it True made
    the design stop reproducing and so drove ``manifest_valid`` False instead.
    """
    spec = LATEST_PILOT_SPEC
    p = prereg_path_for(spec, dataset, path)
    if not p.is_file():
        raise RuntimeError(
            f"RF0 has no frozen {spec.version} manifest to verify at "
            f"{_rel(p)}; freeze it with --preregister --dataset {dataset}.  RF0 "
            f"verifies inputs, it does not invent them")
    doc = json.loads(p.read_text(encoding="utf-8"))
    # A manifest names its own version, so RF0 reports the one it was handed
    # rather than assuming the latest.  The two superseded v2 pilots are still
    # committed and still have to be explainable, and an RF0 that wrote a v3
    # filename and a v2 digest would explain nothing -- it would overwrite the
    # v3 report with the older design's state.
    spec = SPEC_BY_PREREG_KIND.get(doc.get("kind"), spec)
    got = verify_manifest(p)
    req = doc.get("checkpoint_requirements") or {}
    # Live, from the verifier's own probe rather than from the frozen block: see
    # the docstring.  A manifest whose role table predates that shape (v1) has
    # no live readiness and falls back to what it froze.
    ready = got.get("checkpoint_readiness") or {}
    complete = ready.get("complete", req.get("complete"))
    must_train = ready.get("must_be_trained") if ready else (
        req.get("must_be_trained_before_this_pilot_can_run"))
    n_must_train = (ready.get("n_must_be_trained") if ready
                    else req.get("n_must_be_trained"))
    unexpectedly_absent = ready.get("unexpectedly_absent",
                                    req.get("unexpectedly_absent"))

    # The frozen pilot names the held-out images the design was built over.  RF0
    # compares them against what the dataset manifest says NOW, because a
    # manifest that gained or lost an image since freezing makes every cell that
    # was filed against the old one describe a different design.
    frozen_held_out = [e["image_uri"] for e in
                       ((doc.get("held_out_images") or {}).get("images") or [])]
    live_held_out = held_out_uris(load_manifest(dataset))
    drift = sorted(set(frozen_held_out) ^ set(live_held_out))
    held_out = frozen_held_out or live_held_out

    routers = {}
    for seed in doc.get("router_seeds") or []:
        pp = router_prediction_path(dataset, seed)
        entry = {
            "predictions_present": pp.is_file(),
            "predictions_path": _rel(pp),
            "predictions_sha256": sha256_file(pp) if pp.is_file() else None,
            "held_out_accuracy": None,
            "accuracy_error": None,
        }
        if pp.is_file():
            try:
                entry["held_out_accuracy"] = router_held_out_accuracy(
                    dataset, seed, held_out)
            except RuntimeError as exc:
                # reported rather than raised: RF0's job is to say what is
                # wrong with the inputs, and a prediction file that does not
                # cover the design is one of them
                entry["accuracy_error"] = str(exc)
        routers[str(seed)] = entry

    cells_filed, cells_missing = [], []
    for cell in doc.get("cells") or []:
        cid = cell["cell_id"]
        (cells_filed if cell_result_path(dataset, cid, cells).is_file()
         else cells_missing).append(cid)
    why_not = []
    if not got["valid"]:
        why_not.append("the manifest does not verify: "
                       + "; ".join(got["problems"]))
    if drift:
        why_not.append(
            f"the dataset manifest no longer names the held-out images this "
            f"pilot was frozen over ({len(drift)} differ)")
    if not complete:
        why_not.append(f"{n_must_train or 0} role(s) must be trained first: "
                       f"{must_train or 'none named by this manifest'}")
    if unexpectedly_absent:
        why_not.append(
            f"{len(unexpectedly_absent)} role(s) this design declared as "
            f"already on disk are absent: {unexpectedly_absent}, which is a "
            f"broken assumption about the tree rather than training work")
    report = {
        "phase": "RF0",
        "dataset": dataset,
        "design_version": spec.version,
        "manifest": _rel(p),
        "manifest_valid": got["valid"],
        "problems": got["problems"],
        "kind": got["kind"],
        "design_sha256": got["design_sha256"],
        "n_checkpoints_rehashed": got["n_checkpoints_rehashed"],
        "executed": got["executed"],
        "n_cells": doc.get("n_cells"),
        "cells_by_kind": {k: len(v) for k, v in
                          (doc.get("cells_by_kind") or {}).items()},
        # LIVE, not read out of the frozen document.  These four are the answer
        # to "what is still missing", and the frozen document cannot answer it:
        # it was written before the work was done.
        "checkpoint_requirements_complete": complete,
        "must_be_trained_before_this_pilot_can_run": must_train,
        "n_must_be_trained": n_must_train,
        "unexpectedly_absent": unexpectedly_absent,
        "roles_present": sorted((ready.get("present") or {}) or
                                (req.get("verification") or {}).get("present")
                                or {}),
        "roles_that_changed_since_freeze":
            ready.get("roles_whose_presence_changed_since_freeze"),
        "readiness_is_computed_live": bool(ready),
        "routers": routers,
        "held_out_image_drift_since_freeze": drift,
        "n_held_out_images": len(held_out),
        "cells_filed": cells_filed,
        "cells_missing": cells_missing,
        "n_cells_filed": len(cells_filed),
        "n_cells_missing": len(cells_missing),
        "runnable_now": bool(got["valid"] and complete and not drift),
        # EVERY reason, not the first one.  A report that named one and stopped
        # sent the operator to fix it and then answer a second question the
        # report had already known the answer to.
        "why_not_runnable": None if not why_not else "  |  ".join(why_not),
        "runnable_now_is_about_inputs_not_about_results": (
            "True means every weight, prediction file and image the design "
            "names is present and hashed; it says nothing about whether any "
            "gate would pass"),
        "verification_and_readiness_are_separate_questions": (
            "manifest_valid asks whether the design still reproduces from its "
            "own recorded inputs; runnable_now asks whether the work it names "
            "has arrived.  Training an adapter moves the second and must not "
            "move the first -- which is what a design that hashes its own "
            "progress gets wrong"),
        "expected_verdicts": (
            doc.get("verdicts_to_be_reported") or {}
        ).get("expected_on_this_dataset"),
        "run_provenance": start_of_run_provenance(doc, p, spec),
    }
    path_out = report_path(dataset, "RF0", out, spec)
    atomic_write_json(path_out, report)
    logger.info("RF0: filed %s", _rel(path_out))
    return report


#: What D_s's training schedule may override, and nothing else.  These are
#: exactly the values ``RouteSessionV2.train`` reads out of the protocol dict.
#: The LoRA shape is NOT among them: it is fixed by the frozen route module that
#: built g and h, so an override of it could be recorded and never applied -- a
#: configuration the artifact claims and the adapter does not have.
OVERRIDABLE_SCHEDULE_KEYS = ("steps", "warmup", "lr", "repeat")


def _direct_schedule(protocol, overrides):
    """D_s's protocol: the frozen route's, with a selected schedule replaced.

    A COPY and never a mutation, and reachable only from RF1D and RFC: g and h
    are trained under ``frozen_route_protocol()`` itself, so the recipe the route
    was established with is not something this function can move.

    Anything outside ``OVERRIDABLE_SCHEDULE_KEYS`` is REFUSED rather than
    ignored.  Silently dropping an override is the worse failure: the design
    would name a configuration, the adapter would be trained under another, and
    every number filed against it would still look like a measurement of the one
    that was named.
    """
    unknown = sorted(set(overrides) - set(OVERRIDABLE_SCHEDULE_KEYS))
    if unknown:
        raise RuntimeError(
            f"the direct adapter's configuration asks to override {unknown}, "
            f"which is not part of the training schedule.  Only "
            f"{list(OVERRIDABLE_SCHEDULE_KEYS)} reach the trainer; the LoRA "
            f"shape is set by the frozen route module that built g and h, so an "
            f"override of it would be recorded and not applied")
    out = dict(protocol)
    out.update({k: overrides[k] for k in OVERRIDABLE_SCHEDULE_KEYS
                if k in overrides})
    out["schedule_overridden_for_the_direct_adapter"] = OrderedDict(
        (k, {"frozen_route_value": protocol[k], "used": overrides[k]})
        for k in OVERRIDABLE_SCHEDULE_KEYS if k in overrides)
    out["not_overridable"] = (
        "the LoRA rank, alpha, dropout and target modules, the model id and "
        "revision and the dtype: those come from the frozen route module and "
        "are the reason a direct adapter is comparable to the route rather than "
        "merely another model")
    return out


def _direct_train_items(man, direct_training):
    """The image-to-alias pairs D_s trains on.

    The train split, minus whatever development images the design says the
    configuration was selected on.  Read out of the DESIGN and not off a flag: a
    phase that could be pointed at any subset of the training images is a phase
    whose result does not say what it was trained on, and the design is the
    artifact that has to say it.

    The count is checked against what was named.  A hold-out that quietly
    removed fewer images than it declared would leave the selected
    configuration's own selection images inside the confirmatory training set,
    and the accuracy reported against the held-out test images would still be a
    number nobody could distinguish from the honest one.
    """
    items = [it for it in man["items"] if it["split"] == "train"]
    dev = list((direct_training or {}).get("development_image_uris") or ())
    if not dev:
        return items
    drop = set(dev)
    kept = [it for it in items if it["image_uri"] not in drop]
    if len(items) - len(kept) != len(drop):
        raise RuntimeError(
            f"the design names {len(drop)} development images to hold out of "
            f"D_s's training, but only {len(items) - len(kept)} of the "
            f"{len(items)} train images matched one.  A hold-out that does not "
            f"remove what it names is not a hold-out")
    return kept


def design_forget_sets(prereg):
    """The forget sets one design varies, in the order the design lists them.

    Empty for every design frozen so far, which is what makes the qualified cell
    ids and role names below unreachable for them: their cells and their reports
    keep the bytes they have always had.
    """
    return list(prereg.get("forget_sets") or ())


def qualified_cell_id(kind, forget_set, tail):
    """One cell's id, qualified by forget set where the design varies several.

    The KIND STAYS FIRST.  ``load_cell_result`` reads the phase a missing cell
    needs out of ``cell_id.split("__")[0]``, so an id that led with the forget
    set would name no phase and every absence would tell the operator to run
    RF1 -- which ``main`` refuses on sight.
    """
    return f"{kind}__{forget_set}__{tail}" if forget_set else f"{kind}__{tail}"


def select_forget_set(prereg, requested, phase, kind):
    """Which forget set this invocation fills, or a refusal if it is ambiguous.

    None where the design varies one forget set, so the id the phase builds is
    the id it has always built.  Where the design varies several the phase has
    to be TOLD which: defaulting to the first would run a fifth of the design
    and return success, and the aggregate would then refuse with a list of
    missing cells rather than naming the choice that was never made.
    """
    sets = [s["set_id"] for s in design_forget_sets(prereg)]
    if not sets:
        if requested is not None:
            raise RuntimeError(
                f"{phase} was given --forget-set {requested!r} but this design "
                f"varies one forget set and its cell ids do not name one")
        return None
    if requested is None:
        if len(sets) == 1:
            return sets[0]
        raise RuntimeError(
            f"{phase} fills ONE {kind} cell and this design varies "
            f"{len(sets)} forget sets {sets}, so pass --forget-set.  Choosing "
            f"one here would be a selection made at run time by the code being "
            f"run, which is the thing a frozen design exists to prevent")
    if requested not in sets:
        raise RuntimeError(
            f"{phase} was given --forget-set {requested!r}, which this design "
            f"does not vary; it varies {sets}")
    return requested


def phase_rf1b(dataset, prereg=None, device="cuda", resume=False,
               cells_out=None, spec=None, forget_set=None):
    """RF1B: the UNEDITED h on every route code -- the before in before/after.

    ``h_base(do(C=c))`` for every code in the route vocabulary with no image,
    exactly as RF1H does it for an edited h.  The two differ in one respect:
    which adapter answers.  Every forgotten code is required to produce its OWN
    alias here, so that RF1H's Unknown on the same code is a change the edit
    made rather than something this h was doing all along.

    No seed, and so no ``--edit-seed``, ``--router-seed`` or ``--direct-seed``:
    the baseline is the route's own h, which the design declares as a role that
    already exists, so this phase trains nothing.  It can therefore run before
    any of the training work the pilot names, and it should -- the before is
    evidence about the route as it was frozen, and collecting it late risks
    collecting it after something has moved.

    Evaluated ONCE.  It depends on no factor, so filing it per edit seed would
    be one measurement reported three times, which is the duplication the cell
    structure exists to prevent.

    The session seed is 0 because nothing is trained and no LoRA init is drawn
    from it, and decoding is greedy so it cannot move an output.  RF1H passes
    the edit seed because there the seed names the adapter being loaded.
    """
    prereg, prov, spec = _phase_prereg(dataset, prereg, spec)
    prompts = frozen_route_prompts()
    forget_set = select_forget_set(prereg, forget_set, "RF1B", "baseline")
    cell_id = qualified_cell_id("baseline", forget_set, "h_base")
    cells = {c["cell_id"]: c for c in prereg["cells"]}
    if cell_id not in cells:
        raise RuntimeError(
            f"the frozen {spec.version} pilot has no cell {cell_id!r}.  A "
            f"design without a baseline cell measures post-edit behaviour only "
            f"and cannot establish that the edit changed anything, which is the "
            f"defect v3 repairs -- so it is not something to work around by "
            f"running this phase against an older manifest"
            + (f".  The forget sets it does vary are "
               f"{[s['set_id'] for s in design_forget_sets(prereg)]}"
               if design_forget_sets(prereg) else ""))
    design_cell = cells[cell_id]
    role = (prereg["checkpoint_requirements"]["roles"].get("baseline_h"))
    if role is None:
        raise RuntimeError(
            "the frozen pilot declares no baseline_h role, so there is no "
            "adapter this phase is supposed to load")
    ckpt = resolve_recorded_path(role["adapter"])
    if not ckpt.is_file():
        raise RuntimeError(
            f"the unedited h is absent at {_rel(ckpt)}.  This design declared "
            f"it as already present, because it is the route's own h and not "
            f"training work, so its absence is a broken assumption about the "
            f"tree rather than something to wait for")
    # Bound through the table RF2 re-checks against, so the digest filed here is
    # the one RF2 will resolve from the same role declaration rather than from a
    # second derivation of the same path.
    inputs = _input_sha256(prereg, **cell_input_digests(prereg, design_cell))

    def run():
        sess = RouteSessionV2(device, "e2c_rf_h_base", 0)
        try:
            sess.reset_to(ckpt)
            rows = []
            for r in design_cell["rows"]:
                prompt = prompts[r["prompt_key"]].format(code=r["forced_code"])
                # image=None, exactly as in RF1H.  The mediator is hard here
                # too, and a baseline measured with an image on screen would
                # not be the same measurement as the post-edit one it is
                # compared against, so the comparison would not be a
                # comparison.
                raw = sess.generate(None, prompt, max_new_tokens=8)
                rows.append({**r, "h_raw_text": raw, "prompt_used": prompt})
        finally:
            sess.release()
        scored, missing = score_rows_v2(rows, prereg["vocab"])
        return _file_cell(dataset, design_cell, "RF1B", rows, scored, missing,
                          {"prompts": {"h_code_to_alias":
                                       prompts["h_code_to_alias"]},
                           "input_sha256": inputs,
                           "adapter_path": _rel(ckpt),
                           "run_provenance": prov},
                          cells_out, spec.result_kind)

    want = {"cell_id": cell_id, "kind": spec.result_kind, "phase": "RF1B",
            "input_sha256": inputs,
            "prompts": {"h_code_to_alias": prompts["h_code_to_alias"]},
            "n_rows": design_cell["n_rows"]}
    doc, decision = _resume_or_run(
        dataset, cell_id, want, resume, run,
        cell_result_path(dataset, cell_id, cells_out))
    doc = {**doc, "resume": decision}
    write_cell_result(dataset, cell_id, doc, cells_out)
    return doc


def phase_rf1g(dataset, router_seed, prereg=None, device="cuda", resume=False,
               spec=None):
    """RF1G: train and evaluate ONE router seed under the frozen protocol.

    This is what makes the router seed a factor.  It writes two files and the
    design requires both: the adapter, and a held-out prediction file of that
    seed's own.  A checkpoint without predictions cannot compose, and
    predictions without a checkpoint cannot be attributed to a training run.

    There is no output redirect here on purpose: the frozen manifest names the
    paths these two files must occupy, so a router written anywhere else is a
    router the manifest cannot see and RF0 would still report as absent.  The
    paths are READ from the manifest's role declaration rather than recomputed
    here, for the same reason: two derivations of one path are two paths the
    moment either is edited.

    The existing frozen g is REFUSED rather than retrained: retraining it would
    replace the router the route was established with, and the pilot would then
    be measuring a route nobody froze.

    The seed is REFUSED unless the pre-registration names it.  RF1G was the one
    phase that never loaded the manifest, and the seed budget alone permits any
    seed wherever held-out image content exists, so an unpregistered seed could
    consume a full GPU training run that no verdict could use and no design
    named.  A factor has levels; a level nobody pre-registered is not a level.
    """
    # The frozen router is refused BEFORE the manifest is loaded, because this
    # one is a fact about the route rather than about a design: g_17 is the
    # router the route was established with under any pre-registration, so
    # refusing it does not depend on having one to consult -- and answering with
    # "no frozen pilot" when the operator asked to retrain the route's own
    # router would answer a question they did not ask.
    if router_seed == EXISTING_ROUTER_SEED:
        raise RuntimeError(
            f"router seed {router_seed} is the frozen g the route was "
            f"established with; it is READ from "
            f"{_rel(router_checkpoint_path(dataset, router_seed))} and its "
            f"predictions from "
            f"{_rel(router_prediction_path(dataset, router_seed))}, never "
            f"retrained.  Pass a different seed to train a new router")
    prereg, prov, spec = _phase_prereg(dataset, prereg, spec)
    if router_seed not in (prereg.get("router_seeds") or []):
        raise RuntimeError(
            f"router seed {router_seed} is not pre-registered: the frozen "
            f"{spec.version} pilot for {dataset} names router seeds "
            f"{prereg.get('router_seeds')}.  Training a router no design asks "
            f"for spends a GPU run that no cell can consume and no verdict can "
            f"use; add the seed in a NEW pre-registration rather than passing "
            f"it to this one")
    roles = (prereg.get("checkpoint_requirements") or {}).get("roles") or {}
    role_name = f"router_g__seed{router_seed}"
    role = roles.get(role_name)
    if role is None:
        raise RuntimeError(
            f"the frozen pilot declares no checkpoint role {role_name!r}, so "
            f"there is nowhere this router's files belong and nothing that "
            f"would later report them missing; the roles it declares are "
            f"{sorted(roles)}")
    man = load_manifest(dataset)
    check_seed_budget(dataset, [router_seed], [], man=man)
    protocol = frozen_route_protocol()
    prompts = frozen_route_prompts()
    ckpt = resolve_recorded_path(role["adapter"])
    pred_path = resolve_recorded_path(role["held_out_predictions"])
    # The training directory is derived from the checkpoint the frozen manifest
    # names rather than written out a second time: two derivations of one path
    # are two paths the moment either is edited, and the manifest only ever sees
    # one of them.
    out_dir = ckpt.parents[1]
    if pred_path.parent != out_dir:
        raise RuntimeError(
            f"router seed {router_seed}'s checkpoint is at {ckpt} but its "
            f"prediction file is at {pred_path}; RF1G trains into ONE directory "
            f"and the design requires both files to be in it, because a "
            f"checkpoint without its own predictions cannot compose and "
            f"predictions without a checkpoint cannot be attributed to a run")
    result_path = out_dir / "router_result.json"

    want = {"cell_id": f"g_seed_{router_seed}",
            "kind": f"router_training_result_{spec.version}",
            "phase": "RF1G", "router_seed": router_seed,
            "input_sha256": _input_sha256(
                prereg,
                route_protocol=sha256_bytes(
                    canonical_json(protocol).encode("utf-8"))),
            "prompts": {"g_image_to_code": prompts["g_image_to_code"]},
            "n_rows": len(man["items"])}

    def run():
        sess = RouteSessionV2(device, f"e2c_rf_g{router_seed}", router_seed)
        try:
            sess.reset_fresh(router_seed)
            train_items = [it for it in man["items"] if it["split"] == "train"]
            pairs = [(it["image_uri"], prompts["g_image_to_code"],
                      man["code_of"][it["identity_id"]]) for it in train_items]
            sess.train(f"g_seed{router_seed}", pairs, out_dir, protocol)
            if not ckpt.is_file():
                raise RuntimeError(
                    f"training reported success but {ckpt} does not exist; the "
                    f"save path and the design's declared path disagree, and a "
                    f"router with no checkpoint cannot be hashed or attributed")
            codes = route_codes(man)
            rows = []
            for it in man["items"]:
                raw = sess.generate(_open_image(it["image_uri"]),
                                    prompts["g_image_to_code"],
                                    max_new_tokens=12)
                pred = sess.rv._extract_code(raw, codes)
                rows.append({
                    "identity_id": it["identity_id"],
                    "image_uri": str(it["image_uri"]),
                    "split": it.get("split"),
                    "source_file_name": it.get("source_file_name"),
                    "image_sha256": it.get("image_sha256"),
                    "g_raw_text": raw, "pred_code": pred,
                    "code_correct": pred == man["code_of"][it["identity_id"]],
                })
        finally:
            sess.release()
        by_split = {}
        for split in sorted({r["split"] for r in rows}):
            sub = [r for r in rows if r["split"] == split]
            by_split[split] = {"n": len(sub),
                               "accuracy": sum(r["code_correct"] for r in sub)
                               / len(sub)}
        atomic_write_json(pred_path, {
            "kind": f"router_held_out_predictions_{spec.version}",
            "dataset": dataset, "router_seed": router_seed,
            "adapter_sha256": sha256_file(ckpt),
            "protocol": protocol, "prompts": want["prompts"],
            "per_split": by_split, "rows": rows})
        doc = {**want, "dataset": dataset,
               "adapter_path": _rel(ckpt),
               "adapter_sha256": sha256_file(ckpt),
               "adapter_bytes": ckpt.stat().st_size,
               "predictions_path": _rel(pred_path),
               "predictions_sha256": sha256_file(pred_path),
               "per_split": by_split,
               "run_provenance": prov,
               "trained_on": ("the train split only, under the protocol read "
                              "from the frozen route scripts"),
               "held_out_accuracy_is_computed_by_rf2": (
                   "router_held_out_accuracy reads the prediction file rather "
                   "than the number recorded here, so the gate and the "
                   "confusion matrix cannot describe two different routers")}
        atomic_write_json(result_path, doc)
        return doc

    doc, decision = _resume_or_run(dataset, f"g_seed_{router_seed}", want,
                                   resume, run, result_path)
    doc = {**doc, "resume": decision}
    atomic_write_json(result_path, doc)
    return doc


def phase_rf1d(dataset, direct_seed, prereg=None, device="cuda", resume=False,
               cells_out=None, spec=None):
    """RF1D: train and evaluate ONE direct model D_s, image only.

    D_s is a separate adapter on the same training associations -- the train
    split's image-to-alias pairs -- under the same frozen protocol, with its own
    checkpoint and its own digest.  It is never edited when an h is edited, so
    its held-out accuracy is PRE-EDIT by construction and there is no post-edit
    variant to compare it with.

    Files two cells from one adapter load: the direct control D_s(X) and the
    auxiliary hybrid probe D_s(X, C_irrelevant).  They are separate conditions
    with separate gates, because the first asks whether the direct pathway
    survived and the second asks whether a code can capture a pathway never
    trained on codes.
    """
    prereg, prov, spec = _phase_prereg(dataset, prereg, spec)
    man = load_manifest(dataset)
    check_seed_budget(dataset, [], [direct_seed], man=man)
    # The frozen route protocol, and then -- only where the design records a
    # calibration -- the schedule that calibration selected.  Absent for every
    # design frozen before v5, so their D_s is trained under the frozen protocol
    # itself and the digest its cells bind is the digest they have always bound.
    direct_training = prereg.get("direct_training")
    protocol = frozen_route_protocol()
    if direct_training:
        protocol = _direct_schedule(protocol, direct_training["schedule"])
    prompts = frozen_route_prompts()
    cells = {c["cell_id"]: c for c in prereg["cells"]}
    # The namespace comes from the design that recorded the selected
    # configuration, which is the same document that named the adapter directory.
    # Absent for every design frozen before v5, so their RF1D writes the ids it
    # has always written.
    want_cells = [direct_cell_id(k, direct_seed,
                                 (direct_training or {}).get("namespace"))
                  for k in ("direct", "hybrid")]
    for cid in want_cells:
        if cid not in cells:
            raise RuntimeError(
                f"the frozen {spec.version} pilot has no cell {cid!r}; the "
                f"direct seeds it pre-registered are {prereg['direct_seeds']}, "
                f"so --direct-seed {direct_seed} is not one of them")
    roles = (prereg.get("checkpoint_requirements") or {}).get("roles") or {}
    role = roles.get(f"direct_d__seed{direct_seed}")
    if role is None:
        raise RuntimeError(
            f"the frozen pilot declares no checkpoint role "
            f"'direct_d__seed{direct_seed}', so there is nowhere this adapter "
            f"belongs and nothing that would later report it missing")
    # Read from the role the design declares rather than recomputed here: the
    # manifest is the authority on where its own inputs live, and a second
    # derivation is a second path the moment either one is edited.
    ckpt = resolve_recorded_path(role["adapter"])
    # derived from the checkpoint the design names, not written out twice
    out_dir = ckpt.parents[1]
    result_path = out_dir / "direct_result.json"
    digest_before = sha256_file(ckpt) if ckpt.is_file() else None
    want = {
        "cell_id": f"direct_seed_{direct_seed}",
        "kind": f"direct_training_result_{spec.version}",
        "phase": "RF1D", "direct_seed": direct_seed,
        "input_sha256": _input_sha256(
            prereg,
            route_protocol=sha256_bytes(
                canonical_json(protocol).encode("utf-8"))),
        "prompts": {"d_image_to_alias": prompts["d_image_to_alias"]},
        "n_rows": sum(cells[c]["n_rows"] for c in want_cells)}

    def run():
        sess = RouteSessionV2(device, f"e2c_rf_d{direct_seed}", direct_seed)
        try:
            sess.reset_fresh(direct_seed)
            # NOT the identical line in RF1G, which trains g and is left alone:
            # g is the frozen route, it was trained on the whole train split, and
            # holding development images out of it would change the route the
            # design compares D_s against.  The asymmetry that creates is
            # disclosed in the result rather than removed by editing g.
            train_items = _direct_train_items(man, direct_training)
            pairs = [(it["image_uri"], prompts["d_image_to_alias"],
                      man["alias_of"][it["identity_id"]]) for it in train_items]
            sess.train(f"direct_seed{direct_seed}", pairs, out_dir, protocol)
            if not ckpt.is_file():
                raise RuntimeError(
                    f"training reported success but {ckpt} does not exist; the "
                    f"design declares this path, so a checkpoint anywhere else "
                    f"is a checkpoint the frozen manifest cannot see")
            digest = sha256_file(ckpt)
            filed = []
            for cid in want_cells:
                design_cell = cells[cid]
                rows = []
                for r in design_cell["rows"]:
                    prompt = (prompts["d_image_to_alias"]
                              if r["condition"] == "direct_control" else
                              HYBRID_PROBE_PROMPT.format(code=r["forced_code"]))
                    raw = sess.generate(_open_image(r["image_uri"]), prompt,
                                        max_new_tokens=8)
                    rows.append({**r, "d_raw_text": raw, "prompt_used": prompt})
                scored, missing = score_rows_v2(rows, prereg["vocab"])
                filed.append(_file_cell(
                    dataset, design_cell, "RF1D", rows, scored, missing,
                    {"prompts": {"d_image_to_alias":
                                 prompts["d_image_to_alias"],
                                 "hybrid_image_and_code": HYBRID_PROBE_PROMPT},
                     # ``ckpt`` above is ``direct_d__seed{S}.adapter`` resolved
                     # from the same role table CELL_INPUT_BINDINGS names, so
                     # this is the digest RF2 will re-hash, computed once for
                     # both cells this phase files rather than twice.
                     "input_sha256": _input_sha256(prereg,
                                                   direct_adapter=digest),
                     "adapter_path": _rel(ckpt),
                     "direct_seed": direct_seed,
                     "run_provenance": prov},
                    cells_out, spec.result_kind))
        finally:
            sess.release()
        held_out = [r for c in filed if c["kind_of_cell"] == "direct"
                    for r in c["scored"]]
        doc = {
            **want, "dataset": dataset, "adapter_path": _rel(ckpt),
            "adapter_sha256": digest, "adapter_bytes": ckpt.stat().st_size,
            "adapter_sha256_before_this_run": digest_before,
            "run_provenance": prov,
            "cells_filed": [c["cell_id"] for c in filed],
            "direct_pre_edit_held_out_accuracy": _mean_correct(held_out),
            "n_held_out_images": len(held_out),
            "trained_on": (
                "the train split's image-to-alias pairs, the same associations "
                "the route encodes, under the protocol read from the frozen "
                "route scripts" if not direct_training else
                f"{len(train_items)} of the train split's image-to-alias pairs "
                f"-- the train split minus the "
                f"{len(direct_training.get('development_image_uris') or ())} "
                f"development images the configuration was selected on -- under "
                f"the frozen route protocol with the schedule the calibration "
                f"selected"),
            **({"training_data_asymmetry_against_the_route":
                    direct_training["training_data_asymmetry"]}
               if (direct_training or {}).get("training_data_asymmetry")
               else {}),
            "pre_edit_by_construction": (
                "D_s is not modified when an h is edited, so there is no "
                "post-edit variant of this number and nothing to compare it "
                "with; that independence is the property item 2 requires"),
            "does_not_depend_on_forced_codes": (
                "the direct control presents no code at all; only the auxiliary "
                "hybrid probe does, and it is gated separately"),
            "independent_of_edited_h": (
                "a separate checkpoint with a separate digest; v1 reused "
                "edited_h, which was trained on code-to-label pairs and is not "
                "an image model"),
        }
        atomic_write_json(result_path, doc)
        return doc

    doc, decision = _resume_or_run(dataset, f"direct_seed_{direct_seed}", want,
                                   resume, run, result_path)
    doc = {**doc, "resume": decision}
    atomic_write_json(result_path, doc)
    return doc


def phase_rf1h(dataset, edit_seed, prereg=None, device="cuda", resume=False,
               cells_out=None, spec=None, forget_set=None):
    """RF1H: the forced-code interventions for ONE edited h.

    ``h_e(do(C=c))`` for every code in the route vocabulary, with NO image: the
    generate call passes None as the image, which is the whole content of item
    1.  Evaluated once per EDIT seed and never repeated per router seed, because
    the intervention does not involve g -- running it three times and filing
    three copies would count one measurement as three.
    """
    prereg, prov, spec = _phase_prereg(dataset, prereg, spec)
    prompts = frozen_route_prompts()
    forget_set = select_forget_set(prereg, forget_set, "RF1H", "intervention")
    cell_id = qualified_cell_id("intervention", forget_set, f"h{edit_seed}")
    cells = {c["cell_id"]: c for c in prereg["cells"]}
    if cell_id not in cells:
        raise RuntimeError(
            f"the frozen {spec.version} pilot has no cell {cell_id!r}; the edit "
            f"seeds it pre-registered are {prereg['edit_seeds']}"
            + (f" and its forget sets are "
               f"{[s['set_id'] for s in design_forget_sets(prereg)]}"
               if design_forget_sets(prereg) else ""))
    design_cell = cells[cell_id]
    role = (prereg["checkpoint_requirements"]["roles"]
            .get(edited_h_role_name(prereg, edit_seed, forget_set)))
    if role is None:
        raise RuntimeError(f"the frozen pilot declares no edited_h role for "
                           f"seed {edit_seed} under forget set "
                           f"{forget_set or prereg.get('forget_set_id')}")
    # The frozen pilot records checkpoint paths RELATIVE to the dataset
    # root, so that verifying it is not tied to one filesystem root.
    ckpt = resolve_recorded_path(role["adapter"])
    if not ckpt.is_file():
        raise RuntimeError(
            f"edited h for seed {edit_seed} is absent at {_rel(ckpt)}; an "
            f"absent checkpoint is a missing input and must not be read as a "
            f"null result")
    inputs = _input_sha256(prereg, **cell_input_digests(prereg, design_cell))

    def run():
        sess = RouteSessionV2(device, f"e2c_rf_h{edit_seed}", edit_seed)
        try:
            sess.reset_to(ckpt)
            rows = []
            for r in design_cell["rows"]:
                prompt = prompts[r["prompt_key"]].format(code=r["forced_code"])
                # image=None: the mediator is hard.  Passing the identity's
                # image here would make the system f(X, C) and the intervention
                # would stop being about the route.
                raw = sess.generate(None, prompt, max_new_tokens=8)
                rows.append({**r, "h_raw_text": raw, "prompt_used": prompt})
        finally:
            sess.release()
        scored, missing = score_rows_v2(rows, prereg["vocab"])
        return _file_cell(dataset, design_cell, "RF1H", rows, scored, missing,
                          {"prompts": {"h_code_to_alias":
                                       prompts["h_code_to_alias"]},
                           "input_sha256": inputs,
                           "adapter_path": _rel(ckpt),
                           "run_provenance": prov},
                          cells_out, spec.result_kind)

    want = {"cell_id": cell_id, "kind": spec.result_kind, "phase": "RF1H",
            "edit_seed": edit_seed, "input_sha256": inputs,
            "prompts": {"h_code_to_alias": prompts["h_code_to_alias"]},
            "n_rows": design_cell["n_rows"]}
    doc, decision = _resume_or_run(
        dataset, cell_id, want, resume, run,
        cell_result_path(dataset, cell_id, cells_out))
    doc = {**doc, "resume": decision}
    write_cell_result(dataset, cell_id, doc, cells_out)
    return doc


def phase_rf1e(dataset, router_seed, edit_seed, prereg=None, device="cuda",
               resume=False, cells_out=None, spec=None, forget_set=None):
    """RF1E: the natural composition for ONE (router seed, edit seed) pair.

    ``c = g_s(X)`` is REPLAYED from that router seed's own held-out prediction
    file and never re-run, so the composition uses exactly the router whose
    digest RF0 hashed; then ``h_e(c)`` is called with no image.  An image whose
    code did not parse is filed as unroutable rather than dropped, because it is
    a routing failure and the factorization has to account for it.
    """
    man = load_manifest(dataset)
    prereg, prov, spec = _phase_prereg(dataset, prereg, spec)
    prompts = frozen_route_prompts()
    forget_set = select_forget_set(prereg, forget_set, "RF1E", "natural")
    cell_id = qualified_cell_id(
        "natural", forget_set, f"g{router_seed}__h{edit_seed}")
    cells = {c["cell_id"]: c for c in prereg["cells"]}
    if cell_id not in cells:
        raise RuntimeError(
            f"the frozen {spec.version} pilot has no cell {cell_id!r}; it "
            f"pre-registered router seeds {prereg['router_seeds']} and edit "
            f"seeds {prereg['edit_seeds']}"
            + (f" and its forget sets are "
               f"{[s['set_id'] for s in design_forget_sets(prereg)]}"
               if design_forget_sets(prereg) else ""))
    design_cell = cells[cell_id]
    preds = load_router_predictions(dataset, router_seed, held_out_uris(man))
    role = (prereg["checkpoint_requirements"]["roles"]
            .get(edited_h_role_name(prereg, edit_seed, forget_set)))
    if role is None:
        raise RuntimeError(f"the frozen pilot declares no edited_h role for "
                           f"seed {edit_seed} under forget set "
                           f"{forget_set or prereg.get('forget_set_id')}")
    # The frozen pilot records checkpoint paths RELATIVE to the dataset
    # root, so that verifying it is not tied to one filesystem root.
    ckpt = resolve_recorded_path(role["adapter"])
    if not ckpt.is_file():
        raise RuntimeError(f"edited h for seed {edit_seed} is absent at "
                           f"{_rel(ckpt)}")
    inputs = _input_sha256(prereg, **cell_input_digests(prereg, design_cell))

    def run():
        sess = RouteSessionV2(device, f"e2c_rf_e{edit_seed}", edit_seed)
        try:
            sess.reset_to(ckpt)
            rows = []
            for r in design_cell["rows"]:
                got = preds["by_uri"][r["image_uri"]]
                code = got["pred_code"]
                row = {**r, "routed_code": code,
                       "routed_code_raw": got["g_raw_text"],
                       "routed_code_correct": got["code_correct"],
                       "routable": code is not None,
                       "observation_pending": False, "filled_by": "RF1E"}
                if code is None:
                    # No ground-truth fallback: g produced no valid code, so the
                    # composition produced no label.  Substituting the image's
                    # own code would make a routing failure invisible.
                    row["h_raw_text"] = None
                else:
                    prompt = prompts["h_code_to_alias"].format(code=code)
                    row["h_raw_text"] = sess.generate(None, prompt,
                                                      max_new_tokens=8)
                    row["prompt_used"] = prompt
                rows.append(row)
        finally:
            sess.release()
        scored, missing = score_rows_v2(rows, prereg["vocab"])
        return _file_cell(dataset, design_cell, "RF1E", rows, scored, missing,
                          {"prompts": {"h_code_to_alias":
                                       prompts["h_code_to_alias"]},
                           "input_sha256": inputs,
                           "adapter_path": _rel(ckpt),
                           "router_predictions_path": _rel(preds["path"]),
                           "router_predictions_sha256": preds["sha256"],
                           "run_provenance": prov},
                          cells_out, spec.result_kind)

    want = {"cell_id": cell_id, "kind": spec.result_kind, "phase": "RF1E",
            "router_seed": router_seed, "edit_seed": edit_seed,
            "input_sha256": inputs,
            "prompts": {"h_code_to_alias": prompts["h_code_to_alias"]},
            "n_rows": design_cell["n_rows"]}
    doc, decision = _resume_or_run(
        dataset, cell_id, want, resume, run,
        cell_result_path(dataset, cell_id, cells_out))
    doc = {**doc, "resume": decision}
    write_cell_result(dataset, cell_id, doc, cells_out)
    return doc


def _load_cells_and_paths_for(prereg, dataset, cells_out=None):
    """Every filed cell the design names, with the file each was read from.

    The path comes back beside the document because the aggregate hashes the cell
    files it consumed: a report that names its inputs by content can be checked
    against the tree later, while one that names only a directory describes
    inputs anybody can change without touching the report.

    The cell kind is read off the manifest, so a v3 aggregate asks for v3 cells:
    the check exists to stop one design's cells being read as another's, and a
    check that always asked for v2 would refuse exactly the cells the current
    design produces.
    """
    spec = SPEC_BY_PREREG_KIND.get(prereg.get("kind"), LATEST_PILOT_SPEC)
    cells, paths = OrderedDict(), OrderedDict()
    for c in prereg["cells"]:
        cid = c["cell_id"]
        cells[cid] = load_cell_result(dataset, cid, cells_out, spec.result_kind)
        paths[cid] = cell_result_path(dataset, cid, cells_out)
    return cells, paths


def _load_cells_for(prereg, dataset, cells_out=None):
    """Every filed cell the frozen design names, or an error naming the gap."""
    return _load_cells_and_paths_for(prereg, dataset, cells_out)[0]


def phase_rf2(dataset, prereg=None, out=None, cells_out=None, spec=None):
    """RF2: aggregate the filed cells, compute the gates and the intervals."""
    prereg, prov, spec = _phase_prereg(dataset, prereg, spec)
    cells, paths = _load_cells_and_paths_for(prereg, dataset, cells_out)
    report = aggregate_cells_for(spec, prereg, cells, rescore=False,
                                 cell_paths=paths, run_provenance=prov)
    path, digest = atomic_write_json(report_path(dataset, "RF2", out, spec),
                                     report)
    logger.info("RF2: filed %s %s", _rel(path), digest[:16])
    return report


def phase_rf2p(dataset, prereg=None, out=None, cells_out=None, spec=None):
    """RF2P: rescore the stored raw outputs and reproduce RF2 with no GPU.

    Same aggregate, same gates, same verdicts -- the only difference is that
    every score is recomputed from the stored generation instead of read from
    the filed one, and every field that moved is reported.  A reproduction that
    silently disagreed with the original would be worse than no reproduction.
    """
    prereg, prov, spec = _phase_prereg(dataset, prereg, spec)
    cells, paths = _load_cells_and_paths_for(prereg, dataset, cells_out)
    report = aggregate_cells_for(spec, prereg, cells, rescore=True,
                                 cell_paths=paths, run_provenance=prov)
    path, digest = atomic_write_json(report_path(dataset, "RF2P", out, spec),
                                     report)
    logger.info("RF2P: filed %s %s", _rel(path), digest[:16])
    return report


# ---------------------------------------------------------------------------
# pilot entry points: one freeze, one phase dispatcher, both version-aware
# ---------------------------------------------------------------------------

def _freeze_for(spec, args):
    """Freeze one pilot version and verify it from the tracked location.

    The constructor is ``spec``-driven, so freezing v3 cannot quietly change the
    bytes v2 produced and vice versa: both go through here, and both are
    verified afterwards by the same ``verify_manifest``.
    """
    man = load_manifest(args.dataset)
    images = held_out_images(man)
    policy = pilot_seed_policy_v2(args.dataset, man=man)
    selection = None
    if not (args.forget_set or args.forget_ids):
        # The pilot's target comes from the stated selection rule applied to the
        # frozen matrix, not from a default and not from hand-picking a set
        # after inspecting which checkpoints exist.
        edit_seeds = args.edit_seeds or policy["edit_seeds"]
        selection = pilot_forget_set(args.dataset, edit_seeds)
        forget_set, forget_ids = selection["set_id"], list(selection["targets"])
    else:
        forget_ids = args.forget_ids or list(
            man.get("forget_identity_ids") or [])
        forget_set = args.forget_set or (
            "fs_" + "-".join(forget_ids) if forget_ids else None)
        if not forget_ids or not forget_set:
            raise RuntimeError("no forget identities: pass --forget-ids or let "
                               "the selection rule choose")

    block = build_pilot_preregistration_for(
        spec, args.dataset, forget_set, forget_ids, args.router_seeds,
        args.edit_seeds, args.direct_seeds, man=man, images=images,
        selection=selection)
    return _freeze_write_and_verify(spec, args, block)


def _freeze_write_and_verify(spec, args, block):
    """Write one built pre-registration, then verify it from where it landed.

    The tail of every version's freeze.  Shared rather than copied because a
    version whose freeze skipped the verification step would file a manifest
    nothing had ever read back -- and the read-back is the only check that the
    artifact reproduces from its own tracked location rather than from the memory
    of the process that built it.
    """
    # ``--out`` is the DIRECTORY this run writes into, the same reading RF0, RF2
    # and RF2P give it, and the manifest keeps its canonical name inside it.  A
    # staged artifact written under some other name is one the phases will never
    # look for, and one that has to be renamed by hand is one that can be
    # renamed wrongly.
    canonical = prereg_path_for(spec, args.dataset)
    out = Path(args.out) / canonical.name if args.out else canonical

    # Every superseded manifest is named as an input, so every later
    # verification of this file re-hashes them: "preserved byte-identical"
    # becomes a checked property rather than an intention.
    superseded = [DATASET_ROOT / MANIFEST_DIR / n
                  for n in spec.superseded_filenames]
    extra_fn = EXTRA_FREEZE_INPUTS_BY_VERSION.get(spec.version)
    extra = [DATASET_ROOT / MANIFEST_PATHS[args.dataset],
             DATASET_ROOT / G_CACHE_PATHS[args.dataset],
             DATASET_ROOT / MATRIX_MANIFEST_DIR / f"matrix_{args.dataset}.json",
             *superseded,
             *(extra_fn(args.dataset) if extra_fn else ())]
    # The roles declared ``exists_already`` are deliberately NOT listed here.
    # They are gitignored adapters, so on a fresh clone they are legitimately
    # absent, and ``verify_manifest`` reports a hashed input it cannot find as
    # drift -- which would make a correct v3 manifest unverifiable in the one
    # place it most needs to verify, CI.  The weights each RESULT actually used
    # are bound in the result instead, and re-hashed by ``verify_cell_inputs``
    # at RF2.
    path, digest = freeze_manifest(block, out, extra_paths=extra)

    ready = checkpoint_readiness(block)
    ap = block["gate_applicability"]
    sel = block["pilot_forget_set_selection"]
    print(f"frozen  : {path}", file=sys.stderr)
    print(f"sha256  : {digest[:16]}", file=sys.stderr)
    print(f"version : {spec.version} ({spec.prereg_kind})", file=sys.stderr)
    print(f"design  : {block['n_cells']} cells "
          f"{ {k: len(v) for k, v in block['cells_by_kind'].items()} }, "
          f"{block['n_rows_total']} rows, vocab {block['n_vocab']}",
          file=sys.stderr)
    if sel.get("sets") is None:
        print(f"forget set        : {sel['set_id']} "
              f"targets={sel['targets']}", file=sys.stderr)
    else:
        print(f"forget sets       : {sel['n_sets']} "
              f"{[s['set_id'] for s in sel['sets']]} "
              f"targets={sel['targets']}", file=sys.stderr)
    print(f"mediator is hard  : "
          f"{not block['mediator_is_hard']['image_to_h_in_any_condition']} "
          f"(h prompt read from the frozen route scripts)", file=sys.stderr)
    print(f"seeds g/h/d       : {block['router_seeds']} / "
          f"{block['edit_seeds']} / {block['direct_seeds']}", file=sys.stderr)
    print(f"bootstrap         : identity {ap['clustering']['held_out_scope']['n_identity_clusters']}"
          f" primary, image "
          f"{ap['clustering']['held_out_scope']['n_image_clusters']} sensitivity",
          file=sys.stderr)
    print(f"verdicts expected : {block['verdicts_to_be_reported']['expected_on_this_dataset']}",
          file=sys.stderr)
    print(f"checkpoints       : {ready['n_files_present']} files present, "
          f"{ready['n_present']} of {ready['n_required']} roles complete; "
          f"{ready['n_must_be_trained']} must be trained first: "
          f"{ready['must_be_trained']}", file=sys.stderr)
    print(f"readiness         : computed LIVE, outside design_sha256 "
          f"(runnable_now={ready['runnable_now']})", file=sys.stderr)
    print(f"supersedes        : "
          f"{[e['path'] for e in block['supersession']['superseded']]}",
          file=sys.stderr)
    print(f"executed          : {block['executed']} (FROZEN ONLY)",
          file=sys.stderr)

    got = verify_manifest(path)
    if not got["valid"]:
        raise RuntimeError(
            f"the manifest just frozen at {path} does not verify from its own "
            f"tracked location: " + "; ".join(got["problems"]))
    print(f"verified: {got['n_checkpoints_rehashed']} checkpoint digests "
          f"re-hashed, design_sha256 reproduced", file=sys.stderr)
    return 0


def _freeze_v2(args):
    return _freeze_for(PILOT_SPEC_V2, args)


def _freeze_v3(args):
    return _freeze_for(PILOT_SPEC_V3, args)


def _run_phase_for(spec, args, phase):
    """Run one phase against one pilot version.

    ``spec`` decides which manifest the phase loads and which filename it files
    under; the phases themselves read the spec off the document they were given,
    so a caller that hands one over directly is not overridden here.
    """
    ds = args.dataset
    if phase == "RF0":
        rep = phase_rf0(ds, args.manifest, args.out, args.cells)
        print(json.dumps({k: rep[k] for k in
                          ("phase", "dataset", "design_version", "manifest",
                           "manifest_valid",
                           "problems", "kind", "design_sha256", "n_cells",
                           "cells_by_kind", "n_checkpoints_rehashed",
                           "checkpoint_requirements_complete",
                           "n_must_be_trained",
                           "must_be_trained_before_this_pilot_can_run",
                           "unexpectedly_absent",
                           "roles_that_changed_since_freeze",
                           "readiness_is_computed_live",
                           "held_out_image_drift_since_freeze",
                           "n_cells_filed", "n_cells_missing",
                           "runnable_now", "why_not_runnable",
                           "expected_verdicts")},
                         indent=2))
        for seed, entry in sorted(rep["routers"].items()):
            acc = entry["held_out_accuracy"]
            print(f"router {seed}: predictions={entry['predictions_present']} "
                  f"held_out_accuracy="
                  f"{acc['accuracy'] if acc else entry['accuracy_error']}",
                  file=sys.stderr)
        # The exit code reports VERIFICATION, not runnability: a pilot frozen
        # before its training work exists is valid and is supposed to say what
        # is still missing.  Failing here would make the honest state unusable.
        return 0 if rep["manifest_valid"] else 1
    if phase == "RFC":
        if "RFC" not in PHASES_BY_VERSION[spec.version]:
            raise RuntimeError(
                f"{spec.version} has no calibration phase; the phases it has are "
                f"{PHASES_BY_VERSION[spec.version]}.  A version that declared no "
                f"calibration has no configuration to select, and training one "
                f"would produce adapters no design of that version names")
        rep = phase_rfc(ds, args.candidate, args.direct_seed,
                        device=args.device, resume=args.resume)
        print(json.dumps(
            {k: rep.get(k) for k in
             ("phase", "candidate", "seed", "development_accuracy",
              "n_development_images", "n_fit_images", "selected",
              "selection_refused", "resume")}, indent=2))
        return 0
    if phase == "RF1B":
        rep = phase_rf1b(ds, device=args.device, resume=args.resume,
                         cells_out=args.cells, spec=spec,
                         forget_set=args.forget_set)
        print(json.dumps({k: rep.get(k) for k in
                          ("phase", "cell_id", "n_rows", "input_sha256",
                           "adapter_path", "resume")}, indent=2))
        return 0
    if phase == "RF1G":
        if args.router_seed is None:
            raise RuntimeError("RF1G trains ONE router seed: pass --router-seed")
        rep = phase_rf1g(ds, args.router_seed, device=args.device,
                         resume=args.resume, spec=spec)
        print(json.dumps({k: rep.get(k) for k in
                          ("phase", "router_seed", "adapter_sha256",
                           "predictions_sha256", "per_split", "resume",
                           "preregistration_design_sha256",
                           "preregistration_file_sha256")},
                         indent=2))
        return 0
    if phase == "RF1D":
        if args.direct_seed is None:
            raise RuntimeError("RF1D trains ONE direct model: pass "
                               "--direct-seed")
        rep = phase_rf1d(ds, args.direct_seed, device=args.device,
                         resume=args.resume, cells_out=args.cells, spec=spec)
        print(json.dumps({k: rep.get(k) for k in
                          ("phase", "direct_seed", "adapter_sha256",
                           "cells_filed", "direct_pre_edit_held_out_accuracy",
                           "n_held_out_images", "resume")}, indent=2))
        return 0
    if phase == "RF1H":
        if args.edit_seed is None:
            raise RuntimeError("RF1H evaluates ONE edited h: pass --edit-seed")
        rep = phase_rf1h(ds, args.edit_seed, device=args.device,
                         resume=args.resume, cells_out=args.cells, spec=spec,
                         forget_set=args.forget_set)
        print(json.dumps({k: rep.get(k) for k in
                          ("phase", "cell_id", "edit_seed", "n_rows",
                           "input_sha256", "resume")}, indent=2))
        return 0
    if phase == "RF1E":
        if args.router_seed is None or args.edit_seed is None:
            raise RuntimeError("RF1E composes ONE (router, edit) pair: pass "
                               "--router-seed and --edit-seed")
        rep = phase_rf1e(ds, args.router_seed, args.edit_seed,
                         device=args.device, resume=args.resume,
                         cells_out=args.cells, spec=spec,
                         forget_set=args.forget_set)
        print(json.dumps({k: rep.get(k) for k in
                          ("phase", "cell_id", "router_seed", "edit_seed",
                           "n_rows", "input_sha256", "resume")}, indent=2))
        return 0
    if phase in ("RF2", "RF2P"):
        run = phase_rf2 if phase == "RF2" else phase_rf2p
        rep = run(ds, out=args.out, cells_out=args.cells, spec=spec)
        summary = {
            "phase": phase, "dataset": ds, "kind": rep["kind"],
            "n_cells": rep["n_cells"],
            "n_rows": rep["n_rows"],
            "mediation_pass": rep["mediation_pass"],
            "routing_reliability_pass": rep["routing_reliability_pass"],
            "routing_factorization_pass": rep["routing_factorization_pass"],
            "failed_gates": [n for n, g in rep["gates"].items()
                             if g.get("is_a_gate", True) and not g["passed"]],
            "verdict_states": {n: rep["verdicts"][n]["state"]
                               for n in VERDICT_NAMES},
            "no_omnibus_verdict": rep["verdicts"]["no_omnibus_verdict"],
            "input_verification": rep["input_verification"],
            "n_consumed_cell_files": rep["n_consumed_cell_files"],
        }
        if phase == "RF2P":
            summary["rescore_identical_to_filed"] = \
                rep["rescore_agreement"]["identical_to_the_filed_scores"]
            summary["n_fields_that_moved"] = \
                rep["rescore_agreement"]["n_fields_that_moved"]
        print(json.dumps(summary, indent=2))
        return 0
    raise RuntimeError(f"unknown {spec.version} phase {phase!r}; the phases "
                       f"are {PHASES_BY_VERSION[spec.version]}")


def _run_v2_phase(args, phase):
    """The v2 dispatcher, kept so an explicit ``--design-version v2`` runs v2."""
    return _run_phase_for(PILOT_SPEC_V2, args, phase)


def _run_pilot(args, spec):
    if args.verify:
        path = Path(args.manifest or prereg_path_for(spec, args.dataset))
        got = verify_manifest(path)
        print(json.dumps(got, indent=2))
        return 0 if got["valid"] else 1
    if args.preregister_calibration:
        if not spec.calibration_phase:
            raise RuntimeError(
                f"{spec.version} declared no calibration, so there is no "
                f"calibration pre-registration to freeze; the versions that have "
                f"one are "
                f"{sorted(v for v, s in SPEC_BY_VERSION.items() if s.calibration_phase)}")
        return freeze_calibration(args)
    if args.preregister:
        return FREEZE_BY_VERSION.get(spec.version, _freeze_for)(spec, args)
    rc = 0
    for phase in args.phase:
        rc |= _run_phase_for(spec, args, phase)
    return rc


def _run_v2(args):
    return _run_pilot(args, PILOT_SPEC_V2)


def _run_v3(args):
    return _run_pilot(args, PILOT_SPEC_V3)


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if "RF1" in args.phase:
        raise RuntimeError(
            "RF1 no longer exists: it was one stub that raised for every "
            "dataset.  v2 splits it into RF1G (train/evaluate one router seed), "
            "RF1D (train/evaluate one direct-model seed), RF1H (the forced-code "
            "interventions for one edit seed) and RF1E (the natural composition "
            "for one router/edit pair), each naming the single factor it varies")
    if args.design_version == "v1":
        return _run_v1(args)
    # Once, before any phase runs: the provenance every filed record carries
    # includes the complete command line, so it has to be captured from the
    # parsed arguments rather than reconstructed later from a module global
    # that a test or a caller may have left as it found it.
    note_cli(args)
    spec = SPEC_BY_VERSION[args.design_version]
    return _run_pilot(args, spec)


# ===========================================================================
# PART III -- v4: one denominator, declared after the outcome it corrects
# ===========================================================================
#
# Everything v4 adds or changes lives in this one contiguous block, at the end
# of the module, and it works by REBINDING the version tables the code above
# reads: LATEST_PILOT_SPEC, SPEC_BY_VERSION, SPEC_BY_PREREG_KIND,
# PHASES_BY_VERSION, REPAIRS_BY_VERSION, SUPERSESSION_POLICY, SUPERSEDED_NOTES
# and the two dispatch tables the shared builders consult.  That is safe at the
# end because every reference above resolves at call time rather than at import
# time.
#
# Why one block and not eleven edits scattered through eight thousand lines: a
# scope amendment made AFTER a run's verdicts were read has to be inspectable as
# one object.  A reader who wants to know exactly what v4 changes, and what it
# deliberately leaves alone, should read one screen rather than reconstruct it
# out of a diff.  Keeping it here also leaves the constructors above -- the ones
# that produced the frozen v1, v2 and v3 bytes -- textually untouched, which is
# the only reason to believe those bytes still reproduce.
#
# WHAT V4 CHANGES
#   One denominator.  ``no_unparseable_or_multi_label_outputs`` counted every
#   row the run produced, including the rows of ``hybrid_conflict_probe``, which
#   the same design declares ``auxiliary: true``, ``decisive: false`` and
#   "reported separately from every mediated gate" -- and then mapped the count
#   to the mediation verdict.  Each of those two statements is defensible on its
#   own; holding both at once is not, and the run found out by failing a verdict
#   on rows the design had already said were not part of it.
#
#   v4 scopes the verdict-bearing count to the rows the verdict rests on and
#   reports the auxiliary rows' output hygiene as its own diagnostic, which is
#   named in the report and feeds nothing.
#
# WHAT V4 DOES NOT CHANGE
#   No threshold.  No gate-to-verdict mapping.  No condition, no prompt, no row
#   builder, no cell layout, no result kind, no checkpoint role.  A v4 cell and
#   a v3 cell of the same id are the same bytes, which is what makes reusing the
#   filed v3 cells a re-analysis rather than a relabelling.
#
#   ``direct_code_following_rate`` is measured entirely on auxiliary rows and
#   STILL maps to mediation.  It is not a hygiene count: it is the ceiling v2
#   added so that an accuracy floor alone could not be satisfied by a direct
#   adapter that answered from the code and was therefore not a direct model at
#   all.  Moving it out of the verdict would undo that repair, so v4's claim is
#   narrower than "auxiliary rows carry no verdict" and is the one that is true:
#   auxiliary rows are outside every mediated HYGIENE count.
#
# WHAT V4 IS NOT
#   It is not a pre-registration of the run whose verdicts prompted it.  Those
#   cells were produced under v3, are consumed unchanged and are not relabelled;
#   the analysis v4 performs on them is CORRECTIVE.  A confirmatory result under
#   the corrected scope requires a run executed under v4 from the start, and
#   every artifact v4 produces says so in so many words.

#: v4's kinds.  The supersession kind names what it SUPERSEDES, following the
#: convention v2 and v3 set, and v3 supersedes two versions rather than one --
#: so a reader can tell from the kind alone how many frozen designs are behind
#: this one.
KIND_V4 = "route_dependent_forgetting_design_v4"
PREREG_V4_KIND = "route_dependent_forgetting_preregistration_v4"
SUPERSESSION_KIND_V4 = "route_dependent_forgetting_supersession_v1_v2_and_v3"

#: v4 REUSES v3's result kind, and that is forced rather than convenient.
#:
#: ``load_cell_result`` refuses a cell whose ``kind`` is not the aggregating
#: spec's ``result_kind``, so a new kind would refuse all 28 filed cells and
#: repeat 26 GPU phase invocations -- six of them 3000-step training runs -- to
#: produce byte-identical rows.  A version whose only claim is that one gate
#: counts a different subset of the SAME rows has no business re-measuring them.
#:
#: The price is that a v4 report rests on v3-stamped cells, so the pairing is
#: checked (``verify_cell_design_lineage``) and disclosed (the
#: ``cell_design_lineage`` block every v4 report carries) rather than left to a
#: reader who notices the kind string.
RESULT_KIND_V4 = RESULT_KIND_V3

#: The v3 condition table, reused by object.  Nothing v4 says is about what a
#: condition EXECUTES, so nothing here is reworded: ``execution``, ``role`` and
#: ``not_the_mediator_intervention`` are copied onto every row a builder makes,
#: and editing them would make a v4 cell differ in bytes from the v3 cell it is
#: supposed to be identical to.  The correction v4 makes is about how rows are
#: COUNTED, which belongs to the gate and to the declared scope below.
CONDITIONS_V4 = CONDITIONS_V3

FORCED_CODE_CONDITIONS_V4 = tuple(
    n for n, c in CONDITIONS_V4.items()
    if c["prompt_key"] == "h_code_to_alias"
    and "router_seed" not in c["depends_on"])
MEDIATED_CONDITIONS_V4 = tuple(
    n for n, c in CONDITIONS_V4.items()
    if not c.get("auxiliary") and c["prompt_key"] == "h_code_to_alias")
AUXILIARY_CONDITIONS_V4 = tuple(n for n, c in CONDITIONS_V4.items()
                                if c.get("auxiliary"))
DECISIVE_CONDITIONS_V4 = tuple(n for n, c in CONDITIONS_V4.items()
                               if c.get("decisive"))

#: Derived by comparing the two tables rather than written down, so the claim
#: "v4 changes no measurement" is a fact about the objects and not a sentence
#: somebody has to keep true.
V4_CONDITION_TABLE_DIFFERENCE = OrderedDict((
    ("conditions_added",
     tuple(n for n in CONDITIONS_V4 if n not in CONDITIONS_V3)),
    ("conditions_removed",
     tuple(n for n in CONDITIONS_V3 if n not in CONDITIONS_V4)),
    ("condition_entries_changed",
     tuple(n for n in CONDITIONS_V4
           if n in CONDITIONS_V3 and CONDITIONS_V4[n] != CONDITIONS_V3[n])),
    ("same_object", CONDITIONS_V4 is CONDITIONS_V3),
    ("why", ("a row copies its condition's execution text, so rewording the "
             "table would change the bytes of every cell v4 files and break the "
             "one property that makes reusing the v3 cells honest: that they are "
             "the same measurements, not similar ones")),
))

#: v4 changes no threshold and remaps no gate.  Both statements are derived.
GATE_THRESHOLDS_V4 = GATE_THRESHOLDS_V3
GATE_TO_VERDICT_V4 = GATE_TO_VERDICT_V3

V4_THRESHOLD_DIFFERENCE = tuple(
    k for k in GATE_THRESHOLDS_V4
    if k not in GATE_THRESHOLDS_V3 or GATE_THRESHOLDS_V4[k]
    != GATE_THRESHOLDS_V3[k])
V4_GATE_MAP_DIFFERENCE = tuple(
    k for k in GATE_TO_VERDICT_V4
    if k not in GATE_TO_VERDICT_V3 or GATE_TO_VERDICT_V4[k]
    != GATE_TO_VERDICT_V3[k])

#: What the design says about the hygiene gate's denominator, as opposed to what
#: number it compares against.  Same shape as ``PER_SEED_GATE_AGGREGATION`` and
#: for the same reason: the threshold and the row scope are two halves of one
#: rule, and a pre-registration that recorded only the number had not recorded
#: the rule it was frozen with.
DECLARED_HYGIENE_SCOPE_V4 = OrderedDict((
    ("hygiene_gate_row_scope", HYGIENE_SCOPE_NON_AUXILIARY),
    ("which_gate_it_scopes", "no_unparseable_or_multi_label_outputs"),
    ("rule", ("counted over the rows of every condition that is not declared "
              "auxiliary, which is the set of rows the mediation verdict rests "
              "on")),
    ("n_rows_in_the_whole_run_is_still_reported",
     ("the gate names both denominators, so a reader can see what was excluded "
      "and how many rows that was rather than inferring it")),
    ("auxiliary_rows_are_reported_not_dropped",
     ("their unparseable and multi-label counts appear in the report under "
      "auxiliary_output_hygiene, with the offending row ids and the raw outputs "
      "themselves")),
    ("what_it_does_not_move",
     ("direct_code_following_rate is measured entirely on auxiliary rows and "
      "still feeds mediation.  It is not a hygiene count; it is the ceiling v2 "
      "added so an accuracy floor alone could not be met by a direct adapter "
      "that answered from the code.  Scoping it out would undo a repair, so the "
      "claim here is narrower and is the true one: auxiliary rows are outside "
      "every mediated HYGIENE count")),
    ("why_the_scope_belongs_in_the_design",
     ("'zero unparseable outputs' and 'zero unparseable outputs among the rows "
      "this verdict rests on' are two analyses of one run that reach different "
      "verdicts.  A choice between them made in the aggregation code is made "
      "after the rows exist and is invisible in the frozen design, which is the "
      "one artifact that is supposed to say what was pre-registered")),
    ("the_contradiction_it_resolves",
     ("v3 declared hybrid_conflict_probe auxiliary, not decisive and 'reported "
      "separately from every mediated gate', echoed that in its own report as "
      "auxiliary_conditions_reported_separately, and then mapped a count over "
      "ALL rows -- including those -- to mediation.  Both halves are "
      "defensible; a design that states both is not")),
    ("post_outcome_scope_amendment", True),
    ("what_that_means",
     ("this scope was declared after the v3 run's verdicts were read.  Results "
      "aggregated under it from v3's cells are a CORRECTIVE analysis of a "
      "completed run and are not pre-registered, however exactly they are "
      "recomputed.  Only a run executed under v4 from the start is "
      "pre-registered under the corrected scope")),
    ("the_evidence_for_it_is_recorded_in_this_design",
     ("scope_amendment below is computed at freeze time from the committed v3 "
      "reports of BOTH datasets, so a reader can see what the amendment rests "
      "on and that it was not chosen to reach one dataset's conclusion")),
))

#: What the design says about consuming cells another version filed.  The
#: per-dataset half -- the ancestor manifest's own digest -- is filled in by the
#: shared builder, which is where the dataset is known.
DECLARED_CELL_COMPATIBILITY_V4 = OrderedDict((
    ("rule", "cells_are_consumed_from_the_ancestor_design_and_checked"),
    ("ancestor_version", "v3"),
    ("accepted_result_kinds", (RESULT_KIND_V4,)),
    ("why_the_result_kind_is_reused",
     ("load_cell_result refuses a cell whose kind is not the aggregating spec's "
      "result_kind.  v4 changes how rows are counted and not how they are "
      "produced, so a new kind would have refused all 28 filed cells and "
      "repeated 26 GPU phase invocations to write identical bytes back")),
    ("what_is_checked_before_a_cell_is_consumed",
     ("every cell's own recorded preregistration_design_sha256 is compared "
      "against this design's and against the ancestor design frozen into this "
      "manifest; a cell matching neither is refused rather than aggregated")),
    ("states", ("produced_under_this_design",
                "reused_from_the_ancestor_design",
                "unrecognized_design")),
    ("the_cells_are_not_relabelled",
     ("a reused cell keeps its v3 kind, its v3 provenance and its v3 file "
      "bytes.  Relabelling it would destroy the record of which design actually "
      "executed it, and the whole point of the check is that the two designs "
      "are visibly different")),
    ("what_a_v4_report_must_disclose",
     ("input_verification records the pre-registration a cell was produced "
      "under without comparing it to the one doing the aggregating, so a v4 "
      "report over v3 cells would otherwise look self-produced.  Every v4 "
      "report therefore carries cell_design_lineage naming both designs and "
      "how many cells came from each")),
))

PILOT_SPEC_V4 = PilotSpec(
    version="v4", kind=KIND_V4, prereg_kind=PREREG_V4_KIND,
    result_kind=RESULT_KIND_V4, supersession_kind=SUPERSESSION_KIND_V4,
    conditions=CONDITIONS_V4,
    forced_code_conditions=FORCED_CODE_CONDITIONS_V4,
    mediated_conditions=MEDIATED_CONDITIONS_V4,
    auxiliary_conditions=AUXILIARY_CONDITIONS_V4,
    decisive_conditions=DECISIVE_CONDITIONS_V4,
    thresholds=GATE_THRESHOLDS_V4, gate_to_verdict=GATE_TO_VERDICT_V4,
    baseline_cell=True, embed_live_checkpoint_status=False,
    direct_gate_aggregation="every_seed",
    declared_gate_aggregation={
        "direct_gate_aggregation": PER_SEED_GATE_AGGREGATION},
    hygiene_gate_row_scope=HYGIENE_SCOPE_NON_AUXILIARY,
    declared_hygiene_scope=DECLARED_HYGIENE_SCOPE_V4,
    declared_cell_compatibility=DECLARED_CELL_COMPATIBILITY_V4,
    superseded_filenames=("rf_pilot_ppubench.json", "rf_pilot_salmu.json",
                          "rf_pilot_ppubench_v2.json",
                          "rf_pilot_salmu_v2.json",
                          "rf_pilot_ppubench_v3.json",
                          "rf_pilot_salmu_v3.json"))

#: The manifest each version's cells may legitimately come from, if it is not
#: this version's own.  Empty for v2 and v3: neither consumes another design's
#: cells, and a version that has not declared an ancestor has no ancestor to
#: accept a cell from.
ANCESTOR_MANIFEST_BY_VERSION["v4"] = "rf_pilot_{dataset}_v3.json"

#: Named beside the filename rather than parsed out of it, because a version
#: string extracted from a filename is one that a rename silently changes.
ANCESTOR_VERSION_BY_VERSION = {"v4": "v3"}


def ancestor_manifest_path(spec, dataset):
    """The frozen manifest a version's cells may have been produced under."""
    name = ANCESTOR_MANIFEST_BY_VERSION.get(spec.version)
    if not name:
        return None
    return DATASET_ROOT / MANIFEST_DIR / name.format(dataset=dataset)


def ancestor_design_sha256(spec, dataset):
    """The ancestor design's identity, read out of its own committed bytes.

    Read and not verified.  ``verify_manifest`` on a superseded pilot reports
    drift, because such a pilot binds the bytes of this script as it was when it
    was frozen and this script has since moved on; that is the documented state
    of a superseded artifact and not a reason to refuse a cell it produced.  What
    identifies the ancestor here is the design hash it recorded for itself, which
    is a fact about that file and does not change when this one is edited.
    """
    p = ancestor_manifest_path(spec, dataset)
    if p is None:
        return None
    if not p.is_file():
        raise RuntimeError(
            f"{spec.version} declares an ancestor manifest at {_rel(p)} and it "
            f"is absent; without it a cell filed under the ancestor design "
            f"cannot be told from a cell filed under no design at all")
    frozen = json.loads(p.read_text(encoding="utf-8"))
    return OrderedDict((
        ("ancestor_version", ANCESTOR_VERSION_BY_VERSION[spec.version]),
        ("manifest", _rel(p)),
        ("manifest_sha256", sha256_file(p)),
        ("design_sha256", frozen.get("design_sha256")),
        ("kind", frozen.get("kind")),
        ("read_not_verified",
         ("a superseded pilot binds this script's earlier bytes and so reports "
          "drift by design; its own recorded design_sha256 is still the "
          "identity of the design it froze")),
    ))


def v4_scope_amendment_evidence(dataset, spec=None):
    """What the amendment rests on, computed from the committed v3 record.

    Read out of the two v3 RF2 reports and the two v3 manifests rather than
    written down, for the reason every other count in this file is derived: a
    number typed into a rationale is a number nobody can check, and this one is
    doing more work than most -- it is the reason a frozen design changed after
    its verdicts were read.

    BOTH datasets are recorded in BOTH manifests.  An amendment justified by one
    dataset's numbers alone is an amendment tailored to that dataset, and the
    honest test of a scope correction is whether it moves the verdicts it should
    move and leaves the others where they were.  Here it does: PPUBench's
    mediation failed on this gate alone and is rescued, SALMU's failed on two
    gates and is not.

    Binding the reports' digests inside the design is deliberate and is not the
    mistake v2 made.  v2 bound the state of work it had NOT done, so doing the
    work broke the design.  These reports are filed, committed and superseded;
    nothing this pilot will ever do changes them, and if somebody did change them
    the rationale below would be citing bytes that no longer exist -- which is
    exactly when this manifest ought to stop reproducing.
    """
    spec = spec or PILOT_SPEC_V4
    aux = set(spec.auxiliary_conditions)
    # The gates that feed the mediation verdict, read off the map rather than
    # listed: the amendment touches one of them and the block has to show that
    # it touched only that one.
    mediation_gates = {g for g, v in spec.gate_to_verdict.items()
                       if v == "mediation"}
    per_dataset = OrderedDict()
    sources = OrderedDict()
    for ds in ("ppubench", "salmu"):
        rep_path = report_path(ds, "RF2", None, PILOT_SPEC_V3)
        man_path = prereg_path_for(PILOT_SPEC_V3, ds)
        if not rep_path.is_file() or not man_path.is_file():
            raise RuntimeError(
                f"the v4 scope amendment cites the v3 record of {ds} and "
                f"{_rel(rep_path) if not rep_path.is_file() else _rel(man_path)} "
                f"is absent; an amendment whose evidence cannot be read is an "
                f"assertion")
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        man = json.loads(man_path.read_text(encoding="utf-8"))
        sources[ds] = OrderedDict((
            ("v3_report", _rel(rep_path)),
            ("v3_report_sha256", sha256_file(rep_path)),
            ("v3_manifest", _rel(man_path)),
            ("v3_manifest_sha256", sha256_file(man_path)),
            ("v3_design_sha256", rep.get("preregistration_design_sha256")),
            ("v3_report_kind", rep.get("kind")),
        ))
        # Which cells are auxiliary is read off the frozen design's own row
        # table, not guessed from a cell id: the design is what declares a
        # condition auxiliary, so it is what decides which rows fall outside the
        # count.
        aux_cells = set()
        for cell in man.get("cells") or []:
            conds = {r.get("condition") for r in cell.get("rows") or []}
            if conds and conds <= aux:
                aux_cells.add(cell["cell_id"])
        reported_aux = set(rep.get("auxiliary_conditions_reported_separately")
                           or [])
        if reported_aux != aux:
            raise RuntimeError(
                f"the v3 report of {ds} names auxiliary conditions "
                f"{sorted(reported_aux)} and this design names {sorted(aux)}; "
                f"the amendment would be scoping rows out of a count on the "
                f"strength of a table the filed run did not use")
        gate = rep["gates"]["no_unparseable_or_multi_label_outputs"]
        th = man["gate_thresholds"]
        whole = {"unparseable": 0, "multi_label_ambiguous": 0, "n_rows": 0}
        auxc = {"unparseable": 0, "multi_label_ambiguous": 0, "n_rows": 0}
        for cid, cc in (rep.get("cells") or {}).items():
            whole["unparseable"] += cc["n_unparseable"]
            whole["multi_label_ambiguous"] += cc["n_multi_label_ambiguous"]
            whole["n_rows"] += cc["n_rows"]
            if cid in aux_cells:
                auxc["unparseable"] += cc["n_unparseable"]
                auxc["multi_label_ambiguous"] += cc["n_multi_label_ambiguous"]
                auxc["n_rows"] += cc["n_rows"]
        non = {k: whole[k] - auxc[k] for k in whole}
        passes = (non["unparseable"] <= th["max_unparseable_outputs"]
                  and non["multi_label_ambiguous"]
                  <= th["max_multi_label_outputs"])
        v3_mediation = rep["verdicts"]["mediation"]
        still = [n for n in v3_mediation["failed_gates"]
                 if n != "no_unparseable_or_multi_label_outputs"]
        if not passes:
            still = sorted(set(still)
                           | {"no_unparseable_or_multi_label_outputs"})
        # not_established is carried through rather than recomputed: a dataset
        # that cannot support the question supports it no better under a
        # different denominator, and reporting pass or fail for one would invent
        # a measurement -- which is the coercion the three-state verdict exists
        # to prevent.
        under_v4 = (NOT_ESTABLISHED
                    if v3_mediation["state"] == NOT_ESTABLISHED
                    else ("fail" if still else "pass"))
        per_dataset[ds] = OrderedDict((
            ("n_cells", rep.get("n_cells")),
            ("auxiliary_cells", sorted(aux_cells)),
            ("rows", OrderedDict((
                ("in_the_whole_run", whole["n_rows"]),
                ("auxiliary", auxc["n_rows"]),
                ("counted_under_the_v4_scope", non["n_rows"])))),
            ("v3_hygiene_gate", OrderedDict((
                ("row_scope", HYGIENE_SCOPE_ALL_ROWS),
                ("n", gate["n"]),
                ("value", gate["value"]),
                ("passed", gate["passed"])))),
            ("the_same_counts_split", OrderedDict((
                ("auxiliary", {k: auxc[k] for k in
                               ("unparseable", "multi_label_ambiguous")}),
                ("non_auxiliary", {k: non[k] for k in
                                   ("unparseable", "multi_label_ambiguous")})))),
            ("hygiene_gate_under_the_v4_scope", OrderedDict((
                ("row_scope", HYGIENE_SCOPE_NON_AUXILIARY),
                ("n", non["n_rows"]),
                ("value", {"unparseable": non["unparseable"],
                           "multi_label_ambiguous":
                               non["multi_label_ambiguous"]}),
                ("passed", passes),
                ("thresholds_unchanged",
                 {k: th[k] for k in ("max_unparseable_outputs",
                                     "max_multi_label_outputs")})))),
            ("mediation", OrderedDict((
                ("state_under_v3", v3_mediation["state"]),
                ("failed_gates_under_v3", list(v3_mediation["failed_gates"])),
                ("mediated_gates_that_still_fail_under_v4", still),
                ("state_under_v4", under_v4),
                ("moved_by_the_amendment",
                 v3_mediation["state"] != under_v4)))),
            ("the_other_gates_that_feed_mediation_are_untouched", OrderedDict(
                (n, {"passed": rep["gates"][n]["passed"],
                     "value": rep["gates"][n].get("value"),
                     "threshold": rep["gates"][n].get("threshold"),
                     "levels_that_failed":
                         rep["gates"][n].get("levels_that_failed"),
                     "n": rep["gates"][n].get("n")})
                for n in sorted(mediation_gates)
                if n in rep["gates"]
                and n != "no_unparseable_or_multi_label_outputs")),
            ("every_verdict_as_v3_filed_it", OrderedDict(
                (v, rep["verdicts"][v]["state"]) for v in VERDICT_NAMES)),
        ))
    moved = [ds for ds, e in per_dataset.items()
             if e["mediation"]["moved_by_the_amendment"]]
    unmoved = [ds for ds, e in per_dataset.items()
               if not e["mediation"]["moved_by_the_amendment"]]
    return OrderedDict((
        ("what_this_block_is",
         ("the evidence the v4 row scope was amended on, computed at freeze time "
          "from the committed v3 record of both datasets")),
        ("this_manifests_dataset", dataset),
        ("both_datasets_are_recorded_in_both_manifests",
         ("an amendment justified by one dataset's numbers is an amendment "
          "tailored to that dataset; the test of a scope correction is whether "
          "it moves the verdicts it should move and leaves the rest where they "
          "were")),
        ("per_dataset", per_dataset),
        ("sources", sources),
        ("datasets_whose_mediation_verdict_moves", sorted(moved)),
        ("datasets_whose_mediation_verdict_does_not_move", sorted(unmoved)),
        ("the_amendment_is_not_a_waiver",
         ("no threshold is lowered and no gate is unmapped.  Where the excluded "
          "rows also fail a gate that is not a hygiene count, the verdict still "
          "fails, and this block says which datasets those are")),
        ("n_gates_in_the_v4_map", len(GATE_TO_VERDICT_V4)),
        ("n_gates_in_the_v3_map", len(GATE_TO_VERDICT_V3)),
        ("thresholds_that_moved", list(V4_THRESHOLD_DIFFERENCE)),
        ("gate_mappings_that_moved", list(V4_GATE_MAP_DIFFERENCE)),
        ("condition_table_difference", V4_CONDITION_TABLE_DIFFERENCE),
        ("auxiliary_conditions", list(spec.auxiliary_conditions)),
        ("n_gates_that_feed_mediation", len(mediation_gates)),
    ))


#: The repairs v4 makes over v3, in the shape v3's record uses: each item names
#: the two versions it moves between, so the progression reads v1 -> v2 -> v3 ->
#: v4 rather than as one flat list of complaints.
SUPERSESSION_ITEMS_V4 = OrderedDict((
    ("the_hygiene_gate_has_one_denominator", {
        "v3": "no_unparseable_or_multi_label_outputs counted every row the run "
              "produced, including the rows of hybrid_conflict_probe, and "
              "mapped that count to mediation -- while the same design declared "
              "that condition auxiliary, not decisive, and 'reported separately "
              "from every mediated gate', and its own report echoed the "
              "declaration as auxiliary_conditions_reported_separately.  Both "
              "statements are defensible; holding both is not.  The PPUBench run "
              "failed mediation on 9 unparseable outputs, all 9 of them in the "
              "auxiliary probe, with 64 non-auxiliary rows clean",
        "v4": "the row scope is a declared field of the design "
              "(hygiene_gate_row_scope), the verdict-bearing count covers the "
              "non-auxiliary rows the verdict rests on, and the gate names both "
              "denominators so a reader sees what was excluded and how many rows "
              "that was.  A denominator chosen in the aggregation code is chosen "
              "after the rows exist and is invisible in the frozen design"}),
    ("auxiliary_output_hygiene_is_reported_and_feeds_nothing", {
        "v3": "the auxiliary rows' unparseable outputs were visible only as a "
              "count inside a gate they were not supposed to decide, so the one "
              "thing worth knowing about them -- that the probe derails out of "
              "the candidate vocabulary entirely -- was reported as a number and "
              "not as a finding",
        "v4": "they are reported under auxiliary_output_hygiene with "
              "is_a_gate=False and feeds_no_verdict=True, carrying both counts, "
              "the rate, the offending row ids, the raw outputs themselves and "
              "the distinct expected labels missed.  Excluded from a verdict is "
              "not the same as deleted, and a diagnostic nobody reports is a "
              "measurement nobody made"}),
    ("direct_code_following_rate_stays_where_it_was", {
        "v3": "measured entirely on auxiliary rows and mapped to mediation, as "
              "the ceiling v2 added so that an accuracy floor alone could not be "
              "satisfied by a direct adapter answering from the code",
        "v4": "unchanged, deliberately.  It is not a hygiene count, so scoping "
              "auxiliary rows out of hygiene does not touch it.  A repair that "
              "moved it would have traded a v2 defect for a v4 one, and the "
              "claim v4 makes is therefore narrower than 'auxiliary rows carry "
              "no verdict' and is the one that is true"}),
    ("a_cell_from_another_design_is_checked_before_it_is_consumed", {
        "v3": "load_cell_result compared a cell's kind with the aggregating "
              "spec's and nothing compared designs, because until now no version "
              "had a reason to read another's cells.  input_verification "
              "records the pre-registration a cell was produced under without "
              "comparing it to the one doing the aggregating",
        "v4": "verify_cell_design_lineage compares every cell's recorded "
              "preregistration_design_sha256 against this design's and against "
              "the ancestor design frozen into the manifest, refuses a cell "
              "matching neither, and reports how many cells came from which -- "
              "so a v4 report over v3 cells says so on its face instead of "
              "looking self-produced"}),
    ("a_post_outcome_amendment_says_it_is_one", {
        "v3": "had no such concept: every repair v3 made was declared before any "
              "cell existed, so nothing in the artifact distinguished a rule "
              "frozen in advance from one adjusted afterwards",
        "v4": "post_outcome_scope_amendment=true is a top-level field of the "
              "design, the scope_amendment block carries the derived evidence "
              "from BOTH datasets, and every report states that results "
              "aggregated under v4 from v3's cells are a corrective analysis "
              "rather than a pre-registered one.  Only a run executed under v4 "
              "from the start is pre-registered under the corrected scope"}),
))

#: What is true of a superseded v3 pilot, in the shape ``supersession_record``
#: reads.  Keyed on the kind it records rather than on the version replacing it.
SUPERSEDED_NOTES_V3 = {
    "design_is_still_reconstructible": (
        "verify_manifest dispatches on kind and rebuilds a v3 pilot through "
        "build_pilot_preregistration_v3, which is retained for exactly this "
        "reason, so the frozen design_sha256 still reproduces and every "
        "checkpoint digest still matches"),
    "input_digests_will_report_drift": (
        "a v3 manifest binds the bytes of this script as it was when that "
        "manifest was frozen, and this script has since gained v4, so "
        "verify_manifest names the script digest as drifted and returns "
        "valid=False.  That is the expected state of a superseded artifact and "
        "not a defect, and it is not re-frozen because a pre-registration "
        "changed after freezing was never a pre-registration"),
    "its_verdicts_stay_exactly_as_filed": (
        "the v3 reports are committed and are not regenerated, reworded or "
        "replaced.  v4 files its own reports beside them under v4 names; a "
        "corrective analysis that overwrote the analysis it corrects would "
        "leave no way to see that the two disagree"),
    "its_cells_are_consumed_by_v4_and_are_not_relabelled": (
        "v4 reuses v3's result kind, so the 28 filed cells are read as they "
        "stand: same bytes, same kind string, same run_provenance naming v3 and "
        "the commit that executed them.  verify_cell_design_lineage checks each "
        "one against the v3 design hash frozen into the v4 manifest and reports "
        "how many were reused, because a v4 report that did not say which design "
        "produced its rows would look self-produced"),
    "why_it_is_superseded": (
        "for a scope reason and not a scientific one.  v3's science stands: its "
        "router gate, its per-seed direct gates, its baseline cell and its live "
        "readiness all did what they were repaired to do, and the run that "
        "exercised them completed.  What it got wrong is one denominator -- it "
        "counted auxiliary rows toward a mediated verdict it had declared they "
        "were outside -- and that is corrected by declaring the scope, not by "
        "re-measuring anything"),
}

SUPERSESSION_POLICY_V4 = (
    "the v1, v2 and v3 manifests are preserved unmodified and their designs are "
    "still reconstructible; they are marked superseded here rather than edited, "
    "because a pre-registration changed after freezing was never a "
    "pre-registration.  v2 was superseded for a structural reason -- its design "
    "hash covered which adapters were on disk, so it could not survive the "
    "training it required.  v3 is superseded for a scope reason: it declared its "
    "auxiliary condition outside every mediated gate and then counted its rows "
    "toward one.  v4 is the FIRST version in this file whose amendment was made "
    "after a run's verdicts were read, and it says so at the top level of its "
    "own design rather than only in a commit message; the results it computes "
    "over v3's cells are corrective, and a pre-registered result under the "
    "corrected scope requires a run executed under v4 from the start")


def build_design_v4(dataset, forget_set_id, forget_ids, router_seeds,
                    edit_seeds, direct_seeds, man=None, images=None,
                    image_sha_by_uri=None):
    """The v4 design: v3's measurements, one declared denominator.

    Thin, like every other version's constructor: the one implementation is
    ``build_design_for``, and a second copy would be a second place for the
    cell counts, the row counts and the cluster checks to be missing from.
    """
    return build_design_for(PILOT_SPEC_V4, dataset, forget_set_id, forget_ids,
                            router_seeds, edit_seeds, direct_seeds, man=man,
                            images=images, image_sha_by_uri=image_sha_by_uri)


def build_pilot_preregistration_v4(dataset, forget_set_id, forget_ids,
                                   router_seeds=None, edit_seeds=None,
                                   direct_seeds=None, man=None, images=None,
                                   selection=None, superseded_paths=()):
    """The frozen v4 pilot.

    Same construction path as v2 and v3 with a different spec, so every check
    they ran runs here too.  It additionally carries the scope amendment's
    derived evidence and the ancestor design identity its cells are checked
    against, both of which the shared builder fills in from this spec's
    declarations.
    """
    return build_pilot_preregistration_for(
        PILOT_SPEC_V4, dataset, forget_set_id, forget_ids, router_seeds,
        edit_seeds, direct_seeds, man=man, images=images, selection=selection,
        superseded_paths=superseded_paths)


def prereg_path_v4(dataset, path=None):
    return prereg_path_for(PILOT_SPEC_V4, dataset, path)


def load_prereg_v4(dataset, path=None, verify=True):
    return load_prereg_for(PILOT_SPEC_V4, dataset, path, verify)


def _freeze_v4(args):
    return _freeze_for(PILOT_SPEC_V4, args)


def _run_v4_phase(args, phase):
    """The v4 dispatcher, kept so ``--design-version v4`` names a version the
    same way v2 and v3 do rather than only working through the default."""
    return _run_phase_for(PILOT_SPEC_V4, args, phase)


def _run_v4(args):
    return _run_pilot(args, PILOT_SPEC_V4)


def _v4_extra_freeze_inputs(dataset):
    """The files a v4 freeze binds beyond the ones every version binds.

    The two committed v3 RF2 reports, because the scope amendment's evidence is
    computed from them and a rationale that cites bytes nothing re-hashes is a
    rationale nobody can check later.  ``dataset`` is accepted for the shape of
    the table and unused: both reports are bound in both manifests, which is the
    point of recording both.
    """
    return [report_path(ds, "RF2", None, PILOT_SPEC_V3)
            for ds in ("ppubench", "salmu")]


# ---------------------------------------------------------------------------
# Rebind the version tables.  Everything above reads these at call time, so
# declaring v4 here is the same thing as declaring it where the tables are
# built, and the constructors that produced the frozen v1, v2 and v3 bytes stay
# textually untouched.
# ---------------------------------------------------------------------------

LATEST_PILOT_SPEC = PILOT_SPEC_V4
SPEC_BY_VERSION[PILOT_SPEC_V4.version] = PILOT_SPEC_V4
SPEC_BY_PREREG_KIND[PILOT_SPEC_V4.prereg_kind] = PILOT_SPEC_V4
PHASES_BY_VERSION[PILOT_SPEC_V4.version] = _phases_for(PILOT_SPEC_V4)
REPAIRS_BY_VERSION["v4"] = OrderedDict((*SUPERSESSION_ITEMS.items(),
                                        *SUPERSESSION_ITEMS_V3.items(),
                                        *SUPERSESSION_ITEMS_V4.items()))
SUPERSESSION_POLICY["v4"] = SUPERSESSION_POLICY_V4
SUPERSEDED_NOTES[PREREG_V3_KIND] = SUPERSEDED_NOTES_V3
SCOPE_AMENDMENT_EVIDENCE_BY_VERSION["v4"] = v4_scope_amendment_evidence
EXTRA_FREEZE_INPUTS_BY_VERSION["v4"] = _v4_extra_freeze_inputs


# ===========================================================================
# PART IV -- v5: the run the corrected scope actually requires
# ===========================================================================
#
# Everything v5 adds lives in this one contiguous block and works by REBINDING
# the version tables the code above reads, exactly as PART III does for v4.  The
# constructors above -- the ones that produced the frozen v1, v2, v3 and v4 bytes
# -- stay textually untouched, which is the only reason to believe those bytes
# still reproduce.
#
# WHY V5 EXISTS
#   v4 corrected a denominator and then re-read the 28 cells v3 had already
#   filed.  Every one of them is stamped ``reused_from_the_ancestor_design``,
#   and v4's own bytes say what that makes it: "a confirmatory result under the
#   corrected scope requires a run executed under v4 from the start."  v4 is that
#   correction and not that run.  v5 is the run.
#
# WHAT V5 CHANGES
#   1. It consumes NOTHING.  It declares no ancestor, so the lineage state
#      ``reused_from_the_ancestor_design`` is unreachable and a cell filed by any
#      earlier design is refused rather than aggregated.  Its result kind is its
#      own, which is the opposite of v4's choice for the opposite reason: v4
#      reused v3's kind so 28 filed cells would not be re-measured, and v5 must
#      not be able to read them at all.
#   2. It varies FIVE forget sets rather than one, and they are the five no
#      filed design has consumed, chosen by a rule rather than listed.
#   3. It calibrates the direct adapter's training schedule on a development
#      split carved out of the train images, and freezes the selected
#      configuration before any confirmatory cell exists.
#
# WHAT V5 DOES NOT CHANGE
#   No threshold.  No gate-to-verdict mapping.  No condition, no prompt, no row
#   builder, no hygiene denominator.  ``PILOT_SPEC_V5`` holds v4's tables by
#   OBJECT IDENTITY, not by copy, so the claim "the scope and the thresholds are
#   frozen" is a property a test can check with ``is`` rather than a promise.
#
#   The ROUTE is not calibrated and is not touched.  g and h come from
#   ``frozen_route_protocol()`` and the frozen adapters on disk; the calibration
#   is of D_s alone.  A phase named RFC does not imply a router was tuned, and
#   every artifact this block produces says so in words.
#
# WHAT V5 DISCLOSES
#   D_s trains on the train split MINUS the development images, while the frozen
#   g trained on all of it.  That is an asymmetry in the route-versus-direct
#   comparison and its direction favours the route: a direct pathway with less
#   training data is harder to push over the ``min_direct_image_accuracy`` floor,
#   so clearing the floor is more evidence than it would be at parity, not less.
#   The asymmetry is recorded in the design and in every D_s result rather than
#   removed by retraining g, which the frozen route does not permit.

#: v5's kinds.  The supersession kind names what it supersedes, following the
#: convention v3 and v4 set: four designs are behind this one.
KIND_V5 = "route_dependent_forgetting_design_v5"
PREREG_V5_KIND = "route_dependent_forgetting_preregistration_v5"
SUPERSESSION_KIND_V5 = (
    "route_dependent_forgetting_supersession_v1_v2_v3_and_v4")

#: v5's result kind is its OWN, and that is the mechanism rather than a label.
#: ``load_cell_result`` refuses a cell whose kind differs from the aggregating
#: spec's, so a distinct kind is what makes the 28 filed cells unreadable here --
#: a refusal at the door instead of a lineage check performed on cells that were
#: already let in.
RESULT_KIND_V5 = "route_dependent_forgetting_cell_result_v5"
RESULT_KIND_VERSION[RESULT_KIND_V5] = "v5"

#: What v5 repairs about v4, in the same shape the earlier tables use.
SUPERSESSION_ITEMS_V5 = OrderedDict((
    ("corrective_is_not_confirmatory", {
        "v4": ("scoped the hygiene gate to the non-auxiliary rows and then "
               "re-read the 28 cells v3 had filed under the corrected scope.  "
               "Every one of them is stamped reused_from_the_ancestor_design, so "
               "the verdicts v4 reports are a re-analysis of a run that was "
               "pre-registered under a different design"),
        "v5": ("executes under the corrected scope FROM THE START: it declares "
               "no ancestor, so a cell filed by v1-v4 is refused rather than "
               "aggregated, and every cell it reports was produced under the "
               "design that states the thresholds"),
        "also": ("v4's own bytes already say this is what a confirmatory result "
                 "requires; v5 is that statement acted on rather than restated")}),
    ("one_forget_set_is_not_a_factor", {
        "v4": ("varied router seed and edit seed over ONE forget set, so nothing "
               "in it established that the result was about route-dependent "
               "forgetting rather than about the one identity that set happened "
               "to suppress"),
        "v5": ("varies five forget sets as well, all five chosen by a rule from "
               "the sets no filed design consumed, so the crossed matrix is over "
               "targets the design did not pick after seeing an outcome")}),
    ("a_configuration_nobody_chose_in_advance", {
        "v4": ("trained D_s under the route's own schedule because that was the "
               "schedule available, not because anything had established it was "
               "a schedule the direct pathway could clear its floor under"),
        "v5": ("calibrates D_s's schedule on a development split disjoint from "
               "the held-out test images, freezes the selection, and only then "
               "trains the confirmatory adapters -- so the configuration is a "
               "pre-registered choice and not a free parameter")}),
))

SUPERSESSION_POLICY_V5 = (
    "the v1, v2, v3 and v4 manifests are preserved unmodified and their designs "
    "are still reconstructible; they are marked superseded here rather than "
    "edited, because a pre-registration changed after freezing was never a "
    "pre-registration.  v4 is superseded for a STATUS reason rather than a "
    "scientific one: its analysis of v3's cells is correct and stays committed, "
    "but it is corrective, and a corrective analysis of a completed run is not a "
    "confirmatory result under the scope it corrects.  v4's reports are therefore "
    "not relabelled, not withdrawn and not re-derived here")

#: What is true of a superseded v4 artifact.  Keyed on the KIND it records, like
#: every other entry, and required: ``supersession_record`` refuses to describe a
#: file whose kind has no notes rather than say nothing about it.
SUPERSEDED_NOTES_V4 = {
    "design_is_still_reconstructible": (
        "verify_manifest dispatches on kind and rebuilds a v4 pilot through "
        "build_pilot_preregistration_v4, which is retained for exactly this "
        "reason, so the frozen design_sha256 still reproduces"),
    "input_digests_will_report_drift": (
        "a v4 manifest binds the bytes of this script as it was when that "
        "manifest was frozen, and this script has since gained v5, so "
        "verify_manifest names the script digest as drifted and returns "
        "valid=False.  That is the documented state of a superseded artifact and "
        "not a defect: the DESIGN still reproduces from its own recorded "
        "parameters, which is the part that was pre-registered"),
    "its_reports_stay_committed_and_stay_corrective": (
        "rf_report_{ppubench,salmu}_v4.json re-read the 28 v3 cells under the "
        "corrected denominator.  They are not withdrawn by v5 and they are not "
        "confirmatory either: each carries cell_design_lineage naming both "
        "designs and how many cells came from each"),
}

#: v4's scope declaration, minus the three keys that are statements about v4's
#: HISTORY rather than about the scope.
#:
#: Every key that DEFINES the denominator is inherited with v4's own value, so
#: "the scope is frozen" is a fact about v5 and not a restatement.  What is
#: dropped is the claim that the denominator was chosen after the outcome it
#: corrects had been read: that is true of v4 and cannot be true of v5, which
#: freezes its denominator before it has any outcome at all.  Inheriting those
#: keys verbatim would make every v5 artifact announce itself as a post-hoc
#: amendment -- the one thing v5 exists to stop being.
_V5_KEYS_NOT_INHERITED_FROM_V4_SCOPE = (
    "post_outcome_scope_amendment",
    "what_that_means",
    "the_evidence_for_it_is_recorded_in_this_design")

DECLARED_HYGIENE_SCOPE_V5 = OrderedDict((
    *[(k, v) for k, v in DECLARED_HYGIENE_SCOPE_V4.items()
      if k not in _V5_KEYS_NOT_INHERITED_FROM_V4_SCOPE],
    ("scope_inherited_from", "v4"),
    ("every_key_that_defines_the_denominator_is_v4s_own_value",
     sorted(k for k in DECLARED_HYGIENE_SCOPE_V4
            if k not in _V5_KEYS_NOT_INHERITED_FROM_V4_SCOPE)),
    ("post_outcome_scope_amendment", False),
    ("why_the_amendment_fields_are_not_inherited",
     ("they state that v4's denominator was declared after v3's verdicts were "
      "read, and that the evidence for it is derived from v3's committed "
      "reports.  Both are true of v4 and neither is true of v5: v5 declares the "
      "same denominator BEFORE it has a verdict to correct, and derives nothing "
      "from v3's reports because it consumes no cell v3 filed")),
    ("what_v5_is_instead",
     ("the run v4's own bytes say a confirmatory result requires: executed "
      "under the corrected scope from the start, over forget sets no earlier "
      "design consumed, with its direct adapter's configuration selected on a "
      "development split and frozen before its cells were produced")),
))

#: v5 DOES declare a cell compatibility policy, and what it declares is that it
#: has no ancestor.  The policy is what makes the absence a statement rather than
#: an omission: ``verify_cell_design_lineage`` returns None for a version that
#: declared nothing, and a v5 report that carried no lineage block would be
#: indistinguishable from one that never checked.
DECLARED_CELL_COMPATIBILITY_V5 = OrderedDict((
    ("rule", "this_design_only"),
    ("ancestor", None),
    ("why_there_is_no_ancestor",
     ("v4 reused v3's result kind so that 28 filed cells would not be "
      "re-measured, and paid for it by resting a v4 report on v3-stamped cells. "
      "v5 exists to remove exactly that: a confirmatory result has to be produced "
      "by the design that states its thresholds.  So v5 declares no ancestor, "
      "takes a result kind of its own, and a cell filed under any earlier design "
      "is refused at load rather than checked after the fact")),
    ("what_is_checked_before_a_cell_is_consumed",
     ("load_cell_result refuses any cell whose kind is not this design's, which "
      "is every cell v1-v4 filed; verify_cell_design_lineage then compares each "
      "consumed cell's recorded preregistration_design_sha256 against this "
      "design's and refuses a cell that does not match it")),
    ("states", ("produced_under_this_design", "unrecognized_design")),
    ("reused_from_the_ancestor_design_is_unreachable",
     ("the state exists in the earlier declarations and cannot occur here, "
      "because there is no ancestor design hash for a cell to match.  A count of "
      "zero reused cells is therefore not a measurement v5 happened to make; it "
      "is a consequence of what v5 declared")),
    ("what_a_v5_report_must_disclose",
     ("that every cell it aggregates was produced under its own design, and that "
      "the v3 cells v4 analysed take no part in any v5 verdict")),
))

PILOT_SPEC_V5 = PilotSpec(
    version="v5", kind=KIND_V5, prereg_kind=PREREG_V5_KIND,
    result_kind=RESULT_KIND_V5, supersession_kind=SUPERSESSION_KIND_V5,
    conditions=CONDITIONS_V4,
    forced_code_conditions=FORCED_CODE_CONDITIONS_V4,
    mediated_conditions=MEDIATED_CONDITIONS_V4,
    auxiliary_conditions=AUXILIARY_CONDITIONS_V4,
    decisive_conditions=DECISIVE_CONDITIONS_V4,
    thresholds=GATE_THRESHOLDS_V4, gate_to_verdict=GATE_TO_VERDICT_V4,
    baseline_cell=True, embed_live_checkpoint_status=False,
    direct_gate_aggregation="every_seed",
    declared_gate_aggregation={
        "direct_gate_aggregation": PER_SEED_GATE_AGGREGATION},
    hygiene_gate_row_scope=HYGIENE_SCOPE_NON_AUXILIARY,
    declared_hygiene_scope=DECLARED_HYGIENE_SCOPE_V5,
    declared_cell_compatibility=DECLARED_CELL_COMPATIBILITY_V5,
    superseded_filenames=("rf_pilot_ppubench.json", "rf_pilot_salmu.json",
                          "rf_pilot_ppubench_v2.json",
                          "rf_pilot_salmu_v2.json",
                          "rf_pilot_ppubench_v3.json",
                          "rf_pilot_salmu_v3.json",
                          "rf_pilot_ppubench_v4.json",
                          "rf_pilot_salmu_v4.json"),
    calibration_phase=True)


# ---------------------------------------------------------------------------
# the development split
# ---------------------------------------------------------------------------

#: How many of each identity's TRAIN images are held out for development, taken
#: as the highest-indexed ones.  The test images are the manifest's own
#: ``split == "test"`` items and are not named here at all: the split exists to
#: divide the TRAIN images, and a rule that also reached the test images would be
#: a rule about the measurement rather than about the calibration.
#:
#: A COUNT and not a list of indices, because the two datasets do not share an
#: index range.  SALMU has j=0..7 in train and j=8..10 in test for every one of
#: its twelve identities.  PPUBench's indices are sparse, run past forty, and
#: overlap between its own splits -- j=32 is a train image of one identity and a
#: test image of another -- so a rule that named indices would fit SALMU and place
#: some of PPUBench's train images in neither set, leaving a caller to decide
#: where they went.  On SALMU this rule returns j=6 and j=7 for every identity,
#: which is the split the design was sized against: the count is the rule and the
#: indices are what the rule returns on this data.
N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5 = 2


def image_index_within_identity(item):
    """The position one image occupied in its identity's sorted list.

    Read out of the recorded filename rather than recomputed by re-sorting the
    manifest: the generator wrote ``SAL_{iid}_{j:02d}.png`` and derived the split
    from that same j, so the filename IS the record of the position.  Re-deriving
    it by sorting would agree today and be a second rule tomorrow, and the two
    would differ the moment an image was added.
    """
    tail = Path(item["image_uri"]).stem.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise RuntimeError(
            f"cannot read an image index out of {item['image_uri']!r}: the last "
            f"filename component is {tail!r} and not a number.  The development "
            f"split is defined on that index, so a manifest whose filenames do "
            f"not carry it has no development split rather than a guessed one")
    return int(tail)


def dev_split(man, audit=None):
    """The train images, divided into what D_s fits on and what selects its config.

    Derived from the manifest alone and checked against image BYTES rather than
    filenames.  The distinction is not precautionary here: PPUBench's held-out
    filenames are bytes that also appear in its train split, so a split that
    partitioned filenames would partition nothing.  SALMU is asserted to be
    different rather than assumed to be, because the whole purpose of holding a
    development set out is that the selection never sees the test images.
    """
    audit = audit if audit is not None else image_content_audit(man)
    sha_of = audit["images_by_uri_sha256"]
    fit, dev, test = [], [], []
    train = {}
    n_dev = N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5
    for it in man["items"]:
        j = image_index_within_identity(it)
        if it["split"] == "test":
            test.append((it, j))
        else:
            train.setdefault(it["identity_id"], []).append((it, j))
    for iid in sorted(train):
        xs = sorted(train[iid], key=lambda tj: tj[1])
        seen = [j for _, j in xs]
        if len(set(seen)) != len(seen):
            dup = sorted({j for j in seen if seen.count(j) > 1})
            raise RuntimeError(
                f"identity {iid} has more than one train image at image index "
                f"{dup}, so 'the {n_dev} highest-indexed' names no particular "
                f"images.  The rule refuses rather than breaking the tie by list "
                f"order, which would make the split depend on the order the "
                f"manifest's rows happen to be in rather than on the images")
        if len(xs) <= n_dev:
            raise RuntimeError(
                f"identity {iid} has {len(xs)} train image(s) and the rule holds "
                f"out {n_dev} for development, which would leave it nothing to "
                f"fit on.  A development split that consumed an identity's whole "
                f"training set would select a configuration on images that "
                f"identity was never trained on, and would select it differently "
                f"from every identity that had images left")
        dev.extend(xs[-n_dev:])
        fit.extend(xs[:-n_dev])

    def uris(xs):
        return [it["image_uri"] for it, _ in xs]

    def shas(xs):
        return [sha_of[it["image_uri"]] for it, _ in xs]

    fit_sha, dev_sha, test_sha = shas(fit), shas(dev), shas(test)
    overlaps = OrderedDict()
    for a, b, na, nb in ((dev_sha, test_sha, "development", "test"),
                         (fit_sha, test_sha, "fit", "test"),
                         (fit_sha, dev_sha, "fit", "development")):
        shared = sorted(set(a) & set(b))
        overlaps[f"{na}_and_{nb}"] = {
            "n_shared_image_bytes": len(shared), "shared": shared}
        if shared:
            raise RuntimeError(
                f"the {na} and {nb} images share {len(shared)} image "
                f"byte-value(s), e.g. {shared[0]}; a development split that "
                f"overlaps the test images in CONTENT is not disjoint from them, "
                f"and a configuration selected on it would be selected partly on "
                f"the measurement it is supposed to be blind to")
    per_identity = OrderedDict()
    for it, _ in dev:
        per_identity.setdefault(it["identity_id"], []).append(it["image_uri"])
    counts = sorted({len(v) for v in per_identity.values()})
    return OrderedDict((
        ("rule", OrderedDict((
            ("fit", ("split == 'train', all but the highest-indexed "
                     f"{N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5} of each identity")),
            ("development", ("split == 'train', the highest-indexed "
                             f"{N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5} of each "
                             "identity")),
            ("test", ("split == 'test', taken from the manifest and not named by "
                      "this rule at all")),
            ("index_read_from",
             ("the recorded image filename, which is where the generator wrote "
              "the position it derived the split from")),
            ("why_a_count_and_not_a_list_of_indices",
             ("SALMU's train indices are 0..7 for every identity, so a list of "
              "indices works there and reads like a rule.  PPUBench's are sparse, "
              "run past forty and overlap its own test indices, so the same list "
              "would leave some of its train images in neither set.  A count "
              "holds on both, and on SALMU it returns j=6 and j=7")),
        ))),
        ("n_fit_images", len(fit)),
        ("n_development_images", len(dev)),
        ("n_test_images", len(test)),
        ("n_train_images", len(fit) + len(dev)),
        ("fit_image_uris", uris(fit)),
        ("development_image_uris", uris(dev)),
        ("test_image_uris", uris(test)),
        ("fit_image_sha256", fit_sha),
        ("development_image_sha256", dev_sha),
        ("test_image_sha256", test_sha),
        ("disjoint_in_image_content_not_only_in_filename", overlaps),
        ("development_images_per_identity", OrderedDict(
            (k, len(v)) for k, v in sorted(per_identity.items()))),
        ("development_is_balanced_across_identities",
         len(counts) == 1
         and counts[0] == N_DEVELOPMENT_IMAGES_PER_IDENTITY_V5),
        ("every_train_image_is_placed",
         len(fit) + len(dev) == sum(1 for i in man["items"]
                                    if i["split"] == "train")),
    ))


# ---------------------------------------------------------------------------
# the calibration grid and the rule that selects from it
# ---------------------------------------------------------------------------

#: The two seeds each candidate is trained under.  Two rather than one because a
#: candidate that wins on a single LoRA init has not been shown to win.
CALIBRATION_SEEDS_V5 = (17, 42)

#: The subdirectory of a dataset's output tree the calibration writes into.
#: Beside the cells rather than among them, because no design consumes a
#: calibration file: it is evidence about a choice, not an input to a verdict, and
#: a reader who finds it inside the cell table would reasonably assume an
#: aggregate had read it.
CALIBRATION_SUBDIR_V5 = "calibration"

#: The kind Freeze #1 files under.  A constant rather than a literal inside the
#: constructor, because ``verify_manifest`` has to dispatch on the same string:
#: a kind typed in two places is a verifier that cannot rebuild the artifact it
#: is verifying, and the failure lands at the end of the freeze that wrote it.
CALIBRATION_PREREG_KIND_V5 = (
    "route_dependent_forgetting_calibration_preregistration_v5")


def _incumbent_schedule():
    """The schedule the frozen route protocol itself specifies."""
    p = frozen_route_protocol()
    return OrderedDict((k, p[k]) for k in OVERRIDABLE_SCHEDULE_KEYS)


def direct_schedule_candidates_v5():
    """The configurations RFC trains, as overrides of the frozen schedule.

    The incumbent is candidate C0 and is READ from ``frozen_route_protocol()``
    rather than retyped beside it.  A table that retyped 3000/200/2e-5/50 could
    drift from the protocol it claims to reproduce and still call itself the
    incumbent, and then "no change" would be a fifth configuration nobody named.

    Only the SCHEDULE varies.  The LoRA shape cannot be overridden from here at
    all -- see ``OVERRIDABLE_SCHEDULE_KEYS`` -- so D_s's adapter stays the same
    class of adapter as the route's and the comparison between them is about the
    pathway rather than about the recipe.
    """
    base = _incumbent_schedule()
    return OrderedDict((
        ("C0_incumbent", OrderedDict(base)),
        ("C1_half_lr", OrderedDict({**base, "lr": base["lr"] / 2})),
        ("C2_half_as_many_steps_again",
         OrderedDict({**base, "steps": int(base["steps"] * 3 // 2)})),
        ("C3_double_lr", OrderedDict({**base, "lr": base["lr"] * 2})),
    ))


def calibration_floor_v5():
    """The dev accuracy a candidate must reach to be selectable at all.

    Read from the confirmatory gate's own threshold rather than set beside it:
    the floor is that a configuration which cannot clear ``direct_image_accuracy``
    on the development images cannot be expected to clear it on the held-out ones,
    and a threshold typed twice is a threshold that can be changed in one place.
    """
    return PILOT_SPEC_V5.thresholds["min_direct_image_accuracy"]


CALIBRATION_SELECTION_RULE_V5 = OrderedDict((
    ("metric", ("accuracy on the DEVELOPMENT images only, image-to-alias, which "
                "is the same statistic the confirmatory gate "
                "min_direct_image_accuracy uses, measured on images the "
                "confirmatory gates never read")),
    ("scored_over", "the mean of the calibration seeds"),
    ("winner", "the highest mean"),
    ("tie_breaks", (
        ("the smaller maximum absolute difference between its own seeds, because a "
        "configuration that is stable across inits is the one a single "
        "confirmatory seed is more likely to reproduce"),
        ("then the earlier candidate in the frozen table order, which is a fixed "
        "order and not a preference"),
        "then the incumbent, so that a tie is resolved toward changing nothing")),
    ("floor", ("a candidate is selectable only at or above the confirmatory "
               "gate's own threshold on the development images; if NO candidate "
               "reaches it the calibration stops and reports instead of "
               "selecting the best of a lot that cannot pass")),
    ("why_the_floor_refuses_rather_than_degrades",
     ("a selection rule that always returns a winner can select a configuration "
      "the pilot then fails on, and the failure would read as a result about "
      "route-dependent forgetting rather than as a result about the direct "
      "adapter being unable to learn the associations at all")),
    ("frozen_before_any_candidate_is_trained", True),
))


def select_direct_schedule(results):
    """Apply the frozen rule to the calibration's own measurements.

    ``results`` maps candidate name to seed to development accuracy.  The rule is
    applied to whatever it is given and returns the table it reasoned over, so a
    reader can see the winner was the winner under the stated rule and not under
    a rule written after the numbers were known.
    """
    candidates = direct_schedule_candidates_v5()
    unknown = sorted(set(results) - set(candidates))
    if unknown:
        raise RuntimeError(
            f"the calibration produced results for {unknown}, which the frozen "
            f"grid does not contain {list(candidates)}; selecting a "
            f"configuration nobody pre-registered is the thing the grid exists "
            f"to prevent")
    missing = sorted(set(candidates) - set(results))
    if missing:
        raise RuntimeError(
            f"the calibration is missing {missing}; the grid was frozen with "
            f"{len(candidates)} candidates and a selection over part of it is a "
            f"selection over a different grid")
    floor = calibration_floor_v5()
    table = OrderedDict()
    for name, per_seed in results.items():
        seeds = sorted(per_seed)
        if seeds != sorted(CALIBRATION_SEEDS_V5):
            raise RuntimeError(
                f"candidate {name} was measured under seeds {seeds}, expected "
                f"{list(CALIBRATION_SEEDS_V5)}")
        accs = [per_seed[s] for s in seeds]
        table[name] = OrderedDict((
            ("schedule", candidates[name]),
            ("per_seed_development_accuracy",
             OrderedDict((str(s), per_seed[s]) for s in seeds)),
            ("mean_development_accuracy", sum(accs) / len(accs)),
            ("max_seed_spread", max(accs) - min(accs)),
            ("clears_the_floor", (sum(accs) / len(accs)) >= floor),
        ))
    passing = [n for n, e in table.items() if e["clears_the_floor"]]
    out = OrderedDict((
        ("rule", CALIBRATION_SELECTION_RULE_V5),
        ("floor", floor),
        ("floor_read_from", "PILOT_SPEC_V5.thresholds['min_direct_image_accuracy']"),
        ("n_candidates", len(table)),
        ("candidates", table),
        ("n_clearing_the_floor", len(passing)),
        ("selected", None),
        ("selection_refused", None),
    ))
    if not passing:
        out["selection_refused"] = OrderedDict((
            ("reason", ("no candidate reached a mean development accuracy of "
                        f"{floor}; the best was "
                        f"{max(e['mean_development_accuracy'] for e in table.values()):.4f}")),
            ("what_this_means",
             ("the direct X -> Y pathway could not learn the training "
              "associations well enough on held-out-by-calibration images to "
              "clear the confirmatory gate, so there is no configuration to "
              "confirm.  That is a finding about the direct pathway and it is "
              "reported instead of being optimized away by picking a winner")),
            ("what_was_not_done",
             ("the thresholds were not lowered, the floor was not waived, the "
              "development images were not folded into the fit set and the test "
              "images were not consulted")),
        ))
        return out
    order = list(candidates)
    ranked = sorted(
        passing,
        key=lambda n: (-table[n]["mean_development_accuracy"],
                       table[n]["max_seed_spread"],
                       order.index(n)))
    winner = ranked[0]
    tied = [n for n in passing
            if table[n]["mean_development_accuracy"]
            == table[winner]["mean_development_accuracy"]]
    out["selected"] = OrderedDict((
        ("candidate", winner),
        ("schedule", table[winner]["schedule"]),
        ("mean_development_accuracy",
         table[winner]["mean_development_accuracy"]),
        ("max_seed_spread", table[winner]["max_seed_spread"]),
        ("n_tied_on_the_metric", len(tied)),
        ("tied_candidates", tied),
        ("tie_broken_by", (None if len(tied) == 1 else
                           "seed spread, then frozen table order, then the "
                           "incumbent")),
        ("is_the_incumbent", winner == "C0_incumbent"),
        ("ranking", ranked),
    ))
    return out


# ---------------------------------------------------------------------------
# which forget sets v5 covers
# ---------------------------------------------------------------------------

#: The design versions in the order they were frozen.  The rule below needs an
#: order and not a membership test, because "no filed design has consumed this
#: set" has to mean no EARLIER design: a rule that also excluded the sets its own
#: version consumed would return nothing the moment its own manifest was filed,
#: and the manifest could then never be rebuilt from its own parameters.
DESIGN_VERSION_ORDER = ("v1", "v2", "v3", "v4", "v5")


def consumed_forget_sets(dataset, before_version):
    """The forget sets every EARLIER design's committed manifest names.

    Read out of the filed manifests rather than written down, so the answer moves
    when the tree moves and cannot go stale beside it.
    """
    if before_version not in DESIGN_VERSION_ORDER:
        raise RuntimeError(
            f"{before_version!r} is not a design version; the order this rule "
            f"reads is {list(DESIGN_VERSION_ORDER)}")
    cut = DESIGN_VERSION_ORDER.index(before_version)
    earlier = DESIGN_VERSION_ORDER[:cut]
    consumed, per_manifest = set(), OrderedDict()
    for path in sorted((DATASET_ROOT / MANIFEST_DIR).glob(
            f"rf_pilot_{dataset}*.json")):
        frozen = json.loads(path.read_text(encoding="utf-8"))
        version = _version_of(frozen.get("kind"))
        if version is None:
            raise RuntimeError(
                f"{_rel(path)} records kind {frozen.get('kind')!r}, which names "
                f"no design version; a rule that excludes the sets earlier "
                f"designs consumed cannot skip a design it cannot place")
        if version not in earlier:
            continue
        named = ([frozen["forget_set_id"]] if frozen.get("forget_set_id")
                 else []) + [s["set_id"] for s in (frozen.get("forget_sets")
                                                   or ())]
        per_manifest[path.name] = {"version": version, "names": named}
        consumed.update(named)
    return {"consumed": sorted(consumed), "per_manifest": per_manifest,
            "versions_considered": list(earlier)}


def _version_of(kind):
    """The design version a manifest kind belongs to, read off the spec table."""
    for spec in SPEC_BY_VERSION.values():
        if kind in (spec.kind, spec.prereg_kind):
            return spec.version
    return "v1" if kind in (KIND, PREREG_KIND) else None


def pilot_forget_sets_v5(dataset, edit_seeds):
    """Which forget sets the v5 pilot covers, chosen by a stated rule.

    The rule is: every SINGLE-target set the frozen matrix records, whose
    checkpoints and cell results are complete for every required edit seed, and
    which no EARLIER design's committed manifest names.

    Three clauses and each one is doing work.  Single-target, because a
    simultaneous set adds a second forget target whose interactions are a
    different question -- the reason ``pilot_forget_set`` gives.  Complete on
    disk, because v5 trains no edited h and a set missing one could not run
    without training something this design does not name.  And unconsumed,
    because re-running a forget set an earlier design already reported on is not
    a new cell however many seeds are crossed over it -- which is the defect v5
    exists to remove.

    Derived and not listed: the ids below are what the rule returns on the
    committed tree, and a test asserts that, so a set added to the matrix or a
    manifest added to the tree moves the answer visibly.
    """
    mx = _matrix_manifest(dataset)
    declared = list(mx.get("edit_seeds") or [])
    if declared and list(edit_seeds) != declared:
        raise RuntimeError(
            f"edit seeds {list(edit_seeds)} are not the ones the frozen "
            f"{dataset} matrix trained ({declared}); v5 reuses the matrix's "
            f"edited-h adapters and cannot reuse adapters trained under "
            f"different seeds")
    consumed = consumed_forget_sets(dataset, "v5")
    cells = DATASET_ROOT / MATRIX_CELLS[dataset]
    candidates, chosen = [], []
    for e in mx["sets"]:
        if e.get("mode") != "single":
            candidates.append({"set_id": e["set_id"], "targets": e["targets"],
                               "considered": False,
                               "why_not": (f"mode is {e.get('mode')!r}, and the "
                                           f"pilot is single-target")})
            continue
        if e["set_id"] in consumed["consumed"]:
            candidates.append({
                "set_id": e["set_id"], "targets": list(e["targets"]),
                "considered": False,
                "why_not": ("a design frozen before v5 already names it, so its "
                            "cells are not new however many seeds are crossed "
                            "over them")})
            continue
        per_seed = {}
        for s in edit_seeds:
            d = cells / e["set_id"] / f"seed_{s}"
            per_seed[str(s)] = {
                "adapter": (d / "edited_h" / Path(*ADAPTER_RELPATH)).is_file(),
                "cell_results": (d / "cell_results.json").is_file()}
        complete = all(v["adapter"] and v["cell_results"]
                       for v in per_seed.values())
        entry = {"set_id": e["set_id"], "targets": list(e["targets"]),
                 "considered": True, "complete": complete,
                 "per_edit_seed": per_seed,
                 "why_not": (None if complete else
                             "at least one required edit seed has no adapter or "
                             "no cell_results, so v5 could not run it without "
                             "training an h, which it does not")}
        candidates.append(entry)
        if complete:
            chosen.append({"set_id": e["set_id"], "targets": list(e["targets"])})
    if not chosen:
        raise RuntimeError(
            f"no single-target {dataset} forget set is both unconsumed by an "
            f"earlier design and complete for edit seeds {list(edit_seeds)}; v5 "
            f"cannot be frozen over sets that do not exist, and it trains no "
            f"edited h to make one exist")
    return {
        "dataset": dataset,
        "sets": chosen,
        "n_sets": len(chosen),
        "targets": sorted(i for s in chosen for i in s["targets"]),
        "selection_rule": (
            "every single-target set in the frozen matrix whose edited-h "
            "adapter and cell results are complete for every required edit seed "
            "and which no earlier design's committed manifest names"),
        "rule_is_derived_not_preferred": (
            "the rule is stated here and applied to the matrix and the manifests "
            "as they exist, so the sets cannot be chosen after seeing a result. "
            "The unconsumed clause is what makes the cells new: a set an earlier "
            "design already reported on yields cells that are re-runs however "
            "many seeds are crossed over them"),
        "earlier_designs_consulted": consumed,
        "n_candidates": len(candidates),
        "n_considered": sum(1 for c in candidates if c["considered"]),
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# the v5 checkpoint roles and the v5 design
# ---------------------------------------------------------------------------

#: Which cell kinds a forget set changes the CONTENT of.  Derived from the row
#: builders' own signatures and not from a guess: ``build_baseline_rows_v3``,
#: ``build_intervention_rows_v2`` and ``build_natural_rows_v2`` all take
#: ``forget_ids`` and split their rows on it, while ``build_direct_rows_v2`` does
#: not take it at all and sets ``image_is_forgotten`` False on every row.
#:
#: That last fact is why there are three direct cells and not fifteen.  Running
#: D_s once per forget set would make the identical call five times and file it as
#: five observations, which is the defect ``build_design_for``'s docstring names
#: -- and ``build_design_v5`` does not merely assert the rows are the same across
#: sets, it compares them and refuses if they are not.
FORGET_SET_DEPENDENT_KINDS = ("baseline", "intervention", "natural")


def required_checkpoints_v5(dataset, forget_sets, router_seeds, edit_seeds,
                            direct_seeds, direct_namespace):
    """Every checkpoint and prediction file a v5 pilot needs, by role.

    ``required_checkpoints_v2`` with two differences, both forced by what v5
    varies:

      edited_h__{set}__seed{E}   one per (FORGET SET, edit seed).  Five sets
                                 times three seeds is fifteen roles, and a role
                                 name without the set would resolve five
                                 different adapters to one declaration.
      direct_d__seed{S}          under a namespace of its own, because
                                 ``direct_seed_17/`` holds weights whose digest a
                                 committed v3 cell result records.  Overwriting
                                 them would leave a filed report describing an
                                 adapter nobody can inspect.

    The router and baseline roles are unchanged: g and h are the frozen route and
    v5 trains neither.
    """
    cells = MATRIX_CELLS[dataset]
    route = ROUTE_DIRS[dataset]
    req = OrderedDict()
    for seed in router_seeds:
        existing = seed == EXISTING_ROUTER_SEED
        req[f"router_g__seed{seed}"] = {
            "role": ("the frozen router, read not retrained" if existing else
                     "a router trained under the identical frozen protocol with "
                     "a fresh LoRA init at this seed"),
            "adapter": _rel(router_checkpoint_path(dataset, seed)),
            "held_out_predictions": _rel(router_prediction_path(dataset, seed)),
            "exists_already": existing}
    req["baseline_h"] = {
        "role": "the unedited code->label map; the before in before/after",
        "adapter": _rel(DATASET_ROOT / route / "h_C_to_Y"
                        / Path(*ADAPTER_RELPATH)),
        "exists_already": True}
    for fs in forget_sets:
        for seed in edit_seeds:
            req[EDITED_H_ROLE_MULTI.format(forget_set=fs["set_id"],
                                           edit_seed=seed)] = {
                "role": ("the post-edit route for THIS forget set, reused from "
                         "the frozen matrix; v5 trains no edited h"),
                "adapter": _rel(DATASET_ROOT / cells / fs["set_id"]
                                / f"seed_{seed}" / "edited_h"
                                / Path(*ADAPTER_RELPATH)),
                "cell_results": _rel(DATASET_ROOT / cells / fs["set_id"]
                                     / f"seed_{seed}" / "cell_results.json"),
                "exists_already": True}
    for seed in direct_seeds:
        req[f"direct_d__seed{seed}"] = {
            "role": ("a separately trained X -> Y adapter, on the train split "
                     "minus the development images, under the schedule the "
                     "calibration selected; never edited when an h is edited"),
            "adapter": _rel(direct_checkpoint_path(dataset, seed,
                                                   direct_namespace)),
            "exists_already": False}
    # Which roles this design PRODUCES, stated rather than inferred from whether
    # the file happens to be there.  ``exists_already`` answers a question about a
    # disk; this answers a question about the design, and the freeze-ordering
    # listing needs the second one.  Inferring it from presence is what made a
    # router adapter the v3 run trained look like an output v5 had not produced
    # yet, which would have refused every v5 freeze on a tree the route had
    # already been run on.
    for label in req:
        req[label]["produced_by_this_design"] = label.startswith("direct_d__")
    for label, spec in req.items():
        spec["n_files_required"] = sum(1 for k in CHECKPOINT_FILE_KEYS_V2
                                       if spec.get(k))
        if not spec["n_files_required"]:
            raise RuntimeError(
                f"checkpoint role {label!r} declares no file in "
                f"{list(CHECKPOINT_FILE_KEYS_V2)}, so there is nothing to hash "
                f"and nothing that could be reported missing")
    return {
        "dataset": dataset,
        "forget_set_id": None,
        "forget_sets": [{"set_id": s["set_id"], "targets": list(s["targets"])}
                        for s in forget_sets],
        "roles": req,
        "router_seeds": list(router_seeds),
        "edit_seeds": list(edit_seeds),
        "direct_seeds": list(direct_seeds),
        "direct_namespace": direct_namespace,
        "why_the_direct_adapters_are_namespaced": (
            "the unnamespaced directories hold the adapters the v3 run trained "
            "and the v4 analysis read; their digests are recorded in committed "
            "cell results, so writing a differently trained adapter over them "
            "would leave those results describing weights that no longer exist"),
        "what_this_design_produces": (
            "the D_s adapters and their cells, and nothing else.  v5 trains no "
            "router and no edited h: g at every seed it varies is read from the "
            "frozen route's own run and the fifteen edited-h adapters are read "
            "from the frozen matrix.  A role v5 does not produce is an input it "
            "was built over, so the freeze-ordering listing does not count it -- "
            "and ``exists_already`` is left to say separately whether that input "
            "is on this disk, which is a live question ``checkpoint_readiness`` "
            "answers rather than one the design should assert"),
        "edited_h_roles_are_qualified_by_forget_set": (
            "one adapter per (forget set, edit seed); the same edited h cannot "
            "serve two forget sets because it was trained to suppress one"),
        "no_direct_condition_reuses_edited_h": (
            "v1's direct condition reused edited_h and so measured a "
            "code-trained adapter on an image; v5 requires an independent D_s "
            "with its own digest, as v2 and every version since has"),
        "router_seed_is_a_real_factor": (
            "one checkpoint and one held-out prediction file per router seed; "
            "forced-code interventions are one cell per (forget set, EDIT seed) "
            "because they do not depend on g, so they are not repeated across "
            "router seeds and counted as additional evidence"),
        "paths_are_recorded_relative_not_absolute": (
            "so that verifying this table is not tied to one filesystem root"),
    }


def _rename_cell(cell, new_id):
    """One cell under a different id, with every row that belongs to it.

    The id, the ``cell_id`` each row carries and the ``row_id`` prefix move
    together, because ``aggregate_cells_for`` selects a cell's rows by
    ``r["cell_id"] == cell_id``.  Renaming the cell and leaving its rows would be
    worse than renaming nothing: the rows would match no cell, the gates would be
    computed over an empty list, and this module refuses that loudest only where a
    kind ends up with no rows at all.
    """
    old = cell["cell_id"]
    if new_id == old:
        return cell
    rows = []
    for r in cell["rows"]:
        if r["cell_id"] != old:
            raise RuntimeError(
                f"row {r.get('row_id')!r} of cell {old!r} names cell "
                f"{r['cell_id']!r}; a row that does not belong to the cell "
                f"containing it cannot be renamed with it")
        if not r["row_id"].startswith(old):
            raise RuntimeError(
                f"row id {r['row_id']!r} does not begin with its cell id "
                f"{old!r}, so a new id cannot be substituted into it")
        rows.append({**r, "cell_id": new_id,
                     "row_id": new_id + r["row_id"][len(old):]})
    return {**cell, "cell_id": new_id, "rows": rows}


def _qualify_cell(cell, set_id, forget_ids):
    """One cell of a single-forget-set design, made a cell of a specific set.

    The renaming is ``_rename_cell``'s job, and so is its rationale.  What is
    specific to a forget set is that the cell also records WHICH set it measures
    and which identities that set targets, because the statistics read both off
    the cell rather than off the design -- a design-level list of five sets would
    tell an aggregate nothing about which one a given cell suppressed.
    """
    new = qualified_cell_id(cell["kind"], set_id,
                            cell["cell_id"].split("__", 1)[1])
    return {**_rename_cell(cell, new), "forget_set_id": set_id,
            "forget_identity_ids": list(forget_ids)}


#: The namespace v5's confirmatory direct adapters AND their cells are written
#: under.  Both or neither: an adapter at ``direct_seed_17__v5`` whose cell lands
#: at ``direct__d17`` is a v5 measurement filed at a v3 path, and the v3 report
#: that reads ``direct__d17`` would then be describing weights nobody can
#: reproduce.
DIRECT_NAMESPACE_V5 = "v5"


def direct_cell_id(kind, direct_seed, namespace=None):
    """The id of a direct or hybrid cell, namespaced where its adapter is.

    One function and two callers -- the shared design constructor and RF1D -- so
    the id a design names and the id a phase writes cannot drift apart.  With no
    namespace it returns exactly the id every design frozen before v5 has always
    used, which is what makes namespacing v5's cells a change to v5 rather than a
    change to what v1-v4 filed.
    """
    tail = f"d{direct_seed}"
    return f"{kind}__{namespace}__{tail}" if namespace else f"{kind}__{tail}"



def _require_calibration_selection(dataset):
    """The filed calibration selection, or a refusal naming why there is none.

    One function and two callers, because the design constructor and the freeze
    ask the same question and a message written twice is a message that can
    disagree with itself about what is missing.
    """
    p = calibration_selection_path(dataset)
    if not p.is_file():
        raise RuntimeError(
            f"no calibration selection is filed at {_rel(p)}.  The confirmatory "
            f"design embeds the configuration RFC selected, so it cannot be "
            f"frozen before RFC has selected one -- and that order is the point: "
            f"a configuration chosen after the cells exist was chosen with them "
            f"in view")
    doc = json.loads(p.read_text(encoding="utf-8"))
    sel = doc.get("selected")
    if not sel:
        raise RuntimeError(
            f"the calibration filed at {_rel(p)} REFUSED to select: "
            f"{(doc.get('selection_refused') or {}).get('reason')}.  There is no "
            f"configuration to confirm, so there is no v5 design to freeze; the "
            f"finding is that the direct pathway could not clear the floor on "
            f"the development images")
    return doc, sel


def selected_direct_training(dataset, split):
    """The D_s configuration the calibration selected, read out of its filing.

    READ and never passed in.  A design that was handed a configuration could be
    frozen with one that nothing selected, and no downstream check could tell the
    difference -- the artifact would name a schedule and a reader would have no
    way to know a rule had ever been applied to it.  Reading the filed selection
    means the confirmatory design embeds the outcome of the rule that was frozen
    before the candidates were trained, or it does not exist at all.
    """
    p = calibration_selection_path(dataset)
    doc, sel = _require_calibration_selection(dataset)
    return OrderedDict((
        ("selected_candidate", sel["candidate"]),
        ("schedule", OrderedDict(sel["schedule"])),
        ("selected_by", OrderedDict((
            ("rule", doc["rule"]),
            ("floor", doc["floor"]),
            ("mean_development_accuracy", sel["mean_development_accuracy"]),
            ("max_seed_spread", sel["max_seed_spread"]),
            ("is_the_incumbent", sel["is_the_incumbent"]),
            ("n_tied_on_the_metric", sel["n_tied_on_the_metric"]),
            ("selection_artifact", _rel(p)),
            ("selection_artifact_sha256", sha256_file(p)),
        ))),
        ("development_image_uris", list(split["development_image_uris"])),
        ("development_image_sha256", list(split["development_image_sha256"])),
        ("n_fit_images", split["n_fit_images"]),
        ("n_development_images", split["n_development_images"]),
        ("namespace", DIRECT_NAMESPACE_V5),
        ("training_data_asymmetry", OrderedDict((
            ("statement",
             (f"D_s trains on {split['n_fit_images']} images; the frozen g "
             f"trained on all "
             f"{split['n_fit_images'] + split['n_development_images']}")),
            ("direction",
             ("it handicaps the direct pathway, which is the conservative "
             "direction for the mediation claim: the min_direct_image_accuracy "
             "floor is harder to clear with less training data, so clearing it "
             "is more evidence than it would be at parity and not less")),
            ("why_it_is_not_removed",
             ("removing it would mean retraining g on a subset of the train "
             "split, and g is the frozen route -- its recipe, its prompts and "
             "its training data are what the design compares D_s against")),
        ))),
    ))


def build_design_v5(dataset, forget_sets, forget_ids, router_seeds, edit_seeds,
                    direct_seeds, man=None, images=None,
                    image_sha_by_uri=None):
    """The v5 design: several forget sets, one crossed matrix, no reused cell.

    Built by calling the SHARED ``build_design_for`` once per forget set and
    merging the results, rather than by a second constructor.  Each per-set call
    runs ``coverage_audit_v2`` with that set's own ``forget_ids``, so the audit
    that decides whether a cell covers its design is the audit every earlier
    version ran, with the right answer for the set it was given -- and there is no
    second copy of that logic to drift.

    ``build_design_for`` is not edited, because it produced the frozen v1-v4
    bytes and a change to it is a change to what those designs mean.
    """
    if not forget_sets:
        raise RuntimeError("a v5 design with no forget set covers nothing")
    man = man if man is not None else load_manifest(dataset)
    # The audit the caller already paid for is reused where it was supplied: the
    # split only needs the digest table, and re-hashing every image a second time
    # in one construction would be a second answer to the same question.
    audit = ({"images_by_uri_sha256": image_sha_by_uri} if image_sha_by_uri
             else image_content_audit(man))
    direct_training = selected_direct_training(dataset, dev_split(man, audit))
    ids_of = {s["set_id"]: list(s["targets"]) for s in forget_sets}
    union = sorted(i for s in forget_sets for i in ids_of[s["set_id"]])
    if sorted(forget_ids) != union:
        raise RuntimeError(
            f"the design's forget_identity_ids {sorted(forget_ids)} are not the "
            f"union of its forget sets' targets {union}; the union is what the "
            f"design-level field means where several sets are varied, and a "
            f"different list would be a claim about a set no cell measures")
    per_set = []
    for fs in forget_sets:
        per_set.append(build_design_for(
            PILOT_SPEC_V5, dataset, fs["set_id"], ids_of[fs["set_id"]],
            router_seeds, edit_seeds, direct_seeds, man=man, images=images,
            image_sha_by_uri=image_sha_by_uri))

    cells = []
    for fs, one in zip(forget_sets, per_set):
        for c in one["cells"]:
            if c["kind"] in FORGET_SET_DEPENDENT_KINDS:
                cells.append(_qualify_cell(c, fs["set_id"],
                                           ids_of[fs["set_id"]]))
    # The forget-set-INDEPENDENT kinds are taken from one per-set design, and the
    # others are compared against it first.  The comparison is the derivation:
    # "direct does not depend on the forget set" is a claim about the row
    # builders, and this is where the design checks the claim instead of
    # inheriting it.
    for kind in ("direct", "hybrid"):
        copies = [[c for c in one["cells"] if c["kind"] == kind]
                  for one in per_set]
        first = canonical_json(copies[0])
        for fs, other in list(zip(forget_sets, copies))[1:]:
            if canonical_json(other) != first:
                raise RuntimeError(
                    f"the {kind} cells differ between forget set "
                    f"{forget_sets[0]['set_id']!r} and {fs['set_id']!r}, so "
                    f"{kind} is NOT forget-set-independent and cannot be filed "
                    f"once for the whole design.  Filing one copy would report "
                    f"one measurement as covering sets it was never run against")
        # Namespaced with the adapter, and for the same reason.  ``direct__d17``
        # is the id v3 filed a cell under and v5's D_s is a differently trained
        # adapter -- fit on the train split minus the development images, under
        # the schedule RFC selected -- so writing v5's cell at v3's path would
        # leave a committed v3 report describing a measurement nobody can
        # reproduce.  It would also make the freeze-ordering listing report v5's
        # own inputs as outputs it was about to overwrite, and refuse forever.
        cells.extend(_rename_cell(
            c, direct_cell_id(kind, c["direct_seed"], DIRECT_NAMESPACE_V5))
            for c in copies[0])

    by_kind = {}
    for c in cells:
        by_kind.setdefault(c["kind"], []).append(c["cell_id"])
    base = per_set[0]
    out = OrderedDict(base)
    out["forget_set_id"] = None
    out["forget_sets"] = [{"set_id": s["set_id"], "targets": ids_of[s["set_id"]]}
                          for s in forget_sets]
    out["n_forget_sets"] = len(forget_sets)
    out["forget_identity_ids"] = union
    out["forget_identity_ids_are_the_union"] = (
        "no cell was built over the union: each forget-set-dependent cell carries "
        "its own forget_set_id and forget_identity_ids, and those are what the "
        "statistics read.  This field is the design's coverage statement, not a "
        "measurement's input")
    out["cells"] = cells
    out["n_cells"] = len(cells)
    out["cells_by_kind"] = by_kind
    out["n_rows_total"] = sum(c["n_rows"] for c in cells)
    out["checkpoint_requirements"] = required_checkpoints_v5(
        dataset, forget_sets, router_seeds, edit_seeds, direct_seeds,
        DIRECT_NAMESPACE_V5)
    out["forget_set_dependent_kinds"] = list(FORGET_SET_DEPENDENT_KINDS)
    out["forget_set_independent_kinds"] = ["direct", "hybrid"]
    out["rows_are_not_duplicated_across_forget_sets"] = (
        "direct and hybrid rows were compared across all "
        f"{len(forget_sets)} per-set designs and found identical, so one copy of "
        "each is filed; a design that filed five copies would report one "
        "measurement five times, which is the defect the cell structure exists "
        "to prevent")
    out["direct_training"] = direct_training
    # A static statement of the ordering rule, and deliberately NOT the outcome of
    # applying it: the listing of which outputs existed at freeze time is filed
    # beside the manifest by _freeze_v5, because a live listing inside the design
    # would be covered by design_sha256 and would stop reproducing the moment the
    # run it constrains began.  That is the v2 defect, and a check added to
    # prevent a different one is still a way of arriving at it.
    out["confirmatory_ordering_rule"] = (
        "this design is frozen before any cell it names is produced and before "
        "any adapter it requires is trained.  _freeze_v5 lists every output the "
        "design names, REFUSES to freeze if one exists, and files the listing "
        "beside the manifest rather than inside it")
    return out


def build_pilot_preregistration_v5(dataset, forget_sets, forget_ids,
                                   router_seeds=None, edit_seeds=None,
                                   direct_seeds=None, man=None, images=None,
                                   selection=None, superseded_paths=()):
    """The frozen v5 pilot.

    Same construction path as every other version, with the forget sets passed
    through to the shared builder as ``forget_sets`` so the design constructor
    receives the list and the cell-count check receives the multiplier.
    """
    return build_pilot_preregistration_for(
        PILOT_SPEC_V5, dataset, None, forget_ids, router_seeds, edit_seeds,
        direct_seeds, man=man, images=images, selection=selection,
        superseded_paths=superseded_paths, forget_sets=forget_sets)


def prereg_path_v5(dataset, path=None):
    return prereg_path_for(PILOT_SPEC_V5, dataset, path)


def load_prereg_v5(dataset, path=None, verify=True):
    return load_prereg_for(PILOT_SPEC_V5, dataset, path, verify)


# ---------------------------------------------------------------------------
# Freeze #1: the calibration pre-registration
# ---------------------------------------------------------------------------

def calibration_prereg_path(dataset, path=None):
    return (Path(path) if path else
            DATASET_ROOT / MANIFEST_DIR / f"rf_calibration_{dataset}_v5.json")


def calibration_dir(dataset, candidate, seed):
    """Where one (candidate, seed) calibration run writes."""
    return (DATASET_ROOT / CELLS_DIR_V2 / dataset / CALIBRATION_SUBDIR_V5
            / f"{candidate}__seed{seed}")


def calibration_result_path(dataset, candidate, seed):
    return calibration_dir(dataset, candidate, seed) / "calibration_result.json"


def calibration_selection_path(dataset):
    """Where the selection rule's outcome is filed, after the measurements."""
    return (DATASET_ROOT / REPORTS_DIR
            / f"rf_calibration_selection_{dataset}_v5.json")


def load_calibration_prereg(dataset, path=None, verify=True):
    """Freeze #1, verified before RFC trains anything against it.

    The discipline ``load_prereg_for`` applies to a pilot, applied to the
    calibration: RFC trains the grid that document declares, so a document that no
    longer reproduces from its own recorded inputs is a grid nobody
    pre-registered.  Reading it with ``json.loads`` instead would have made an
    edit after freezing invisible in the selection it produced, which is the one
    failure mode a pre-registration exists to make impossible.
    """
    p = calibration_prereg_path(dataset, path)
    if not p.is_file():
        raise RuntimeError(
            f"RFC has no frozen calibration pre-registration at {_rel(p)}; "
            f"freeze it first with --preregister-calibration --dataset "
            f"{dataset}.  A calibration whose grid was not declared before the "
            f"training is a search whose result was chosen twice")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("kind") != CALIBRATION_PREREG_KIND_V5:
        raise RuntimeError(
            f"{_rel(p)} has kind {doc.get('kind')!r}, expected "
            f"{CALIBRATION_PREREG_KIND_V5!r}; this is not a v5 calibration "
            f"pre-registration and RFC will not train a grid it does not "
            f"declare")
    if verify:
        got = verify_manifest(p)
        if not got["valid"]:
            raise RuntimeError(
                f"the frozen calibration pre-registration at {_rel(p)} does not "
                f"verify: " + "; ".join(got["problems"]))
    return doc


def build_calibration_preregistration(dataset, man=None, audit=None):
    """Freeze #1: what the calibration will train, score and select between.

    Frozen BEFORE any candidate is trained, which is the only thing that makes
    the selection a pre-registration rather than a description of whichever
    configuration happened to look best.  It carries the split rule, the grid,
    the seeds, the metric, the tie-breaks and the floor, and it binds the dataset
    manifest the split is derived from.

    It carries NO threshold from the confirmatory design except by reference: the
    floor is ``PILOT_SPEC_V5.thresholds['min_direct_image_accuracy']``, read at
    freeze time, so a floor typed beside the gate could not disagree with it.
    """
    man = man if man is not None else load_manifest(dataset)
    audit = audit if audit is not None else image_content_audit(man)
    split = dev_split(man, audit)
    if not split["development_is_balanced_across_identities"]:
        raise RuntimeError(
            f"the development split is not balanced across identities: "
            f"{split['development_images_per_identity']}.  A calibration scored "
            f"on images from some identities and not others would select a "
            f"configuration for the identities it happened to include")
    if not split["every_train_image_is_placed"]:
        raise RuntimeError("the split did not place every train image")
    return OrderedDict((
        ("kind", CALIBRATION_PREREG_KIND_V5),
        ("dataset", dataset),
        ("version", "v5"),
        ("preregistered", True),
        ("executed", False),
        ("what_is_being_calibrated", OrderedDict((
            ("component", "D_s, the direct X -> Y adapter"),
            ("what_is_NOT_being_calibrated",
             ("g and h.  The router is used exactly as the frozen route left it "
              "and no router or edited h is trained or selected here, so a phase "
              "named RFC is not evidence that the route was tuned")),
            ("why_only_the_schedule",
             ("the LoRA rank, alpha, dropout and target modules are fixed by the "
              "frozen route module that built g and h.  Overriding them for D_s "
              "alone would make the direct pathway a different class of adapter "
              "from the route it is compared against, and the comparison would "
              "then be about the recipe rather than about the pathway")),
        ))),
        ("development_split", split),
        ("grid", OrderedDict((
            ("candidates", direct_schedule_candidates_v5()),
            ("n_candidates", len(direct_schedule_candidates_v5())),
            ("seeds", list(CALIBRATION_SEEDS_V5)),
            ("n_trainings",
             len(direct_schedule_candidates_v5()) * len(CALIBRATION_SEEDS_V5)),
            ("overridable_keys", list(OVERRIDABLE_SCHEDULE_KEYS)),
            ("incumbent_read_from", "frozen_route_protocol()"),
            ("incumbent_is_a_candidate",
             ("C0_incumbent is the frozen schedule itself, so 'change nothing' is "
             "an outcome the grid can return; a grid that cannot return it has "
             "already decided to change something")),
        ))),
        ("training", OrderedDict((
            ("trains_on", ("the fit images: the train split minus the "
                          "development images")),
            ("n_fit_images", split["n_fit_images"]),
            ("scored_on", "the development images only"),
            ("n_development_images", split["n_development_images"]),
            ("never_reads",
             (f"the {split['n_test_images']} held-out test images, which are what "
             f"the confirmatory gates measure")),
        ))),
        ("selection", CALIBRATION_SELECTION_RULE_V5),
        ("floor", calibration_floor_v5()),
        ("confirmatory_gate_thresholds_are_v4s",
         "PILOT_SPEC_V5.thresholds is GATE_THRESHOLDS_V4, the same object"),
        ("what_is_frozen_later",
         ("the confirmatory design.  It embeds the candidate this calibration "
          "selects and every cell it will run, and it is frozen before any of "
          "those cells exists")),
    ))


def freeze_calibration(args):
    """Write Freeze #1 and verify it from its tracked location."""
    man = load_manifest(args.dataset)
    block = build_calibration_preregistration(args.dataset, man=man)
    canonical = calibration_prereg_path(args.dataset)
    out = Path(args.out) / canonical.name if args.out else canonical
    if out.is_file():
        raise RuntimeError(
            f"{_rel(out)} already exists.  A calibration pre-registration is "
            f"frozen once; re-freezing it after the candidates have been trained "
            f"would produce a document that agrees with the outcome, which is "
            f"the thing it exists to make impossible")
    root = DATASET_ROOT / CELLS_DIR_V2 / args.dataset / CALIBRATION_SUBDIR_V5
    existing = sorted(root.glob("*")) if root.is_dir() else []
    if existing:
        raise RuntimeError(
            f"{len(existing)} calibration output(s) already exist under "
            f"{_rel(root)}, e.g. {existing[0].name}; the pre-registration has to "
            f"precede the training it constrains, so freezing now would produce "
            f"a grid that agrees with measurements already taken")
    extra = [DATASET_ROOT / MANIFEST_PATHS[args.dataset]]
    path, digest = freeze_manifest(block, out, extra_paths=extra)
    print(f"calibration frozen : {_rel(path)}", file=sys.stderr)
    print(f"sha256             : {digest[:16]}", file=sys.stderr)
    print(f"development split  : {block['development_split']['n_fit_images']} "
          f"fit / {block['development_split']['n_development_images']} dev / "
          f"{block['development_split']['n_test_images']} test (untouched)",
          file=sys.stderr)
    print(f"grid               : {block['grid']['n_candidates']} candidates x "
          f"{len(CALIBRATION_SEEDS_V5)} seeds = "
          f"{block['grid']['n_trainings']} trainings", file=sys.stderr)
    print(f"floor              : {block['floor']} on development accuracy, read "
          f"from the confirmatory gate", file=sys.stderr)
    got = verify_manifest(path)
    print(f"verified           : valid={got['valid']} problems={got['problems']}",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# RFC: the calibration phase
# ---------------------------------------------------------------------------

def phase_rfc(dataset, candidate=None, seed=None, prereg=None, device="cuda",
              resume=False):
    """RFC: train one D_s candidate on the fit split and score it on the dev split.

    With no candidate this is the SELECTION step instead: it reads every filed
    measurement, applies the rule the calibration pre-registration froze, and
    files the outcome.  That step needs no GPU and refuses to run before the grid
    is complete, so a selection can never be made over part of a grid.

    The development images are the only ones scored here.  Reading the held-out
    test images during calibration would make the confirmatory gates a
    measurement the configuration was chosen against, which is the one thing the
    disjoint split exists to prevent.
    """
    # The path is asked for as well as the document, because RFC binds the
    # digest of the pre-registration it trained under into every result it
    # files -- a binding that named the document but not its bytes would record
    # that a grid existed and not which grid.
    cal_path = calibration_prereg_path(dataset, prereg)
    cal = load_calibration_prereg(dataset, prereg)
    man = load_manifest(dataset)
    audit = image_content_audit(man)
    split = dev_split(man, audit)
    frozen_split = cal["development_split"]
    for key in ("development_image_uris", "fit_image_uris", "test_image_uris"):
        if split[key] != frozen_split[key]:
            raise RuntimeError(
                f"the development split derived now disagrees with the frozen "
                f"one on {key}; the calibration was pre-registered against a "
                f"different partition of the images")
    candidates = direct_schedule_candidates_v5()
    grid = cal["grid"]["candidates"]
    if canonical_json(grid) != canonical_json(
            {k: dict(v) for k, v in candidates.items()}):
        raise RuntimeError(
            "the grid derived now is not the grid the calibration froze; a "
            "candidate added after the fact is a candidate the pre-registration "
            "does not cover")
    if candidate is None:
        return _rfc_select(dataset, cal, candidates)

    if candidate not in candidates:
        raise RuntimeError(
            f"candidate {candidate!r} is not in the frozen grid "
            f"{list(candidates)}")
    if seed is None:
        raise RuntimeError("RFC trains ONE (candidate, seed) pair: pass "
                           "--direct-seed")
    if seed not in CALIBRATION_SEEDS_V5:
        raise RuntimeError(
            f"calibration seed {seed} is not one of "
            f"{list(CALIBRATION_SEEDS_V5)}; a seed outside the frozen grid is a "
            f"measurement the selection rule was not written for")
    prompts = frozen_route_prompts()
    vocab = label_vocab(man)
    protocol = _direct_schedule(frozen_route_protocol(), candidates[candidate])
    out_dir = calibration_dir(dataset, candidate, seed)
    result_path = calibration_result_path(dataset, candidate, seed)
    # Derived from out_dir and never from a second path rule: the trainer writes
    # where it is told, so a checkpoint path computed independently is a path
    # that reports a successful training as a missing file.
    ckpt = out_dir / Path(*ADAPTER_RELPATH)
    dev_by_identity = OrderedDict()
    for it in _items_by_uri(man, split["development_image_uris"]).values():
        dev_by_identity.setdefault(it["identity_id"], []).append(it)

    # Bound through the same fields a cell result is bound through, so --resume
    # reuses a measurement of THIS schedule on THIS split.  Without the schedule
    # and split digests a candidate re-trained under a different grid would be
    # read as already measured, and the selection would be made over numbers
    # nobody produced.
    bindings = OrderedDict((
        ("schedule", sha256_bytes(
            canonical_json(candidates[candidate]).encode("utf-8"))),
        ("protocol", sha256_bytes(
            canonical_json(protocol).encode("utf-8"))),
        ("fit_images", sha256_bytes(
            canonical_json(split["fit_image_uris"]).encode("utf-8"))),
        ("development_images", sha256_bytes(
            canonical_json(split["development_image_uris"]).encode("utf-8"))),
        ("calibration_preregistration", sha256_file(cal_path)),
    ))
    want = {"cell_id": f"calibration__{candidate}__seed{seed}",
            "kind": "direct_calibration_result_v5", "phase": "RFC",
            "candidate": candidate, "seed": seed,
            "schedule": dict(candidates[candidate]),
            "input_sha256": bindings,
            "n_development_images": len(split["development_image_uris"])}

    def run():
        sess = RouteSessionV2(device, f"e2c_rfc_{candidate}_{seed}", seed)
        try:
            sess.reset_fresh(seed)
            train_items = _direct_train_items(man, {
                "development_image_uris": split["development_image_uris"]})
            pairs = [(it["image_uri"], prompts["d_image_to_alias"],
                      man["alias_of"][it["identity_id"]])
                     for it in train_items]
            sess.train(f"calibration_{candidate}_seed{seed}", pairs, out_dir,
                       protocol)
            if not ckpt.is_file():
                raise RuntimeError(
                    f"training reported success but {ckpt} does not exist; a "
                    f"calibration whose adapter cannot be found cannot be "
                    f"re-run or inspected")
            rows = build_direct_rows_v2(
                man, f"calibration__{candidate}__seed{seed}", seed,
                image_sha_by_uri=audit["images_by_uri_sha256"],
                images=dev_by_identity)
            for r in rows:
                r["d_raw_text"] = sess.generate(
                    _open_image(r["image_uri"]), prompts["d_image_to_alias"],
                    max_new_tokens=8)
                r["prompt_used"] = prompts["d_image_to_alias"]
        finally:
            sess.release()
        scored, missing = score_rows_v2(rows, vocab)
        if missing:
            raise RuntimeError(
                f"{len(missing)} development image(s) produced no generation, "
                f"e.g. {missing[0]}; a calibration scored on part of the "
                f"development split would select against images it never saw")
        doc = {
            **want, "dataset": dataset,
            "adapter_path": _rel(ckpt),
            "adapter_sha256": sha256_file(ckpt),
            "adapter_bytes": ckpt.stat().st_size,
            "protocol": protocol,
            "n_fit_images": len(train_items),
            "development_accuracy": _mean_correct(scored),
            "n_unparseable": sum(1 for r in scored if r.get("unparseable")),
            "n_multi_label_ambiguous": sum(1 for r in scored
                                           if r.get("multi_label_ambiguous")),
            "calibration_preregistration_sha256": sha256_file(cal_path),
            "development_images_are_not_the_test_images": (
                "the images scored here are the train-split images at the frozen "
                "development indices; the held-out test images are disjoint from "
                "them in CONTENT, which the pre-registration asserts and this "
                "phase never reads"),
            "scored": scored,
        }
        atomic_write_json(result_path, doc)
        return doc

    doc, decision = _resume_or_run(dataset, f"rfc_{candidate}_{seed}", want,
                                   resume, run, result_path)
    doc = {**doc, "resume": decision}
    atomic_write_json(result_path, doc)
    return doc


def _items_by_uri(man, uris):
    """The manifest items at these URIs, in the order given."""
    want = set(uris)
    out = OrderedDict()
    for it in man["items"]:
        if it["image_uri"] in want:
            out[it["image_uri"]] = it
    if len(out) != len(want):
        raise RuntimeError(
            f"{len(want) - len(out)} of the requested image URIs are not in the "
            f"manifest; a split naming images the dataset does not have is not a "
            f"split of it")
    return out


def _rfc_select(dataset, cal, candidates):
    """Apply the frozen selection rule to the filed measurements, or refuse."""
    results, absent = OrderedDict(), []
    for name in candidates:
        per_seed = OrderedDict()
        for seed in CALIBRATION_SEEDS_V5:
            p = calibration_result_path(dataset, name, seed)
            if not p.is_file():
                absent.append(f"{name}/seed{seed}")
                continue
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc.get("calibration_preregistration_sha256") != \
                    sha256_file(calibration_prereg_path(dataset)):
                raise RuntimeError(
                    f"{_rel(p)} was produced against a different calibration "
                    f"pre-registration than the one now frozen; a measurement "
                    f"from another grid is not evidence about this one")
            per_seed[seed] = doc["development_accuracy"]
        if per_seed:
            results[name] = per_seed
    if absent:
        raise RuntimeError(
            f"RFC cannot select: {len(absent)} of "
            f"{len(candidates) * len(CALIBRATION_SEEDS_V5)} calibration runs "
            f"have no filed measurement, e.g. {absent[0]}.  Selecting over part "
            f"of a grid is selecting over a different grid, and the rule was "
            f"frozen for this one")
    out = select_direct_schedule(results)
    out["dataset"] = dataset
    out["phase"] = "RFC"
    out["calibration_preregistration"] = _rel(calibration_prereg_path(dataset))
    out["calibration_preregistration_sha256"] = sha256_file(
        calibration_prereg_path(dataset))
    out["measurements_consumed"] = OrderedDict(
        (name, OrderedDict((str(s), calibration_result_path(dataset, name, s))
                           for s in CALIBRATION_SEEDS_V5))
        for name in candidates)
    if out["selected"] is None:
        out["the_confirmatory_design_is_not_frozen"] = (
            "the rule refused to select, so there is no configuration to embed "
            "and no v5 pilot to freeze.  That is the outcome the floor exists to "
            "produce, and it is reported rather than worked around")
    path, digest = atomic_write_json(calibration_selection_path(dataset), out)
    logger.info("RFC selection: filed %s %s", _rel(path), digest[:16])
    return out


# ---------------------------------------------------------------------------
# Freeze #2, and the ordering it has to hold
# ---------------------------------------------------------------------------

def confirmatory_outputs_present(dataset, block, cells_out=None):
    """Every output this design names that ALREADY EXISTS.

    The checkable form of "frozen before producing their outputs".  A promise
    about ordering is a promise; a listing taken at freeze time, refused against,
    and filed beside the manifest is a fact a reader can check later and a test
    can assert on without trusting anybody's memory of which command ran first.

    Roles this design does not PRODUCE are skipped: the router at every seed, the
    unedited h and the matrix's fifteen edited-h adapters are all inputs v5 was
    built over rather than outputs of it.  Counting them would report every v5
    freeze as a freeze after the fact on any tree the route had already been run
    on -- a check that fails so reliably it would be disabled.

    Skipped on ``produced_by_this_design`` and not on ``exists_already``, because
    the two answer different questions and only the first is about the design.
    Whether an input is actually present is ``checkpoint_readiness``'s live
    answer, and it is unchanged by this.
    """
    present = []
    for c in block["cells"]:
        p = cell_result_path(dataset, c["cell_id"], cells_out)
        if p.is_file():
            present.append({"what": "cell", "cell_id": c["cell_id"],
                            "path": _rel(p), "sha256": sha256_file(p)})
    roles = (block.get("checkpoint_requirements") or {}).get("roles") or {}
    for label, role in sorted(roles.items()):
        if "produced_by_this_design" not in role:
            raise RuntimeError(
                f"checkpoint role {label!r} does not say whether this design "
                f"produces it, so the freeze-ordering listing cannot tell an "
                f"output it has not made yet from an input it was built over.  "
                f"Guessing from presence is the inference this check replaced")
        if not role["produced_by_this_design"]:
            continue
        for key in CHECKPOINT_FILE_KEYS_V2:
            raw = role.get(key)
            if not raw:
                continue
            p = resolve_recorded_path(raw)
            if p.is_file():
                present.append({"what": "checkpoint", "role": label,
                                "file_key": key, "path": _rel(p),
                                "sha256": sha256_file(p)})
    return present


def freeze_ordering_report_path(dataset):
    """Where the freeze-time listing is filed, beside the manifest not inside it.

    Outside ``design_sha256`` deliberately.  A listing inside the design would be
    covered by the hash and would stop reproducing the moment the run it
    constrains began -- which is the v2 defect, arriving through a check that was
    added to prevent a different one.
    """
    return (DATASET_ROOT / REPORTS_DIR
            / f"rf_freeze_ordering_{dataset}_v5.json")


def _freeze_v5(spec, args):
    """Freeze #2: the confirmatory v5 pilot.

    Ordered by what it reads rather than by convention.  The design constructor
    refuses unless a calibration selection is filed, so a v5 manifest cannot
    exist before RFC has selected a configuration; and this function refuses
    unless every output the design names is absent, so a v5 manifest cannot exist
    after the run it confirms has begun.
    """
    if spec.version != "v5":
        raise RuntimeError(
            f"_freeze_v5 was handed the {spec.version} spec; the forget-set rule "
            f"and the design constructor it calls are v5's")
    # Checked FIRST, before the forget-set rule runs and before any design is
    # built.  The ordering v5 exists to enforce is that the configuration was
    # chosen before the confirmatory design was frozen; a freeze that failed
    # later for some unrelated reason would not have demonstrated that ordering
    # at all, it would only have failed.
    _require_calibration_selection(args.dataset)
    man = load_manifest(args.dataset)
    images = held_out_images(man)
    policy = pilot_seed_policy_v2(args.dataset, man=man)
    edit_seeds = args.edit_seeds or policy["edit_seeds"]
    if args.forget_ids:
        raise RuntimeError(
            "v5's forget sets come from the selection rule and not from "
            "--forget-ids; a design whose targets were passed in is a design "
            "whose targets somebody chose")
    selection = pilot_forget_sets_v5(args.dataset, edit_seeds)
    block = build_pilot_preregistration_v5(
        args.dataset, selection["sets"], selection["targets"],
        args.router_seeds, edit_seeds, args.direct_seeds, man=man,
        images=images, selection=selection)

    already = confirmatory_outputs_present(args.dataset, block, args.cells)
    listing = OrderedDict((
        ("kind", "route_dependent_forgetting_freeze_ordering_v5"),
        ("dataset", args.dataset),
        ("version", "v5"),
        ("what_was_listed", OrderedDict((
            ("cells", len(block["cells"])),
            ("checkpoint_roles_that_are_training_work",
             sum(1 for r in block["checkpoint_requirements"]["roles"].values()
                 if not r.get("exists_already"))),
        ))),
        ("n_outputs_that_already_existed", len(already)),
        ("outputs_that_already_existed", already),
        ("the_freeze_was_therefore", "before its outputs" if not already
         else "REFUSED as after the fact"),
        ("why_this_is_filed_outside_the_design",
         ("a listing inside the manifest would be covered by design_sha256 and "
         "would stop reproducing the moment the run it constrains began, which "
         "is the v2 defect arriving through a check added to prevent another")),
    ))
    if already:
        atomic_write_json(freeze_ordering_report_path(args.dataset), listing)
        raise RuntimeError(
            f"{len(already)} output(s) this design names already exist, e.g. "
            f"{already[0]['path']}.  A confirmatory design has to be frozen "
            f"before the cells it confirms are produced; the listing is filed at "
            f"{_rel(freeze_ordering_report_path(args.dataset))}")
    atomic_write_json(freeze_ordering_report_path(args.dataset), listing)
    print(f"ordering           : 0 of {len(block['cells'])} cells and 0 of the "
          f"adapters it requires exist yet -- frozen before its outputs",
          file=sys.stderr)
    return _freeze_write_and_verify(PILOT_SPEC_V5, args, block)


def _v5_extra_freeze_inputs(dataset):
    """The files a v5 freeze binds beyond the ones every version binds.

    Both calibration artifacts, because the design's ``direct_training`` block is
    READ out of the selection and the selection was made under the
    pre-registration: a design that embedded a configuration without binding the
    two documents that produced it would name a choice nothing could re-check.
    """
    return [calibration_prereg_path(dataset),
            calibration_selection_path(dataset)]


def _run_v5_phase(args, phase):
    """The v5 dispatcher, kept so ``--design-version v5`` names a version the
    same way v2, v3 and v4 do rather than only working through the default."""
    return _run_phase_for(PILOT_SPEC_V5, args, phase)


def _run_v5(args):
    return _run_pilot(args, PILOT_SPEC_V5)


# ---------------------------------------------------------------------------
# Rebind the version tables.  Everything above reads these at call time, so
# declaring v5 here is the same thing as declaring it where the tables are built,
# and the constructors that produced the frozen v1, v2, v3 and v4 bytes stay
# textually untouched.
# ---------------------------------------------------------------------------

LATEST_PILOT_SPEC = PILOT_SPEC_V5
SPEC_BY_VERSION[PILOT_SPEC_V5.version] = PILOT_SPEC_V5
SPEC_BY_PREREG_KIND[PILOT_SPEC_V5.prereg_kind] = PILOT_SPEC_V5
PHASES_BY_VERSION[PILOT_SPEC_V5.version] = _phases_for(PILOT_SPEC_V5)
REPAIRS_BY_VERSION["v5"] = OrderedDict((*SUPERSESSION_ITEMS.items(),
                                        *SUPERSESSION_ITEMS_V3.items(),
                                        *SUPERSESSION_ITEMS_V4.items(),
                                        *SUPERSESSION_ITEMS_V5.items()))
SUPERSESSION_POLICY["v5"] = SUPERSESSION_POLICY_V5

#: v4 becomes superseded, and its kind needs notes before ``supersession_record``
#: will describe a v4 manifest: it refuses to state nothing about an artifact it
#: marks superseded, which is the right refusal.
SUPERSEDED_NOTES[PREREG_V4_KIND] = SUPERSEDED_NOTES_V4

#: v5 registers its own forget-set rule and its own extra freeze inputs, and
#: registers NO scope-amendment evidence function -- the absence is the statement
#: that v5 did not amend a scope after reading its own outcome.
FORGET_SET_SELECTION_BY_VERSION["v5"] = pilot_forget_sets_v5
EXTRA_FREEZE_INPUTS_BY_VERSION["v5"] = _v5_extra_freeze_inputs
FREEZE_BY_VERSION["v5"] = _freeze_v5


if __name__ == "__main__":
    sys.exit(main())
