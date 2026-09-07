#!/usr/bin/env python3
"""E2C-v3 prompt-robustness panel (PP0-PP4, PPR): a HELD-OUT template sweep.

WHY THIS EXISTS
===============
Every E2C-v3 conclusion so far is scoped to one prompt string.  The granularity
runner says so itself: proximity to a retraining reference is "scoped to the
evaluated code prompts and candidate-label space".  Route h is trained and
measured with ``rd.CODE_TO_ALIAS_PROMPT`` and nothing else, so an edit that
only survives that exact rendering would look identical to one that survives
paraphrase.  This script builds a frozen six-template panel and measures the
difference.  It retrains nothing: all checkpoints already exist, and
``mx.ModelSession.reset_to`` swaps adapters in place, so the sweep is one model
load plus a fast reload per checkpoint.

THE PANEL IS HELD OUT, AND THAT IS ENFORCED RATHER THAN ASSERTED
================================================================
* PP0 builds and freezes it.  Later phases recompute its digest and ABORT on
  any mismatch with the committed manifest, so a template cannot be edited
  after its results are seen.
* It is committed before any evaluation runs.
* The pass criteria come from ``scripts/e2c_v3_granularity.py`` and are applied
  UNCHANGED as read-only per-template columns.  No threshold anywhere in this
  repo is derived from panel output, and no gate reads it.
* ``template_worst_case`` is a ``min`` over all six pre-frozen templates and
  ``max_template_spread`` a ``max - min``, so a template that behaves badly
  cannot be dropped from the headline after the fact.

SIX TEMPLATE ROLES (identical roles on both routes)
===================================================
canonical           the production prompt, byte-identical
concise_paraphrase  same request, fewer words
question_form       interrogative rather than imperative
instruction_form    explicit "return only ... do not explain"
format_variation    canonical modulo case/punctuation/whitespace ONLY, so it
                    isolates surface formatting from lexical change
distractor          carries a decoy drawn from the SAME output space, chosen to
                    be neutral for that identity (see below)

DISTRACTOR NEUTRALITY -- the one real design constraint
=======================================================
The panel is frozen ONCE per identity, but which labels are source/target
differs across the 21 sets.  A decoy that is neutral for one set could be the
desired label for another, which would make the distractor template easier than
the canonical one and turn a robustness measurement into a leak.  So the decoy
for an identity is drawn from the vocabulary MINUS every label that identity is
ever associated with in any set: its baseline alias, its whole taxonomic chain
(specific / level-1 / level-2, because a set may transform it to an ancestor),
and every source and target it is assigned in any of the 21 sets.  The pool and
the exclusions are recorded per identity.  If the pool were ever empty the
builder falls back to a label from a different taxonomic branch and records the
exact ``(set_id, role)`` collisions, so the report can separate those rows
instead of averaging over them.

ROUTE g: NO g-SIDE INTERVENTION EXISTS, SO NOTHING HERE MAY CLAIM ONE
=====================================================================
There is exactly one trained visual router (``g_X_to_C``).  The four model
classes -- baseline, edited, matched-retrain, LOO-retrain -- are all h-side
adapters, and the granularity matrix deliberately REPLAYS cached g rows instead
of re-running g.  Route g therefore measures two separate things:

PP2  Baseline prompt sensitivity.  The FROZEN ``g_X_to_C`` router over the six
     image-conditioned templates: code accuracy per template, six-template
     consistency, worst-template accuracy, maximum template spread, and
     invalid / off-support output rates.

PP3  Cross-route edit spillover.  Every in-scope h-side adapter loaded onto the
     same base model and run over the IDENTICAL panel, scored RELATIVE TO
     FROZEN BASE g: prediction-flip rate, change in code accuracy,
     candidate-distribution distance where available, target-person versus
     retained-person effects, and per-template plus worst-template spillover.
     The null expectation is that an edit targeting h(C->A) leaves g(X->C)
     approximately unchanged.

The confirmatory spillover reference is ALWAYS frozen base g.  Distances to the
h-side matched/LOO adapters are retained only as a secondary diagnostic named
``g_spillover_reference_distance`` and labelled a cross-route reference
comparison.  They are NOT oracle distances: there was no g-side matched or LOO
retraining.  No g result may be called "edit success", "g-side unlearning" or
"retraining equivalence", and :func:`_guard_g_vocabulary` raises if one is.

Phases: PP0 build+freeze the panel | PP1 route h sweep (full SALMU matrix)
        PP2 route g baseline | PP3 route g spillover
        PP4 aggregate + report | PPR CPU-only re-aggregation

Usage:
    python scripts/e2c_v3_prompt_panel.py --dataset salmu --phase PP0
    python scripts/e2c_v3_prompt_panel.py --dataset salmu --phase all \
        --device cuda:0
    python scripts/e2c_v3_prompt_panel.py --dataset salmu --phase PP1 \
        --only-sets gx_sal_mix_0 --only-seeds 17        # subset
    python scripts/e2c_v3_prompt_panel.py --dataset salmu --phase PPR  # CPU

GPU: launched on one device via CUDA_VISIBLE_DEVICES with --device cuda:0
inside the masked namespace.  Qwen3.5-9B bf16 needs ~19-20 GB resident.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e2c_v3_pp")

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The same low-level facilities the granularity runner reuses: strict parsing,
# full-sequence candidate scoring, candidate/OTHER accounting, gated distances,
# model session, provenance.  Nothing here trains or edits a model.
rv = _load_sibling("e2c_rv_pp", "e2c_v3_research_validity.py")
rd = _load_sibling("e2c_rd_pp", "e2c_v3_realdata.py")
mx = _load_sibling("e2c_mx_pp", "e2c_v3_matrix.py")
gx = _load_sibling("e2c_gx_pp", "e2c_v3_granularity.py")
gxm = _load_sibling("e2c_gxm_pp", "e2c_v3_granularity_matrix.py")

PANEL_ROOT = Path("e2c_prompt_panel")
PANEL_MANIFEST_DIR = PANEL_ROOT / "manifests"
PANEL_OUT_ROOT = PANEL_ROOT / "outputs"
PANEL_REPORT_DIR = PANEL_ROOT / "reports"

#: The one trained visual router.  There is no edited, matched or LOO variant
#: of it anywhere in this project, which is why route g is reported as
#: prompt sensitivity plus cross-route spillover and never as an edit.
SALMU_ROUTE_G = Path("e2c_salmu/outputs/salmu/g_X_to_C/adapter_final/"
                     "adapter_model.safetensors")

#: Captured in ``main`` BEFORE any result file is written, so every artifact
#: names the code that actually produced it rather than whatever HEAD is by the
#: time the sweep finishes (the bug fixed in commit 9e488a3).
EXECUTING_COMMIT = None

#: The six roles, in the order they are always reported.  Frozen: the worst
#: case is a min over exactly these, so the list cannot shrink after results.
TEMPLATE_ROLES = ("canonical", "concise_paraphrase", "question_form",
                  "instruction_form", "format_variation", "distractor")

#: Route h (code -> alias).  ``canonical`` MUST stay byte-identical to
#: ``rd.CODE_TO_ALIAS_PROMPT``; a test asserts it, because the whole point is
#: that the canonical column reproduces the existing evidence.
H_TEMPLATES = {
    "canonical": rd.CODE_TO_ALIAS_PROMPT,
    "concise_paraphrase": "Alias for code {code}:",
    "question_form": "What alias corresponds to identity code {code}?",
    "instruction_form": ("Return only the alias associated with identity code "
                         "{code}. Do not explain."),
    "format_variation": "identity code: {code}\ngenerate the alias",
    "distractor": ("Identity code: {code}. Note that {decoy} is a known alias "
                   "in this dataset. Generate the alias for {code}."),
}

#: Route g (image -> code).  ``canonical`` MUST stay byte-identical to
#: ``rd.IMG_TO_CODE_PROMPT``.  The decoy here is another identity's CODE,
#: because codes are this route's output space.
G_TEMPLATES = {
    "canonical": rd.IMG_TO_CODE_PROMPT,
    "concise_paraphrase": "Identity code?",
    "question_form": ("Which identity code belongs to the person in this "
                      "image?"),
    "instruction_form": ("Identify the person in this image and return only "
                         "their identity code. Do not explain."),
    "format_variation": "what is the identity code for this person",
    "distractor": ("What is the identity code for this person? Note that "
                   "{decoy} appears elsewhere in this dataset."),
}

#: Determinism for every seeded choice in the panel (decoy selection).  Matches
#: the matrix's own seed so the panel is reproducible from the frozen inputs.
PANEL_SEED = gx.MATRIX_SEED

#: Distances are gated exactly as everywhere else in this project: below this
#: candidate mass a renormalized distance is a numerical artifact, not a
#: finding, so it is reported as null / "not established".
MIN_CANDIDATE_MASS = rv.MIN_CANDIDATE_MASS

#: Route g scope.  The full h matrix is 106 checkpoints; route g additionally
#: pays for image encoding per prompt, so spillover runs over the same
#: representative sets GX2B uses -- one per transformation type -- across all
#: three edit seeds.  Extending g later is a scope change, not a redesign, and
#: cannot invalidate the frozen panel.
G_SPILLOVER_REP_SETS = tuple(gxm.BALANCED_REP_SETS["salmu"])

#: Words that would be false about route g, because there was no g-side
#: intervention and no g-specific matched/LOO retraining.  Enforced on every
#: g-side claim and field name.
PROHIBITED_G_TERMS = ("edit success", "g-side unlearning",
                      "retraining equivalence")

#: What the panel may never be used for.  Recorded in the artifact so a reader
#: holding only the JSON can see the constraint, not just the numbers.
HELD_OUT_RECORD = {
    "status": "held-out behavioral evaluation",
    "frozen_before_evaluation": True,
    "not_used_for": ["model selection", "threshold setting",
                     "template selection", "training recipe changes",
                     "any gate or promotion criterion"],
    "criteria_source": ("scripts/e2c_v3_granularity.py PASS_CRITERIA, applied "
                        "unchanged as read-only per-template columns"),
    "worst_case_rule": ("min over the six pre-frozen template roles; no "
                        "template may be dropped after results are seen"),
    "prohibited_g_terms": list(PROHIBITED_G_TERMS),
}


# ====================================================================== #
# PP0: build and freeze the panel
# ====================================================================== #
def _labels_excluded_for_identity(ctx, matrix, iid):
    """Every alias this identity is EVER associated with, in any set.

    A decoy drawn from this set would sometimes BE the desired label, which
    would make the distractor template easier than the canonical one and invert
    the measurement.  The taxonomic chain is excluded whole, not just the
    specific alias, because a set may transform this identity to its own
    level-1 or level-2 ancestor.
    """
    excluded = {ctx["baseline_alias_of"][iid]}
    excluded.update(ctx["hierarchy_of"][iid])
    collisions = []
    for entry in matrix["sets"]:
        assignment = entry["assignments"].get(iid)
        if not assignment:
            continue
        for role in ("source", "target"):
            label = assignment.get(role)
            if label:
                excluded.add(label)
                collisions.append({"set_id": entry["set_id"],
                                   "role": role, "label": label})
    return sorted(excluded), collisions


def _pick_decoy(pool, iid, identity_ids):
    """Deterministic decoy from ``pool``, seeded by the identity's position.

    Position-seeded rather than hash-seeded so the choice is reproducible
    across processes and Python versions without depending on hash
    randomization or on dict ordering.
    """
    if not pool:
        return None
    rng = random.Random(PANEL_SEED + identity_ids.index(iid))
    return rng.choice(sorted(pool))


def _h_decoy_fallback(ctx, matrix, iid, identity_ids):
    """A label from a DIFFERENT taxonomic branch, when the neutral pool is empty.

    Only reachable if an identity is associated with essentially the whole
    vocabulary across the 21 sets.  The fallback is recorded as a collision so
    the report separates those rows rather than presenting them as neutral.
    """
    own_branch = set(ctx["hierarchy_of"][iid])
    candidates = [lab for lab in ctx["vocab"]
                  if lab != gx.DELETED_LABEL and lab not in own_branch
                  and lab != ctx["baseline_alias_of"][iid]]
    return _pick_decoy(candidates, iid, identity_ids)


def _first_test_image(manifest, iid):
    """The first held-out test image for one identity, by sorted file name.

    Test rather than train on purpose: the frozen router is 96/96 on train and
    29/36 on held-out, so a train image would sit at a ceiling and hide exactly
    the template sensitivity this panel exists to measure.
    """
    rows = sorted((it for it in manifest["items"]
                   if it["identity_id"] == iid and it["split"] == "test"),
                  key=lambda it: str(it["image_uri"]))
    if not rows:
        raise RuntimeError(f"no held-out test image for identity {iid}")
    row = rows[0]
    return {"image_uri": row["image_uri"], "image_sha256": row["image_sha256"],
            "source_file_name": row.get("source_file_name"),
            "split": row["split"]}


def _render_h(templates, code, decoy):
    return {role: text.format(code=code, decoy=decoy)
            for role, text in templates.items()}


def _render_g(templates, decoy):
    return {role: text.format(decoy=decoy)
            for role, text in templates.items()}


def build_panel(ds, manifest, matrix, ctx):
    """The frozen panel: six templates x every identity, for both routes."""
    if ds != "salmu":
        raise RuntimeError(
            f"the prompt panel is instantiated for salmu only (got {ds!r}); "
            f"celeba_numeric has no image router and its matrix is incomplete")
    identity_ids = list(ctx["identity_ids"])
    codes = list(ctx["code_of"].values())

    h_rows, g_rows = [], []
    for iid in identity_ids:
        code = ctx["code_of"][iid]
        excluded, assigned = _labels_excluded_for_identity(ctx, matrix, iid)
        pool = [lab for lab in ctx["vocab"] if lab not in excluded]
        decoy = _pick_decoy(pool, iid, identity_ids)
        neutral = decoy is not None
        collisions = []
        if not neutral:
            decoy = _h_decoy_fallback(ctx, matrix, iid, identity_ids)
            collisions = [{"set_id": c["set_id"], "role": c["role"],
                           "label": c["label"], "reason": "assigned to this "
                                                         "identity"}
                          for c in assigned]
            if decoy is None:
                raise RuntimeError(
                    f"identity {iid}: no distractor label available even by "
                    f"fallback; the vocabulary is exhausted for this identity")
            logger.warning("identity %s: neutral distractor pool empty; fell "
                           "back to %r and recorded %d collision(s)",
                           iid, decoy, len(collisions))
        # Route g decoy: another identity's CODE.  Any other code is wrong for
        # this image by construction, so every choice is neutral here; the seed
        # is for reproducibility, not for avoiding a leak.
        g_pool = [c for c in codes if c != code]
        g_decoy = _pick_decoy(g_pool, iid, identity_ids)
        h_rows.append({
            "identity_id": iid,
            "code": code,
            "baseline_alias": ctx["baseline_alias_of"][iid],
            "taxonomic_chain": list(ctx["hierarchy_of"][iid]),
            "distractor_decoy": decoy,
            "distractor_neutral_for_all_sets": neutral,
            "distractor_collisions": collisions,
            "distractor_neutral_pool_size": len(pool),
            "distractor_excluded_labels": excluded,
            "prompts": _render_h(H_TEMPLATES, code, decoy),
        })
        image = _first_test_image(manifest, iid)
        g_rows.append({
            "identity_id": iid,
            "code": code,
            **image,
            "distractor_decoy_code": g_decoy,
            "prompts": _render_g(G_TEMPLATES, g_decoy),
        })

    return {
        "kind": "e2c_v3_prompt_panel_v1",
        "dataset": ds,
        "produced_by": "scripts/e2c_v3_prompt_panel.py",
        "panel_seed": PANEL_SEED,
        "template_roles": list(TEMPLATE_ROLES),
        "held_out": dict(HELD_OUT_RECORD),
        "candidate_spaces": {
            "h": {"output_space": "alias", "vocab": list(ctx["vocab"]),
                  "deleted_label": gx.DELETED_LABEL},
            "g": {"output_space": "code", "codes": codes,
                  "deleted_label": None},
        },
        "routes": {
            "h": {"templates": dict(H_TEMPLATES), "n_rows": len(h_rows),
                  "rows": h_rows},
            "g": {"templates": dict(G_TEMPLATES), "n_rows": len(g_rows),
                  "rows": g_rows},
        },
    }


def panel_digest(panel):
    """SHA-256 over the panel's canonical JSON, excluding the digest field.

    Hashing the serialized content rather than the file bytes means a
    re-serialization (key order, indentation) does not look like a changed
    panel, while any change to a template, a decoy or an image does.
    """
    body = {k: v for k, v in panel.items() if k != "panel_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def panel_path(ds):
    return PANEL_MANIFEST_DIR / f"prompt_panel_{ds}.json"


def freeze_panel(ds, panel, args):
    """Write the panel once.  Refuses to overwrite unless --refreeze."""
    PANEL_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = panel_path(ds)
    if path.exists() and not args.refreeze:
        existing = json.loads(path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"{path} is already frozen (digest "
            f"{existing.get('panel_sha256', '?')[:16]}...).  A frozen panel is "
            f"the whole guarantee that no template was adjusted after its "
            f"results were seen, so overwriting it needs an explicit "
            f"--refreeze, and any results already produced against the old "
            f"digest must be discarded and re-run.")
    panel["inputs"] = {
        "salmu_manifest_sha256": rv.sha256_file(gxm.SALMU_MANIFEST),
        "matrix_salmu_sha256": rv.sha256_file(
            gxm.MANIFEST_DIR / "matrix_salmu.json"),
    }
    panel["builder"] = {
        "git_commit": rv.git_commit_sha(),
        "git_dirty": rv.git_worktree_dirty(),
        "script_sha256": rv.script_sha256(),
    }
    panel["built_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    panel["panel_sha256"] = panel_digest(panel)
    path.write_text(json.dumps(panel, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    logger.info("PP0: froze %s (digest %s, %d identities x %d templates x "
                "%d routes)", path, panel["panel_sha256"][:16],
                panel["routes"]["h"]["n_rows"], len(TEMPLATE_ROLES), 2)
    return panel


def load_frozen_panel(ds):
    """Load the panel and VERIFY its digest, or abort.

    This is the held-out guarantee at run time: a template edited after results
    were seen changes the digest and stops the sweep, rather than silently
    producing numbers that no longer correspond to the committed panel.
    """
    path = panel_path(ds)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found; run --phase PP0 first and commit the panel "
            f"before evaluating it")
    panel = json.loads(path.read_text(encoding="utf-8"))
    recorded = panel.get("panel_sha256")
    actual = panel_digest(panel)
    if recorded != actual:
        raise RuntimeError(
            f"{path}: panel digest mismatch (recorded {str(recorded)[:16]}, "
            f"recomputed {actual[:16]}).  The frozen panel was modified after "
            f"it was written, so it is no longer the held-out panel any "
            f"previous result was measured against.  Restore it from git or "
            f"re-freeze explicitly and discard the old results.")
    return panel


# ====================================================================== #
# Model inventory -- what is evaluated, and what "desired" means for each
# ====================================================================== #
RETRAIN_FAMILIES = ("matched_retrain", "loo_retrain")


def _adapter_path(directory):
    return Path(directory) / "adapter_final" / "adapter_model.safetensors"


def h_model_specs(ds, out_base, matrix, args):
    """Every route-h checkpoint in scope: baseline, edited cells, retrain oracles.

    A missing checkpoint is recorded as ``pending`` rather than skipped
    silently, so an incomplete sweep is visible in the report as incomplete
    instead of looking like a smaller matrix that happened to pass.
    """
    cells_root = out_base / "cells"
    oracle_root = out_base / "oracles"
    only_sets = set(args.only_sets) if args.only_sets else None
    only_seeds = set(args.only_seeds) if args.only_seeds else None
    specs = []

    def add(model_id, model_class, set_id, seed, ckpt):
        specs.append({"model_id": model_id, "model_class": model_class,
                      "set_id": set_id, "seed": seed, "checkpoint": str(ckpt),
                      "present": Path(ckpt).exists()})

    add("baseline_h", "baseline", None, None, gxm._baseline_ckpt(ds, out_base))
    for entry in matrix["sets"]:
        sid = entry["set_id"]
        if only_sets and sid not in only_sets:
            continue
        for seed in matrix["edit_seeds"]:
            if only_seeds and seed not in only_seeds:
                continue
            add(f"edited__{sid}__seed{seed}", "edited", sid, seed,
                _adapter_path(cells_root / sid / f"seed_{seed}" / "edited_h"))
        for family in RETRAIN_FAMILIES:
            add(f"{family}__{sid}", family, sid, None,
                _adapter_path(oracle_root / gxm.retrain_oracle_dir(
                    family, sid, gxm.ORACLE_SEED)))
    return specs


def g_model_specs(ds, out_base, matrix, args):
    """Route-g scope: the frozen router plus every in-scope h-side adapter.

    The frozen ``g_X_to_C`` router is the CONFIRMATORY reference for spillover
    and is always present.  The h-side adapters are the same four classes as
    route h, restricted to the representative sets because route g additionally
    pays for image encoding on every prompt.
    """
    specs = [{"model_id": "frozen_g", "model_class": "frozen_g",
              "set_id": None, "seed": None,
              "checkpoint": str(SALMU_ROUTE_G),
              "present": SALMU_ROUTE_G.exists()}]
    rep = set(G_SPILLOVER_REP_SETS)
    for spec in h_model_specs(ds, out_base, matrix, args):
        if spec["model_class"] == "baseline" or spec["set_id"] in rep:
            specs.append(dict(spec))
    return specs


def expected_of_for(spec, ctx, entry_by_id):
    """The desired label per identity for THIS model, plus its accuracy scope.

    ``loo_retrain`` is trained on the retained mapping only and never sees the
    transformation targets, so a target identity has no desired label for that
    model: its row is scored for its candidate distribution -- which is exactly
    what the edit-versus-LOO distance needs -- but is not counted as an
    accuracy failure.  Reporting it as one would be a claim about a model that
    was never asked to do the thing.
    """
    ids = list(ctx["identity_ids"])
    if spec["model_class"] == "baseline" or spec["set_id"] is None:
        return {i: ctx["baseline_alias_of"][i] for i in ids}, "all"
    entry = entry_by_id[spec["set_id"]]
    if spec["model_class"] == "loo_retrain":
        return ({i: (None if i in entry["assignments"]
                     else ctx["baseline_alias_of"][i]) for i in ids},
                "retained_only")
    return ({i: gxm.expected_label(ctx, entry, i) for i in ids}, "all")


def groups_for(spec, ctx, entry_by_id):
    """Per-identity reporting group, so blocks never average together.

    Transformation targets and refusal controls are kept apart (the existing
    aggregation rule: a refusal control must never leak into a granularity
    headline), and sibling / cousin / retain follow ``gxm.control_group``.
    """
    if spec["set_id"] is None:
        return {i: "baseline_reference" for i in ctx["identity_ids"]}, {}
    entry = entry_by_id[spec["set_id"]]
    groups, operations = {}, {}
    for iid in ctx["identity_ids"]:
        assignment = entry["assignments"].get(iid)
        if assignment:
            operations[iid] = assignment["operation"]
            groups[iid] = ("refusal_control"
                           if assignment["operation"] == "refusal"
                           else "transformation_target")
        else:
            groups[iid] = gxm.control_group(ctx, entry, iid)
    return groups, operations


# ====================================================================== #
# Scoring: strict parsing + candidate support, per template
# ====================================================================== #
def score_route_h(session, panel, ctx, args):
    """Hard generation and full-sequence candidate support, per template.

    Soft scoring goes through ``rv.full_sequence_label_probs`` with
    ``prompt_text`` so route h stays on the identical code path that produced
    every canonical-prompt artifact already on disk.  Template results are
    therefore comparable to that evidence rather than measured on a second
    scorer.
    """
    import torch

    backend = session.backend()
    session.model.eval()
    vocab = panel["candidate_spaces"]["h"]["vocab"]
    rows = {}
    for role in TEMPLATE_ROLES:
        per_id = {}
        for row in panel["routes"]["h"]["rows"]:
            iid = row["identity_id"]
            prompt = row["prompts"][role]
            with torch.no_grad():
                gen = backend.generate(None, prompt,
                                       max_new_tokens=args.max_gen_tokens)
            raw = gen.text.strip()
            recognized = rv.recognized_labels_in(raw, vocab)
            parsed = rv.parse_recognized_label(raw, vocab)
            probs = rv.full_sequence_label_probs(
                session.adapter, session.model, session.processor,
                row["code"], vocab, args.device, prompt_text=prompt)
            prob_by_label = {l: probs.get(l, {}).get("prob", 0.0)
                             for l in vocab}
            summary = rv.build_candidate_summary(prob_by_label, vocab,
                                                 gx.DELETED_LABEL)
            per_id[iid] = {
                "raw": raw,
                "parsed_label": parsed,
                "recognized_labels": recognized,
                # Strict scoring: more than one distinct recognized label is
                # INVALID, never resolved to the first match.
                "multi_label_invalid": len(recognized) > 1,
                "unparseable": parsed is None,
                "distractor_echoed": (role == "distractor"
                                      and row["distractor_decoy"] in recognized),
                "probs": summary["probs"],
                "candidate_mass": summary["candidate_mass"],
                "other_mass": summary["other_mass"],
            }
        rows[role] = per_id
    return rows


def score_route_g(session, panel, args):
    """Image-conditioned hard generation and code-space candidate support.

    Uses ``backend.score_candidates``, which builds the multimodal prefix once
    and teacher-forces each candidate -- the same conditional-probability
    definition as route h's scorer, but the only one that can carry an image.
    The two scorers are cross-checked on the canonical template in PP1 so a
    difference in scale is measured rather than assumed away.
    """
    import torch

    backend = session.backend()
    session.model.eval()
    codes = panel["candidate_spaces"]["g"]["codes"]
    rows = {}
    for role in TEMPLATE_ROLES:
        per_id = {}
        for row in panel["routes"]["g"]["rows"]:
            iid = row["identity_id"]
            prompt = row["prompts"][role]
            image = rv._load_image(row["image_uri"])
            with torch.no_grad():
                gen = backend.generate(image, prompt,
                                       max_new_tokens=args.max_gen_tokens)
            raw = gen.text.strip()
            parsed = rv._extract_code(raw, codes)
            recognized = [t for t in raw.split()
                          if t.strip(".,!?;:'\"()[]{}") in codes]
            resp = backend.score_candidates(image, prompt, list(codes))
            prob_by_code = {s.candidate: math.exp(s.log_probability)
                            for s in (resp.candidate_scores or [])}
            # deleted_label=None: codes have no refusal label, so the
            # alias-only view collapses onto the candidate view.
            summary = rv.build_candidate_summary(prob_by_code, codes, None)
            per_id[iid] = {
                "raw": raw,
                "parsed_code": parsed,
                "recognized_codes": sorted(set(
                    t.strip(".,!?;:'\"()[]{}") for t in recognized)),
                "multi_label_invalid": len({t.strip(".,!?;:'\"()[]{}")
                                            for t in recognized}) > 1,
                "unparseable": parsed is None,
                "distractor_echoed": (
                    role == "distractor"
                    and row["distractor_decoy_code"] in recognized),
                "probs": summary["probs"],
                "candidate_mass": summary["candidate_mass"],
                "other_mass": summary["other_mass"],
            }
        rows[role] = per_id
    return rows


# ====================================================================== #
# Per-model result files: written on completion, skipped when present
# ====================================================================== #
def _result_path(ds, route, model_id):
    return PANEL_OUT_ROOT / ds / route / f"{model_id}.json"


def _load_cached(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                      # noqa: BLE001
            logger.warning("%s unreadable (%s); re-running", path, exc)
    return None


def evaluate_models(ds, route, specs, panel, ctx, args, entry_by_id, out_base):
    """Run one route over every in-scope checkpoint, resuming from disk.

    One ``ModelSession`` for the whole sweep: ``reset_to`` reloads the adapter
    in place, so 106 checkpoints cost one 9B model load rather than 106.
    """
    results, session, pending = {}, None, []
    todo = [s for s in specs
            if _load_cached(_result_path(ds, route, s["model_id"])) is None]
    for spec in specs:
        cached = _load_cached(_result_path(ds, route, spec["model_id"]))
        if cached is not None:
            results[spec["model_id"]] = cached
            logger.info("[%s] cached", spec["model_id"])
    if not todo:
        logger.info("%s: all %d models cached", route, len(specs))
        return results, pending

    missing = [s for s in todo if not s["present"]]
    for spec in missing:
        pending.append({"model_id": spec["model_id"],
                        "reason": f"checkpoint absent: {spec['checkpoint']}"})
        logger.warning("[%s] PENDING (no checkpoint)", spec["model_id"])
    todo = [s for s in todo if s["present"]]
    if not todo:
        return results, pending

    try:
        session = mx.ModelSession(args, f"e2c_pp_{ds}_{route}")
        for n, spec in enumerate(todo, 1):
            t0 = time.time()
            session.reset_to(spec["checkpoint"])
            mx.seed_everything(args.seed)
            if route == "h":
                expected, scope = expected_of_for(spec, ctx, entry_by_id)
                groups, operations = groups_for(spec, ctx, entry_by_id)
                rows = score_route_h(session, panel, ctx, args)
            else:
                expected = {i: ctx["code_of"][i] for i in ctx["identity_ids"]}
                scope, groups, operations = "all", {}, {}
                rows = score_route_g(session, panel, args)
            elapsed = time.time() - t0
            n_prompts = len(TEMPLATE_ROLES) * len(ctx["identity_ids"])
            record = {
                "model_id": spec["model_id"],
                "model_class": spec["model_class"],
                "set_id": spec["set_id"],
                "seed": spec["seed"],
                "route": route,
                "checkpoint": spec["checkpoint"],
                "checkpoint_sha256": rv.sha256_file(spec["checkpoint"]),
                "expected_of": expected,
                "accuracy_scope": scope,
                "groups": groups,
                "operations": operations,
                "template_roles": list(TEMPLATE_ROLES),
                "rows": rows,
                "timing": {"seconds": round(elapsed, 2),
                           "n_prompts": n_prompts,
                           "seconds_per_prompt": round(elapsed / n_prompts, 4)},
                "provenance": {"git_commit": EXECUTING_COMMIT,
                               "script_sha256": rv.script_sha256(),
                               "panel_sha256": panel["panel_sha256"],
                               "device": args.device},
            }
            path = _result_path(ds, route, spec["model_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False)
                            + "\n", encoding="utf-8")
            results[spec["model_id"]] = record
            logger.info("[%d/%d %s] %s  %.1fs (%.3fs/prompt)", n, len(todo),
                        route, spec["model_id"], elapsed,
                        elapsed / n_prompts)
    finally:
        if session is not None:
            session.release()
    return results, pending


# ====================================================================== #
# Scorer cross-check: the two conditional-probability paths must agree
# ====================================================================== #
def scorer_agreement(session, panel, ctx, args, tol=1e-9):
    """Compare route h's two scorers on the CANONICAL template only.

    ``rv.full_sequence_label_probs`` renders the chat template through the
    tokenizer; ``backend.score_candidates`` renders it through the processor.
    Route h uses the first (the path every existing artifact was produced by)
    and route g can only use the second (it is the one that carries an image),
    so if the two disagree, g's distances are on a different scale from h's and
    the report must say so instead of inviting a naive comparison.
    """
    import torch

    vocab = panel["candidate_spaces"]["h"]["vocab"]
    row = panel["routes"]["h"]["rows"][0]
    prompt = row["prompts"]["canonical"]
    backend = session.backend()
    with torch.no_grad():
        via_rv = rv.full_sequence_label_probs(
            session.adapter, session.model, session.processor, row["code"],
            vocab, args.device, prompt_text=prompt)
        resp = backend.score_candidates(None, prompt, list(vocab))
    via_backend = {s.candidate: math.exp(s.log_probability)
                   for s in (resp.candidate_scores or [])}
    diffs = {l: abs(via_rv.get(l, {}).get("prob", 0.0) - via_backend.get(l, 0.0))
             for l in vocab}
    worst = max(diffs.values()) if diffs else 0.0
    return {"identity_id": row["identity_id"], "template_role": "canonical",
            "max_abs_prob_difference": worst, "agrees": worst <= tol,
            "tolerance": tol,
            "worst_label": max(diffs, key=diffs.get) if diffs else None}


# ====================================================================== #
# Aggregation: route h
# ====================================================================== #
def _accuracy(rows, ids, expected_of):
    """Strict desired-label accuracy over ``ids``, or None when none apply."""
    scored = [i for i in ids if expected_of.get(i) is not None]
    if not scored:
        return None, 0
    ok = sum(1 for i in scored if rows[i]["parsed_label"] == expected_of[i])
    return ok / len(scored), len(scored)


def h_template_block(role_rows, expected_of, groups, entry, vocab):
    """One template role's metrics, in blocks that are never averaged together.

    Transformation targets, refusal controls and retained identities stay
    separate, which is the existing aggregation rule: a refusal control must
    not leak into a granularity headline.  Sibling accuracy is null -- not 1.0
    -- when a set has no retained sibling, with the coverage counts beside it.
    """
    ids = list(expected_of)

    def pick(group):
        return [i for i in ids if groups.get(i) == group]

    targets = pick("transformation_target")
    refusals = pick("refusal_control")
    retained = pick("retain")
    sibling = pick("sibling")
    cousin = pick("cousin")

    source_of = {i: a["source"] for i, a in (entry or {"assignments": {}})
                 ["assignments"].items()}
    leaked = [i for i in targets
              if source_of.get(i) in role_rows[i]["recognized_labels"]]
    p_source = [role_rows[i]["probs"].get(source_of[i], 0.0)
                for i in targets if i in source_of]
    p_desired = [role_rows[i]["probs"].get(expected_of[i], 0.0)
                 for i in targets if expected_of.get(i) is not None]

    target_acc, n_target = _accuracy(role_rows, targets, expected_of)
    refusal_acc, n_refusal = _accuracy(role_rows, refusals, expected_of)
    retain_acc, n_retain = _accuracy(role_rows, retained, expected_of)
    sib_acc, n_sib = _accuracy(role_rows, sibling, expected_of)
    cousin_acc, n_cousin = _accuracy(role_rows, cousin, expected_of)
    all_rows = [role_rows[i] for i in ids]
    return {
        "desired_label_accuracy": {
            "transformation_targets": target_acc,
            "refusal_controls": refusal_acc,
            "retained": retain_acc,
            "sibling": sib_acc,
            "cousin": cousin_acc,
        },
        "coverage": {"transformation_targets": n_target,
                     "refusal_controls": n_refusal, "retained": n_retain,
                     # Sibling coverage sits next to the sibling metric so a
                     # null is visibly "no control available" and not "passed".
                     "sibling_available": n_sib, "sibling_total": len(targets),
                     "cousin": n_cousin},
        "source_label_leakage": {
            "hard_leaked_targets": leaked,
            "hard_leak_rate": (len(leaked) / len(targets)) if targets else None,
            "max_p_source": max(p_source) if p_source else None,
            "min_p_desired": min(p_desired) if p_desired else None,
        },
        "candidate_support": {
            "min_candidate_mass": min(r["candidate_mass"] for r in all_rows),
            "max_other_mass": max(r["other_mass"] for r in all_rows),
            "unparseable_rate": sum(r["unparseable"] for r in all_rows)
            / len(all_rows),
            "multi_label_invalid_rate": sum(r["multi_label_invalid"]
                                            for r in all_rows) / len(all_rows),
            "distractor_echoed": [i for i in ids
                                  if role_rows[i]["distractor_echoed"]],
        },
    }


#: The metrics a worst case and a spread are taken over.  Each is a path into
#: the per-template block, with the direction that makes a value "worse".
H_WORST_METRICS = (
    ("desired_label_accuracy.transformation_targets", "min"),
    ("desired_label_accuracy.refusal_controls", "min"),
    ("desired_label_accuracy.retained", "min"),
    ("desired_label_accuracy.sibling", "min"),
    ("source_label_leakage.hard_leak_rate", "max"),
    ("source_label_leakage.max_p_source", "max"),
    ("source_label_leakage.min_p_desired", "min"),
    ("candidate_support.min_candidate_mass", "min"),
    ("candidate_support.unparseable_rate", "max"),
    ("candidate_support.multi_label_invalid_rate", "max"),
)


def _dig(block, path):
    node = block
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def worst_case_and_spread(per_template, metrics):
    """Worst template and maximum spread, over the PRE-FROZEN template set.

    The worst case is a min (or max, for a leakage-style metric) over all six
    roles with no ability to exclude one, and the spread is max - min.  A
    template that behaves badly therefore shows up in the headline rather than
    being averaged away by five that behave.
    """
    worst, spread, arg_worst = {}, {}, {}
    for path, direction in metrics:
        values = {role: _dig(per_template[role], path)
                  for role in TEMPLATE_ROLES}
        present = {r: v for r, v in values.items() if v is not None}
        if not present:
            worst[path] = None
            spread[path] = None
            arg_worst[path] = None
            continue
        if direction == "min":
            worst[path] = min(present.values())
            arg_worst[path] = min(present, key=present.get)
        else:
            worst[path] = max(present.values())
            arg_worst[path] = max(present, key=present.get)
        spread[path] = max(present.values()) - min(present.values())
    return {"template_worst_case": worst, "max_template_spread": spread,
            "worst_template_by_metric": arg_worst,
            "per_template_values": {
                path: {role: _dig(per_template[role], path)
                       for role in TEMPLATE_ROLES}
                for path, _ in metrics}}


def criteria_readonly(block):
    """The frozen GX criteria applied to ONE template, as read-only columns.

    Deliberately not a gate and deliberately not new thresholds: these are
    ``gx.PASS_CRITERIA`` unchanged, evaluated per template so a reader can see
    which templates would have passed the existing bar.  Nothing in the repo
    reads this, and the panel never promotes or blocks anything.
    """
    crit = gx.PASS_CRITERIA
    acc = block["desired_label_accuracy"]
    leak = block["source_label_leakage"]
    support = block["candidate_support"]
    checks = {
        "min_target_p_desired>=crit": (
            leak["min_p_desired"] is not None
            and leak["min_p_desired"] >= crit["min_target_p_desired"]),
        "max_target_p_source<=crit": (
            leak["max_p_source"] is not None
            and leak["max_p_source"] <= crit["max_target_p_source"]),
        "min_candidate_mass>=crit": (
            support["min_candidate_mass"] >= crit["min_candidate_mass"]),
        "retained_strict_accuracy==1.0": acc["retained"] == 1.0,
        "sibling_strict_accuracy==1.0_or_null": (
            acc["sibling"] is None or acc["sibling"] == 1.0),
        "unparseable_rate==0": support["unparseable_rate"] == 0.0,
        "multi_label_invalid_rate==0":
            support["multi_label_invalid_rate"] == 0.0,
    }
    return {"criteria": checks,
            "all_met": all(checks.values()),
            "source": "scripts/e2c_v3_granularity.py PASS_CRITERIA (unchanged)",
            "is_a_gate": False}


def _summary_from_probs(probs, vocab):
    return rv.build_candidate_summary(probs, vocab, gx.DELETED_LABEL)


def oracle_distances_by_template(cell_rec, matched_rec, loo_rec, vocab,
                                 entry):
    """D(edit, matched_retrain) and D(edit, loo_retrain), per template.

    Computed on the transformation targets only, from the stored candidate
    distributions, and gated on candidate mass exactly as everywhere else in
    this project: below the threshold the distance is null ("not established")
    rather than a small number that looks like proximity.
    """
    targets = [i for i, g in (cell_rec.get("groups") or {}).items()
               if g == "transformation_target"]
    out = {}
    for role in TEMPLATE_ROLES:
        per_target = {}
        for iid in targets:
            e_rows = cell_rec["rows"][role][iid]
            rec = {"matched_retrain": None, "loo_retrain": None,
                   "delta_retrain": None, "reliable": False, "reason": None}
            summaries = {"edit": _summary_from_probs(e_rows["probs"], vocab)}
            for fam, other in (("matched_retrain", matched_rec),
                               ("loo_retrain", loo_rec)):
                if other is None:
                    rec["reason"] = f"{fam} not evaluated"
                    continue
                summaries[fam] = _summary_from_probs(
                    other["rows"][role][iid]["probs"], vocab)
            if "matched_retrain" in summaries and "loo_retrain" in summaries:
                d_m, ok_m, why_m = rv.gated_distance(
                    summaries["edit"], summaries["matched_retrain"],
                    "candidate", vocab, MIN_CANDIDATE_MASS)
                d_l, ok_l, why_l = rv.gated_distance(
                    summaries["edit"], summaries["loo_retrain"],
                    "candidate", vocab, MIN_CANDIDATE_MASS)
                rec["matched_retrain"] = d_m["l2"] if d_m else None
                rec["loo_retrain"] = d_l["l2"] if d_l else None
                rec["reliable"] = bool(ok_m and ok_l)
                rec["reason"] = why_m or why_l
                if rec["reliable"]:
                    rec["delta_retrain"] = (rec["loo_retrain"]
                                            - rec["matched_retrain"])
            per_target[iid] = rec
        deltas = [v["delta_retrain"] for v in per_target.values()
                  if v["delta_retrain"] is not None]
        out[role] = {
            "per_target": per_target,
            "worst_case_delta_retrain": min(deltas) if deltas else None,
            "all_reliable": all(v["reliable"] for v in per_target.values())
            if per_target else False,
            "closer_to_matched_than_loo": all(
                v["delta_retrain"] > 0 for v in per_target.values()
                if v["delta_retrain"] is not None) if deltas else None,
        }
    return out


def aggregate_route_h(ds, results, pending, matrix, ctx, entry_by_id):
    """Per-cell blocks, worst cases, spreads and oracle distances."""
    vocab = ctx["vocab"]
    cells = {}
    for model_id, rec in results.items():
        if rec["model_class"] != "edited":
            continue
        sid, seed = rec["set_id"], rec["seed"]
        entry = entry_by_id[sid]
        per_template = {
            role: h_template_block(rec["rows"][role], rec["expected_of"],
                                   rec["groups"], entry, vocab)
            for role in TEMPLATE_ROLES}
        matched = results.get(f"matched_retrain__{sid}")
        loo = results.get(f"loo_retrain__{sid}")
        dist = oracle_distances_by_template(rec, matched, loo, vocab, entry)
        deltas = {role: dist[role]["worst_case_delta_retrain"]
                  for role in TEMPLATE_ROLES}
        present = [v for v in deltas.values() if v is not None]
        cells[model_id] = {
            "set_id": sid, "seed": seed, "mode": entry["mode"],
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "accuracy_scope": rec["accuracy_scope"],
            "per_template": per_template,
            **worst_case_and_spread(per_template, H_WORST_METRICS),
            "criteria_readonly": {role: criteria_readonly(per_template[role])
                                  for role in TEMPLATE_ROLES},
            "distance_to_oracle_by_template": dist,
            "delta_retrain_by_template": deltas,
            "delta_retrain_worst_case": min(present) if present else None,
            "delta_retrain_sign_preserved_across_templates": (
                all(v > 0 for v in present) if present else None),
            "oracles_evaluated": {"matched_retrain": matched is not None,
                                  "loo_retrain": loo is not None},
        }
    # Oracle and baseline blocks, so the sweep reports what it measured and not
    # only the derived cell view.
    others = {}
    for model_id, rec in results.items():
        if rec["model_class"] == "edited":
            continue
        entry = entry_by_id.get(rec["set_id"]) if rec["set_id"] else None
        per_template = {
            role: h_template_block(rec["rows"][role], rec["expected_of"],
                                   rec.get("groups") or {}, entry, vocab)
            for role in TEMPLATE_ROLES}
        others[model_id] = {
            "model_class": rec["model_class"], "set_id": rec["set_id"],
            "accuracy_scope": rec["accuracy_scope"],
            "per_template": per_template,
            **worst_case_and_spread(per_template, H_WORST_METRICS),
        }
    n_by_class = {}
    for rec in results.values():
        n_by_class[rec["model_class"]] = n_by_class.get(rec["model_class"], 0) + 1
    return {"route": "h", "dataset": ds,
            "n_models_evaluated": len(results),
            "n_models_by_class": n_by_class,
            "pending": pending,
            "expected_models": len(h_model_specs(ds, OUT_ROOT / ds, matrix,
                                                 _ALL_ARGS)) if _ALL_ARGS
            else None,
            "edited_cells": cells, "reference_models": others}


# ====================================================================== #
# Aggregation: route g
# ====================================================================== #
def _guard_g_vocabulary(text, where):
    """Refuse to emit a g-side claim that the design cannot support.

    There was no g-side intervention and no g-specific matched or LOO
    retraining, so "edit success", "g-side unlearning" and "retraining
    equivalence" would all be false statements about route g no matter what the
    numbers say.  Raising is better than a report that has to be walked back.
    """
    lowered = str(text).lower()
    hits = [t for t in PROHIBITED_G_TERMS if t in lowered]
    if hits:
        raise RuntimeError(
            f"{where}: route-g wording contains prohibited term(s) {hits}.  "
            f"Route g has no edited, matched or LOO variant -- only the frozen "
            f"router and h-side adapters measured against it -- so these terms "
            f"would claim an intervention that does not exist.")


def g_baseline_block(rec, ctx):
    """PP2: the frozen router's own prompt sensitivity."""
    ids = list(ctx["identity_ids"])
    per_template = {}
    for role in TEMPLATE_ROLES:
        rows = rec["rows"][role]
        correct = [i for i in ids if rows[i]["parsed_code"] == ctx["code_of"][i]]
        invalid = [i for i in ids if rows[i]["multi_label_invalid"]
                   or rows[i]["unparseable"]]
        off_support = [i for i in ids
                       if rows[i]["candidate_mass"] < MIN_CANDIDATE_MASS]
        per_template[role] = {
            "code_accuracy": len(correct) / len(ids),
            "n_correct": len(correct), "n": len(ids),
            "invalid_rate": len(invalid) / len(ids),
            "invalid_ids": invalid,
            "off_support_rate": len(off_support) / len(ids),
            "off_support_ids": off_support,
            "min_candidate_mass": min(rows[i]["candidate_mass"] for i in ids),
            "distractor_echoed": [i for i in ids
                                  if rows[i]["distractor_echoed"]],
        }
    accuracies = {r: per_template[r]["code_accuracy"] for r in TEMPLATE_ROLES}
    # Six-template consistency: does the router give the SAME code for an image
    # however it is asked?  Agreement is measured per identity across all six.
    agree = [i for i in ids
             if len({rec["rows"][r][i]["parsed_code"] for r in TEMPLATE_ROLES})
             == 1]
    worst_role = min(accuracies, key=accuracies.get)
    return {
        "reference": "frozen g_X_to_C router (the only trained visual router)",
        "per_template": per_template,
        "code_accuracy_by_template": accuracies,
        "six_template_consistency": len(agree) / len(ids),
        "six_template_consistent_ids": agree,
        "worst_template": worst_role,
        "worst_template_code_accuracy": accuracies[worst_role],
        "max_template_spread": max(accuracies.values()) - min(accuracies.values()),
    }


def g_spillover_block(rec, frozen_rec, ctx, entry_by_id):
    """PP3: one h-side adapter on the route-g panel, vs FROZEN BASE g.

    The confirmatory reference is always the frozen router.  Everything here is
    a deviation from it, so a zero flip rate and a zero accuracy change is the
    null expectation: an edit aimed at h(C->A) should leave g(X->C) alone.
    """
    ids = list(ctx["identity_ids"])
    entry = entry_by_id.get(rec["set_id"]) if rec["set_id"] else None
    targets = set(entry["assignments"]) if entry else set()
    per_template = {}
    for role in TEMPLATE_ROLES:
        rows, base = rec["rows"][role], frozen_rec["rows"][role]
        flips = [i for i in ids if rows[i]["parsed_code"] != base[i]["parsed_code"]]
        acc = sum(1 for i in ids if rows[i]["parsed_code"] == ctx["code_of"][i])
        base_acc = sum(1 for i in base
                       if base[i]["parsed_code"] == ctx["code_of"][i])
        dists, unreliable = {}, []
        for iid in ids:
            m = rv.build_candidate_summary(rows[iid]["probs"],
                                           list(ctx["code_of"].values()), None)
            b = rv.build_candidate_summary(base[iid]["probs"],
                                           list(ctx["code_of"].values()), None)
            d, ok, why = rv.gated_distance(
                m, b, "candidate", list(ctx["code_of"].values()),
                MIN_CANDIDATE_MASS)
            if ok:
                dists[iid] = d["l2"]
            else:
                dists[iid] = None
                unreliable.append({"identity_id": iid, "reason": why})
        per_template[role] = {
            "prediction_flip_rate": len(flips) / len(ids),
            "flipped_ids": flips,
            "code_accuracy": acc / len(ids),
            "code_accuracy_change_vs_frozen_g": (acc - base_acc) / len(ids),
            "mean_candidate_distance_vs_frozen_g": (
                sum(v for v in dists.values() if v is not None)
                / len([v for v in dists.values() if v is not None])
                if any(v is not None for v in dists.values()) else None),
            "max_candidate_distance_vs_frozen_g": max(
                [v for v in dists.values() if v is not None], default=None),
            "distance_not_established": unreliable,
            "target_person": {
                "n": len(targets & set(ids)),
                "flip_rate": (len([i for i in flips if i in targets])
                              / len(targets)) if targets else None,
            },
            "retained_person": {
                "n": len(set(ids) - targets),
                "flip_rate": (len([i for i in flips if i not in targets])
                              / len(set(ids) - targets))
                if (set(ids) - targets) else None,
            },
            "distractor_echoed": [i for i in ids
                                  if rows[i]["distractor_echoed"]],
        }
    flips = {r: per_template[r]["prediction_flip_rate"] for r in TEMPLATE_ROLES}
    worst_role = max(flips, key=flips.get)
    return {
        "model_class": rec["model_class"], "set_id": rec["set_id"],
        "seed": rec["seed"], "checkpoint_sha256": rec["checkpoint_sha256"],
        "confirmatory_reference": "frozen_g (g_X_to_C)",
        "null_expectation": ("an edit targeting h(C->A) leaves g(X->C) "
                             "approximately unchanged: flip rate 0, accuracy "
                             "change 0"),
        "per_template": per_template,
        "prediction_flip_rate_by_template": flips,
        "worst_template": worst_role,
        "worst_template_flip_rate": flips[worst_role],
        "max_template_spread_flip_rate": max(flips.values()) - min(flips.values()),
        "worst_template_accuracy_change": min(
            per_template[r]["code_accuracy_change_vs_frozen_g"]
            for r in TEMPLATE_ROLES),
    }
