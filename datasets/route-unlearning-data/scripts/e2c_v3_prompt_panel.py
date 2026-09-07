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
import hashlib
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
def _roles(args):
    """The template roles this invocation evaluates: all six, unless --smoke.

    ``--only-templates`` exists so the smoke run is cheap.  It is refused
    outside ``--smoke`` and its output goes to a separate ``*_smoke`` tree,
    because the report's worst case is a min over the SIX pre-frozen roles and
    a partial panel must never be aggregable into it.
    """
    chosen = getattr(args, "only_templates", None)
    return list(chosen) if chosen else list(TEMPLATE_ROLES)


def score_route_h(session, panel, args):
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
    for role in _roles(args):
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
    for role in _roles(args):
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
            # Whitespace-split then punctuation-stripped, longest match wins:
            # codes are "SAL_<iid>", so a code embedded in prose is still found.
            found = sorted({t.strip(".,!?;:'\"()[]{}") for t in raw.split()
                            if t.strip(".,!?;:'\"()[]{}") in codes})
            resp = backend.score_candidates(image, prompt, list(codes))
            prob_by_code = {s.candidate: math.exp(s.log_probability)
                            for s in (resp.candidate_scores or [])}
            # deleted_label=None: codes have no refusal label, so the
            # alias-only view collapses onto the candidate view.
            summary = rv.build_candidate_summary(prob_by_code, codes, None)
            per_id[iid] = {
                "raw": raw,
                "parsed_code": parsed,
                "recognized_codes": found,
                # Strict scoring: more than one distinct recognized code is
                # INVALID, never resolved to the first match.
                "multi_label_invalid": len(found) > 1,
                "unparseable": parsed is None,
                "distractor_echoed": (role == "distractor"
                                      and row["distractor_decoy_code"] in found),
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
        except Exception as exc:
            logger.warning("%s unreadable (%s); re-running", path, exc)
    return None


def evaluate_models(ds, route, specs, panel, ctx, args, entry_by_id,
                    after_first=None):
    """Run one route over every in-scope checkpoint, resuming from disk.

    One ``ModelSession`` for the whole sweep: ``reset_to`` reloads the adapter
    in place, so 106 checkpoints cost one 9B model load rather than 106.

    ``after_first`` is called once, with the session positioned on the first
    checkpoint actually evaluated, and its return value is handed back as the
    third element.  That is how the scorer cross-check rides along with the
    sweep instead of paying for a second model load of its own.
    """
    results, session, pending, extras = {}, None, [], None
    roles = _roles(args)
    for spec in specs:
        cached = _load_cached(_result_path(ds, route, spec["model_id"]))
        if cached is not None:
            results[spec["model_id"]] = cached
    logger.info("%s: %d/%d models already on disk", route, len(results),
                len(specs))
    todo = [s for s in specs if s["model_id"] not in results]
    if not todo:
        return results, pending, extras

    missing = [s for s in todo if not s["present"]]
    for spec in missing:
        pending.append({"model_id": spec["model_id"],
                        "reason": f"checkpoint absent: {spec['checkpoint']}"})
        logger.warning("[%s] PENDING (no checkpoint)", spec["model_id"])
    todo = [s for s in todo if s["present"]]
    if not todo:
        return results, pending, extras

    try:
        session = mx.ModelSession(args, f"e2c_pp_{ds}_{route}")
        for n, spec in enumerate(todo, 1):
            t0 = time.time()
            session.reset_to(spec["checkpoint"])
            mx.seed_everything(args.seed)
            if after_first is not None and n == 1:
                extras = after_first(session)
            if route == "h":
                expected, scope = expected_of_for(spec, ctx, entry_by_id)
                groups, operations = groups_for(spec, ctx, entry_by_id)
                rows = score_route_h(session, panel, args)
            else:
                expected = {i: ctx["code_of"][i] for i in ctx["identity_ids"]}
                scope, groups, operations = "all", {}, {}
                rows = score_route_g(session, panel, args)
            elapsed = time.time() - t0
            n_prompts = len(roles) * len(ctx["identity_ids"])
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
                "template_roles": roles,
                "rows": rows,
                "timing": {"seconds": round(elapsed, 2),
                           "n_prompts": n_prompts,
                           "seconds_per_prompt": round(elapsed / n_prompts, 4)},
                "provenance": {
                    "git_commit": EXECUTING_COMMIT,
                    # Named as the granularity runner names them: the RUNNER
                    # that produced this record, and the shared scoring
                    # library it scored through.  rv.script_sha256() is the
                    # library's own hash, so calling it "script_sha256" here
                    # would have pointed a reader at the wrong file.
                    "runner_script_sha256": rv.sha256_file(
                        Path(__file__).resolve()),
                    "shared_scoring_script_sha256": rv.script_sha256(),
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
    return results, pending, extras


# ====================================================================== #
# Scorer cross-check: the two conditional-probability paths must agree
# ====================================================================== #
def scorer_agreement(session, panel, ctx, args, tol=1e-9):
    """Compare route h's two scorers on the CANONICAL template only.

    ``rv.full_sequence_label_probs`` renders the chat template through the
    TOKENIZER; ``backend.score_candidates`` renders it through the PROCESSOR.
    Route h uses the first (the path every existing artifact was produced by)
    and route g can only use the second (it is the one that carries an image),
    so if the two disagree, g's distances are on a different scale from h's and
    the report must say so instead of inviting a naive comparison.

    The rendered prompt ids are compared as well as the probabilities, because
    "the numbers differ" is only actionable once you know whether the two paths
    even asked the model the same question.
    """
    import torch

    vocab = panel["candidate_spaces"]["h"]["vocab"]
    row = panel["routes"]["h"]["rows"][0]
    prompt = row["prompts"]["canonical"]
    backend = session.backend()
    rv_ids = rv._build_prompt_ids(session.processor, row["code"],
                                  prompt_text=prompt).tolist()
    prefix = backend._build_prefix(None, prompt)
    be_ids = prefix["input_ids"][0].tolist()
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
            "prompt_text": prompt,
            "prompt_ids_identical": rv_ids == be_ids,
            "prompt_ids_len_rv": len(rv_ids),
            "prompt_ids_len_backend": len(be_ids),
            "max_abs_prob_difference": worst, "agrees": worst <= tol,
            "tolerance": tol,
            "worst_label": max(diffs, key=diffs.get) if diffs else None,
            "interpretation": (
                "route h and route g use different renderers for the same "
                "prompt text; this measures whether their candidate "
                "probabilities are on one scale, so an h distance and a g "
                "distance are only comparable if 'agrees' is true")}


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
            "distractor_echo_rate": sum(
                r["distractor_echoed"] for r in all_rows) / len(all_rows),
            # Does this template's output distribution meet the project's OWN
            # frozen support criterion (gx.PASS_CRITERIA["min_candidate_mass"])?
            # Not a new threshold and not a gate: it separates templates whose
            # renormalized distances mean something from templates where every
            # model has fallen off the candidate space and a distance is noise.
            "on_support": (min(r["candidate_mass"] for r in all_rows)
                           >= gx.PASS_CRITERIA["min_candidate_mass"]),
            "off_support_ids": [
                i for i in ids
                if role_rows[i]["candidate_mass"]
                < gx.PASS_CRITERIA["min_candidate_mass"]],
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
    ("candidate_support.distractor_echo_rate", "max"),
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


def support_profile_by_class(results):
    """Per model class and template: does output stay on the candidate space?

    This decides how every other number should be read.  If only the EDITED
    models fall off the candidate space away from the canonical prompt, that is
    a property of the edit.  If the unedited baseline and both retrain
    references fall off it too, it is a property of how route h is TRAINED, and
    the panel cannot separate the two -- a much narrower claim, and the honest
    one.  Reporting the second case as the first would blame the unlearning
    operation for the route.

    Uses ``gx.PASS_CRITERIA["min_candidate_mass"]`` -- the project's own frozen
    criterion, not a new threshold and not a gate.
    """
    crit = gx.PASS_CRITERIA["min_candidate_mass"]
    by_class = {}
    for rec in results.values():
        by_class.setdefault(rec["model_class"], []).append(rec)
    profile = {}
    for klass, recs in sorted(by_class.items()):
        per_template = {}
        for role in TEMPLATE_ROLES:
            usable = [r for r in recs if role in (r.get("rows") or {})]
            if not usable:
                continue
            masses, unparsable, multi, echoed, n_rows = [], 0, 0, 0, 0
            ok, scored = 0, 0
            for rec in usable:
                expected = rec["expected_of"]
                for iid, row in rec["rows"][role].items():
                    masses.append(row["candidate_mass"])
                    unparsable += bool(row["unparseable"])
                    multi += bool(row["multi_label_invalid"])
                    echoed += bool(row["distractor_echoed"])
                    n_rows += 1
                    if expected.get(iid) is not None:
                        scored += 1
                        ok += row["parsed_label"] == expected[iid]
            per_template[role] = {
                "n_models": len(usable), "n_rows": n_rows,
                "mean_candidate_mass": (sum(masses) / len(masses)
                                        if masses else None),
                "min_candidate_mass": min(masses) if masses else None,
                "strict_accuracy": (ok / scored) if scored else None,
                "n_scored": scored,
                "unparseable_rate": unparsable / n_rows if n_rows else None,
                "multi_label_invalid_rate": multi / n_rows if n_rows else None,
                "distractor_echo_rate": echoed / n_rows if n_rows else None,
                "meets_frozen_support_criterion": (
                    bool(masses) and min(masses) >= crit),
            }
        profile[klass] = per_template

    def classes_on(role):
        return sorted(k for k, pt in profile.items()
                      if pt.get(role, {}).get(
                          "meets_frozen_support_criterion"))

    def classes_off(role):
        return sorted(k for k, pt in profile.items()
                      if role in pt
                      and not pt[role]["meets_frozen_support_criterion"])

    baseline_off = [r for r in TEMPLATE_ROLES if "baseline" in classes_off(r)]
    edited_off = [r for r in TEMPLATE_ROLES if "edited" in classes_off(r)]
    # Edit-specific would mean: the edited models are off support on a template
    # where the never-edited baseline is still on it.
    edit_specific = [r for r in edited_off if r not in baseline_off]
    return {
        "criterion": ("min over that template's rows of candidate_mass >= "
                      f"gx.PASS_CRITERIA['min_candidate_mass'] = {crit}"),
        "is_a_gate": False,
        "per_class": profile,
        "classes_on_support_by_template": {
            r: classes_on(r) for r in TEMPLATE_ROLES},
        "classes_off_support_by_template": {
            r: classes_off(r) for r in TEMPLATE_ROLES},
        "templates_where_the_baseline_is_off_support": baseline_off,
        "templates_where_only_edited_models_are_off_support": edit_specific,
        "collapse_is_edit_specific": bool(edit_specific),
        "reading": (
            "off-support templates are ones where the model's output mass has "
            "left the candidate space, so a renormalized distance there "
            "compares two artifacts rather than two behaviors; where the "
            "never-edited baseline is off support too, the collapse is a "
            "property of how the route was trained and this panel cannot "
            "attribute it to the edit"),
    }


def aggregate_route_h(ds, results, pending, expected_models, ctx,
                      entry_by_id):
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
        # Which templates this cell's OWN output stayed on the candidate space
        # for, by the project's frozen criterion.  A renormalized distance on a
        # template outside this list compares two artifacts rather than two
        # behaviors, so the unrestricted numbers are kept -- nothing is hidden
        # -- but the interpretable subset is reported beside them.
        on_support = [role for role in TEMPLATE_ROLES
                      if per_template[role]["candidate_support"]["on_support"]]
        for role in TEMPLATE_ROLES:
            dist[role]["cell_on_support"] = role in on_support
        supported = {role: deltas[role] for role in on_support}
        supported_present = [v for v in supported.values() if v is not None]
        cells[model_id] = {
            "set_id": sid, "seed": seed, "mode": entry["mode"],
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "accuracy_scope": rec["accuracy_scope"],
            "per_template": per_template,
            **worst_case_and_spread(per_template, H_WORST_METRICS),
            "criteria_readonly": {role: criteria_readonly(per_template[role])
                                  for role in TEMPLATE_ROLES},
            "templates_on_support": on_support,
            "n_templates_on_support": len(on_support),
            "distance_to_oracle_by_template": dist,
            "delta_retrain_by_template": deltas,
            "delta_retrain_worst_case": min(present) if present else None,
            "delta_retrain_sign_preserved_across_templates": (
                all(v > 0 for v in present) if present else None),
            "delta_retrain_on_supported_templates": supported,
            "delta_retrain_worst_case_on_supported_templates": (
                min(supported_present) if supported_present else None),
            "delta_retrain_sign_preserved_on_supported_templates": (
                all(v > 0 for v in supported_present)
                if supported_present else None),
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
            # In scope but not on disk: an incomplete sweep must be visible as
            # incomplete rather than looking like a smaller matrix that passed.
            "expected_models": expected_models,
            "complete": expected_models is None or len(results) >= expected_models,
            # Read this before any per-template number: it says which templates
            # every model class stayed on the candidate space for, and whether
            # a collapse is specific to the edit or shared with the baseline.
            "prompt_support_profile": support_profile_by_class(results),
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


#: How the secondary cross-route diagnostic must be labelled wherever it
#: appears.  The name and the label travel together on purpose: a bare
#: distance between an edited adapter and a matched/LOO adapter invites the
#: reading "oracle distance", which for route g would be false.
G_REFERENCE_LABEL = ("cross-route reference comparison, NOT an oracle "
                     "distance: route g has no matched or LOO retraining")


def g_reference_distances(rec, matched_rec, loo_rec, ctx):
    """Secondary diagnostic: h-side adapters compared ON g prompts.

    Both operands are h-side adapters (an edited cell and its set's
    matched/LOO retrain references) measured on the route-g panel.  It answers
    a spillover question -- does the h-side edit/retrain separation show up on
    image prompts at all -- and is never a retraining claim about g, because
    nothing on the g side was retrained.
    """
    codes = list(ctx["code_of"].values())
    ids = list(ctx["identity_ids"])
    per_template = {}
    for role in TEMPLATE_ROLES:
        row = {}
        for fam, other in (("matched_retrain", matched_rec),
                           ("loo_retrain", loo_rec)):
            if other is None:
                row[fam] = None
                continue
            dists = []
            for iid in ids:
                a = rv.build_candidate_summary(
                    rec["rows"][role][iid]["probs"], codes, None)
                b = rv.build_candidate_summary(
                    other["rows"][role][iid]["probs"], codes, None)
                d, ok, _why = rv.gated_distance(
                    a, b, "candidate", codes, MIN_CANDIDATE_MASS)
                dists.append(d["l2"] if ok else None)
            established = [v for v in dists if v is not None]
            row[fam] = {
                "mean_l2": (sum(established) / len(established)
                            if established else None),
                "max_l2": max(established) if established else None,
                "n_established": len(established),
                "n_not_established": len(dists) - len(established),
            }
        m, l = row["matched_retrain"], row["loo_retrain"]
        row["delta_g_reference"] = (
            l["mean_l2"] - m["mean_l2"]
            if (m and l and m["mean_l2"] is not None
                and l["mean_l2"] is not None) else None)
        per_template[role] = row
    deltas = [r["delta_g_reference"] for r in per_template.values()]
    present = [v for v in deltas if v is not None]
    return {
        "metric_name": "g_spillover_reference_distance",
        "label": G_REFERENCE_LABEL,
        "is_an_oracle_distance": False,
        "per_template": per_template,
        "delta_g_reference_by_template": dict(zip(TEMPLATE_ROLES, deltas)),
        "worst_case_delta_g_reference": min(present) if present else None,
    }


def _guard_g_tree(obj, where):
    """Apply :func:`_guard_g_vocabulary` to every string in a nested structure.

    Guarding the whole g-side tree rather than a hand-picked list of fields
    means a future field cannot introduce a prohibited claim by being added
    somewhere nobody thought to check.
    """
    if isinstance(obj, str):
        _guard_g_vocabulary(obj, where)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _guard_g_tree(value, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for n, value in enumerate(obj):
            _guard_g_tree(value, f"{where}[{n}]")


# ====================================================================== #
# Result loading and reporting plumbing (CPU only)
# ====================================================================== #
def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def _hms(seconds):
    total = round(seconds)
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours}h{mins:02d}m{secs:02d}s" if hours else f"{mins}m{secs:02d}s"


def _gpu_name():
    try:
        import torch
        return (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu")
    except Exception:
        return "unknown"


def _gran_out_base(ds):
    """The committed granularity outputs the checkpoints are read from.

    This panel trains nothing, so it reads the SAME checkpoint tree the
    granularity matrix wrote -- never a panel-local copy that could drift.
    """
    return gxm.OUT_ROOT / ds


def _panel_out_ds(ds, args):
    """Where panel RESULTS go.

    Smoke output is separated by dataset name so a partial panel (one
    template) can never be picked up by PP4 and aggregated into a report whose
    worst case claims to be a min over all six.
    """
    return f"{ds}_smoke" if args.smoke else ds


def load_route_results(ds_out, route, specs):
    """Read the stored per-model files for a route.  No GPU, no model load."""
    results, pending = {}, []
    for spec in specs:
        rec = _load_cached(_result_path(ds_out, route, spec["model_id"]))
        if rec is None:
            pending.append({"model_id": spec["model_id"],
                            "reason": "no result file on disk"})
        else:
            results[spec["model_id"]] = rec
    return results, pending


def assert_full_panel(results_by_route):
    """Refuse to aggregate anything measured on fewer than the six templates.

    ``template_worst_case`` is only a worst case if every frozen template is
    in it.  A record produced by a smoke run carries a shorter role list, and
    aggregating it would silently report a min over a subset as a min over the
    panel.
    """
    want = list(TEMPLATE_ROLES)
    short = []
    for route, results in results_by_route.items():
        for model_id, rec in results.items():
            if list(rec.get("template_roles") or []) != want:
                short.append(f"{route}/{model_id}: "
                             f"{rec.get('template_roles')}")
    if short:
        raise RuntimeError(
            f"refusing to aggregate {len(short)} result file(s) that do not "
            f"cover all {len(want)} frozen templates: {short[:5]}.  These are "
            f"smoke or pilot outputs; a worst case over a subset of the panel "
            f"is not the panel's worst case.")


def _pilot_specs(ds_out, route, specs, n):
    """Truncate to the first ``n`` NOT-YET-EVALUATED models (cached ones free).

    A pilot exists to measure cost, so it must run models that have no timing
    yet; re-running cached ones would measure nothing and report a fake ETA.
    """
    cached = {s["model_id"] for s in specs
              if _load_cached(_result_path(ds_out, route, s["model_id"]))
              is not None}
    todo = [s for s in specs if s["model_id"] not in cached]
    keep = cached | {s["model_id"] for s in todo[:n]}
    return [s for s in specs if s["model_id"] in keep]


def _shard(specs, args):
    """This process's slice of the sweep: every ``shard_count``-th model.

    Sharding is what makes running several processes at once safe.  The slices
    are DISJOINT and their union is the whole sweep, so two writers never share
    a result file -- which means no lock, no partial-write race, and no
    duplicated GPU-hours.  That matters more than it looks: results are cached
    by presence, so two processes racing on one model would not merely repeat
    work, they would interleave writes into one JSON file and leave a record
    that is corrupt but present, and therefore skipped forever.

    Stride rather than contiguous blocks so every shard gets the same mix of
    edited cells and retrain oracles.  A contiguous split would hand one
    process all 63 cells and another all 42 oracles, and the two would finish
    hours apart for no reason.
    """
    count = args.shard_count
    if count <= 1:
        return list(specs)
    if not 0 <= args.shard_index < count:
        raise RuntimeError(
            f"--shard-index must be in [0, {count}) (got {args.shard_index}); "
            f"an out-of-range index would silently evaluate nothing and the "
            f"sweep would look complete when it was not")
    return [s for i, s in enumerate(specs) if i % count == args.shard_index]


def _shard_tag(args):
    """Suffix for artifacts ONE process owns, so parallel shards do not clash.

    The run manifest and the scorer cross-check are read-modify-write or
    write-once files shared by the whole sweep.  Per-model results are already
    disjoint by construction; these two are not, so each shard writes its own
    and PP4 collects the set.
    """
    if args.shard_count <= 1:
        return ""
    return f".shard{args.shard_index}of{args.shard_count}"


def _shard_label(args):
    return ("unsharded" if args.shard_count <= 1
            else f"shard {args.shard_index}/{args.shard_count}")


def _shard_artifacts(ds_out, stem):
    """Every shard's copy of a shared artifact, sorted for determinism."""
    directory = PANEL_OUT_ROOT / ds_out
    return sorted(directory.glob(f"{stem}*.json")) if directory.exists() else []


def print_eta(route, specs, ds_out, label="unsharded"):
    """Turn measured seconds-per-prompt into a projected wall clock.

    Uses timings already on disk, so it is exact rather than an estimate from
    a guess about forward-pass cost, and it can be re-read at any time with
    ``--eta-only`` on CPU.  ``specs`` is THIS shard's slice, so the projection
    is the wall clock of the process printing it, not of the whole sweep.
    """
    measured, remaining, n_prompts = [], [], None
    for spec in specs:
        rec = _load_cached(_result_path(ds_out, route, spec["model_id"]))
        if rec is None:
            remaining.append(spec["model_id"])
            continue
        timing = rec.get("timing") or {}
        if timing.get("seconds_per_prompt"):
            measured.append(timing["seconds_per_prompt"])
            n_prompts = timing.get("n_prompts", n_prompts)
    if not measured or not n_prompts:
        logger.info("%s ETA: nothing measured yet -- run --pilot 2 for a real "
                    "number instead of an estimate", route)
        return {"route": route, "projected_seconds": None,
                "reason": "no measured timings on disk"}
    mean = sum(measured) / len(measured)
    total = mean * n_prompts * len(remaining)
    eta = {"route": route, "shard": label,
           "models_measured": len(measured),
           "models_remaining": len(remaining),
           "mean_seconds_per_prompt": round(mean, 4),
           "prompts_per_model": n_prompts,
           "projected_seconds": round(total, 1),
           "projected_hms": _hms(total),
           "excludes": "one 9B model load per phase (~1-2 min), not per model"}
    logger.info("%s ETA (%s): %d models left x %d prompts x %.3fs/prompt = %s",
                route, label, len(remaining), n_prompts, mean,
                eta["projected_hms"])
    return eta


# ====================================================================== #
# Phase runners
# ====================================================================== #
def run_pp1(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start):
    """Route h: the full matrix -- baseline, edited cells, retrain oracles."""
    ds_out = _panel_out_ds(ds, args)
    label = _shard_label(args)
    whole = h_model_specs(ds, _gran_out_base(ds), matrix, args)
    specs = _shard(whole, args)
    if args.pilot:
        before = len(specs)
        specs = _pilot_specs(ds_out, "h", specs, args.pilot)
        logger.info("PP1 pilot (%s): %d of %d shard models", label,
                    len(specs), before)
    logger.info("PP1 (%s): %d of %d route-h models in scope",
                label, len(specs), len(whole))
    agreement_path = PANEL_OUT_ROOT / ds_out / \
        f"scorer_agreement{_shard_tag(args)}.json"
    cached_agreement = _load_cached(agreement_path)

    def after_first(session):
        if cached_agreement is not None:
            logger.info("scorer agreement: cached (%s)",
                        "agrees" if cached_agreement.get("agrees")
                        else "DIFFERS")
            return cached_agreement
        rec = scorer_agreement(session, panel, ctx, args)
        _write_json(agreement_path, rec)
        logger.info("scorer agreement on canonical: prompt_ids_identical=%s "
                    "max_abs_prob_diff=%.3e agrees=%s",
                    rec["prompt_ids_identical"],
                    rec["max_abs_prob_difference"], rec["agrees"])
        if not rec["agrees"]:
            logger.warning(
                "the two conditional-probability paths DISAGREE beyond %g.  "
                "Route h keeps using rv.full_sequence_label_probs (the path "
                "every existing artifact used) and route g can only use "
                "backend.score_candidates, so their distances are reported as "
                "separate blocks and are not directly comparable.",
                rec["tolerance"])
        return rec

    results, pending, extras = evaluate_models(
        ds_out, "h", specs, panel, ctx, args, entry_by_id,
        after_first=after_first)
    update_run_manifest(ds, ds_out, panel, provenance, "h", results, t_start,
                        args)
    eta = print_eta("h", specs, ds_out, label)
    logger.info("PP1 (%s): %d/%d whole-sweep route-h models now on disk, "
                "%d pending in this shard", label, len(results), len(whole),
                len(pending))
    return {"results": results, "pending": pending, "agreement": extras,
            "eta": eta, "specs": whole}


def run_pp2(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start):
    """Route g baseline: the frozen router's own prompt sensitivity.

    The frozen router is a single model, so under sharding exactly one shard
    owns it and the others find an empty slice and return at once.  That is the
    intended behaviour: the router is the confirmatory reference and must be
    written by one process, not several.
    """
    ds_out = _panel_out_ds(ds, args)
    label = _shard_label(args)
    all_g = g_model_specs(ds, _gran_out_base(ds), matrix, args)
    specs = _shard([s for s in all_g if s["model_class"] == "frozen_g"], args)
    if args.pilot:
        specs = _pilot_specs(ds_out, "g", specs, args.pilot)
    if not specs:
        logger.info("PP2 (%s): frozen router belongs to another shard; "
                    "nothing to do", label)
        return {"results": {}, "pending": []}
    results, pending, _ = evaluate_models(
        ds_out, "g", specs, panel, ctx, args, entry_by_id)
    update_run_manifest(ds, ds_out, panel, provenance, "g", results, t_start,
                        args)
    print_eta("g", _shard(all_g, args), ds_out, label)
    logger.info("PP2 (%s): frozen router evaluated over %d templates x %d "
                "images", label, len(_roles(args)), len(ctx["identity_ids"]))
    return {"results": results, "pending": pending}


def run_pp3(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start):
    """Route g spillover: every in-scope h-side adapter on the g panel.

    The spec list includes the frozen router, so PP3 is self-sufficient: run
    after PP2 it costs nothing extra (cached), run alone it still has its
    confirmatory reference.
    """
    ds_out = _panel_out_ds(ds, args)
    label = _shard_label(args)
    whole = g_model_specs(ds, _gran_out_base(ds), matrix, args)
    specs = _shard(whole, args)
    if args.pilot:
        before = len(specs)
        specs = _pilot_specs(ds_out, "g", specs, args.pilot)
        logger.info("PP3 pilot (%s): %d of %d shard models", label,
                    len(specs), before)
    results, pending, _ = evaluate_models(
        ds_out, "g", specs, panel, ctx, args, entry_by_id)
    update_run_manifest(ds, ds_out, panel, provenance, "g", results, t_start,
                        args)
    print_eta("g", specs, ds_out, label)
    logger.info("PP3 (%s): %d/%d whole-sweep route-g models now on disk, "
                "%d pending in this shard", label, len(results), len(whole),
                len(pending))
    return {"results": results, "pending": pending}


#: Fields that legitimately differ between two aggregations of the SAME stored
#: results (wall clock, which process, which device).  Excluded from the core
#: digest so PPR can answer the question it exists for: is the report a
#: function of the stored per-model files alone?
VOLATILE_REPORT_KEYS = ("generated_at", "elapsed_sec", "provenance", "gpu",
                        "reaggregation")


def _report_core_sha(report):
    body = {k: v for k, v in report.items() if k not in VOLATILE_REPORT_KEYS}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _reaggregation_check(path, report):
    """Did re-aggregating the stored files on CPU reproduce the report?"""
    core = _report_core_sha(report)
    previous = _load_cached(path)
    prev_core = _report_core_sha(previous) if previous else None
    return {"previous_report_found": previous is not None,
            "previous_core_sha256": prev_core,
            "reaggregated_core_sha256": core,
            "identical": prev_core == core,
            "volatile_fields_excluded": list(VOLATILE_REPORT_KEYS),
            "note": ("PPR re-derives the report from the stored per-model "
                     "files on CPU with no checkpoint in sight; identical "
                     "core digests mean the report is a function of those "
                     "files alone and can be revised without a GPU")}


def run_pp4(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start,
            cpu_only=False):
    """Aggregate everything on disk into the report.  Never loads a model.

    Deliberately UNSHARDED: it reads the whole spec list for both routes no
    matter what ``--shard-*`` says, because the report describes the sweep and
    not one process's slice of it.  A shard that never finished shows up as
    ``pending`` and ``complete: false`` rather than as a smaller matrix.
    """
    ds_out = _panel_out_ds(ds, args)
    out_base = _gran_out_base(ds)
    h_specs = h_model_specs(ds, out_base, matrix, args)
    g_specs = g_model_specs(ds, out_base, matrix, args)
    h_results, h_pending = load_route_results(ds_out, "h", h_specs)
    g_results, g_pending = load_route_results(ds_out, "g", g_specs)
    assert_full_panel({"h": h_results, "g": g_results})
    if not h_results and not g_results:
        raise RuntimeError(
            f"{ds_out}: nothing to aggregate; run PP1/PP2/PP3 first")

    frozen = g_results.get("frozen_g")
    if g_results and frozen is None:
        raise RuntimeError(
            "route g has results but no frozen_g record.  Frozen base g is the "
            "confirmatory spillover reference, so without it every g number "
            "would be a deviation from nothing; run PP2.")

    route_h = aggregate_route_h(ds, h_results, h_pending, len(h_specs), ctx,
                                entry_by_id) if h_results else None
    g_baseline = g_baseline_block(frozen, ctx) if frozen else None
    spillover, g_refs = {}, {}
    for model_id, rec in sorted(g_results.items()):
        if rec["model_class"] == "frozen_g":
            continue
        spillover[model_id] = g_spillover_block(rec, frozen, ctx, entry_by_id)
        if rec["model_class"] == "edited" and rec["set_id"]:
            g_refs[model_id] = g_reference_distances(
                rec,
                g_results.get(f"matched_retrain__{rec['set_id']}"),
                g_results.get(f"loo_retrain__{rec['set_id']}"), ctx)

    checkpoints = {}
    for route, results in (("h", h_results), ("g", g_results)):
        for model_id, rec in results.items():
            if rec.get("checkpoint_sha256"):
                checkpoints[f"{route}_{model_id}"] = rec["checkpoint_sha256"]

    # Parallel shards each wrote their own copy of the two shared artifacts;
    # collect every one so the report names all of them rather than whichever
    # shard happened to run last.
    agreements = {}
    for path in _shard_artifacts(ds_out, "scorer_agreement"):
        rec = _load_cached(path)
        if rec is not None:
            agreements[path.name] = rec
    manifests = [{"file": path.name, "sha256": rv.sha256_file(path)}
                 for path in _shard_artifacts(ds_out, "run_manifest")]
    disagreeing = [name for name, rec in agreements.items()
                   if not rec.get("agrees")]

    report = build_report(ds, ds_out, panel, route_h, g_baseline, spillover,
                          g_refs, agreements, provenance, t_start, cpu_only,
                          checkpoints, {"h": h_pending, "g": g_pending},
                          {"h": len(h_specs), "g": len(g_specs)}, manifests,
                          disagreeing)
    path = PANEL_REPORT_DIR / f"prompt_robustness_{ds_out}.json"
    if cpu_only:
        report["reaggregation"] = _reaggregation_check(path, report)
        logger.info("PPR: core digest %s; identical to previous: %s",
                    report["reaggregation"]["reaggregated_core_sha256"][:16],
                    report["reaggregation"]["identical"])
    _write_json(path, report)
    logger.info("PP%s: report written to %s", "R" if cpu_only else "4", path)
    _log_headline(report)
    return report


def build_report(ds, ds_out, panel, route_h, g_baseline, spillover, g_refs,
                 agreements, provenance, t_start, cpu_only, checkpoints,
                 pending, expected, manifests=None, disagreeing=None):
    """The report: three separate blocks plus scoped claims.

    Route h, route-g baseline and route-g spillover are never merged into one
    number, because they measure different things about different models and
    only route h has retraining references.
    """
    claims = build_claims(ds, route_h, g_baseline, spillover, panel)
    g_blocks = {"route_g_baseline": g_baseline,
                "route_g_spillover": {
                    "confirmatory_reference":
                        "frozen_g (g_X_to_C), the only trained visual router",
                    "null_expectation":
                        "an edit targeting h(C->A) leaves g(X->C) approximately "
                        "unchanged: prediction-flip rate 0, accuracy change 0",
                    "intervention_applied_to_g": False,
                    "per_adapter": spillover,
                    "g_spillover_reference_distance": {
                        "metric_name": "g_spillover_reference_distance",
                        "label": G_REFERENCE_LABEL,
                        "is_an_oracle_distance": False,
                        "per_adapter": g_refs},
                }}
    # Enforced here, on the assembled tree, rather than on each writer: a
    # prohibited g-side claim cannot reach the artifact.
    _guard_g_tree(g_blocks, "report")
    _guard_g_tree({k: v for k, v in claims.items()
                   if k.startswith("route_g") or k == "not_claimed"}, "claims")
    return {
        "kind": "e2c_v3_prompt_robustness_report_v1",
        "dataset": ds,
        "produced_by": "scripts/e2c_v3_prompt_panel.py",
        "mode": "cpu_reaggregation" if cpu_only else "aggregate",
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "provenance": provenance,
        "held_out": panel["held_out"],
        "panel": {
            "panel_sha256": panel["panel_sha256"],
            "template_roles": panel["template_roles"],
            "n_identities": panel["routes"]["h"]["n_rows"],
            "h_templates": panel["routes"]["h"]["templates"],
            "g_templates": panel["routes"]["g"]["templates"],
            "candidate_spaces": panel["candidate_spaces"],
            "distractor_neutral_for_all_identities": all(
                r["distractor_neutral_for_all_sets"]
                for r in panel["routes"]["h"]["rows"]),
            "distractor_collisions": [
                {"identity_id": r["identity_id"],
                 "collisions": r["distractor_collisions"]}
                for r in panel["routes"]["h"]["rows"]
                if r["distractor_collisions"]],
            "route_g_images": [{"identity_id": r["identity_id"],
                                "image_uri": r["image_uri"],
                                "image_sha256": r["image_sha256"],
                                "split": r["split"]}
                               for r in panel["routes"]["g"]["rows"]],
        },
        "scorer_agreement": agreements,
        "scorer_agreement_all_agree": not (disagreeing or []),
        "scorer_agreement_disagreeing": list(disagreeing or []),
        "coverage": {"expected_models": expected, "pending": pending,
                     "checkpoints_sha256": checkpoints,
                     "n_checkpoints": len(checkpoints),
                     "run_manifests": list(manifests or [])},
        "route_h": route_h,
        **g_blocks,
        "claims": claims,
        "elapsed_sec": round(time.time() - t_start, 1),
        "gpu": _gpu_name(),
    }


def _nums(values):
    return [v for v in values if v is not None]


def _f(value, nd=4):
    return "not established" if value is None else f"{value:.{nd}f}"


def build_claims(ds, route_h, g_baseline, spillover, panel):
    """Claims, each scoped to the evaluated templates and candidate space.

    Follows this project's existing ``claims`` convention: a claim states what
    was measured AND the scope it was measured in, so it cannot be lifted out
    as a general property of the edit.  The h and g claims are separate
    sentences about separate model families and are never combined.
    """
    n_roles = len(TEMPLATE_ROLES)
    h_space = panel["candidate_spaces"]["h"]
    g_space = panel["candidate_spaces"]["g"]
    n_ids = panel["routes"]["h"]["n_rows"]
    scope = (
        f"every number in this report was measured on the {n_roles} "
        f"pre-frozen prompt templates ({', '.join(TEMPLATE_ROLES)}) over the "
        f"{len(h_space['vocab'])}-label route-h candidate space and the "
        f"{len(g_space['codes'])}-code route-g candidate space, dataset {ds}, "
        f"{n_ids} identities; the panel was frozen and committed before any "
        f"model was evaluated, and no template, threshold or recipe was "
        f"chosen using it")

    cells = (route_h or {}).get("edited_cells", {})
    signs = [c["delta_retrain_sign_preserved_across_templates"]
             for c in cells.values()]
    deltas = _nums([c["delta_retrain_worst_case"] for c in cells.values()])
    canonical = _nums([
        c["distance_to_oracle_by_template"]["canonical"][
            "worst_case_delta_retrain"] for c in cells.values()])
    target_acc = _nums([
        c["template_worst_case"][
            "desired_label_accuracy.transformation_targets"]
        for c in cells.values()])
    leak = _nums([
        c["template_worst_case"]["source_label_leakage.hard_leak_rate"]
        for c in cells.values()])
    sign_true = sum(1 for s in signs if s is True)

    # ---- prompt support: the fact that decides how the rest may be read ---- #
    profile = (route_h or {}).get("prompt_support_profile") or {}
    per_class = profile.get("per_class") or {}
    on_by_role = profile.get("classes_on_support_by_template") or {}
    crit = gx.PASS_CRITERIA["min_candidate_mass"]
    # A template can carry a retraining comparison only if the three model
    # classes being compared are all still on the candidate space there.
    supported_roles = [r for r in TEMPLATE_ROLES
                       if {"edited", "matched_retrain", "loo_retrain"}
                       <= set(on_by_role.get(r, []))]
    baseline_off = profile.get(
        "templates_where_the_baseline_is_off_support") or []
    edit_specific = profile.get(
        "templates_where_only_edited_models_are_off_support") or []
    base = per_class.get("baseline") or {}

    def _class_stat(klass, role, field):
        return ((per_class.get(klass) or {}).get(role) or {}).get(field)

    if base:
        off_mass = _nums([_class_stat("baseline", r, "min_candidate_mass")
                          for r in TEMPLATE_ROLES if r != "canonical"])
        worst_base_mass = min(off_mass) if off_mass else None
        base_acc = {r: _class_stat("baseline", r, "strict_accuracy")
                    for r in TEMPLATE_ROLES}
        off_acc = _nums([v for r, v in base_acc.items() if r != "canonical"])
        best_off_acc = _f(max(off_acc), 3) if off_acc else "not established"
        # The attribution has to follow the data.  Asserting "a property of how
        # the route is trained" when nothing collapsed, or "specific to the
        # edit" when the baseline collapsed too, would each be a conclusion the
        # numbers do not support.
        if not baseline_off and not edit_specific:
            attribution = (f"every model class stays on the candidate space on "
                           f"all {n_roles} templates, so no collapse was "
                           f"observed in this sweep")
        elif baseline_off and not edit_specific:
            attribution = ("prompt sensitivity in this sweep is a property of "
                           "how route h is TRAINED and the panel cannot "
                           "attribute it to the edit")
        elif edit_specific and not baseline_off:
            attribution = ("the collapse IS specific to the edited models, on "
                           f"{', '.join(edit_specific)}, where the "
                           "never-edited baseline stays on support")
        else:
            attribution = (
                f"{', '.join(baseline_off)} collapse for the baseline too and "
                f"so are a route-training property, while "
                f"{', '.join(edit_specific)} collapse only for edited models "
                f"and so are edit-specific")
        route_h_support_claim = (
            f"route h prompt support (read this before any other route-h "
            f"number): {len(supported_roles)} of the {n_roles} templates "
            f"({', '.join(supported_roles) or 'none'}) leave the edited cells "
            f"AND both retrain references on the candidate space under the "
            f"frozen min_candidate_mass={crit} criterion.  The never-edited "
            f"baseline_h is off support on {len(baseline_off)} of {n_roles} "
            f"templates ({', '.join(baseline_off) or 'none'}), reaching a "
            f"minimum candidate mass of {_f(worst_base_mass)} away from "
            f"canonical, where its strict accuracy is "
            f"{_f(base_acc.get('canonical'))} on canonical and at best "
            f"{best_off_acc} on the other {n_roles - 1}.  "
            f"{len(edit_specific)} template(s) are off support for edited "
            f"models but not for the baseline: {attribution}.")
    else:
        route_h_support_claim = (
            "route h prompt support: not evaluated in this run")

    sup_signs = [c["delta_retrain_sign_preserved_on_supported_templates"]
                 for c in cells.values()]
    sup_deltas = _nums([
        c["delta_retrain_worst_case_on_supported_templates"]
        for c in cells.values()])
    sup_true = sum(1 for s in sup_signs if s is True)
    robust = bool(sup_deltas) and sup_true == len(sup_signs)
    route_h_claim = (
        f"route h (code -> alias): Delta_retrain = D(edit, loo_retrain) - "
        f"D(edit, matched_retrain) is interpreted only on the "
        f"{len(supported_roles)} template(s) where the edited cell and both "
        f"retrain references are all on the candidate space "
        f"({', '.join(supported_roles) or 'none'}); there it stays positive in "
        f"{sup_true} of {len(sup_signs)} cells with a worst case of "
        f"{_f(min(sup_deltas) if sup_deltas else None)}, against "
        f"{_f(min(canonical) if canonical else None)} on the canonical prompt "
        f"alone, so the retraining-equivalence finding is "
        f"{'robust across the templates that remain on support' if robust else 'not established beyond the templates that remain on support'}"
        f".  The unrestricted worst case over all {n_roles} templates is "
        f"{_f(min(deltas) if deltas else None)} with the sign holding in "
        f"{sign_true} of {len(signs)} cells, but the templates that make it "
        f"worse are off support, where a renormalized distance compares two "
        f"artifacts rather than two behaviors, so those values are reported "
        f"and not interpreted.  Worst-case strict desired-label accuracy on "
        f"transformation targets is "
        f"{_f(min(target_acc) if target_acc else None)} and worst-case hard "
        f"source leakage is {_f(max(leak) if leak else None)}")

    if g_baseline:
        accs = g_baseline["code_accuracy_by_template"]
        route_g_baseline_claim = (
            f"route g baseline prompt sensitivity (no intervention): the "
            f"frozen g_X_to_C router's code accuracy over {n_ids} held-out "
            f"test images ranges from {_f(min(accs.values()))} "
            f"(worst template: {g_baseline['worst_template']}) to "
            f"{_f(max(accs.values()))}, a maximum template spread of "
            f"{_f(g_baseline['max_template_spread'])}; six-template "
            f"consistency (the same code however it is asked) is "
            f"{_f(g_baseline['six_template_consistency'])}")
    else:
        route_g_baseline_claim = "route g baseline: not evaluated in this run"

    if spillover:
        flips = _nums([b["worst_template_flip_rate"]
                       for b in spillover.values()])
        changes = _nums([b["worst_template_accuracy_change"]
                         for b in spillover.values()])
        route_g_spillover_claim = (
            f"route g cross-route spillover (no intervention on g): "
            f"{len(spillover)} h-side adapters loaded onto the same base model "
            f"and run over the identical panel reach a worst-template "
            f"prediction-flip rate of {_f(max(flips) if flips else None)} and "
            f"a worst-template code-accuracy change of "
            f"{_f(min(changes) if changes else None)} relative to frozen base "
            f"g; the null expectation is that an h-targeted edit leaves the "
            f"visual router approximately unchanged")
    else:
        route_g_spillover_claim = "route g spillover: not evaluated in this run"

    not_claimed = [
        ("no intervention of any kind was applied to the visual router -- no "
         "edit, no unlearning run, no g-specific matched or LOO retraining -- "
         "so no route-g number describes one"),
        ("distances between h-side adapters measured on route-g prompts are "
         "reported as g_spillover_reference_distance, a cross-route reference "
         "comparison; they are not oracle distances and support no claim "
         "about the visual router"),
        (f"no claim reaches beyond the {n_roles} evaluated templates or the "
         "recorded candidate spaces: a prompt outside the panel is "
         "unmeasured, not passing"),
        ("no threshold, gate, promotion criterion or training recipe in this "
         "repository changes as a result of this panel"),
        ("the panel is a held-out behavioral evaluation; it selects no model, "
         "no template and no checkpoint"),
    ]
    return {
        "scope": scope,
        "route_h_prompt_support": route_h_support_claim,
        "route_h": route_h_claim,
        "route_g_baseline": route_g_baseline_claim,
        "route_g_cross_route_spillover": route_g_spillover_claim,
        "not_claimed": not_claimed,
    }


def _log_headline(report):
    for key in ("route_h_prompt_support", "route_h", "route_g_baseline",
                "route_g_cross_route_spillover"):
        logger.info("CLAIM %s: %s", key, report["claims"][key])


def update_run_manifest(ds, ds_out, panel, provenance, route, results,
                        t_start, args):
    """Cumulative provenance for a resumable, multi-process sweep.

    Each phase may run in its own process, so the manifest merges rather than
    overwrites: every checkpoint evaluated keeps the commit that evaluated it,
    and ``provenance_history`` keeps one entry per invocation.  The commit is
    captured in ``main`` BEFORE the first result file is written.

    Under sharding each process owns its OWN manifest file.  Merging across
    processes would mean concurrent read-modify-write of one JSON file, and a
    torn write there would be indistinguishable from a manifest that simply
    recorded fewer checkpoints; PP4 collects the per-shard files instead.
    """
    path = PANEL_OUT_ROOT / ds_out / f"run_manifest{_shard_tag(args)}.json"
    existing = _load_cached(path) or {}
    checkpoints = dict(existing.get("checkpoints_sha256") or {})
    for model_id, rec in results.items():
        if rec.get("checkpoint_sha256"):
            checkpoints[f"{route}_{model_id}"] = rec["checkpoint_sha256"]
    manifest = {
        "experiment": f"e2c_v3_prompt_panel_{ds}",
        "produced_by": "scripts/e2c_v3_prompt_panel.py",
        "shard": _shard_label(args),
        "provenance": provenance,
        "provenance_history": (existing.get("provenance_history") or []) + [
            {"route": route, "phase": provenance.get("phase"),
             "commit": provenance.get("commit"),
             "started_utc": provenance.get("started_utc")}],
        "panel_sha256": panel["panel_sha256"],
        "inputs_sha256": panel.get("inputs", {}),
        "routes_executed": sorted(
            set((existing.get("routes_executed") or []) + [route])),
        "checkpoints_sha256": checkpoints,
        "results": {"n_checkpoints_evaluated": len(checkpoints)},
        "cumulative_elapsed_sec": round(
            (existing.get("cumulative_elapsed_sec") or 0.0)
            + (time.time() - t_start), 1),
        "gpu": _gpu_name(),
        "updated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    _write_json(path, manifest)
    logger.info("%s (%s): run manifest updated (%d checkpoints)", route,
                _shard_label(args), len(checkpoints))
    return manifest


# ====================================================================== #
# Frozen inputs and entry point
# ====================================================================== #
def _load_matrix(ds):
    """The committed matrix, re-derived and GX0-validated, or a hard stop.

    Verify-not-rewrite, as in the granularity runner: the matrix is a committed
    input to a held-out evaluation, so a difference between the file and the
    builder is a stop, not a regeneration.

    One subtlety decides whether that comparison means anything.
    ``gx.validate_set`` POPULATES ``entry["controls"]`` and
    ``entry["control_notes"]`` as a side effect, so the committed matrix equals
    the builder's output only AFTER a GX0 pass has run over it.  Comparing
    first would report drift where there is none; skipping the validation to
    make the comparison pass would let a panel be evaluated against a matrix
    the project's own hard gate rejects.  Running the same pass in the same
    order avoids both.
    """
    path = gxm.MANIFEST_DIR / f"matrix_{ds}.json"
    if not path.exists():
        raise RuntimeError(
            f"{path} not found; build and commit the granularity matrix "
            f"before evaluating a prompt panel against it")
    committed = json.loads(path.read_text(encoding="utf-8"))
    with open(gxm.SALMU_MANIFEST, encoding="utf-8") as f:
        # build_salmu_matrix returns (matrix, builder_ctx); only the matrix is
        # the committed artifact, exactly as the granularity runner compares it
        rebuilt, _builder_ctx = gx.build_salmu_matrix(json.load(f))
    ctx = gxm.dataset_ctx(ds, rebuilt)
    issues = []
    for entry in rebuilt["sets"]:
        issues.extend(f"{entry['set_id']}: {issue}"
                      for issue in gx.validate_set(entry, ctx))
    vocab_issues, _collisions = gx.validate_vocab(ctx["vocab"])
    issues.extend(vocab_issues)
    if issues:
        for issue in issues:
            logger.error("GX0 ISSUE: %s", issue)
        raise RuntimeError(
            f"the granularity matrix failed GX0 validation with {len(issues)} "
            f"issue(s); a held-out panel must not be evaluated against a "
            f"matrix the project's own hard gate rejects")
    if rebuilt != committed:
        raise RuntimeError(
            f"{path} differs from the frozen builder in code; the matrix is a "
            f"committed input and must not drift under a held-out panel")
    logger.info("matrix %s: GX0 validation passed and the committed file "
                "matches the frozen builder (%d sets, vocab %d)", ds,
                len(rebuilt["sets"]), len(ctx["vocab"]))
    return rebuilt


#: The code this runner executes.  Provenance binds to THESE files being
#: committed; result files are allowed to be dirty, because the tree currently
#: holds in-flight granularity outputs from a parallel run that are not this
#: script's to commit (the GX2B/GX2S relaxation from commit febc662).
PP_CODE = ["scripts/e2c_v3_prompt_panel.py",
           "scripts/e2c_v3_research_validity.py",
           "scripts/e2c_v3_granularity.py",
           "scripts/e2c_v3_granularity_matrix.py",
           "scripts/e2c_v3_matrix.py",
           "scripts/e2c_v3_realdata.py"]


def _dirty_tracked_code():
    """Tracked-file changes among the EXECUTED code (not result outputs)."""
    code = [p for p in PP_CODE if Path(p).exists()]
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", *code],
            text=True)
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return ["<git status failed>"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["salmu"],
                   help="salmu only: it is the one dataset whose granularity "
                        "matrix is complete, and the one with a trained "
                        "visual router")
    p.add_argument("--phase", default="all",
                   choices=["all", "PP0", "PP1", "PP2", "PP3", "PP4", "PPR"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, nargs="+", default=gx.SEEDS_DEFAULT)
    p.add_argument("--only-sets", nargs="*", default=None)
    p.add_argument("--only-seeds", type=int, nargs="*", default=None,
                   help="restrict EVALUATION to these edit seeds (the frozen "
                        "matrix always declares 17/42/123)")
    p.add_argument("--only-templates", nargs="*", default=None,
                   choices=list(TEMPLATE_ROLES),
                   help="smoke/pilot only: evaluate a subset of the frozen "
                        "templates (refused without --smoke)")
    p.add_argument("--max-gen-tokens", type=int, default=12)
    p.add_argument("--smoke", action="store_true",
                   help="1 model, 1 template, separate *_smoke output tree")
    p.add_argument("--pilot", type=int, default=0, metavar="N",
                   help="evaluate only the first N pending models, then print "
                        "a measured ETA for the rest and stop")
    p.add_argument("--shard-index", type=int, default=0,
                   help="which slice of the sweep THIS process evaluates "
                        "(0-based); slices are disjoint so parallel processes "
                        "never write the same result file")
    p.add_argument("--shard-count", type=int, default=1,
                   help="how many processes the sweep is split across; one "
                        "per GPU that can hold the model")
    p.add_argument("--eta-only", action="store_true",
                   help="CPU only: print ETAs from timings already on disk")
    p.add_argument("--refreeze", action="store_true",
                   help="PP0 only: overwrite an already-frozen panel. Any "
                        "result produced against the old digest is invalid.")
    return p.parse_args()


def main():
    global EXECUTING_COMMIT
    args = parse_args()
    t_start = time.time()
    if args.only_templates and not args.smoke:
        raise RuntimeError(
            "--only-templates is a smoke/pilot affordance and requires "
            "--smoke: the report's template_worst_case is a min over all six "
            "frozen roles, so a partial panel must stay in its own *_smoke "
            "output tree and never reach PP4.")
    if args.smoke:
        args.only_seeds = args.only_seeds or [17]
        args.only_templates = args.only_templates or ["canonical"]
    if args.shard_count < 1:
        raise RuntimeError(f"--shard-count must be >= 1 (got {args.shard_count})")
    if not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError(
            f"--shard-index must be in [0, {args.shard_count}) "
            f"(got {args.shard_index})")
    args.seed = (args.only_seeds or args.seeds)[0]
    ds = args.dataset
    if args.phase == "PPR":
        # CPU-only re-aggregation: the report must be reproducible from the
        # stored per-model files without a GPU or a checkpoint in sight.
        args.device = "cpu"

    # Captured BEFORE anything is written, so every artifact names the code
    # that produced it rather than whatever HEAD is when the sweep finishes.
    EXECUTING_COMMIT = rv.git_commit_sha()
    dirty_code = _dirty_tracked_code()
    provenance = {
        "phase": args.phase,
        "commit": EXECUTING_COMMIT,
        "runner_script_sha256": rv.sha256_file(Path(__file__).resolve()),
        "shared_scoring_script_sha256": rv.script_sha256(),
        "dirty": rv.git_worktree_dirty(),
        "dirty_executed_code": dirty_code,
        "clean_code_required": not args.smoke,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "shard": _shard_label(args),
    }
    logger.info("Provenance: commit=%s runner=%s rv=%s dirty=%s shard=%s",
                EXECUTING_COMMIT[:12],
                provenance["runner_script_sha256"][:12],
                provenance["shared_scoring_script_sha256"][:12],
                provenance["dirty"], provenance["shard"])
    if dirty_code and not args.smoke:
        raise RuntimeError(
            f"{args.phase}: executed CODE not committed: {dirty_code}.  "
            f"Result files may be dirty (a parallel granularity run owns "
            f"them), but the code that produces a held-out panel may not be.")

    matrix = _load_matrix(ds)
    ctx = gxm.dataset_ctx(ds, matrix)
    entry_by_id = {e["set_id"]: e for e in matrix["sets"]}
    logger.info("%s: %d sets, %d edit seeds, %d identities, vocab %d",
                ds, len(matrix["sets"]), len(matrix["edit_seeds"]),
                len(ctx["identity_ids"]), len(ctx["vocab"]))

    if args.phase == "PP0":
        with open(gxm.SALMU_MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)
        freeze_panel(ds, build_panel(ds, manifest, matrix, ctx), args)
        logger.info("PP0 COMPLETE (%s) -- commit the panel before evaluating",
                    _hms(time.time() - t_start))
        return 0

    panel = load_frozen_panel(ds)
    if panel["inputs"]["matrix_salmu_sha256"] != rv.sha256_file(
            gxm.MANIFEST_DIR / f"matrix_{ds}.json"):
        raise RuntimeError(
            "the frozen panel was built against a different matrix than the "
            "one now committed; the panel and its inputs move together or not "
            "at all")
    if args.only_templates and set(args.only_templates) != set(
            panel["template_roles"]):
        logger.warning("evaluating a SUBSET of the frozen panel: %s",
                       args.only_templates)

    if args.eta_only:
        out_base = _gran_out_base(ds)
        ds_out = _panel_out_ds(ds, args)
        label = _shard_label(args)
        print_eta("h", _shard(h_model_specs(ds, out_base, matrix, args), args),
                  ds_out, label)
        print_eta("g", _shard(g_model_specs(ds, out_base, matrix, args), args),
                  ds_out, label)
        return 0

    if args.phase in ("all", "PP1"):
        run_pp1(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start)
    if args.phase in ("all", "PP2"):
        run_pp2(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start)
    if args.phase in ("all", "PP3"):
        run_pp3(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start)
    # Under --phase all with sharding, every shard would otherwise write the
    # SAME report file concurrently.  The report describes the whole sweep, so
    # it is written once, unsharded, after every shard has finished.  An
    # explicit --phase PP4 still aggregates whatever is on disk, shards or not.
    if args.phase == "all" and args.shard_count > 1:
        logger.info("PP4 skipped: shard %d of %d does not own the report.  Run "
                    "--phase PP4 (unsharded) once every shard has finished.",
                    args.shard_index, args.shard_count)
    elif args.phase in ("all", "PP4"):
        run_pp4(args, ds, matrix, ctx, entry_by_id, panel, provenance, t_start)
    if args.phase == "PPR":
        run_pp4(args, ds, matrix, ctx, entry_by_id, panel, provenance,
                t_start, cpu_only=True)
    logger.info("=" * 60)
    logger.info("PROMPT PANEL (%s) PHASE %s %s COMPLETE (%s)", ds, args.phase,
                _shard_label(args), _hms(time.time() - t_start))
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
