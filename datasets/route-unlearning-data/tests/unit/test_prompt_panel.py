"""CPU tests for the E2C-v3 prompt-robustness panel (PP0-PP4 / PPR).

The panel is a HELD-OUT behavioral evaluation: it is frozen before any model is
scored, its digest is re-verified by every later phase, and nothing in the
repository reads its output as a gate.  These tests pin the properties that make
"held out" mean something rather than being a comment:

- panel shape, and canonical byte-identity with the production prompts, so the
  canonical column reproduces the existing evidence instead of restating it;
- ``format_variation`` differs from canonical in case / punctuation / whitespace
  ONLY, so it isolates surface formatting from lexical change;
- distractor neutrality across ALL 21 sets -- a decoy that is some set's desired
  label would make the distractor template EASIER than the canonical one and
  invert the measurement;
- freeze semantics: no overwrite without ``--refreeze``, and a mutated panel
  aborts the evaluator;
- the one shared-primitive change (``prompt_text``) leaves the default path
  byte-identical, so no existing artifact's numbers move;
- strict parsing: a decoy echo is leakage, never the desired label, and a
  multi-label output is invalid rather than resolved to its first match;
- worst case is a min over the SIX frozen roles and cannot be taken over a
  subset; sibling absence stays null rather than becoming 1.0;
- distance gating yields None ("not established"), never 0.0;
- route-g spillover is measured against frozen base g, split target / retained;
- prohibited g-side vocabulary raises, including inside the report's own claims.

No test here loads a model or touches a GPU.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pp = _load("pp_panel_lib", "e2c_v3_prompt_panel.py")
rv, rd, gx, gxm = pp.rv, pp.rd, pp.gx, pp.gxm


# ------------------------------------------------------------------ #
# real committed SALMU inputs -> the panel, built with no GPU
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def real():
    """The frozen SALMU inputs and the panel built from them.

    ``build_panel`` is a pure function of committed files, so this needs no
    checkpoint and no device -- which is the point: the panel is frozen before
    any model is in the picture.
    """
    manifest = json.loads(gxm.SALMU_MANIFEST.read_text(encoding="utf-8"))
    path = gxm.MANIFEST_DIR / "matrix_salmu.json"
    matrix = json.loads(path.read_text(encoding="utf-8"))
    ctx = gxm.dataset_ctx("salmu", matrix)
    return {"manifest": manifest, "matrix": matrix, "ctx": ctx,
            "panel": pp.build_panel("salmu", manifest, matrix, ctx)}


def _lexical(text):
    """Case / punctuation / whitespace stripped: the lexical content only."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


# ------------------------------------------------------------------ #
# panel shape
# ------------------------------------------------------------------ #
def test_panel_is_six_roles_by_every_identity_by_two_routes(real):
    panel, ctx = real["panel"], real["ctx"]
    assert list(pp.TEMPLATE_ROLES) == [
        "canonical", "concise_paraphrase", "question_form", "instruction_form",
        "format_variation", "distractor"]
    assert panel["template_roles"] == list(pp.TEMPLATE_ROLES)
    n = len(ctx["identity_ids"])
    assert n == 12, "SALMU is frozen at 12 identities"
    for route in ("h", "g"):
        block = panel["routes"][route]
        assert block["n_rows"] == n and len(block["rows"]) == n
        assert set(block["templates"]) == set(pp.TEMPLATE_ROLES)
        for row in block["rows"]:
            assert set(row["prompts"]) == set(pp.TEMPLATE_ROLES)
            assert len(set(row["prompts"].values())) == len(pp.TEMPLATE_ROLES), \
                "the six templates must be distinct strings, not six aliases"


def test_canonical_templates_are_byte_identical_to_the_production_prompts(real):
    """The canonical column must BE the prompt every existing artifact used."""
    panel = real["panel"]
    assert panel["routes"]["h"]["templates"]["canonical"] == \
        rd.CODE_TO_ALIAS_PROMPT
    assert panel["routes"]["g"]["templates"]["canonical"] == \
        rd.IMG_TO_CODE_PROMPT
    for row in panel["routes"]["h"]["rows"]:
        assert row["prompts"]["canonical"] == \
            rd.CODE_TO_ALIAS_PROMPT.format(code=row["code"])
    for row in panel["routes"]["g"]["rows"]:
        assert row["prompts"]["canonical"] == rd.IMG_TO_CODE_PROMPT


def test_format_variation_changes_surface_form_only(real):
    """Formatting, not wording: otherwise it is a seventh paraphrase."""
    for route in ("h", "g"):
        tpl = real["panel"]["routes"][route]["templates"]
        assert tpl["format_variation"] != tpl["canonical"]
        assert _lexical(tpl["format_variation"]) == _lexical(tpl["canonical"])
        for row in real["panel"]["routes"][route]["rows"]:
            assert _lexical(row["prompts"]["format_variation"]) == \
                _lexical(row["prompts"]["canonical"])


def test_other_roles_do_change_lexical_content(real):
    """Guard the guard: only ``format_variation`` may be lexically identical."""
    for route in ("h", "g"):
        tpl = real["panel"]["routes"][route]["templates"]
        for role in pp.TEMPLATE_ROLES:
            if role == "canonical":
                continue
            same = _lexical(tpl[role]) == _lexical(tpl["canonical"])
            assert same is (role == "format_variation"), \
                f"{route}/{role} should {'not ' if not same else ''}differ " \
                f"from canonical in wording"


# ------------------------------------------------------------------ #
# distractor neutrality
# ------------------------------------------------------------------ #
def test_the_matrix_really_has_21_sets_and_3_edit_seeds(real):
    matrix = real["matrix"]
    assert len(matrix["sets"]) == 21
    assert matrix["edit_seeds"] == [17, 42, 123]


def test_load_matrix_verifies_the_committed_matrix_against_the_builder():
    """The matrix is a committed input to a held-out panel, so it is verified
    against the frozen builder rather than trusted -- and rebuilt, never
    rewritten.  ``build_salmu_matrix`` returns ``(matrix, builder_ctx)``, and
    only the matrix is the artifact; comparing the tuple would always differ."""
    matrix = pp._load_matrix("salmu")
    assert matrix["dataset"] == "salmu"
    assert len(matrix["sets"]) == 21
    assert matrix["edit_seeds"] == [17, 42, 123]


def test_load_matrix_runs_the_gx0_pass_that_populates_controls():
    """``gx.validate_set`` fills ``controls`` / ``control_notes`` IN PLACE, so
    the committed matrix only equals the builder's output after that pass.
    Comparing before validating reports drift where there is none."""
    with open(gxm.SALMU_MANIFEST, encoding="utf-8") as f:
        raw, _builder_ctx = gx.build_salmu_matrix(json.load(f))
    assert all(entry.get("controls") is None for entry in raw["sets"]), \
        "fixture stale: the builder now populates controls itself"
    matrix = pp._load_matrix("salmu")
    assert all(entry["controls"] for entry in matrix["sets"])
    committed = json.loads((gxm.MANIFEST_DIR / "matrix_salmu.json").read_text(
        encoding="utf-8"))
    assert matrix == committed


def test_a_matrix_that_fails_gx0_is_refused(monkeypatch):
    """A held-out panel must not be evaluated against a matrix the project's
    own hard gate rejects -- and GX0 is a precondition, not a comment."""
    real_builder = gx.build_salmu_matrix

    def broken(manifest):
        matrix, builder_ctx = real_builder(manifest)
        entry = matrix["sets"][0]
        iid = next(iter(entry["assignments"]))
        entry["assignments"][iid]["target"] = "a label nobody froze"
        return matrix, builder_ctx

    monkeypatch.setattr(gx, "build_salmu_matrix", broken)
    with pytest.raises(RuntimeError, match="failed GX0 validation"):
        pp._load_matrix("salmu")


def test_a_drifted_matrix_is_refused(tmp_path, monkeypatch):
    body = json.loads((gxm.MANIFEST_DIR / "matrix_salmu.json").read_text(
        encoding="utf-8"))
    body["vocab"] = sorted(body["vocab"] + ["a label nobody froze"])
    monkeypatch.setattr(gxm, "MANIFEST_DIR", tmp_path)
    (tmp_path / "matrix_salmu.json").write_text(json.dumps(body, indent=2),
                                                encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not drift"):
        pp._load_matrix("salmu")


def test_a_missing_matrix_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(gxm, "MANIFEST_DIR", tmp_path)
    with pytest.raises(RuntimeError,
                       match="build and commit the granularity matrix"):
        pp._load_matrix("salmu")


def test_distractor_decoy_is_neutral_for_every_set(real):
    """A decoy that is ever a source / target / expected label is a leak."""
    panel, ctx, matrix = real["panel"], real["ctx"], real["matrix"]
    for row in panel["routes"]["h"]["rows"]:
        iid, decoy = row["identity_id"], row["distractor_decoy"]
        assert decoy in ctx["vocab"]
        assert decoy != ctx["baseline_alias_of"][iid]
        assert decoy not in ctx["hierarchy_of"][iid], \
            f"{iid}: decoy is on its own taxonomic chain"
        for entry in matrix["sets"]:
            assignment = entry["assignments"].get(iid)
            if assignment:
                assert decoy != assignment.get("source"), \
                    f"{iid}: decoy is the source in {entry['set_id']}"
                assert decoy != assignment.get("target"), \
                    f"{iid}: decoy is the target in {entry['set_id']}"
            assert decoy != gxm.expected_label(ctx, entry, iid), \
                f"{iid}: decoy is the expected label in {entry['set_id']}"
        assert row["distractor_neutral_for_all_sets"] is True
        assert row["distractor_collisions"] == []
        assert row["distractor_neutral_pool_size"] > 0


def test_decoy_selection_is_deterministic_and_position_seeded(real):
    """Rebuilding must reproduce the panel exactly, or the freeze is hollow."""
    again = pp.build_panel("salmu", real["manifest"], real["matrix"],
                           real["ctx"])
    assert pp.panel_digest(again) == pp.panel_digest(real["panel"])
    assert pp._pick_decoy(["a", "b", "c"], "i1", ["i1", "i2"]) == \
        pp._pick_decoy(["a", "b", "c"], "i1", ["i1", "i2"])


def test_excluded_labels_cover_everything_an_identity_is_ever_associated_with(
        real):
    ctx, matrix = real["ctx"], real["matrix"]
    for iid in ctx["identity_ids"]:
        excluded, _assigned = pp._labels_excluded_for_identity(ctx, matrix, iid)
        assert ctx["baseline_alias_of"][iid] in excluded
        assert set(ctx["hierarchy_of"][iid]) <= set(excluded)
        for entry in matrix["sets"]:
            assignment = entry["assignments"].get(iid)
            if assignment:
                assert assignment["source"] in excluded
                assert assignment["target"] in excluded
            assert gxm.expected_label(ctx, entry, iid) in excluded


def test_the_fallback_decoy_leaves_the_identitys_own_branch(real):
    ctx, matrix = real["ctx"], real["matrix"]
    ids = list(ctx["identity_ids"])
    for iid in ids:
        fallback = pp._h_decoy_fallback(ctx, matrix, iid, ids)
        assert fallback is not None
        assert fallback != ctx["baseline_alias_of"][iid]
        assert fallback not in ctx["hierarchy_of"][iid]
        assert fallback != gx.DELETED_LABEL


def test_build_panel_records_collisions_when_the_neutral_pool_is_empty(
        real, monkeypatch):
    """A non-neutral decoy must be flagged, not averaged away silently."""
    original = pp._pick_decoy
    counts = {i: sum(1 for e in real["matrix"]["sets"]
                     if i in e["assignments"])
              for i in real["ctx"]["identity_ids"]}
    target = max(counts, key=counts.get)
    assert counts[target] > 0, "no identity is ever assigned; fixture is stale"
    starved = {"done": False}

    def starve_h_pool(pool, iid, identity_ids):
        # route-h pools hold aliases, route-g pools hold "SAL_*" codes; starve
        # only the FIRST h call so the fallback path still gets to run
        if (iid == target and not starved["done"] and pool
                and not str(pool[0]).startswith("SAL_")):
            starved["done"] = True
            return None
        return original(pool, iid, identity_ids)

    monkeypatch.setattr(pp, "_pick_decoy", starve_h_pool)
    panel = pp.build_panel("salmu", real["manifest"], real["matrix"],
                           real["ctx"])
    row = next(r for r in panel["routes"]["h"]["rows"]
               if r["identity_id"] == target)
    assert row["distractor_neutral_for_all_sets"] is False
    assert row["distractor_neutral_pool_size"] > 0
    _excluded, assigned = pp._labels_excluded_for_identity(
        real["ctx"], real["matrix"], target)
    assert len(row["distractor_collisions"]) == len(assigned)
    assert len(assigned) >= counts[target], \
        "every assignment must contribute at least one recorded collision"
    assert {"set_id", "role", "label", "reason"} <= set(
        row["distractor_collisions"][0])
    assert row["distractor_decoy"] != real["ctx"]["baseline_alias_of"][target]
    assert row["distractor_decoy"] in row["prompts"]["distractor"]
    # the other identities are untouched by one starved pool
    assert all(r["distractor_neutral_for_all_sets"] is True
               for r in panel["routes"]["h"]["rows"]
               if r["identity_id"] != target)


def test_route_g_decoy_is_another_identitys_code(real):
    ctx = real["ctx"]
    codes = set(ctx["code_of"].values())
    for row in real["panel"]["routes"]["g"]["rows"]:
        assert row["distractor_decoy_code"] in codes
        assert row["distractor_decoy_code"] != row["code"]
        assert row["distractor_decoy_code"] in row["prompts"]["distractor"]


def test_route_g_images_are_the_first_held_out_test_image(real):
    """Test split on purpose: the frozen router is 96/96 on train, so a train
    image would sit at a ceiling and hide template sensitivity."""
    manifest = real["manifest"]
    by_uri = {it["image_uri"]: it for it in manifest["items"]}
    for row in real["panel"]["routes"]["g"]["rows"]:
        item = by_uri[row["image_uri"]]
        assert item["split"] == "test"
        assert item["identity_id"] == row["identity_id"]
        assert row["image_sha256"] == item["image_sha256"]
        same = sorted(i["image_uri"] for i in manifest["items"]
                      if i["identity_id"] == row["identity_id"]
                      and i["split"] == "test")
        assert same and row["image_uri"] == same[0]


# ------------------------------------------------------------------ #
# freeze / held-out enforcement
# ------------------------------------------------------------------ #
def test_freeze_refuses_to_overwrite_without_refreeze(tmp_path, monkeypatch,
                                                      real):
    monkeypatch.setattr(pp, "PANEL_MANIFEST_DIR", tmp_path / "manifests")
    panel = json.loads(json.dumps(real["panel"]))
    pp.freeze_panel("salmu", panel, SimpleNamespace(refreeze=False))
    with pytest.raises(RuntimeError, match="already frozen"):
        pp.freeze_panel("salmu", json.loads(json.dumps(panel)),
                        SimpleNamespace(refreeze=False))
    # an explicit refreeze is allowed, and identical content refreezes to the
    # SAME digest: determinism is what makes the freeze auditable
    again = pp.freeze_panel("salmu", json.loads(json.dumps(panel)),
                            SimpleNamespace(refreeze=True))
    assert again["panel_sha256"] == panel["panel_sha256"]
    changed = json.loads(json.dumps(panel))
    changed["routes"]["h"]["templates"]["concise_paraphrase"] = "Alias?"
    moved = pp.freeze_panel("salmu", changed, SimpleNamespace(refreeze=True))
    assert moved["panel_sha256"] != panel["panel_sha256"], \
        "a changed template must change the digest, or the freeze proves nothing"


def test_a_mutated_panel_aborts_the_evaluator(tmp_path, monkeypatch, real):
    monkeypatch.setattr(pp, "PANEL_MANIFEST_DIR", tmp_path / "manifests")
    panel = json.loads(json.dumps(real["panel"]))
    pp.freeze_panel("salmu", panel, SimpleNamespace(refreeze=False))
    assert pp.load_frozen_panel("salmu")["panel_sha256"] == \
        panel["panel_sha256"]
    path = pp.panel_path("salmu")
    body = json.loads(path.read_text(encoding="utf-8"))
    # exactly the edit a tuning temptation would make after seeing results
    body["routes"]["h"]["templates"]["instruction_form"] = "Say the alias."
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="panel digest mismatch"):
        pp.load_frozen_panel("salmu")


def test_a_reserialized_panel_is_still_the_same_panel(tmp_path, monkeypatch,
                                                      real):
    """Key order and indentation are not a changed panel; a template is."""
    monkeypatch.setattr(pp, "PANEL_MANIFEST_DIR", tmp_path / "manifests")
    panel = json.loads(json.dumps(real["panel"]))
    pp.freeze_panel("salmu", panel, SimpleNamespace(refreeze=False))
    path = pp.panel_path("salmu")
    body = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
    assert pp.load_frozen_panel("salmu")["panel_sha256"] == \
        panel["panel_sha256"]


def test_loading_a_panel_that_was_never_frozen_is_an_error(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(pp, "PANEL_MANIFEST_DIR", tmp_path / "manifests")
    with pytest.raises(RuntimeError, match="run --phase PP0 first"):
        pp.load_frozen_panel("salmu")


def test_the_panel_records_that_it_selects_nothing(real):
    held = real["panel"]["held_out"]
    assert held["status"] == "held-out behavioral evaluation"
    assert held["frozen_before_evaluation"] is True
    assert "threshold setting" in held["not_used_for"]
    assert "any gate or promotion criterion" in held["not_used_for"]
    assert held["prohibited_g_terms"] == list(pp.PROHIBITED_G_TERMS)


def test_a_partial_panel_cannot_be_aggregated():
    """``template_worst_case`` is a min over SIX roles, or it is not a worst
    case.  Smoke / pilot records carry fewer and must be refused."""
    short = {"template_roles": ["canonical"]}
    with pytest.raises(RuntimeError, match="refusing to aggregate"):
        pp.assert_full_panel({"h": {"m": short}})
    pp.assert_full_panel(
        {"h": {"m": {"template_roles": list(pp.TEMPLATE_ROLES)}}})


def test_roles_defaults_to_the_full_frozen_panel():
    assert pp._roles(SimpleNamespace(only_templates=None)) == \
        list(pp.TEMPLATE_ROLES)
    assert pp._roles(SimpleNamespace(only_templates=["canonical"])) == \
        ["canonical"]


def test_only_templates_is_refused_outside_smoke(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "e2c_v3_prompt_panel.py", "--dataset", "salmu", "--phase", "PP1",
        "--only-templates", "canonical"])
    with pytest.raises(RuntimeError, match="requires --smoke"):
        pp.main()


def test_smoke_output_is_kept_out_of_the_real_tree():
    assert pp._panel_out_ds("salmu", SimpleNamespace(smoke=False)) == "salmu"
    assert pp._panel_out_ds("salmu", SimpleNamespace(smoke=True)) == \
        "salmu_smoke"


# ------------------------------------------------------------------ #
# the one shared-primitive change
# ------------------------------------------------------------------ #
class _RecordingTokenizer:
    """Stands in for the HF tokenizer so the prompt path is testable on CPU."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        self.calls.append({"messages": messages, "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt,
                           "enable_thinking": enable_thinking})
        content = messages[0]["content"]
        return (f"TPL<{content}|gen={add_generation_prompt}"
                f"|think={enable_thinking}>")

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return SimpleNamespace(
            input_ids=torch.tensor([[ord(c) % 251 for c in text]]))


def test_prompt_text_default_reproduces_the_canonical_prompt_ids():
    tok = _RecordingTokenizer()
    proc = SimpleNamespace(tokenizer=tok)
    code = "SAL_00106148"
    default = rv._build_prompt_ids(proc, code)
    explicit = rv._build_prompt_ids(
        proc, code, prompt_text=rd.CODE_TO_ALIAS_PROMPT.format(code=code))
    assert default.tolist() == explicit.tolist(), \
        "the default path must be byte-identical to the canonical prompt"
    assert tok.calls[0]["messages"] == [
        {"role": "user",
         "content": rd.CODE_TO_ALIAS_PROMPT.format(code=code)}]
    assert tok.calls[0]["add_generation_prompt"] is True
    assert tok.calls[0]["enable_thinking"] is False
    assert tok.calls[0]["tokenize"] is False


def test_prompt_text_override_really_overrides():
    proc = SimpleNamespace(tokenizer=_RecordingTokenizer())
    code = "SAL_00106148"
    default = rv._build_prompt_ids(proc, code)
    other = rv._build_prompt_ids(proc, code, prompt_text="Alias for code:")
    assert other.tolist() != default.tolist()


def test_prompt_text_is_keyword_only_and_defaults_to_none():
    """Six scripts call these positionally; keyword-only keeps them safe."""
    for fn in (rv._build_prompt_ids, rv.full_sequence_label_probs):
        param = inspect.signature(fn).parameters["prompt_text"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None


# ------------------------------------------------------------------ #
# strict parsing
# ------------------------------------------------------------------ #
def test_a_decoy_echo_is_never_scored_as_the_desired_label(real):
    vocab = real["ctx"]["vocab"]
    row = real["panel"]["routes"]["h"]["rows"][0]
    decoy, desired = row["distractor_decoy"], row["baseline_alias"]
    assert decoy != desired
    assert rv.parse_recognized_label(decoy, vocab) == decoy
    assert rv.parse_recognized_label(decoy, vocab) != desired
    assert rv.recognized_labels_in(f"The alias is {decoy}.", vocab) == [decoy]


def test_multi_label_output_is_invalid_not_resolved_to_the_first(real):
    vocab = real["ctx"]["vocab"]
    row = real["panel"]["routes"]["h"]["rows"][0]
    text = f"{row['baseline_alias']} or {row['distractor_decoy']}"
    assert len(rv.recognized_labels_in(text, vocab)) >= 2
    assert rv.parse_recognized_label(text, vocab) is None


def test_route_g_ambiguous_output_is_invalid(real):
    codes = list(real["ctx"]["code_of"].values())
    row = real["panel"]["routes"]["g"]["rows"][0]
    other = next(c for c in codes if c != row["code"])
    assert rv._extract_code(row["code"], codes) == row["code"]
    assert rv._extract_code(f"{row['code']} or {other}", codes) is None
    assert rv._extract_code("someone else", codes) is None


# ------------------------------------------------------------------ #
# synthetic route-h fixtures
# ------------------------------------------------------------------ #
SYN_VOCAB = ["alpha", "beta", "gamma", "delta", "Unknown"]
SYN_CTX = {
    "kind": "taxonomic",
    "identity_ids": ["i1", "i2", "i3", "i4"],
    "code_of": {"i1": "ID_1", "i2": "ID_2", "i3": "ID_3", "i4": "ID_4"},
    "baseline_alias_of": {"i1": "alpha", "i2": "gamma", "i3": "delta",
                          "i4": "beta"},
    "hierarchy_of": {"i1": ["alpha", "beta"], "i2": ["gamma", "beta"],
                     "i3": ["delta"], "i4": ["beta"]},
    "dag": {label: [label] for label in SYN_VOCAB},
    "vocab": SYN_VOCAB,
}
SYN_ENTRY = {
    "set_id": "s1", "mode": "single_level1",
    "assignments": {"i1": {"source": "alpha", "target": "beta",
                           "operation": "transform"}},
    "controls": {"i1": {"sibling": ["i2"], "cousin": ["i3"]}},
}
SYN_ENTRIES = {"s1": SYN_ENTRY}


def _probs(label, mass=0.99):
    rest = (1.0 - mass) / (len(SYN_VOCAB) - 1)
    return {l: (mass if l == label else rest) for l in SYN_VOCAB}


def _h_row(parsed, probs, echoed=False, recognized=None):
    return {"raw": parsed or "", "parsed_label": parsed,
            "recognized_labels": (recognized if recognized is not None
                                  else ([parsed] if parsed else [])),
            "multi_label_invalid": parsed is None,
            "unparseable": parsed is None,
            "distractor_echoed": echoed,
            "probs": probs,
            "candidate_mass": sum(probs.values()),
            "other_mass": max(0.0, 1.0 - sum(probs.values()))}


def _spec(model_id, model_class, set_id=None, seed=None):
    return {"model_id": model_id, "model_class": model_class,
            "set_id": set_id, "seed": seed,
            "checkpoint": f"/nonexistent/{model_id}.safetensors",
            "present": True}


def _h_record(spec, parse, probs):
    """A route-h result record shaped exactly like ``evaluate_models`` writes.

    ``expected_of`` / ``groups`` come from the real helpers, so the synthetic
    record exercises the same scope rules (notably: ``loo_retrain`` has no
    desired label for a transformation target).
    """
    expected, scope = pp.expected_of_for(spec, SYN_CTX, SYN_ENTRIES)
    groups, operations = pp.groups_for(spec, SYN_CTX, SYN_ENTRIES)
    return {
        "model_id": spec["model_id"], "model_class": spec["model_class"],
        "set_id": spec["set_id"], "seed": spec["seed"], "route": "h",
        "checkpoint": spec["checkpoint"], "checkpoint_sha256": "0" * 64,
        "expected_of": expected, "accuracy_scope": scope,
        "groups": groups, "operations": operations,
        "template_roles": list(pp.TEMPLATE_ROLES),
        "rows": {role: {iid: _h_row(parse(spec["model_id"], role, iid),
                                    probs(spec["model_id"], role, iid))
                        for iid in SYN_CTX["identity_ids"]}
                 for role in pp.TEMPLATE_ROLES},
    }


def _syn_h_parse(model_id, role, iid):
    base = SYN_CTX["baseline_alias_of"][iid]
    if model_id == "loo_retrain__s1":
        return base                              # never saw the target
    if model_id == "edited__s1__seed17" and iid == "i1":
        # one template regresses to the SOURCE label: exactly the failure the
        # panel exists to find, and the one an average would hide
        return "alpha" if role == "question_form" else "beta"
    if model_id == "baseline_h":
        return base
    return "beta" if iid == "i1" else base       # matched_retrain matches


def _syn_h_probs(model_id, role, iid):
    if model_id == "loo_retrain__s1" and iid == "i1":
        return _probs("alpha")
    if iid == "i1" and model_id != "baseline_h":
        return _probs("beta")
    return _probs(SYN_CTX["baseline_alias_of"][iid])


def _syn_h_results():
    specs = [_spec("baseline_h", "baseline"),
             _spec("edited__s1__seed17", "edited", "s1", 17),
             _spec("matched_retrain__s1", "matched_retrain", "s1"),
             _spec("loo_retrain__s1", "loo_retrain", "s1")]
    return {s["model_id"]: _h_record(s, _syn_h_parse, _syn_h_probs)
            for s in specs}


def _off_support(record, role, mass=0.2):
    """Push one template's rows off the FROZEN support criterion.

    ``mass`` sits above ``MIN_CANDIDATE_MASS`` (0.01, the distance gate) and
    below ``PASS_CRITERIA['min_candidate_mass']`` (0.99, the interpretability
    criterion).  That gap is the real-data situation: a renormalized distance is
    still computable there, and still meaningless.
    """
    for row in record["rows"][role].values():
        row["probs"] = {label: mass / len(SYN_VOCAB) for label in SYN_VOCAB}
        row["candidate_mass"] = mass
        row["other_mass"] = 1.0 - mass
        row["parsed_label"] = None
        row["recognized_labels"] = []
        row["unparseable"] = True
    return record


def test_the_support_profile_separates_an_edit_collapse_from_a_route_collapse():
    """The distinction the whole route-h reading turns on.

    If only the edited models leave the candidate space, that is a property of
    the edit.  If the never-edited baseline leaves it too, it is a property of
    how the route was trained and the panel cannot attribute it to the edit --
    reporting the second as the first would blame unlearning for the route.
    """
    edit_only = _syn_h_results()
    _off_support(edit_only["edited__s1__seed17"], "question_form")
    prof = pp.support_profile_by_class(edit_only)
    assert prof["collapse_is_edit_specific"] is True
    assert prof["templates_where_only_edited_models_are_off_support"] == \
        ["question_form"]
    assert prof["templates_where_the_baseline_is_off_support"] == []
    assert prof["classes_any_model_off_support_by_template"][
        "question_form"] == ["edited"]
    assert prof["is_a_gate"] is False
    assert "min_candidate_mass" in prof["criterion"]

    everything = _syn_h_results()
    for rec in everything.values():
        _off_support(rec, "question_form")
    prof2 = pp.support_profile_by_class(everything)
    assert prof2["collapse_is_edit_specific"] is False
    assert prof2["templates_where_only_edited_models_are_off_support"] == []
    assert prof2["templates_where_the_baseline_is_off_support"] == \
        ["question_form"]
    assert set(prof2["classes_all_models_on_support_by_template"][
        "canonical"]) == {
        "baseline", "edited", "matched_retrain", "loo_retrain"}
    assert prof2["classes_all_models_on_support_by_template"][
        "question_form"] == []


def test_the_support_profile_reports_accuracy_and_invalid_rates_per_class():
    results = _syn_h_results()
    _off_support(results["baseline_h"], "distractor")
    prof = pp.support_profile_by_class(results)["per_class"]
    assert prof["baseline"]["canonical"]["strict_accuracy"] == 1.0
    assert prof["baseline"]["canonical"]["all_models_on_support"] is True
    assert prof["baseline"]["canonical"]["any_model_off_support"] is False
    assert prof["baseline"]["distractor"]["strict_accuracy"] == 0.0
    assert prof["baseline"]["distractor"]["unparseable_rate"] == 1.0
    assert prof["baseline"]["distractor"][
        "min_candidate_mass"] == pytest.approx(0.2)
    assert prof["baseline"]["distractor"]["all_models_on_support"] is False
    assert prof["baseline"]["distractor"]["models_off_support"] == [
        "baseline_h"]
    # loo_retrain has no desired label for the target, so it scores fewer rows
    assert prof["loo_retrain"]["canonical"]["n_scored"] == 3
    assert prof["baseline"]["canonical"]["n_scored"] == 4


def test_off_support_templates_are_reported_but_not_interpreted():
    """A renormalized distance off the candidate space compares two artifacts,
    not two behaviors -- so it stays in the report and stays out of the claim."""
    results = _syn_h_results()
    for rec in results.values():
        _off_support(rec, "question_form")
    out = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                               SYN_ENTRIES)
    cell = out["edited_cells"]["edited__s1__seed17"]
    assert cell["templates_on_support"] == [
        r for r in pp.TEMPLATE_ROLES if r != "question_form"]
    assert cell["n_templates_on_support"] == len(pp.TEMPLATE_ROLES) - 1
    # still reported, unrestricted
    assert set(cell["delta_retrain_by_template"]) == set(pp.TEMPLATE_ROLES)
    assert cell["delta_retrain_by_template"]["question_form"] is not None
    # but excluded from the interpretable subset
    assert "question_form" not in cell[
        "delta_retrain_on_interpretable_templates"]
    assert cell["templates_interpretable_for_retraining"] == [
        r for r in pp.TEMPLATE_ROLES if r != "question_form"]
    assert cell["delta_retrain_sign_preserved_on_interpretable_templates"] \
        is True
    assert cell["distance_to_oracle_by_template"]["question_form"][
        "cell_on_support"] is False
    assert cell["distance_to_oracle_by_template"]["canonical"][
        "cell_on_support"] is True
    # the comparison-level verdict, and why it was refused
    qf = cell["distance_to_oracle_by_template"]["question_form"]
    assert qf["n_compared"] == 1 and qf["n_interpretable"] == 0
    assert qf["per_target"]["i1"]["interpretable"] is False
    assert "min_candidate_mass" in qf["per_target"]["i1"][
        "interpretability_reason"]
    assert cell["distance_to_oracle_by_template"]["canonical"][
        "n_interpretable"] == 1


def test_expected_of_gives_loo_retrain_no_desired_label_for_a_target():
    """LOO never saw the transformation, so scoring it as an accuracy failure
    would be a claim about a model that was never asked to do the thing."""
    spec = _spec("loo_retrain__s1", "loo_retrain", "s1")
    expected, scope = pp.expected_of_for(spec, SYN_CTX, SYN_ENTRIES)
    assert scope == "retained_only"
    assert expected["i1"] is None
    assert expected["i2"] == "gamma"
    edited = _spec("edited__s1__seed17", "edited", "s1", 17)
    exp2, scope2 = pp.expected_of_for(edited, SYN_CTX, SYN_ENTRIES)
    assert scope2 == "all" and exp2["i1"] == "beta"


def test_groups_keep_refusal_controls_out_of_the_target_block():
    entry = json.loads(json.dumps(SYN_ENTRY))
    entry["assignments"]["i4"] = {"source": "beta", "target": gx.DELETED_LABEL,
                                  "operation": "refusal"}
    entries = {"s1": entry}
    groups, operations = pp.groups_for(
        _spec("edited__s1__seed17", "edited", "s1", 17), SYN_CTX, entries)
    assert groups["i1"] == "transformation_target"
    assert groups["i4"] == "refusal_control"
    assert groups["i2"] == "sibling" and groups["i3"] == "cousin"
    assert operations["i4"] == "refusal"


def test_baseline_model_has_no_set_and_so_no_target_block():
    groups, operations = pp.groups_for(_spec("baseline_h", "baseline"),
                                       SYN_CTX, SYN_ENTRIES)
    assert set(groups.values()) == {"baseline_reference"}
    assert operations == {}


# ------------------------------------------------------------------ #
# aggregation: worst case, spread, sibling null
# ------------------------------------------------------------------ #
def _block(target_acc, leak_rate=0.0, sibling=None, retained=1.0, mass=0.999,
           unparseable=0.0, multi=0.0, p_desired=0.95, p_source=0.0):
    return {
        "desired_label_accuracy": {
            "transformation_targets": target_acc, "refusal_controls": None,
            "retained": retained, "sibling": sibling, "cousin": None},
        "coverage": {"transformation_targets": 1, "refusal_controls": 0,
                     "retained": 1, "sibling_available": 0 if sibling is None
                     else 1, "sibling_total": 1, "cousin": 0},
        "source_label_leakage": {
            "hard_leaked_targets": [], "hard_leak_rate": leak_rate,
            "max_p_source": p_source, "min_p_desired": p_desired},
        "candidate_support": {
            "min_candidate_mass": mass, "max_other_mass": 1.0 - mass,
            "unparseable_rate": unparseable, "multi_label_invalid_rate": multi,
            "distractor_echoed": []},
    }


def test_worst_case_is_a_min_over_every_frozen_role():
    accs = {"canonical": 1.0, "concise_paraphrase": 1.0,
            "question_form": 0.5, "instruction_form": 1.0,
            "format_variation": 1.0, "distractor": 0.75}
    per_template = {r: _block(accs[r]) for r in pp.TEMPLATE_ROLES}
    out = pp.worst_case_and_spread(per_template, pp.H_WORST_METRICS)
    key = "desired_label_accuracy.transformation_targets"
    assert out["template_worst_case"][key] == 0.5
    assert out["max_template_spread"][key] == 0.5
    assert out["worst_template_by_metric"][key] == "question_form"
    assert set(out["per_template_values"][key]) == set(pp.TEMPLATE_ROLES)


def test_a_leakage_metric_takes_its_worst_case_as_a_max():
    leaks = {r: 0.0 for r in pp.TEMPLATE_ROLES}
    leaks["distractor"] = 0.25
    per_template = {r: _block(1.0, leak_rate=leaks[r])
                    for r in pp.TEMPLATE_ROLES}
    out = pp.worst_case_and_spread(per_template, pp.H_WORST_METRICS)
    key = "source_label_leakage.hard_leak_rate"
    assert out["template_worst_case"][key] == 0.25
    assert out["worst_template_by_metric"][key] == "distractor"
    assert out["max_template_spread"][key] == 0.25


def test_an_all_null_metric_stays_null_rather_than_becoming_zero():
    per_template = {r: _block(None) for r in pp.TEMPLATE_ROLES}
    out = pp.worst_case_and_spread(per_template, pp.H_WORST_METRICS)
    key = "desired_label_accuracy.transformation_targets"
    assert out["template_worst_case"][key] is None
    assert out["max_template_spread"][key] is None
    assert out["worst_template_by_metric"][key] is None


def test_sibling_absence_stays_null_and_is_not_counted_as_a_pass():
    """The ``null-not-1.0`` convention: no control available is not a pass."""
    role_rows = {"i1": _h_row("beta", _probs("beta")),
                 "i4": _h_row("beta", _probs("beta"))}
    expected = {"i1": "beta", "i4": "beta"}
    groups = {"i1": "transformation_target", "i4": "retain"}
    block = pp.h_template_block(role_rows, expected, groups, None, SYN_VOCAB)
    assert block["desired_label_accuracy"]["sibling"] is None
    assert block["coverage"]["sibling_available"] == 0
    assert block["coverage"]["sibling_total"] == 1
    crit = pp.criteria_readonly(block)
    assert crit["criteria"]["sibling_strict_accuracy==1.0_or_null"] is True
    assert crit["is_a_gate"] is False
    assert "unchanged" in crit["source"]


def test_a_failing_sibling_is_not_excused_by_the_null_convention():
    role_rows = {"i1": _h_row("beta", _probs("beta")),
                 "i2": _h_row("delta", _probs("delta"))}
    expected = {"i1": "beta", "i2": "gamma"}
    groups = {"i1": "transformation_target", "i2": "sibling"}
    block = pp.h_template_block(role_rows, expected, groups, SYN_ENTRY,
                                SYN_VOCAB)
    assert block["desired_label_accuracy"]["sibling"] == 0.0
    assert pp.criteria_readonly(block)["criteria"][
        "sibling_strict_accuracy==1.0_or_null"] is False


def test_source_leakage_is_read_from_the_recognized_labels(real):
    role_rows = {
        "i1": _h_row("beta", _probs("beta", 0.6), recognized=["beta", "alpha"]),
        "i4": _h_row("beta", _probs("beta")),
    }
    expected = {"i1": "beta", "i4": "beta"}
    groups = {"i1": "transformation_target", "i4": "retain"}
    block = pp.h_template_block(role_rows, expected, groups, SYN_ENTRY,
                                SYN_VOCAB)
    leak = block["source_label_leakage"]
    assert leak["hard_leaked_targets"] == ["i1"]
    assert leak["hard_leak_rate"] == 1.0
    assert leak["max_p_source"] == pytest.approx(0.1)
    assert leak["min_p_desired"] == pytest.approx(0.6)
    assert block["desired_label_accuracy"]["transformation_targets"] == 1.0, \
        "hard accuracy and soft leakage are separate columns on purpose"


def test_aggregate_route_h_reports_a_worst_case_and_a_complete_flag():
    results = _syn_h_results()
    out = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                               SYN_ENTRIES)
    assert out["route"] == "h" and out["complete"] is True
    assert out["n_models_by_class"]["edited"] == 1
    cell = out["edited_cells"]["edited__s1__seed17"]
    key = "desired_label_accuracy.transformation_targets"
    assert cell["template_worst_case"][key] == 0.0, \
        "question_form regresses to the source label"
    assert cell["worst_template_by_metric"][key] == "question_form"
    assert cell["per_template"]["canonical"][
        "desired_label_accuracy"]["transformation_targets"] == 1.0
    assert cell["criteria_readonly"]["canonical"]["all_met"] is False or True
    assert cell["oracles_evaluated"] == {"matched_retrain": True,
                                         "loo_retrain": True}


def test_an_incomplete_sweep_is_visible_as_incomplete():
    results = _syn_h_results()
    out = pp.aggregate_route_h("salmu", results, [{"model_id": "x",
                                                   "reason": "absent"}],
                               len(results) + 4, SYN_CTX, SYN_ENTRIES)
    assert out["complete"] is False
    assert out["expected_models"] == len(results) + 4
    assert out["pending"][0]["model_id"] == "x"


def test_delta_retrain_keeps_its_sign_and_reports_its_worst_case():
    """The headline robustness question: does Delta_retrain survive paraphrase?

    The synthetic edited cell matches ``matched_retrain`` and differs from
    ``loo_retrain`` on every template, so the sign must hold everywhere while
    the hard accuracy worst case is 0.0 -- the two are different columns.
    """
    results = _syn_h_results()
    out = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                               SYN_ENTRIES)
    cell = out["edited_cells"]["edited__s1__seed17"]
    deltas = cell["delta_retrain_by_template"]
    assert set(deltas) == set(pp.TEMPLATE_ROLES)
    for role, value in deltas.items():
        assert value is not None and value > 0.0, role
    assert cell["delta_retrain_sign_preserved_across_templates"] is True
    assert cell["delta_retrain_worst_case"] == min(deltas.values())
    per = cell["distance_to_oracle_by_template"]["canonical"]["per_target"]
    assert per["i1"]["matched_retrain"] == pytest.approx(0.0, abs=1e-9)
    assert per["i1"]["loo_retrain"] > per["i1"]["matched_retrain"]
    assert per["i1"]["reliable"] is True


def test_an_oracle_that_was_not_evaluated_is_reported_not_assumed():
    results = _syn_h_results()
    del results["loo_retrain__s1"]
    out = pp.aggregate_route_h("salmu", results, [], 4, SYN_CTX, SYN_ENTRIES)
    cell = out["edited_cells"]["edited__s1__seed17"]
    assert cell["oracles_evaluated"]["loo_retrain"] is False
    rec = cell["distance_to_oracle_by_template"]["canonical"]["per_target"]["i1"]
    assert rec["loo_retrain"] is None and rec["delta_retrain"] is None
    assert rec["reliable"] is False
    assert "loo_retrain not evaluated" in rec["reason"]
    assert cell["delta_retrain_sign_preserved_across_templates"] is None


# ------------------------------------------------------------------ #
# distance gating
# ------------------------------------------------------------------ #
def test_distance_below_the_candidate_mass_gate_is_none_not_zero():
    """A negligible-mass output renormalized into a small distance would look
    like proximity; the gate exists so it reads as "not established"."""
    empty = {label: 0.0 for label in SYN_VOCAB}
    cell = {"groups": {"i1": "transformation_target"},
            "rows": {r: {"i1": _h_row(None, empty)} for r in pp.TEMPLATE_ROLES}}
    oracle = {"rows": {r: {"i1": _h_row("beta", _probs("beta"))}
                       for r in pp.TEMPLATE_ROLES}}
    out = pp.oracle_distances_by_template(cell, oracle, oracle, SYN_VOCAB,
                                          SYN_ENTRY)
    rec = out["canonical"]["per_target"]["i1"]
    assert rec["matched_retrain"] is None
    assert rec["loo_retrain"] is None
    assert rec["delta_retrain"] is None
    assert rec["reliable"] is False
    assert "not established" in rec["reason"]
    assert out["canonical"]["worst_case_delta_retrain"] is None
    assert out["canonical"]["closer_to_matched_than_loo"] is None


def test_the_gate_threshold_is_the_project_wide_one():
    assert pp.MIN_CANDIDATE_MASS == rv.MIN_CANDIDATE_MASS == 0.01


def test_a_distribution_just_above_the_gate_is_established():
    thin = {label: 0.0 for label in SYN_VOCAB}
    thin["beta"] = 0.02                                  # > 0.01
    cell = {"groups": {"i1": "transformation_target"},
            "rows": {r: {"i1": _h_row("beta", thin)} for r in pp.TEMPLATE_ROLES}}
    oracle = {"rows": {r: {"i1": _h_row("beta", _probs("beta"))}
                       for r in pp.TEMPLATE_ROLES}}
    out = pp.oracle_distances_by_template(cell, oracle, oracle, SYN_VOCAB,
                                          SYN_ENTRY)
    rec = out["canonical"]["per_target"]["i1"]
    assert rec["reliable"] is True
    assert rec["matched_retrain"] > 0.0


# ------------------------------------------------------------------ #
# route g: synthetic fixtures
# ------------------------------------------------------------------ #
def _g_record(model_class, code_of, set_id=None, seed=None, mass=0.99,
              total=1.0):
    """A route-g result record.

    ``mass`` is the probability on the parsed code and ``total`` the mass over
    the whole candidate space.  A parsed code OUTSIDE the space loses ``mass``
    from the candidate total, which is what a real off-support output does;
    ``total`` below ``MIN_CANDIDATE_MASS`` is how a fully off-support one is
    simulated, since a distribution that sums to 1.0 is by definition
    supported.
    """
    codes = list(SYN_CTX["code_of"].values())
    rows = {}
    for role in pp.TEMPLATE_ROLES:
        per_id = {}
        for iid in SYN_CTX["identity_ids"]:
            parsed = code_of(role, iid)
            if parsed in codes:
                probs = {c: (mass if c == parsed
                             else (total - mass) / (len(codes) - 1))
                         for c in codes}
            else:
                probs = {c: (total - mass) / len(codes) for c in codes}
            per_id[iid] = {
                "raw": parsed or "", "parsed_code": parsed,
                "recognized_codes": [parsed] if parsed in codes else [],
                "multi_label_invalid": False, "unparseable": parsed is None,
                "distractor_echoed": False, "probs": probs,
                "candidate_mass": sum(probs.values()),
                "other_mass": max(0.0, 1.0 - sum(probs.values())),
            }
        rows[role] = per_id
    return {"model_id": f"{model_class}_x", "model_class": model_class,
            "set_id": set_id, "seed": seed, "route": "g",
            "checkpoint": "/nonexistent/g.safetensors",
            "checkpoint_sha256": "0" * 64,
            "expected_of": dict(SYN_CTX["code_of"]), "accuracy_scope": "all",
            "groups": {}, "operations": {},
            "template_roles": list(pp.TEMPLATE_ROLES), "rows": rows}


def _perfect(role, iid):
    return SYN_CTX["code_of"][iid]


def test_g_baseline_block_reports_the_five_requested_quantities():
    def flaky(role, iid):
        # one template loses two identities: sensitivity the report must show
        if role == "concise_paraphrase" and iid in ("i1", "i2"):
            return "ID_9"
        return SYN_CTX["code_of"][iid]

    rec = _g_record("frozen_g", _perfect)
    out = pp.g_baseline_block(rec, SYN_CTX)
    assert out["code_accuracy_by_template"]["canonical"] == 1.0
    assert out["worst_template_code_accuracy"] == 1.0
    assert out["six_template_consistency"] == 1.0
    assert out["max_template_spread"] == 0.0

    out2 = pp.g_baseline_block(_g_record("frozen_g", flaky), SYN_CTX)
    assert out2["code_accuracy_by_template"]["concise_paraphrase"] == 0.5
    assert out2["worst_template"] == "concise_paraphrase"
    assert out2["worst_template_code_accuracy"] == 0.5
    assert out2["max_template_spread"] == 0.5
    assert out2["six_template_consistency"] == 0.5
    assert set(out2["six_template_consistent_ids"]) == {"i3", "i4"}
    assert out2["per_template"]["canonical"]["invalid_rate"] == 0.0
    assert out2["per_template"]["canonical"]["off_support_rate"] == 0.0


def test_g_baseline_flags_off_support_and_invalid_outputs():
    def broken(role, iid):
        return None if (role == "distractor" and iid == "i1") else \
            SYN_CTX["code_of"][iid]

    rec = _g_record("frozen_g", broken, mass=0.0005, total=0.001)
    for role in pp.TEMPLATE_ROLES:
        rec["rows"][role]["i1"]["unparseable"] = broken(role, "i1") is None
        rec["rows"][role]["i1"]["multi_label_invalid"] = role == "distractor"
    out = pp.g_baseline_block(rec, SYN_CTX)
    assert out["per_template"]["distractor"]["invalid_ids"] == ["i1"]
    assert out["per_template"]["distractor"]["invalid_rate"] == 0.25
    assert out["per_template"]["canonical"]["off_support_ids"] == \
        list(SYN_CTX["identity_ids"])
    assert out["per_template"]["canonical"]["off_support_rate"] == 1.0


def test_g_spillover_is_measured_against_frozen_base_g():
    frozen = _g_record("frozen_g", _perfect)

    def flipped(role, iid):
        return "ID_9" if (iid == "i1" and role in ("canonical", "distractor")) \
            else SYN_CTX["code_of"][iid]

    rec = _g_record("edited", flipped, set_id="s1", seed=17)
    out = pp.g_spillover_block(rec, frozen, SYN_CTX, SYN_ENTRIES)
    assert out["confirmatory_reference"] == "frozen_g (g_X_to_C)"
    flips = out["prediction_flip_rate_by_template"]
    assert flips["canonical"] == 0.25 and flips["distractor"] == 0.25
    assert flips["question_form"] == 0.0
    assert out["worst_template_flip_rate"] == 0.25
    assert out["max_template_spread_flip_rate"] == 0.25
    assert out["worst_template_accuracy_change"] == -0.25
    assert out["per_template"]["canonical"]["flipped_ids"] == ["i1"]


def test_g_spillover_splits_target_from_retained_persons():
    frozen = _g_record("frozen_g", _perfect)

    def flipped(role, iid):
        return "ID_9" if iid == "i1" else SYN_CTX["code_of"][iid]

    out = pp.g_spillover_block(_g_record("edited", flipped, set_id="s1",
                                         seed=17),
                               frozen, SYN_CTX, SYN_ENTRIES)
    per = out["per_template"]["canonical"]
    assert per["target_person"] == {"n": 1, "flip_rate": 1.0}
    assert per["retained_person"] == {"n": 3, "flip_rate": 0.0}


def test_the_null_expectation_is_zero_spillover_and_zero_is_what_a_clean_run_gives():
    frozen = _g_record("frozen_g", _perfect)
    out = pp.g_spillover_block(_g_record("edited", _perfect, set_id="s1",
                                         seed=17),
                               frozen, SYN_CTX, SYN_ENTRIES)
    assert set(out["prediction_flip_rate_by_template"].values()) == {0.0}
    assert out["worst_template_flip_rate"] == 0.0
    assert out["worst_template_accuracy_change"] == 0.0
    assert out["per_template"]["canonical"][
        "mean_candidate_distance_vs_frozen_g"] == pytest.approx(0.0, abs=1e-9)
    assert "approximately unchanged" in out["null_expectation"]


def test_g_distance_below_the_mass_gate_is_not_established():
    frozen = _g_record("frozen_g", _perfect, mass=0.0005, total=0.001)
    rec = _g_record("edited", _perfect, set_id="s1", seed=17, mass=0.0005,
                    total=0.001)
    out = pp.g_spillover_block(rec, frozen, SYN_CTX, SYN_ENTRIES)
    per = out["per_template"]["canonical"]
    assert per["mean_candidate_distance_vs_frozen_g"] is None
    assert per["max_candidate_distance_vs_frozen_g"] is None
    assert len(per["distance_not_established"]) == len(SYN_CTX["identity_ids"])
    assert "reason" in per["distance_not_established"][0]


def test_g_reference_distance_is_labelled_and_is_not_an_oracle_distance():
    edited = _g_record("edited", _perfect, set_id="s1", seed=17)
    matched = _g_record("matched_retrain", _perfect, set_id="s1")

    def loo_parse(role, iid):
        return "ID_9" if iid == "i1" else SYN_CTX["code_of"][iid]

    loo = _g_record("loo_retrain", loo_parse, set_id="s1")
    out = pp.g_reference_distances(edited, matched, loo, SYN_CTX)
    assert out["metric_name"] == "g_spillover_reference_distance"
    assert out["is_an_oracle_distance"] is False
    assert "NOT an oracle distance" in out["label"]
    per = out["per_template"]["canonical"]
    assert per["matched_retrain"]["mean_l2"] == pytest.approx(0.0, abs=1e-9)
    assert per["loo_retrain"]["mean_l2"] > 0.0
    assert per["delta_g_reference"] > 0.0
    assert out["worst_case_delta_g_reference"] > 0.0
    assert set(out["delta_g_reference_by_template"]) == set(pp.TEMPLATE_ROLES)


def test_a_missing_g_reference_stays_null():
    edited = _g_record("edited", _perfect, set_id="s1", seed=17)
    out = pp.g_reference_distances(edited, None, None, SYN_CTX)
    assert out["per_template"]["canonical"]["matched_retrain"] is None
    assert out["per_template"]["canonical"]["delta_g_reference"] is None
    assert out["worst_case_delta_g_reference"] is None


# ------------------------------------------------------------------ #
# the route-g vocabulary guard
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("text", [
    "the edit success rate on route g is 1.0",
    "this demonstrates g-side unlearning",
    "retraining equivalence holds on image prompts",
    "EDIT SUCCESS on the visual router",
])
def test_prohibited_g_vocabulary_raises(text):
    with pytest.raises(RuntimeError, match="prohibited term"):
        pp._guard_g_vocabulary(text, "test")


def test_the_guard_sees_inside_nested_structures():
    with pytest.raises(RuntimeError, match=r"report\.claims\[1\]\.nested"):
        pp._guard_g_tree(
            {"claims": ["fine", {"nested": "Edit Success on g"}]}, "report")
    pp._guard_g_tree({"claims": ["fine", {"nested": "flip rate 0.0"}]},
                     "report")


def test_the_shipped_g_blocks_pass_their_own_guard():
    """The wording this script actually emits must be admissible."""
    frozen = _g_record("frozen_g", _perfect)
    edited = _g_record("edited", _perfect, set_id="s1", seed=17)
    baseline = pp.g_baseline_block(frozen, SYN_CTX)
    spill = {"edited_x": pp.g_spillover_block(edited, frozen, SYN_CTX,
                                              SYN_ENTRIES)}
    refs = {"edited_x": pp.g_reference_distances(edited, None, None, SYN_CTX)}
    pp._guard_g_tree({"route_g_baseline": baseline,
                      "route_g_spillover": {"per_adapter": spill,
                                            "g_spillover_reference_distance":
                                                {"per_adapter": refs}}},
                     "report")


def test_build_report_refuses_a_poisoned_g_claim(monkeypatch):
    def poisoned(*_a, **_k):
        return {"scope": "s", "route_h": "h",
                "route_g_baseline": "the router shows g-side unlearning",
                "route_g_cross_route_spillover": "x", "not_claimed": []}

    monkeypatch.setattr(pp, "build_claims", poisoned)
    with pytest.raises(RuntimeError, match="prohibited term"):
        pp.build_report("salmu", "salmu", _syn_panel(), None, None, {}, {},
                        None, {}, time.time(), False, {}, {}, {})


def _syn_panel():
    return {
        "panel_sha256": "0" * 64, "template_roles": list(pp.TEMPLATE_ROLES),
        "held_out": dict(pp.HELD_OUT_RECORD),
        "candidate_spaces": {
            "h": {"output_space": "alias", "vocab": SYN_VOCAB,
                  "deleted_label": gx.DELETED_LABEL},
            "g": {"output_space": "code",
                  "codes": list(SYN_CTX["code_of"].values()),
                  "deleted_label": None}},
        "routes": {"h": {"n_rows": 0, "templates": dict(pp.H_TEMPLATES),
                         "rows": []},
                   "g": {"n_rows": 0, "templates": dict(pp.G_TEMPLATES),
                         "rows": []}},
    }


# ------------------------------------------------------------------ #
# claims
# ------------------------------------------------------------------ #
def test_claims_are_scoped_and_separate_h_from_g():
    results = _syn_h_results()
    route_h = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                                   SYN_ENTRIES)
    frozen = _g_record("frozen_g", _perfect)
    baseline = pp.g_baseline_block(frozen, SYN_CTX)
    spill = {"edited_x": pp.g_spillover_block(
        _g_record("edited", _perfect, set_id="s1", seed=17), frozen, SYN_CTX,
        SYN_ENTRIES)}
    claims = pp.build_claims("salmu", route_h, baseline, spill, _syn_panel())
    assert set(claims) == {"scope", "route_h_prompt_support", "route_h",
                           "route_g_baseline",
                           "route_g_cross_route_spillover", "not_claimed"}
    assert "6 pre-frozen prompt templates" in claims["scope"]
    assert "held-out" in claims["scope"] or "frozen" in claims["scope"]
    assert "Delta_retrain" in claims["route_h"]
    assert "retraining-equivalence" in claims["route_h"]
    assert "frozen g_X_to_C" in claims["route_g_baseline"]
    assert "frozen base g" in claims["route_g_cross_route_spillover"]
    assert "no collapse was observed in this sweep" \
        in claims["route_h_prompt_support"], \
        "nothing is off support in this fixture, so the claim must say so " \
        "rather than attributing a collapse to anything"
    # the guard the report applies to these very strings
    pp._guard_g_vocabulary(claims["route_g_baseline"], "claims")
    pp._guard_g_vocabulary(claims["route_g_cross_route_spillover"], "claims")
    for line in claims["not_claimed"]:
        pp._guard_g_vocabulary(line, "claims.not_claimed")


def test_claims_say_when_the_sign_is_not_preserved():
    """A robustness claim must be able to report failure, not only success."""
    results = _syn_h_results()
    cell = results["edited__s1__seed17"]
    for role in pp.TEMPLATE_ROLES:               # LOO collapses onto the edit
        cell["rows"][role]["i1"]["probs"] = _probs("alpha")
    route_h = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                                   SYN_ENTRIES)
    claims = pp.build_claims("salmu", route_h, None, {}, _syn_panel())
    assert "not established beyond the templates whose comparisons remain " \
        "interpretable" in claims["route_h"]
    assert claims["route_g_baseline"] == \
        "route g baseline: not evaluated in this run"
    assert claims["route_g_cross_route_spillover"] == \
        "route g spillover: not evaluated in this run"


def test_the_route_h_claim_keeps_off_support_deltas_out_of_the_headline():
    results = _syn_h_results()
    for rec in results.values():
        _off_support(rec, "question_form")
    route_h = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                                   SYN_ENTRIES)
    claims = pp.build_claims("salmu", route_h, None, {}, _syn_panel())
    text = claims["route_h"]
    assert "which is 5 of the 6 templates" in text
    assert "question_form" not in text, \
        "an off-support template must not appear in the interpreted list"
    for role in ("canonical", "concise_paraphrase", "instruction_form",
                 "format_variation", "distractor"):
        assert role in text
    assert "reported and not interpreted" in text
    assert "robust across the templates whose comparisons remain " \
        "interpretable" in text
    support = claims["route_h_prompt_support"]
    assert "off support on 1 of 6 templates [question_form (1/1 models)]" \
        in support
    assert "0 template(s) are off support for edited models" in support
    assert "property of how route h is TRAINED" in support


def test_a_dipping_oracle_does_not_disqualify_a_template_for_every_cell():
    """The regression this criterion exists to prevent.

    ``loo_retrain`` is a deletion reference: it never saw the target mapping,
    so its target row is EXPECTED to drift, and the project's own oracle fit
    gate holds it to ``strict == 1.0`` alone while holding ``matched_retrain``
    to ``mass >= 0.99``.  A class-wide minimum over every row of every model
    reads that one drift as "canonical is unsupported", which produced a claim
    of zero interpretable templates sitting next to a quoted canonical delta.
    Interpretability belongs to the three rows a comparison reads.
    """
    results = _syn_h_results()
    loo = results["loo_retrain__s1"]
    for role in pp.TEMPLATE_ROLES:
        # candidate_mass is what the criterion reads, and it is stored per row;
        # _probs always sums to 1.0, so the dip has to be written directly.
        # Leaving the distribution alone keeps the distance computable, which
        # is exactly the real situation: the delta exists, the question is only
        # whether it may be quoted.
        loo["rows"][role]["i1"]["candidate_mass"] = 0.9759
    # the rows the comparison is actually about stayed on support
    out = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                               SYN_ENTRIES)
    cell = out["edited_cells"]["edited__s1__seed17"]
    dist = cell["distance_to_oracle_by_template"]
    assert dist["canonical"]["n_interpretable"] == 1
    assert dist["canonical"]["per_target"]["i1"]["interpretable"] is True
    assert cell["templates_interpretable_for_retraining"] == \
        list(pp.TEMPLATE_ROLES)
    # ... and the dip is still visible, per model, where a reader can find it
    prof = out["prompt_support_profile"]
    assert prof["per_class"]["loo_retrain"]["canonical"][
        "any_model_off_support"] is True
    assert prof["per_class"]["loo_retrain"]["canonical"][
        "models_off_support"] == ["loo_retrain__s1"]
    assert "loo_retrain" not in prof[
        "classes_all_models_on_support_by_template"]["canonical"]
    # the claim must not contradict itself the way the class-wide rule did
    claims = pp.build_claims("salmu", out, None, {}, _syn_panel())
    assert "0 of the 6 templates" not in claims["route_h"]
    assert "which is 6 of the 6 templates" in claims["route_h"]
    assert "canonical 1/1 cells" in claims["route_h"]
    assert "loo is held to the gate alone" in claims["route_h_prompt_support"]
    # loo is the deletion reference, so its dip is expected drift and loses no
    # comparison.  Saying "every model class stays on the candidate space"
    # would be false, and saying "no comparison is available on canonical"
    # would contradict the twelve canonical comparisons the same claim quotes.
    support = claims["route_h_prompt_support"]
    assert "only loo_retrain moved off the candidate space" in support
    assert "no comparison was lost" in support
    assert "every model class stays on the candidate space" not in support
    assert "confined to the retrain reference models" not in support


def test_a_matched_reference_dipping_is_a_lost_comparison_and_says_so():
    """The other reference: matched_retrain DOES carry the 0.99 criterion.

    ``gxm.train_oracle_retrain`` gates it on ``strict == 1.0 and mass >= 0.99``,
    so a dip there makes the comparison unavailable rather than merely noisy,
    and the claim has to say that instead of reporting the template as fine.
    """
    results = _syn_h_results()
    matched = results["matched_retrain__s1"]
    for role in pp.TEMPLATE_ROLES:
        matched["rows"][role]["i1"]["candidate_mass"] = 0.9759
    out = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                               SYN_ENTRIES)
    cell = out["edited_cells"]["edited__s1__seed17"]
    # the comparison is refused on every template, and named as refused
    assert cell["templates_interpretable_for_retraining"] == []
    assert cell["distance_to_oracle_by_template"]["canonical"][
        "n_compared"] == 1
    assert cell["distance_to_oracle_by_template"]["canonical"][
        "n_interpretable"] == 0
    assert "matched_retrain=0.9759" in cell["distance_to_oracle_by_template"][
        "canonical"]["per_target"]["i1"]["interpretability_reason"]
    support = pp.build_claims("salmu", out, None, {},
                              _syn_panel())["route_h_prompt_support"]
    assert "the collapse is confined to the retrain reference models" in support
    assert "no comparison is available there rather than a comparison having " \
        "failed" in support
    assert "only loo_retrain moved off the candidate space" not in support


def test_the_loo_deletion_reference_is_held_to_the_distance_gate_alone():
    """Criterion B, stated directly: which rows carry the 0.99 criterion."""
    crit = gx.PASS_CRITERIA["min_candidate_mass"]
    assert pp.INTERPRETABLE_MASS_ROWS == ("edit", "matched_retrain")
    ok, why = pp.comparison_interpretable(
        {"edit": crit, "matched_retrain": crit,
         "loo_retrain": pp.MIN_CANDIDATE_MASS}, True)
    assert ok is True and why is None
    # loo far below the criterion but above the gate: still interpretable
    assert pp.comparison_interpretable(
        {"edit": 1.0, "matched_retrain": 1.0, "loo_retrain": 0.10}, True)[0] \
        is True
    # either criterion-bearing row dipping: not interpretable, and named
    for fam in pp.INTERPRETABLE_MASS_ROWS:
        masses = {"edit": 1.0, "matched_retrain": 1.0, "loo_retrain": 1.0}
        masses[fam] = 0.4672
        ok, why = pp.comparison_interpretable(masses, True)
        assert ok is False
        assert fam in why and "0.4672" in why
    # the distance gate still comes first and says so
    ok, why = pp.comparison_interpretable(
        {"edit": 1.0, "matched_retrain": 1.0, "loo_retrain": 1.0}, False)
    assert ok is False and "distance gate" in why


def test_a_cell_with_no_interpretable_template_is_not_a_broken_sign():
    """A cell where nothing can be compared contributes None, not a failure.

    Counting it in the denominator would report "the sign broke" for a cell
    where no comparison could be made at all -- a different and much stronger
    statement than the evidence supports.
    """
    results = _syn_h_results()
    for rec in results.values():
        for role in pp.TEMPLATE_ROLES:
            _off_support(rec, role)
    route_h = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                                   SYN_ENTRIES)
    cell = route_h["edited_cells"]["edited__s1__seed17"]
    assert cell["templates_interpretable_for_retraining"] == []
    assert cell["n_interpretable_target_comparisons"] == 0
    assert cell["n_target_comparisons"] == len(pp.TEMPLATE_ROLES)
    assert cell[
        "delta_retrain_sign_preserved_on_interpretable_templates"] is None
    text = pp.build_claims("salmu", route_h, None, {}, _syn_panel())["route_h"]
    assert "stays positive in 0 of 0 cells" in text
    assert "1 of 1 cells have no template on which the comparison is " \
        "interpretable at all" in text
    assert "which is 0 of the 6 templates [none]" in text


def test_an_edit_specific_collapse_is_named_as_such():
    """The other direction: when the baseline holds and the edit does not, the
    claim must say so instead of hiding behind the route-training excuse."""
    results = _syn_h_results()
    _off_support(results["edited__s1__seed17"], "question_form")
    route_h = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                                   SYN_ENTRIES)
    support = pp.build_claims("salmu", route_h, None, {},
                              _syn_panel())["route_h_prompt_support"]
    assert "1 template(s) are off support for edited models" in support
    assert "the collapse IS specific to the edited models, on question_form" \
        in support
    assert "property of how route h is TRAINED" not in support


def test_a_mixed_collapse_separates_the_two_causes():
    results = _syn_h_results()
    for rec in results.values():
        _off_support(rec, "question_form")          # route-wide
    _off_support(results["edited__s1__seed17"], "distractor")   # edit only
    route_h = pp.aggregate_route_h("salmu", results, [], len(results), SYN_CTX,
                                   SYN_ENTRIES)
    prof = route_h["prompt_support_profile"]
    assert prof["collapse_is_edit_specific"] is True
    assert prof["templates_where_the_baseline_is_off_support"] == \
        ["question_form"]
    assert prof["templates_where_only_edited_models_are_off_support"] == \
        ["distractor"]
    support = pp.build_claims("salmu", route_h, None, {},
                              _syn_panel())["route_h_prompt_support"]
    assert "question_form collapse for the baseline too" in support
    assert "distractor collapse only for edited models" in support


def test_not_claimed_names_what_the_design_cannot_support():
    claims = pp.build_claims("salmu", None, None, {}, _syn_panel())
    joined = " ".join(claims["not_claimed"]).lower()
    assert "no intervention of any kind was applied to the visual router" \
        in joined
    assert "g_spillover_reference_distance" in joined
    assert "not oracle distances" in joined
    assert "no threshold, gate, promotion criterion or training recipe" \
        in joined


# ------------------------------------------------------------------ #
# model inventory: the scope the sweep actually covers
# ------------------------------------------------------------------ #
def _args(**over):
    base = {"only_sets": None, "only_seeds": None, "smoke": False,
            "only_templates": None, "max_gen_tokens": 12, "seed": 17,
            "device": "cpu", "pilot": 0, "shard_index": 0, "shard_count": 1}
    base.update(over)
    return SimpleNamespace(**base)


def test_route_h_scope_is_the_full_matrix(real):
    matrix = real["matrix"]
    specs = pp.h_model_specs("salmu", pp._gran_out_base("salmu"), matrix,
                             _args())
    classes = {}
    for spec in specs:
        classes[spec["model_class"]] = classes.get(spec["model_class"], 0) + 1
    assert classes == {"baseline": 1, "edited": 63, "matched_retrain": 21,
                       "loo_retrain": 21}
    assert len(specs) == 106


def test_route_g_scope_is_the_representative_sets_plus_the_frozen_router(real):
    matrix = real["matrix"]
    specs = pp.g_model_specs("salmu", pp._gran_out_base("salmu"), matrix,
                             _args())
    ids = [s["model_id"] for s in specs]
    assert ids[0] == "frozen_g"
    assert sum(1 for s in specs if s["model_class"] == "edited") == 9
    assert sum(1 for s in specs if s["model_class"] == "baseline") == 1
    assert sum(1 for s in specs if s["model_class"] in pp.RETRAIN_FAMILIES) == 6
    assert len(specs) == 17
    for sid in pp.G_SPILLOVER_REP_SETS:
        assert any(s["set_id"] == sid for s in specs)


def test_only_sets_and_only_seeds_narrow_the_sweep(real):
    matrix = real["matrix"]
    specs = pp.h_model_specs(
        "salmu", pp._gran_out_base("salmu"), matrix,
        _args(only_sets=["gx_sal_mix_0"], only_seeds=[17]))
    assert {s["model_id"] for s in specs} == {
        "baseline_h", "edited__gx_sal_mix_0__seed17",
        "matched_retrain__gx_sal_mix_0", "loo_retrain__gx_sal_mix_0"}


@pytest.mark.skipif(not (pp._gran_out_base("salmu") / "cells").exists(),
                    reason="granularity checkpoint tree not present")
def test_every_in_scope_checkpoint_exists_on_disk(real):
    """Checked before spending a GPU: the sweep must not discover gaps at
    hour three, and a gap must be a PENDING row rather than a silent skip."""
    matrix = real["matrix"]
    base = pp._gran_out_base("salmu")
    absent = [s["model_id"] for s in pp.h_model_specs("salmu", base, matrix,
                                                      _args())
              if not s["present"]]
    absent += [s["model_id"] for s in pp.g_model_specs("salmu", base, matrix,
                                                       _args())
               if not s["present"]]
    assert absent == []


def test_the_frozen_router_checkpoint_is_the_one_the_project_trained():
    assert pp.SALMU_ROUTE_G.exists(), \
        f"frozen g router missing at {pp.SALMU_ROUTE_G}"
    assert pp.SALMU_ROUTE_G.name == "adapter_model.safetensors"


# ------------------------------------------------------------------ #
# sharding: parallel processes must never share a result file
# ------------------------------------------------------------------ #
def test_shards_partition_the_sweep_disjointly_and_completely(real):
    """The union is the whole sweep and no model appears twice.

    Results are cached by presence, so two processes racing on one model would
    not merely repeat work: they would interleave writes into one JSON file and
    leave a record that is corrupt but present, and therefore skipped forever.
    """
    specs = pp.h_model_specs("salmu", pp._gran_out_base("salmu"),
                             real["matrix"], _args())
    for count in (2, 3, 4, 7):
        parts = [pp._shard(specs, _args(shard_index=i, shard_count=count))
                 for i in range(count)]
        ids = [s["model_id"] for part in parts for s in part]
        assert len(ids) == len(set(ids)), f"count={count}: a model is in two shards"
        assert sorted(ids) == sorted(s["model_id"] for s in specs), \
            f"count={count}: the shards do not cover the sweep"


def test_route_g_shards_are_disjoint_and_complete(real):
    specs = pp.g_model_specs("salmu", pp._gran_out_base("salmu"),
                             real["matrix"], _args())
    for count in (2, 3, 4):
        parts = [pp._shard(specs, _args(shard_index=i, shard_count=count))
                 for i in range(count)]
        ids = [s["model_id"] for part in parts for s in part]
        assert len(ids) == len(set(ids)) == len(specs)


def test_exactly_one_shard_owns_the_frozen_router(real):
    """PP2 is a single model, so one shard writes it and the rest do nothing."""
    specs = [s for s in pp.g_model_specs("salmu", pp._gran_out_base("salmu"),
                                         real["matrix"], _args())
             if s["model_class"] == "frozen_g"]
    assert len(specs) == 1
    for count in (2, 3, 4):
        owners = [i for i in range(count)
                  if pp._shard(specs, _args(shard_index=i, shard_count=count))]
        assert owners == [0]


def test_stride_sharding_gives_every_shard_a_mix_of_model_classes(real):
    """Stride, not contiguous blocks: a contiguous split would hand one process
    all 63 cells and another all 42 oracles, and they would finish hours apart
    for no reason."""
    specs = pp.h_model_specs("salmu", pp._gran_out_base("salmu"),
                             real["matrix"], _args())
    for count in (2, 3):
        sizes = []
        for i in range(count):
            part = pp._shard(specs, _args(shard_index=i, shard_count=count))
            classes = {s["model_class"] for s in part}
            assert "edited" in classes and classes & set(pp.RETRAIN_FAMILIES)
            sizes.append(len(part))
        assert max(sizes) - min(sizes) <= 1, "shards must be evenly balanced"


def test_an_out_of_range_shard_index_is_refused():
    """An index outside the range would evaluate nothing at all, and the sweep
    would look complete when three quarters of it never ran."""
    for index, count in ((2, 2), (3, 2), (-1, 3)):
        with pytest.raises(RuntimeError, match="shard-index"):
            pp._shard([{"model_id": "a"}],
                      _args(shard_index=index, shard_count=count))


def test_an_unsharded_run_keeps_the_plain_artifact_names():
    """Shard suffixes appear only when sharding, so a single-process sweep
    writes exactly the files it always wrote."""
    assert pp._shard_tag(_args()) == ""
    assert pp._shard_label(_args()) == "unsharded"
    assert pp._shard_tag(_args(shard_index=1, shard_count=3)) == ".shard1of3"
    assert pp._shard_label(_args(shard_index=1, shard_count=3)) == "shard 1/3"
    specs = [{"model_id": "a"}, {"model_id": "b"}]
    assert pp._shard(specs, _args()) == specs


def test_pp4_aggregates_the_whole_sweep_not_one_shard():
    """The report describes the sweep, so run_pp4 must not shard its spec list
    -- otherwise a shard's report would silently describe a fraction of the
    matrix while claiming to be the matrix."""
    source = inspect.getsource(pp.run_pp4)
    assert "_shard(" not in source
    assert "h_model_specs" in source and "g_model_specs" in source


def test_a_sharded_phase_all_leaves_the_report_to_a_single_process():
    """Every shard writing the same report file concurrently would race; the
    report is written once, unsharded, after the shards finish."""
    source = inspect.getsource(pp.main)
    assert 'args.phase == "all" and args.shard_count > 1' in source
    assert "PP4 skipped" in source


def test_shard_artifacts_are_collected_for_the_report(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "PANEL_OUT_ROOT", tmp_path)
    for name in ("run_manifest.json", "run_manifest.shard0of2.json",
                 "run_manifest.shard1of2.json", "scorer_agreement.json"):
        pp._write_json(tmp_path / "salmu" / name, {"file": name})
    found = [p.name for p in pp._shard_artifacts("salmu", "run_manifest")]
    assert found == ["run_manifest.json", "run_manifest.shard0of2.json",
                     "run_manifest.shard1of2.json"]
    assert [p.name for p in pp._shard_artifacts("salmu", "scorer_agreement")] \
        == ["scorer_agreement.json"]
    assert pp._shard_artifacts("absent_dataset", "run_manifest") == []


# ------------------------------------------------------------------ #
# PPR: the report is a function of the stored files alone
# ------------------------------------------------------------------ #
def test_the_core_digest_ignores_only_volatile_fields():
    report = {"kind": "k", "generated_at": "2026-01-01T00:00:00+00:00",
              "elapsed_sec": 1.0, "provenance": {"commit": "a"},
              "gpu": "x", "claims": {"route_h": "h"}}
    other = dict(report, generated_at="2027-01-01T00:00:00+00:00",
                 elapsed_sec=9.0, provenance={"commit": "b"}, gpu="y")
    assert pp._report_core_sha(report) == pp._report_core_sha(other)
    moved = dict(report, claims={"route_h": "different"})
    assert pp._report_core_sha(report) != pp._report_core_sha(moved)


def test_reaggregation_reports_whether_the_core_moved(tmp_path):
    report = {"kind": "k", "claims": {"route_h": "h"},
              "generated_at": "2026-01-01T00:00:00+00:00"}
    missing = pp._reaggregation_check(tmp_path / "none.json", report)
    assert missing["previous_report_found"] is False
    assert missing["identical"] is False
    pp._write_json(tmp_path / "r.json", report)
    same = pp._reaggregation_check(tmp_path / "r.json", dict(report))
    assert same["previous_report_found"] is True and same["identical"] is True
    changed = pp._reaggregation_check(
        tmp_path / "r.json", dict(report, claims={"route_h": "other"}))
    assert changed["identical"] is False


def test_pilot_never_spends_time_on_models_that_are_already_measured(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(pp, "PANEL_OUT_ROOT", tmp_path)
    specs = [_spec(f"m{n}", "edited", "s1", 17) for n in range(4)]
    pp._write_json(pp._result_path("salmu", "h", "m0"),
                   {"template_roles": list(pp.TEMPLATE_ROLES),
                    "timing": {"seconds": 10.0, "n_prompts": 24,
                               "seconds_per_prompt": 0.4}})
    chosen = pp._pilot_specs("salmu", "h", specs, 2)
    assert [s["model_id"] for s in chosen] == ["m0", "m1", "m2"]


def test_eta_is_measured_from_stored_timings_not_guessed(tmp_path, monkeypatch,
                                                         caplog):
    monkeypatch.setattr(pp, "PANEL_OUT_ROOT", tmp_path)
    specs = [_spec(f"m{n}", "edited", "s1", 17) for n in range(3)]
    pp._write_json(pp._result_path("salmu", "h", "m0"),
                   {"template_roles": list(pp.TEMPLATE_ROLES),
                    "timing": {"seconds": 60.0, "n_prompts": 24,
                               "seconds_per_prompt": 2.5}})
    eta = pp.print_eta("h", specs, "salmu")
    assert eta["models_measured"] == 1 and eta["models_remaining"] == 2
    assert eta["mean_seconds_per_prompt"] == 2.5
    assert eta["projected_seconds"] == pytest.approx(2 * 24 * 2.5)
    nothing = pp.print_eta("g", specs, "salmu")
    assert nothing["projected_seconds"] is None
    assert "no measured timings" in nothing["reason"]


def test_the_runner_binds_provenance_to_executed_code_not_result_files():
    """The GX2B/GX2S relaxation: result files may be dirty, code may not."""
    source = inspect.getsource(pp._dirty_tracked_code)
    assert "--untracked-files=no" in source
    assert "scripts/e2c_v3_prompt_panel.py" in pp.PP_CODE
    assert "scripts/e2c_v3_research_validity.py" in pp.PP_CODE
    assert isinstance(pp._dirty_tracked_code(), list)


def test_the_scoring_library_hash_is_not_the_runner_hash():
    """``rv.script_sha256()`` hashes rv's OWN file.

    A result record that stored it under a bare ``script_sha256`` would name
    the scoring library as its producer and point any reader at the wrong
    file, permanently: records are written once and skipped when present, so
    the mistake could not be corrected without re-running the sweep.  The
    record therefore carries both, named as the granularity runner names them.
    """
    runner = rv.sha256_file(Path(pp.__file__).resolve())
    assert rv.script_sha256() != runner
    source = inspect.getsource(pp.evaluate_models)
    assert "runner_script_sha256" in source
    assert "shared_scoring_script_sha256" in source
    assert '"script_sha256": rv.script_sha256()' not in source
