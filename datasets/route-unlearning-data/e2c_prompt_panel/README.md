# E2C-v3 prompt-robustness panel (held out)

Every E2C-v3 conclusion is scoped to one prompt string. The granularity runner
says so itself: proximity to a retraining reference is *"scoped to the evaluated
code prompts and candidate-label space"*. Route h is trained and measured with
`rd.CODE_TO_ALIAS_PROMPT` and nothing else, so an edit that survives only that
exact rendering is indistinguishable, in the existing evidence, from one that
survives paraphrase.

This directory holds the measurement of that difference. It retrains nothing.
All 252 checkpoints already exist; `mx.ModelSession.reset_to` swaps adapters in
place, so the sweep is one 9B model load per process plus a fast reload per
checkpoint.

Producer: `scripts/e2c_v3_prompt_panel.py` (phases `PP0`–`PP4`, `PPR`).

## Held out, and enforced rather than asserted

The panel is a **held-out behavioral evaluation**. Four mechanisms make that
checkable instead of a comment:

1. **Frozen before evaluation, and committed before evaluation.** `PP0` writes
   `manifests/prompt_panel_salmu.json` and refuses to overwrite it without an
   explicit `--refreeze`. The panel was committed at digest
   `65cd6a3324d0c7d1973c4ab1430987c1b88a9becfefd2ef0624c6d5f4ea34883` before any
   model was scored.
2. **Digest re-verification at run time.** Every later phase recomputes the
   digest and aborts on a mismatch, so a template softened after its results
   were seen stops the sweep instead of quietly producing numbers that no longer
   correspond to the panel anyone can read. Re-serializing the file (key order,
   indentation) is *not* a change: the digest is over canonical JSON.
3. **Criteria applied unchanged, as read-only columns.** The pass criteria come
   from `scripts/e2c_v3_granularity.py` `PASS_CRITERIA` and are evaluated per
   template with `is_a_gate: false`. No new threshold is introduced anywhere and
   no gate in this repository reads panel output.
4. **Worst case is a `min` over all six pre-frozen roles.** No template can be
   dropped from the headline after the fact, and `assert_full_panel` refuses to
   aggregate any result file that does not cover all six — which is why smoke and
   pilot output goes to a separate `*_smoke` tree.

The panel also carries its own restrictions inside the artifact
(`held_out.not_used_for`): model selection, threshold setting, template
selection, training-recipe changes, and any gate or promotion criterion. A
reader holding only the JSON sees the constraint, not just the numbers.

**Nothing here changes because of what the panel finds.** If a template exposes a
failure, it is reported. No threshold, gate, promotion criterion or training
recipe moves, and the route architecture (g, h, prompts, LoRA config) is not
touched — route establishment and unlearning optimization stay decoupled.

## The panel

Six template roles, identical roles on both routes, frozen as literal strings:

| role | what it varies |
| --- | --- |
| `canonical` | nothing — byte-identical to the production prompt |
| `concise_paraphrase` | lexical content, same request in fewer words |
| `question_form` | interrogative rather than imperative |
| `instruction_form` | explicit "return only … do not explain" |
| `format_variation` | case, punctuation and whitespace **only** |
| `distractor` | adds a decoy drawn from the same output space |

Route h (`code -> alias`) canonical is `rd.CODE_TO_ALIAS_PROMPT`;
route g (`image -> code`) canonical is `rd.IMG_TO_CODE_PROMPT`. Keeping those
byte-identical is what lets the canonical column *reproduce* the existing
evidence on the existing code path rather than restate it from a second
implementation. `format_variation` isolates surface formatting from lexical
change, and a test asserts it is the **only** role whose lexical content matches
canonical — otherwise it would silently become a seventh paraphrase.

**Distractor neutrality** is the one real design constraint. The panel is frozen
once per identity, but which labels are source and target differs across the 21
sets, so a decoy that is neutral for one set could be the desired label for
another. That would make the distractor template *easier* than the canonical one
and invert the measurement. Each decoy is therefore drawn, deterministically at
panel seed 17, from the vocabulary minus every label that identity is ever
associated with in any set: its baseline alias, its whole taxonomic chain (a set
may transform it to an ancestor, not only to its specific alias), and every
source and target assigned to it — which is where `Unknown` enters for the three
identities carrying a refusal assignment. That leaves a pool of 26–27 of the 30
labels; all twelve decoys came from it, so `distractor_neutral_for_all_sets` is
true twelve times over and every collision list is empty. The fallback path (a
label from a different branch, with the exact `(set_id, role)` collisions
recorded so the report separates those rows instead of averaging over them)
exists and is tested, and was not needed.

Route g decoys are another identity's **code**, since codes are that route's
output space, and every choice is neutral there by construction. Route g images
are the **first held-out test image** per identity (`SAL_<iid>_08.png`), not a
train image: the frozen router is 96/96 on train and 29/36 held out, so a train
image would sit at a ceiling and hide precisely the sensitivity being measured.
Each row records `image_uri` and `image_sha256`.

## Scope

Dataset **SALMU** only; `celeba_numeric` has 2 of 84 matrix cells on disk and no
trained image router.

- 12 identities, 12 codes `SAL_<iid>`, 30-label route-h vocabulary (including
  `Unknown`), 12-code route-g space.
- **Route h — full matrix, 106 checkpoints:** `baseline_h`, 21 sets x 3 edit
  seeds = 63 edited cells, 21 `matched_retrain`, 21 `loo_retrain`.
- **Route g — 17 models:** the frozen `g_X_to_C` router, `baseline_h`, the 3
  representative sets x 3 seeds = 9 edited cells, and their 6 retrain
  references. One image per identity.
- 6 templates x 12 identities = **72 prompts per model per route**.

The `*_finetune` oracle families are not evaluated: only the `*_retrain`
families (fresh base + fresh LoRA) support a retraining claim. GX2S oracle-seed
variants (`__oseed42`, `__oseed123`) are out of scope — this panel varies
templates, not oracle seeds.

## What is measured

Per template, reported **separately** for transformation targets / refusal
controls / retained / sibling / cousin / unrelated, never averaged together
(matching the existing rule that a refusal control must not leak into a
granularity headline):

- desired-label accuracy (strict) and `p_desired`;
- source-label leakage: hard leaked rate and `max p(source)`;
- retention and sibling accuracy, with the existing `null`-not-`1.0` convention
  where a set has no retained sibling;
- candidate support: `candidate_mass`, `other_mass`, unparseable rate,
  multi-label-invalid rate;
- `template_worst_case` (`min` over six) and `max_template_spread` (`max - min`);
- distance to oracle **by template**: `D(E, matched_retrain)`,
  `D(E, loo_retrain)`, `Delta_retrain = D(E,LOO) - D(E,matched)`, each gated on
  `candidate_mass >= 0.01` and otherwise `null` / "not established".

Strict parsing throughout: token-exact, and more than one distinct recognized
label is **invalid**, never resolved to the first match. A decoy echoed back is
scored as leakage or a flip, never as the desired label.

The headline robustness question is whether `Delta_retrain` keeps its sign and
margin under paraphrase. On the canonical prompt alone it is currently
`D(E,matched) <= 1e-05` against `D(E,loo)` of 1.12–1.27.

## Route g: no intervention exists, so nothing may claim one

There is exactly one trained visual router. The four model classes — baseline,
edited, matched-retrain, LOO-retrain — are all **h-side** adapters, and the
granularity matrix deliberately replays cached g rows instead of re-running g.
Route g therefore measures two separate things:

- **`PP2` baseline prompt sensitivity** of the frozen router across the six
  image-conditioned templates: code accuracy per template, six-template
  consistency, worst-template accuracy, maximum template spread, and invalid /
  off-support output rates.
- **`PP3` cross-route edit spillover**: every in-scope h-side adapter loaded
  onto the same base model and run over the *identical* panel, scored relative
  to **frozen base g** — prediction-flip rate, change in code accuracy,
  candidate-distribution distance where available, target-person versus
  retained-person effects, and per-template plus worst-template spillover. The
  null expectation is that an edit targeting `h(C->A)` leaves `g(X->C)`
  approximately unchanged.

The confirmatory spillover reference is **always frozen base g**. Distances
between h-side adapters measured on g prompts are reported as
`g_spillover_reference_distance`, labelled a *cross-route reference comparison*,
and are **not** oracle distances.

`_guard_g_vocabulary` raises if any g-side field or claim contains
`edit success`, `g-side unlearning` or `retraining equivalence`, and
`_guard_g_tree` applies it to every string in the assembled g blocks and claims.
There was no g-side intervention and no g-specific matched or LOO retraining, so
those phrases would be false about route g no matter what the numbers say.

## Layout

```
manifests/prompt_panel_salmu.json   the frozen panel (committed before evaluation)
outputs/salmu/h/<model_id>.json     one route-h result per checkpoint
outputs/salmu/g/<model_id>.json     one route-g result per model
outputs/salmu/scorer_agreement*.json  the two scoring paths, cross-checked
outputs/salmu/run_manifest*.json    cumulative provenance, one file per shard
outputs/salmu_smoke/...             smoke / pilot output, never aggregated
reports/prompt_robustness_salmu.json  the report
```

Per-model result files are written on completion and skipped when present, so
an interrupted sweep **resumes** instead of restarting.

## Reproducing

```bash
PY=/scratch/wutiantong/miniconda3/envs/midp-qwen35/bin/python

# build and freeze the panel (refuses to overwrite without --refreeze)
$PY scripts/e2c_v3_prompt_panel.py --dataset salmu --phase PP0

# one model, one template, into the throwaway *_smoke tree
CUDA_VISIBLE_DEVICES=3 $PY scripts/e2c_v3_prompt_panel.py --dataset salmu \
    --phase PP1 --smoke --pilot 1 --device cuda:0

# measure cost before committing to the long sweep, then read it back on CPU
$PY scripts/e2c_v3_prompt_panel.py --dataset salmu --phase PP1 --pilot 2 \
    --device cuda:0
$PY scripts/e2c_v3_prompt_panel.py --dataset salmu --eta-only

# the sweep, one shard per GPU that can hold the model
bash /scratch/wutiantong/launch_pp_parallel.sh

# aggregate (CPU only), then prove it is reproducible from the stored files
$PY scripts/e2c_v3_prompt_panel.py --dataset salmu --phase PP4 --device cpu
$PY scripts/e2c_v3_prompt_panel.py --dataset salmu --phase PPR --device cpu
```

`--shard-index i --shard-count N` splits the sweep by **stride** over the model
list. The slices are disjoint and their union is the whole sweep, which is what
makes several processes safe without a lock: results are cached by file
presence, so two processes racing on one model would interleave writes into one
JSON file and leave a record that is corrupt but present, and therefore skipped
forever. Stride rather than contiguous blocks so every shard gets the same mix
of edited cells and retrain oracles. The run manifest and the scorer cross-check
carry a `.shard<i>of<N>` suffix; `PP4` collects every copy and is itself never
sharded, because the report describes the sweep and not one process's slice of
it.

Qwen3.5-9B in bf16 holds about 18.7 GB resident. A GPU with less free than that
cannot take a shard, and the launcher skips it rather than risking an OOM for
itself or a co-resident job.

## Provenance

Each per-model record carries the **executing** commit, captured before any
result was written, plus `runner_script_sha256` (this producer) and
`shared_scoring_script_sha256` (the scoring library) separately — naming them
apart matters, because the library's own hash is not the runner's. Records also
carry the checkpoint's sha256 and the panel digest they were measured against.
`run_manifest*.json` is cumulative and keeps a `provenance_history` entry per
invocation, so a resumable multi-process sweep still shows which code evaluated
which checkpoint.

The dirty-tree gate binds **committed code** and tolerates dirty result files
(the GX2B/GX2S relaxation), because the worktree holds in-flight granularity
outputs from parallel runs that are not this panel's to commit. A dirty script
is a hard stop.

`_load_matrix` re-derives the granularity matrix from source, runs the project's
own GX0 validation over it, and only then compares it to the committed file.
Order matters: `gx.validate_set` populates `controls` and `control_notes` in
place, so the committed matrix equals the builder's output only *after* a GX0
pass. Comparing first reports drift where there is none; skipping the validation
to make the comparison pass would let a held-out panel be evaluated against a
matrix the project's own hard gate rejects.

`PPR` re-derives the report on CPU and digests it minus
`VOLATILE_REPORT_KEYS` (`generated_at`, `elapsed_sec`, `provenance`, `gpu`,
`reaggregation`), recording whether the core moved. Whole-file hashes would
always differ and so would prove nothing; the point is to show the report is a
function of the stored per-model files alone and can be revised without a GPU.

## Findings

*(written once the sweep completes; see `reports/prompt_robustness_salmu.json`.)*
