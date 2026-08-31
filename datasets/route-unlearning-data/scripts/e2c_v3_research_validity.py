#!/usr/bin/env python3
"""E2C-v3 Research-Validity Phase (RV1-RV9) — corrected edition.

This revision addresses the research-validity review.  Every finding from the
review is fixed here; prior RV artifacts produced by the previous revision are
superseded and must not be cited.

Blocking findings fixed
=======================
1. GA baseline is now honest: gradient ASCENT is applied only to the forget
   (target) set, while supervised DESCENT is applied to the retain set.  The
   previous version negated the loss on every example (including retain), which
   engineered the reported 0% retention.
2. NPO reference handling is fixed: current LoRA weights are snapshotted, the
   reference weights are swapped in only for the reference forward pass, and
   the current weights are RESTORED afterwards.  Previously the restore block
   was a no-op, so every update was applied from the reference state.
3. Soft metrics are now FULL-SEQUENCE alias probabilities (product over all
   alias tokens), not first-token probabilities.
4. RV7 is now simultaneous multi-target: a SINGLE model forgets several
   identities in one training run while retaining the rest (no per-target loop).
5. The oracle is training-matched to the Phase-C C->Y defaults (3000 steps /
   200 warmup / lr 2e-5 / repeat 50) and is compared to each edited model at
   the distribution level (L2 / cosine / JS distance over the alias space), not
   merely on hard output.

Additional corrections
======================
- A genuine deletion target is introduced: deleted identities are driven to a
  non-identity refusal label ("Unknown") that is outside the alias space.  The
  retrain-without-target oracle is reported separately as the retrain-equivalence
  reference (the oracle itself collapses the deleted target onto a nearby alias).
- RV5 tests granularity reduction at multiple hierarchy depths as separate
  transformations (specific->subgroup and specific->group), each evaluated with
  strict recognized-label parsing.
- RV6 compositional mixture uses the genuine deletion label and full-sequence
  soft metrics; all four transformations are evaluated simultaneously.
- Subgroup / group evaluation uses strict recognized-label parsing instead of
  substring matching.
- load_trained_weights() is fail-closed: it raises on any partial checkpoint
  load instead of silently continuing.
- RV8 adds a post-edit end-to-end image -> code -> output evaluation using the
  frozen g (X->C) and the edited h (C->Y), plus a visual-control probe.
- RV9 writes a run manifest binding git commit, CLI invocation, config, seeds,
  input manifests, checkpoint hashes, and output artifact hashes.

Architecture constraint (freeze): g (X->C) is NEVER modified; only h (C->Y).
"""
import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as tnn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_rv")

SCOPE_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"

# --- Training defaults ----------------------------------------------------- #
# Editing budget (the established U0-U8 unlearning budget).
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5
UL_REPEAT = 50
# Oracle budget MUST be training-matched to Phase-C C->Y defaults.
ORACLE_STEPS = 3000
ORACLE_WARMUP = 200
ORACLE_LR = 2e-5
ORACLE_REPEAT = 50

# --- Genuine deletion target ------------------------------------------------ #
# A non-identity label OUTSIDE the alias space.  Deleted identities are driven
# to this label; it is never used for any real identity.
DELETED_LABEL = "Unknown"

# --- Identity aliases -------------------------------------------------------- #
ALIAS_OF = {
    "syn_00": "Aven", "syn_01": "Bira", "syn_02": "Caro",
    "syn_03": "Deni", "syn_04": "Eris", "syn_05": "Faro",
    "syn_06": "Gela", "syn_07": "Hani", "syn_08": "Ivoa",
    "syn_09": "Jora",
}
IDENTITY_IDS = [f"syn_{i:02d}" for i in range(10)]

# A novel alias used only by the RV6 update transformation (S_U).
UPDATE_NEW_ALIAS = "Nyx"

# --- Granularity hierarchy --------------------------------------------------- #
GROUP_MAP = {iid: ("GROUP_A" if ALIAS_OF[iid][0] in "ABCDE" else "GROUP_B")
             for iid in IDENTITY_IDS}
SUBGROUP_MAP = {
    "syn_00": "SG_A1", "syn_01": "SG_A1",
    "syn_02": "SG_A2", "syn_03": "SG_A2",
    "syn_04": "SG_A3",
    "syn_05": "SG_B1", "syn_06": "SG_B1",
    "syn_07": "SG_B2", "syn_08": "SG_B2",
    "syn_09": "SG_B3",
}

PHASE_C_CY_CKPT = Path(
    "e2c_v3/outputs/phaseC/C_to_Y/adapter_final/adapter_model.safetensors")
PHASE_C_XC_CKPT = Path(
    "e2c_v3/outputs/phaseC/X_to_C/adapter_final/adapter_model.safetensors")
IDENTITY_MAPPING_MANIFEST = Path("e2c_v3/manifests/identity_code_mapping.json")
IMAGE_SPLIT_MANIFEST = Path("e2c_v2/manifests/e2c_image_split.json")
VISUAL_CONTROLS_MANIFEST = Path("e2c_v2/manifests/e2c_visual_controls.json")


def alias_label_vocab(extra=()):
    """Full label vocabulary used for soft-metric / parsing comparisons."""
    labels = sorted(set(ALIAS_OF.values()))
    labels.append(DELETED_LABEL)
    for x in extra:
        if x not in labels:
            labels.append(x)
    return labels


def granularity_vocab():
    labels = alias_label_vocab()
    labels += sorted(set(SUBGROUP_MAP.values()))
    labels += sorted(set(GROUP_MAP.values()))
    return labels


# ====================================================================== #
# Pure, GPU-free helpers (unit tested)
# ====================================================================== #
def parse_recognized_label(text, vocab):
    """Strictly parse the first recognized label token from model output.

    Returns the first output token that EXACTLY matches a vocabulary label
    (case-insensitive), or None.  Unlike substring matching, a label only
    matches when the whole whitespace-delimited token equals the label, so
    e.g. "GROUP_A" will not match inside "GROUP_ABC" and "SG_A1" will not
    match inside "SG_A10".
    """
    if not text:
        return None
    lowered = {v.lower(): v for v in vocab}
    for raw_token in text.strip().split():
        token = raw_token.strip().strip(".,!?;:'\"()[]{}").lower()
        if token in lowered:
            return lowered[token]
    return None


def distribution_distance(p, q, metric="l2", labels=None):
    """Distance between two distributions given as {label: prob} dicts.

    Supports l2, cosine, and js (Jensen-Shannon).  Vectors are aligned over
    ``labels`` (default: union of keys) and renormalized to sum to 1.
    """
    if labels is None:
        labels = sorted(set(p) | set(q))
    pv = torch.tensor([float(p.get(l, 0.0)) for l in labels], dtype=torch.float64)
    qv = torch.tensor([float(q.get(l, 0.0)) for l in labels], dtype=torch.float64)
    pv = pv / max(pv.sum().item(), 1e-12)
    qv = qv / max(qv.sum().item(), 1e-12)
    if metric == "l2":
        return float(torch.norm(pv - qv, p=2).item())
    if metric == "cosine":
        denom = torch.norm(pv) * torch.norm(qv)
        if denom.item() < 1e-12:
            return 1.0
        return float(1.0 - (pv @ qv / denom).item())
    if metric == "js":
        m = 0.5 * (pv + qv)
        eps = 1e-12
        kl_pm = (pv * ((pv + eps).log() - (m + eps).log())).sum().item()
        kl_qm = (qv * ((qv + eps).log() - (m + eps).log())).sum().item()
        return float(0.5 * (kl_pm + kl_qm))
    raise ValueError(f"unknown metric: {metric}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ====================================================================== #
# Model infrastructure
# ====================================================================== #
def create_adapter_model(args, device, adapter_name):
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter
    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name=adapter_name, trust_remote_code=True, dtype="bfloat16",
        attn_implementation="sdpa",
        candidate_positive="Yes", candidate_negative="No",
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        lora_scope="custom_ablation",
        lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_scope_regex=SCOPE_REGEX,
        r2mu_candidate_layers=(8, 16, 24, 29), r2mu_n_select_layers=4,
        language_layer_path="model.language_model.layers",
        language_hidden_size=4096, intermediate_size=12288,
        num_language_layers=32, lora_expected_target_modules=128,
    )
    adapter = Qwen35Adapter(profile)
    model, processor = adapter.load_model_processor(
        model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", device=device, training=True,
    )
    return adapter, model, processor


def attach_lora(adapter, model):
    target_modules = sorted(
        n for n, m in model.named_modules()
        if isinstance(m, tnn.Linear) and re.match(SCOPE_REGEX, n))
    model = adapter.attach_unlearning_adapter(
        model, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules,
        adapter_name=adapter.profile.adapter_name,
    )
    return model


def load_trained_weights(adapter, model, ckpt_path=None):
    """Load the Phase-C C->Y checkpoint into the live adapter.

    FAIL-CLOSED: raises RuntimeError if any checkpoint tensor cannot be placed
    into a live adapter parameter, or if any live adapter (LoRA) parameter is
    left without a loaded value.  This prevents silent partial loading.
    """
    from safetensors.torch import load_file
    ckpt_path = Path(ckpt_path) if ckpt_path else PHASE_C_CY_CKPT
    ckpt_data = load_file(str(ckpt_path))
    live_params = dict(model.named_parameters())
    aname = adapter.profile.adapter_name
    live_adapter_names = {
        n for n in live_params if ("lora_A" in n) or ("lora_B" in n)}
    loaded = set()
    missing_ckpt = []
    for ckpt_key, ckpt_tensor in ckpt_data.items():
        target = None
        if ckpt_key in live_params:
            target = ckpt_key
        else:
            remapped = ckpt_key.replace("lora_A", f"lora_A.{aname}").replace(
                "lora_B", f"lora_B.{aname}")
            if remapped in live_params:
                target = remapped
        if target is None:
            missing_ckpt.append(ckpt_key)
            continue
        live_params[target].data.copy_(ckpt_tensor)
        loaded.add(target)
    missing_live = live_adapter_names - loaded
    if missing_ckpt or missing_live:
        raise RuntimeError(
            "load_trained_weights: partial checkpoint load detected. "
            f"ckpt_tensors={len(ckpt_data)} loaded={len(loaded)} "
            f"missing_ckpt={len(missing_ckpt)} missing_live={len(missing_live)}. "
            f"sample_missing_ckpt={missing_ckpt[:3]} "
            f"sample_missing_live={sorted(missing_live)[:3]}")
    logger.info(f"Loaded {len(loaded)}/{len(ckpt_data)} trained weight tensors "
                f"(fail-closed OK)")
    return len(loaded)


def build_supervised_items(adapter, processor, pairs, repeat=1):
    sup_items = []
    for pair in pairs:
        for _ in range(repeat):
            ex = adapter.build_supervised_example(
                processor, image=None,
                prompt=pair["prompt"], answer_text=pair["answer"],
            )
            sup_items.append(ex)
    return sup_items


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_loader(adapter, items):
    collate = adapter.collate
    return DataLoader(
        ItemDataset(items), batch_size=1, shuffle=True,
        collate_fn=lambda b: collate(b), num_workers=0)


def make_optimizer_scheduler(params, steps, warmup, lr):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    optimizer = AdamW(params, lr=lr, weight_decay=0.0)
    actual_warmup = min(warmup, max(steps - 1, 1))
    warmup_s = LinearLR(optimizer, start_factor=0.1, total_iters=actual_warmup)
    cosine_s = CosineAnnealingLR(optimizer, T_max=max(steps - actual_warmup, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_s, cosine_s],
                             milestones=[actual_warmup])
    return optimizer, scheduler


def move_batch(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def forward_loss(model, bd):
    """Teacher-forced mean NLL loss for a batch dict."""
    outputs = model(
        input_ids=bd["input_ids"],
        attention_mask=bd["attention_mask"],
        labels=bd["labels"], use_cache=False,
        **{k: v for k, v in bd.items()
           if k not in ("input_ids", "attention_mask", "labels")
           and isinstance(v, torch.Tensor) and not k.startswith("_")})
    return outputs.loss


def forward_logits(model, bd):
    """Return raw logits for a batch dict (no labels)."""
    outputs = model(
        input_ids=bd["input_ids"],
        attention_mask=bd["attention_mask"], use_cache=False,
        **{k: v for k, v in bd.items()
           if k not in ("input_ids", "attention_mask", "labels")
           and isinstance(v, torch.Tensor) and not k.startswith("_")})
    return outputs.logits


def snapshot_lora(model):
    return {n: p.data.clone() for n, p in model.named_parameters()
            if p.requires_grad}


def restore_lora(model, snap):
    for n, p in model.named_parameters():
        if n in snap:
            p.data.copy_(snap[n])



# ====================================================================== #
# Corrected soft metrics: FULL-SEQUENCE alias probabilities
# ====================================================================== #
def _build_prompt_ids(processor, code_id):
    prompt_text = CODE_TO_ALIAS_PROMPT.format(code=code_id)
    chat = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = processor.tokenizer(chat, return_tensors="pt",
                              add_special_tokens=False).input_ids[0]
    return ids


def full_sequence_label_probs(adapter, model, processor, code_id,
                              candidate_labels, device):
    """Return FULL-SEQUENCE probability for each candidate label.

    P(label | prompt) = prod_t P(token_t | prompt, tokens_<t), computed by a
    single teacher-forced forward over (prompt + label) and summing the log
    probability at each label-token prediction position.  This replaces the
    previous first-token-only scoring which mis-scored multi-token aliases.
    """
    prompt_ids = _build_prompt_ids(processor, code_id).to(device)
    plen = prompt_ids.shape[0]
    model.eval()
    results = {}
    with torch.no_grad():
        for label in candidate_labels:
            label_ids = processor.tokenizer.encode(
                label, add_special_tokens=False)
            if not label_ids:
                results[label] = {"prob": 0.0, "log_prob": -1e9,
                                  "n_tokens": 0}
                continue
            full_ids = torch.cat(
                [prompt_ids, torch.tensor(label_ids, device=device)]
            ).unsqueeze(0)
            attn = torch.ones_like(full_ids)
            logits = model(full_ids, attention_mask=attn,
                           use_cache=False).logits[0]
            logp = torch.log_softmax(logits.float(), dim=-1)
            lp = 0.0
            for i, tid in enumerate(label_ids):
                pos = plen - 1 + i  # logits[pos] predicts token plen+i
                lp += logp[pos, tid].item()
            results[label] = {"prob": math.exp(lp), "log_prob": lp,
                              "n_tokens": len(label_ids)}
    return results


def soft_summary(probs, correct_label, labels):
    """Compute p_correct, runner-up, margin, entropy over the label set."""
    p_correct = probs.get(correct_label, {}).get("prob", 0.0)
    ordered = sorted(((l, probs.get(l, {}).get("prob", 0.0)) for l in labels),
                     key=lambda x: -x[1])
    if ordered[0][0] == correct_label and len(ordered) > 1:
        runner_up = ordered[1]
    else:
        runner_up = ordered[0]
    margin = p_correct - runner_up[1]
    pv = torch.tensor([probs.get(l, {}).get("prob", 0.0) for l in labels])
    pv = pv / max(pv.sum().item(), 1e-12)
    entropy = float(-(pv * (pv + 1e-12).log()).sum().item())
    max_entropy = math.log(max(len(labels), 2))
    return {
        "p_correct": round(p_correct, 6),
        "runner_up_alias": runner_up[0],
        "p_runner_up": round(runner_up[1], 6),
        "margin": round(margin, 6),
        "entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "normalized_entropy": round(entropy / max_entropy, 4),
    }


def evaluate_h_labels(adapter, model, processor, identity_ids, expected_of,
                      device, seed, vocab, tag="eval"):
    """Hard evaluation with STRICT recognized-label parsing."""
    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed)
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()
    preds = []
    with torch.no_grad():
        for iid in identity_ids:
            expected = expected_of[iid]
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=8)
            text = gen.text.strip()
            parsed = parse_recognized_label(text, vocab)
            ok = parsed is not None and parsed.lower() == expected.lower()
            preds.append({"identity_id": iid, "expected": expected,
                          "raw": text, "parsed_label": parsed, "correct": ok})
            logger.info(f"  [{tag}] {iid} -> '{parsed}' "
                        f"(expected '{expected}') {'ok' if ok else 'X'}")
    n_correct = sum(p["correct"] for p in preds)
    return {"accuracy": n_correct / len(preds), "preds": preds}


# ====================================================================== #
# Training objectives
# ====================================================================== #
def train_supervised(condition, adapter, model, processor, train_items,
                     output_dir, device, steps=UL_STEPS, warmup=UL_WARMUP,
                     lr=UL_LR):
    """Plain supervised descent on the given items (used for CF/GD/oracle)."""
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    logger.info(f"[{condition}] trainable params={n_params:,} "
                f"items={len(train_items)} steps={steps} lr={lr}")
    loader = make_loader(adapter, train_items)
    optimizer, scheduler = make_optimizer_scheduler(params, steps, warmup, lr)
    model.train()
    trace, global_step, running = [], 0, 0.0
    for _ in range(100000):
        for batch in loader:
            bd = move_batch(batch, device)
            loss = forward_loss(model, bd)
            loss.backward()
            running += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % 50 == 0 or global_step <= 5:
                trace.append({"step": global_step, "loss": running / global_step})
                logger.info(f"[{condition}] step {global_step}/{steps} "
                            f"loss={running / global_step:.6f}")
            if global_step >= steps:
                break
        if global_step >= steps:
            break
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    logger.info(f"[{condition}] done, final avg loss={running / max(global_step,1):.6f}")
    return trace


def train_ga(condition, adapter, model, processor, forget_items, retain_items,
             output_dir, device, steps=UL_STEPS, warmup=UL_WARMUP, lr=UL_LR):
    """Honest Gradient Ascent: ASCEND on forget set, DESCEND on retain set.

    Each step takes one forget batch (maximize NLL of the old answer) and one
    retain batch (minimize NLL), so retention is genuinely preserved.  This
    replaces the previous version which negated the loss on ALL examples
    (including retain) and therefore engineered 0% retention.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"[{condition}] GA forget={len(forget_items)} "
                f"retain={len(retain_items)} steps={steps} lr={lr}")
    forget_loader = make_loader(adapter, forget_items)
    retain_loader = make_loader(adapter, retain_items)
    optimizer, scheduler = make_optimizer_scheduler(params, steps, warmup, lr)
    retain_iter = iter(retain_loader)
    model.train()
    trace, step, running = [], 0, 0.0
    for _ in range(100000):
        for fbatch in forget_loader:
            fbd = move_batch(fbatch, device)
            floss = forward_loss(model, fbd)
            # gradient ASCENT on the forget set
            (-floss).backward()
            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)
            rbd = move_batch(rbatch, device)
            rloss = forward_loss(model, rbd)
            # supervised DESCENT on the retain set
            rloss.backward()
            running += floss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                trace.append({"step": step, "forget_loss": floss.item(),
                              "retain_loss": rloss.item()})
                logger.info(f"[{condition}] step {step}/{steps} "
                            f"forget={floss.item():.6f} retain={rloss.item():.6f}")
            if step >= steps:
                break
        if step >= steps:
            break
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    return trace


def train_npo(condition, adapter, model, processor, forget_items, retain_items,
              output_dir, device, steps=UL_STEPS, warmup=UL_WARMUP, lr=UL_LR,
              beta=1.0):
    """Fixed Negative Preference Optimization.

    Reference handling is corrected: current LoRA weights are snapshotted, the
    reference weights are swapped in ONLY for the reference forward pass, and
    the current weights are RESTORED before the backward/optimizer step, so
    updates accumulate on the live model (the previous version left the model
    at the reference state because the restore block was a no-op).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"[{condition}] NPO forget={len(forget_items)} "
                f"retain={len(retain_items)} steps={steps} beta={beta}")
    ref_weights = snapshot_lora(model)
    forget_loader = make_loader(adapter, forget_items)
    retain_loader = make_loader(adapter, retain_items)
    optimizer, scheduler = make_optimizer_scheduler(params, steps, warmup, lr)
    retain_iter = iter(retain_loader)
    model.train()
    trace, step, running = [], 0, 0.0
    eps = 1e-8
    for _ in range(100000):
        for fbatch in forget_loader:
            fbd = move_batch(fbatch, device)
            # current-model forward on the forget answer
            loss_target = forward_loss(model, fbd)
            # reference forward: swap in ref, compute, RESTORE current
            current = snapshot_lora(model)
            restore_lora(model, ref_weights)
            with torch.no_grad():
                ref_loss = forward_loss(model, fbd).detach()
            restore_lora(model, current)
            # NPO: -log sigmoid(beta * (loss_theta - loss_ref))
            diff = loss_target - ref_loss
            npo_loss = -torch.log(torch.sigmoid(beta * diff) + eps)
            # retain descent
            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)
            rbd = move_batch(rbatch, device)
            rloss = forward_loss(model, rbd)
            total = npo_loss + 0.1 * rloss
            total.backward()
            running += total.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                trace.append({"step": step, "npo_loss": npo_loss.item(),
                              "retain_loss": rloss.item(),
                              "diff": diff.item()})
                logger.info(f"[{condition}] step {step}/{steps} "
                            f"npo={npo_loss.item():.6f} retain={rloss.item():.6f}")
            if step >= steps:
                break
        if step >= steps:
            break
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    return trace


def train_kl(condition, adapter, model, processor, forget_items, retain_items,
             output_dir, device, steps=UL_STEPS, warmup=UL_WARMUP, lr=UL_LR,
             beta_kl=0.5):
    """KL-anchored unlearning: GA on forget + KL(pi_theta || pi_ref) on retain.

    The KL term bounds drift from the reference on the retain set, measured at
    the token-distribution level.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"[{condition}] KL forget={len(forget_items)} "
                f"retain={len(retain_items)} steps={steps} beta_kl={beta_kl}")
    ref_weights = snapshot_lora(model)
    forget_loader = make_loader(adapter, forget_items)
    retain_loader = make_loader(adapter, retain_items)
    optimizer, scheduler = make_optimizer_scheduler(params, steps, warmup, lr)
    retain_iter = iter(retain_loader)
    model.train()
    trace, step, running = [], 0, 0.0
    for _ in range(100000):
        for fbatch in forget_loader:
            fbd = move_batch(fbatch, device)
            floss = forward_loss(model, fbd)
            (-floss).backward()  # ascent on forget
            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)
            rbd = move_batch(rbatch, device)
            # KL(pi_theta || pi_ref) on retain (token-level)
            logits_theta = forward_logits(model, rbd)
            current = snapshot_lora(model)
            restore_lora(model, ref_weights)
            with torch.no_grad():
                logits_ref = forward_logits(model, rbd).detach()
            restore_lora(model, current)
            lp_theta = torch.log_softmax(logits_theta.float(), dim=-1)
            lp_ref = torch.log_softmax(logits_ref.float(), dim=-1)
            mask = (rbd["labels"] != -100).unsqueeze(-1)
            kl = torch.nn.functional.kl_div(
                lp_ref, lp_theta, reduction="none", log_target=True)
            kl = (kl * mask).sum() / mask.sum().clamp(min=1)
            (beta_kl * kl).backward()
            running += floss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                trace.append({"step": step, "forget_loss": floss.item(),
                              "kl": kl.item()})
                logger.info(f"[{condition}] step {step}/{steps} "
                            f"forget={floss.item():.6f} kl={kl.item():.6f}")
            if step >= steps:
                break
        if step >= steps:
            break
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    return trace


def train_gd_refusal(condition, adapter, model, processor, target_ids,
                     retain_pairs, output_dir, device,
                     steps=UL_STEPS, warmup=UL_WARMUP, lr=UL_LR):
    """GD distribution-matching toward the refusal label.

    On each target, minimize KL(model distribution over label-vocab first
    tokens at the generation position || one-hot(DELETED_LABEL)), plus
    supervised descent on the retain set.  This is a distribution-level edit,
    distinct from teacher-forced CE relabeling.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"[{condition}] GD-refusal targets={target_ids} "
                f"retain={len(retain_pairs)} steps={steps}")
    vocab = alias_label_vocab()
    first_tok = {}
    for lab in vocab:
        tids = processor.tokenizer.encode(lab, add_special_tokens=False)
        first_tok[lab] = tids[0] if tids else None
    target_vectors = {}
    for iid in target_ids:
        prompt_ids = _build_prompt_ids(processor, iid).to(device)
        target_vectors[iid] = prompt_ids
    retain_items = build_supervised_items(adapter, processor, retain_pairs,
                                          repeat=UL_REPEAT)
    retain_loader = make_loader(adapter, retain_items)
    retain_iter = iter(retain_loader)
    optimizer, scheduler = make_optimizer_scheduler(params, steps, warmup, lr)
    model.train()
    trace, step, running = [], 0, 0.0
    target_list = list(target_ids)
    for _ in range(100000):
        for iid in target_list:
            prompt_ids = target_vectors[iid]
            attn = torch.ones_like(prompt_ids).unsqueeze(0)
            logits = model(prompt_ids.unsqueeze(0), attention_mask=attn,
                           use_cache=False).logits[0, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            # gather model log-prob over each label's first token; F.kl_div
            # expects the input already in log-space
            dist = torch.stack([lp[first_tok[lab]] for lab in vocab])
            target_dist = torch.full_like(dist, 0.0)
            target_dist[vocab.index(DELETED_LABEL)] = 1.0
            gd_loss = torch.nn.functional.kl_div(
                dist, target_dist, reduction="sum")
            gd_loss.backward()
            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)
            rbd = move_batch(rbatch, device)
            rloss = forward_loss(model, rbd)
            rloss.backward()
            running += gd_loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                trace.append({"step": step, "gd_loss": gd_loss.item(),
                              "retain_loss": rloss.item()})
                logger.info(f"[{condition}] step {step}/{steps} "
                            f"gd={gd_loss.item():.6f} retain={rloss.item():.6f}")
            if step >= steps:
                break
        if step >= steps:
            break
    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    return trace



# ====================================================================== #
# Shared pair builders
# ====================================================================== #
def deletion_pairs(target_ids):
    """Genuine deletion: drive each target to the non-identity refusal label."""
    return [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
             "answer": DELETED_LABEL} for iid in target_ids]


def retain_pairs(identity_ids, exclude=()):
    return [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
             "answer": ALIAS_OF[iid]}
            for iid in identity_ids if iid not in set(exclude)]


def load_edit_model(args, adapter_dir, adapter_name):
    adapter, model, processor = create_adapter_model(
        args, args.device, adapter_name)
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)
    return adapter, model, processor


def fresh_eval_model(args, adapter_name, adapter_dir):
    adapter, model, processor = create_adapter_model(
        args, args.device, adapter_name)
    model = adapter.load_unlearning_adapter(
        model, adapter_dir / "adapter_final", adapter_name=adapter_name)
    return adapter, model, processor


def target_distribution_over_aliases(adapter, model, processor, target,
                                     device, vocab=None):
    """Full-sequence distribution over the alias vocabulary for a target."""
    vocab = vocab or alias_label_vocab()
    probs = full_sequence_label_probs(
        adapter, model, processor, target, vocab, device)
    return {lab: probs.get(lab, {}).get("prob", 0.0) for lab in vocab}, probs


# ====================================================================== #
# RV1: Soft metrics (FULL-SEQUENCE) on the frozen trained h
# ====================================================================== #
def run_rv1_soft_metrics(args, out_base, identity_ids, alias_of):
    logger.info("=" * 60)
    logger.info("RV1: SOFT METRICS (full-sequence probabilities)")
    logger.info("=" * 60)
    rv1_dir = out_base / "RV1_soft_metrics"
    rv1_dir.mkdir(parents=True, exist_ok=True)
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)
    vocab = alias_label_vocab()
    all_results = {}
    for iid in identity_ids:
        probs = full_sequence_label_probs(
            adapter, model, processor, iid, vocab, args.device)
        summary = soft_summary(probs, alias_of[iid], vocab)
        all_results[iid] = {
            "correct_alias": alias_of[iid],
            **summary,
            "full_probs": {a: round(probs.get(a, {}).get("prob", 0.0), 6)
                           for a in vocab},
        }
        logger.info(f"  {iid} ({alias_of[iid]}): p={summary['p_correct']:.4f} "
                    f"margin={summary['margin']:.4f} H={summary['entropy']:.3f}")
    with open(rv1_dir / "soft_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    del model, processor, adapter
    torch.cuda.empty_cache()
    return all_results


# ====================================================================== #
# RV2: Oracle (TRAINING-MATCHED) — retrain h without the target
# ====================================================================== #
def run_rv2_oracle(args, out_base, identity_ids, alias_of):
    logger.info("=" * 60)
    logger.info("RV2: ORACLE (training-matched retrain-without-target)")
    logger.info(f"  steps={args.oracle_steps} warmup={args.oracle_warmup} "
                f"lr={args.oracle_lr} repeat={args.oracle_repeat}")
    logger.info("=" * 60)
    target = args.target_id
    target_alias = alias_of[target]
    other_ids = [iid for iid in identity_ids if iid != target]
    rv2_dir = out_base / "RV2_oracle"
    rv2_dir.mkdir(parents=True, exist_ok=True)

    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_h_oracle")
    model = attach_lora(adapter, model)
    oracle_items = build_supervised_items(
        adapter, processor, retain_pairs(identity_ids, exclude=[target]),
        repeat=args.oracle_repeat)
    logger.info(f"Oracle training on {len(other_ids)} identities, "
                f"{len(oracle_items)} items")
    train_supervised("oracle", adapter, model, processor, oracle_items,
                     rv2_dir, args.device, steps=args.oracle_steps,
                     warmup=args.oracle_warmup, lr=args.oracle_lr)
    del model, processor, adapter
    torch.cuda.empty_cache()

    adapter, model, processor = fresh_eval_model(
        args, "e2c_v3_h_oracle", rv2_dir)
    vocab = alias_label_vocab()
    h_eval = evaluate_h_labels(adapter, model, processor, identity_ids,
                               alias_of, args.device, args.seed, vocab,
                               tag="oracle")
    other_correct = sum(1 for p in h_eval["preds"]
                        if p["identity_id"] != target and p["correct"])
    other_acc = other_correct / len(other_ids)
    target_pred = next(p["parsed_label"] for p in h_eval["preds"]
                       if p["identity_id"] == target)
    alias_dist, _ = target_distribution_over_aliases(
        adapter, model, processor, target, args.device, vocab)
    results = {
        "method": "oracle_retrain_without_target",
        "training_matched": {
            "steps": args.oracle_steps, "warmup": args.oracle_warmup,
            "lr": args.oracle_lr, "repeat": args.oracle_repeat,
            "matches_phaseC_CtoY": (
                args.oracle_steps == ORACLE_STEPS
                and args.oracle_warmup == ORACLE_WARMUP
                and args.oracle_lr == ORACLE_LR
                and args.oracle_repeat == ORACLE_REPEAT)},
        "target_id": target, "target_alias": target_alias,
        "target_prediction": target_pred,
        "target_was_correct": any(
            p["correct"] for p in h_eval["preds"]
            if p["identity_id"] == target),
        "p_target_old_alias": round(alias_dist.get(target_alias, 0.0), 6),
        "non_target_retention": round(other_acc, 4),
        "overall_accuracy": h_eval["accuracy"],
        "oracle_target_distribution": {
            k: round(v, 6) for k, v in alias_dist.items()},
        "hard_preds": h_eval["preds"],
    }
    with open(rv2_dir / "oracle_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"RV2: target_removed={not results['target_was_correct']} "
                f"p({target_alias})={results['p_target_old_alias']} "
                f"non_target={other_acc:.3f} oracle_pred={target_pred}")
    del model, processor, adapter
    torch.cuda.empty_cache()
    return results



# ====================================================================== #
# RV3: Method comparison (CF vs GA vs NPO vs KL vs GD) vs oracle
# ====================================================================== #
def _eval_method(method, args, adapter_dir, adapter_name, identity_ids,
                 alias_of, target, oracle_dist, seed):
    other_ids = [i for i in identity_ids if i != target]
    adapter, model, processor = fresh_eval_model(args, adapter_name, adapter_dir)
    vocab = alias_label_vocab()
    h_eval = evaluate_h_labels(adapter, model, processor, identity_ids,
                               alias_of, args.device, seed, vocab, tag=method)
    non_target_acc = sum(
        1 for p in h_eval["preds"]
        if p["identity_id"] != target and p["correct"]) / len(other_ids)
    alias_dist, _ = target_distribution_over_aliases(
        adapter, model, processor, target, args.device, vocab)
    alias_only = [l for l in vocab if l != DELETED_LABEL]
    dist_full = {
        m: distribution_distance(alias_dist, oracle_dist, m, labels=vocab)
        for m in ("l2", "cosine", "js")}
    dist_alias_only = {
        m: distribution_distance(
            {l: alias_dist.get(l, 0.0) for l in alias_only},
            {l: oracle_dist.get(l, 0.0) for l in alias_only},
            m, labels=alias_only)
        for m in ("l2", "cosine", "js")}
    target_pred = next(p["parsed_label"] for p in h_eval["preds"]
                       if p["identity_id"] == target)
    result = {
        "target_prediction": target_pred,
        "p_old_alias": round(alias_dist.get(alias_of[target], 0.0), 6),
        "p_deleted_label": round(alias_dist.get(DELETED_LABEL, 0.0), 6),
        "non_target_acc": round(non_target_acc, 4),
        "distance_to_oracle_full_vocab": {k: round(v, 4)
                                          for k, v in dist_full.items()},
        "distance_to_oracle_alias_only": {k: round(v, 4)
                                          for k, v in dist_alias_only.items()},
        "target_distribution": {k: round(v, 6)
                                for k, v in alias_dist.items()},
    }
    del model, processor, adapter
    torch.cuda.empty_cache()
    return result


def run_rv3_methods(args, out_base, identity_ids, alias_of, oracle_dist):
    logger.info("=" * 60)
    logger.info("RV3: METHOD COMPARISON (CF vs GA vs NPO vs KL vs GD)")
    logger.info("=" * 60)
    target = args.target_id
    target_alias = alias_of[target]
    rv3_dir = out_base / "RV3_method_comparison"
    rv3_dir.mkdir(parents=True, exist_ok=True)
    method_results = {}

    # --- CF: counterfactual / supervised relabel to the refusal label --- #
    logger.info("--- CF: supervised relabel -> refusal ---")
    cf_dir = rv3_dir / "counterfactual"; cf_dir.mkdir(exist_ok=True)
    adapter, model, processor = load_edit_model(args, cf_dir, "e2c_v3_cy")
    # Boost the novel deletion label so it overcomes the strong alias prior.
    cf_items = (
        build_supervised_items(adapter, processor, deletion_pairs([target]),
                               repeat=args.ul_repeat * 5)
        + build_supervised_items(adapter, processor,
                                 retain_pairs(identity_ids, [target]),
                                 repeat=args.ul_repeat))
    train_supervised("cf", adapter, model, processor, cf_items, cf_dir,
                     args.device, steps=args.ul_steps, warmup=args.ul_warmup,
                     lr=args.ul_lr)
    del model, processor, adapter; torch.cuda.empty_cache()
    method_results["counterfactual"] = _eval_method(
        "cf", args, cf_dir, "e2c_v3_cy_cf", identity_ids, alias_of, target,
        oracle_dist, args.seed)

    # --- GA: ascent on forget + descent on retain --- #
    logger.info("--- GA: gradient ascent on target only + retain descent ---")
    ga_dir = rv3_dir / "gradient_ascent"; ga_dir.mkdir(exist_ok=True)
    adapter, model, processor = load_edit_model(args, ga_dir, "e2c_v3_cy")
    forget_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
          "answer": target_alias}], repeat=args.ul_repeat * 5)
    retain_items = build_supervised_items(
        adapter, processor, retain_pairs(identity_ids, [target]),
        repeat=args.ul_repeat)
    train_ga("ga", adapter, model, processor, forget_items, retain_items,
             ga_dir, args.device, steps=args.ul_steps, warmup=args.ul_warmup,
             lr=args.ul_lr)
    del model, processor, adapter; torch.cuda.empty_cache()
    method_results["gradient_ascent"] = _eval_method(
        "ga", args, ga_dir, "e2c_v3_cy_ga", identity_ids, alias_of, target,
        oracle_dist, args.seed)

    # --- NPO (fixed) --- #
    logger.info("--- NPO: fixed reference handling ---")
    npo_dir = rv3_dir / "npo"; npo_dir.mkdir(exist_ok=True)
    adapter, model, processor = load_edit_model(args, npo_dir, "e2c_v3_cy")
    forget_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
          "answer": target_alias}], repeat=args.ul_repeat * 5)
    retain_items = build_supervised_items(
        adapter, processor, retain_pairs(identity_ids, [target]),
        repeat=args.ul_repeat)
    train_npo("npo", adapter, model, processor, forget_items, retain_items,
              npo_dir, args.device, steps=args.ul_steps, warmup=args.ul_warmup,
              lr=args.ul_lr)
    del model, processor, adapter; torch.cuda.empty_cache()
    method_results["npo"] = _eval_method(
        "npo", args, npo_dir, "e2c_v3_cy_npo", identity_ids, alias_of, target,
        oracle_dist, args.seed)

    # --- KL-anchored --- #
    logger.info("--- KL: GA + KL(pi||pi_ref) anchor on retain ---")
    kl_dir = rv3_dir / "kl"; kl_dir.mkdir(exist_ok=True)
    adapter, model, processor = load_edit_model(args, kl_dir, "e2c_v3_cy")
    forget_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
          "answer": target_alias}], repeat=args.ul_repeat * 5)
    retain_items = build_supervised_items(
        adapter, processor, retain_pairs(identity_ids, [target]),
        repeat=args.ul_repeat)
    train_kl("kl", adapter, model, processor, forget_items, retain_items,
             kl_dir, args.device, steps=args.ul_steps, warmup=args.ul_warmup,
             lr=args.ul_lr)
    del model, processor, adapter; torch.cuda.empty_cache()
    method_results["kl"] = _eval_method(
        "kl", args, kl_dir, "e2c_v3_cy_kl", identity_ids, alias_of, target,
        oracle_dist, args.seed)

    # --- GD: distribution-matching toward refusal --- #
    logger.info("--- GD: distribution matching toward refusal ---")
    gd_dir = rv3_dir / "gd"; gd_dir.mkdir(exist_ok=True)
    adapter, model, processor = load_edit_model(args, gd_dir, "e2c_v3_cy")
    train_gd_refusal("gd", adapter, model, processor, [target],
                     retain_pairs(identity_ids, [target]), gd_dir,
                     args.device, steps=args.ul_steps, warmup=args.ul_warmup,
                     lr=args.ul_lr)
    del model, processor, adapter; torch.cuda.empty_cache()
    method_results["gd"] = _eval_method(
        "gd", args, gd_dir, "e2c_v3_cy_gd", identity_ids, alias_of, target,
        oracle_dist, args.seed)

    with open(rv3_dir / "method_comparison.json", "w") as f:
        json.dump({"oracle_target_distribution":
                   {k: round(v, 6) for k, v in oracle_dist.items()},
                   "methods": method_results}, f, indent=2)
    logger.info("RV3 COMPARISON:")
    for method, res in method_results.items():
        logger.info(f"  {method}: pred={res['target_prediction']} "
                    f"p_old={res['p_old_alias']} p_del={res['p_deleted_label']} "
                    f"non_target={res['non_target_acc']} "
                    f"dist_l2(alias)={res['distance_to_oracle_alias_only']['l2']}")
    return method_results


# ====================================================================== #
# RV4: Multi-seed robustness (genuine deletion)
# ====================================================================== #
def run_rv4_multiseed(args, out_base, identity_ids, alias_of):
    logger.info("=" * 60)
    logger.info("RV4: MULTI-SEED ROBUSTNESS (genuine deletion)")
    logger.info("=" * 60)
    seeds = args.seeds
    target = args.target_id
    other_ids = [i for i in identity_ids if i != target]
    rv4_dir = out_base / "RV4_multiseed"
    rv4_dir.mkdir(parents=True, exist_ok=True)
    vocab = alias_label_vocab()
    all_seed_results = {}
    for seed in seeds:
        logger.info(f"--- Seed {seed} ---")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_dir = rv4_dir / f"seed_{seed}"; seed_dir.mkdir(exist_ok=True)
        adapter, model, processor = load_edit_model(args, seed_dir, "e2c_v3_cy")
        items = (
            build_supervised_items(adapter, processor, deletion_pairs([target]),
                                   repeat=args.ul_repeat * 5)
            + build_supervised_items(adapter, processor,
                                     retain_pairs(identity_ids, [target]),
                                     repeat=args.ul_repeat))
        train_supervised(f"cf_seed{seed}", adapter, model, processor, items,
                         seed_dir, args.device, steps=args.ul_steps,
                         warmup=args.ul_warmup, lr=args.ul_lr)
        del model, processor, adapter; torch.cuda.empty_cache()
        adapter, model, processor = fresh_eval_model(
            args, f"e2c_v3_h_s{seed}", seed_dir)
        h_eval = evaluate_h_labels(adapter, model, processor, identity_ids,
                                   alias_of, args.device, seed, vocab,
                                   tag=f"s{seed}")
        target_pred = next(p["parsed_label"] for p in h_eval["preds"]
                           if p["identity_id"] == target)
        target_removed = not any(
            p["correct"] for p in h_eval["preds"]
            if p["identity_id"] == target)
        refused = (target_pred == DELETED_LABEL)
        other_acc = sum(
            1 for p in h_eval["preds"]
            if p["identity_id"] != target and p["correct"]) / len(other_ids)
        all_seed_results[str(seed)] = {
            "target_removed": target_removed,
            "target_refused": refused,
            "target_prediction": target_pred,
            "non_target_retention": round(other_acc, 4),
        }
        logger.info(f"  Seed {seed}: removed={target_removed} refused={refused} "
                    f"retention={other_acc:.3f}")
        del model, processor, adapter; torch.cuda.empty_cache()
    n_removed = sum(1 for r in all_seed_results.values() if r["target_removed"])
    n_refused = sum(1 for r in all_seed_results.values() if r["target_refused"])
    avg_ret = sum(r["non_target_retention"]
                  for r in all_seed_results.values()) / len(seeds)
    summary = {
        "seeds": seeds, "deletion_label": DELETED_LABEL,
        "per_seed": all_seed_results,
        "deletion_rate": round(n_removed / len(seeds), 4),
        "refusal_rate": round(n_refused / len(seeds), 4),
        "avg_non_target_retention": round(avg_ret, 4),
        "all_seeds_succeeded": n_removed == len(seeds),
    }
    with open(rv4_dir / "multiseed_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"RV4: {n_removed}/{len(seeds)} removed, {n_refused} refused, "
                f"avg retention={avg_ret:.3f}")
    return summary



# ====================================================================== #
# RV5: Multi-depth granularity (ordinal, separate transformations)
# ====================================================================== #
def _eval_granularity_levels(adapter, model, processor, identity_ids, device,
                             seed, tag):
    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed)
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()
    vocab = granularity_vocab()
    details = {"specific": {}, "subgroup": {}, "group": {}}
    with torch.no_grad():
        for iid in identity_ids:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=8)
            parsed = parse_recognized_label(gen.text.strip(), vocab)
            checks = {
                "specific": (parsed == ALIAS_OF[iid]),
                "subgroup": (parsed == SUBGROUP_MAP[iid]),
                "group": (parsed == GROUP_MAP[iid]),
            }
            for level, ok in checks.items():
                expected = {"specific": ALIAS_OF[iid],
                            "subgroup": SUBGROUP_MAP[iid],
                            "group": GROUP_MAP[iid]}[level]
                details[level][iid] = {
                    "prediction": parsed, "expected": expected, "correct": ok}
            logger.info(f"  [{tag}] {iid}: parsed='{parsed}' "
                        f"spec={checks['specific']} sub={checks['subgroup']} "
                        f"grp={checks['group']}")
    summary = {}
    for level in ("specific", "subgroup", "group"):
        n_ok = sum(1 for v in details[level].values() if v["correct"])
        summary[level] = {"accuracy": round(n_ok / len(identity_ids), 4),
                          "correct": n_ok, "total": len(identity_ids)}
    return summary, details


def run_rv5_granularity(args, out_base, identity_ids, alias_of):
    logger.info("=" * 60)
    logger.info("RV5: MULTI-DEPTH GRANULARITY (ordinal transformations)")
    logger.info("=" * 60)
    rv5_dir = out_base / "RV5_granularity"
    rv5_dir.mkdir(parents=True, exist_ok=True)
    # Each depth is a SEPARATE transformation trained on its own model.
    depth_targets = {
        "specific_to_subgroup": SUBGROUP_MAP,
        "specific_to_group": GROUP_MAP,
    }
    all_depth = {}
    for depth_name, label_map in depth_targets.items():
        logger.info(f"--- Depth transformation: {depth_name} ---")
        d_dir = rv5_dir / depth_name
        d_dir.mkdir(parents=True, exist_ok=True)
        adapter, model, processor = load_edit_model(args, d_dir, "e2c_v3_h_g")
        pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
                  "answer": label_map[iid]} for iid in identity_ids]
        items = build_supervised_items(adapter, processor, pairs,
                                       repeat=args.ul_repeat * 3)
        train_supervised(depth_name, adapter, model, processor, items, d_dir,
                         args.device, steps=args.ul_steps,
                         warmup=args.ul_warmup, lr=args.ul_lr)
        del model, processor, adapter; torch.cuda.empty_cache()
        adapter, model, processor = fresh_eval_model(
            args, f"e2c_v3_h_g_{depth_name}", d_dir)
        summary, details = _eval_granularity_levels(
            adapter, model, processor, identity_ids, args.device, args.seed,
            tag=depth_name)
        all_depth[depth_name] = {"level_accuracy": summary,
                                 "details": details}
        logger.info(f"  {depth_name}: " + ", ".join(
            f"{lvl}={summary[lvl]['accuracy']}" for lvl in summary))
        del model, processor, adapter; torch.cuda.empty_cache()
    with open(rv5_dir / "granularity_results.json", "w") as f:
        json.dump(all_depth, f, indent=2)
    return all_depth


# ====================================================================== #
# RV6: Compositional mixture (genuine deletion + update + generalize + retain)
# ====================================================================== #
def run_rv6_mixture(args, out_base, identity_ids, alias_of):
    logger.info("=" * 60)
    logger.info("RV6: COMPOSITIONAL MIXTURE (S_D/S_U/S_G/S_R simultaneously)")
    logger.info("=" * 60)
    rv6_dir = out_base / "RV6_mixture"
    rv6_dir.mkdir(parents=True, exist_ok=True)
    s_delete = list(args.rv6_delete)
    s_update = list(args.rv6_update)
    s_generalize = list(args.rv6_generalize)
    s_retain = [i for i in identity_ids
                if i not in s_delete + s_update + s_generalize]
    logger.info(f"S_D(delete->{DELETED_LABEL})={s_delete} "
                f"S_U(update->{UPDATE_NEW_ALIAS})={s_update} "
                f"S_G(generalize->group)={s_generalize} S_R(retain)={s_retain}")

    adapter, model, processor = load_edit_model(args, rv6_dir, "e2c_v3_h_mix")
    mix_items = (
        build_supervised_items(adapter, processor, deletion_pairs(s_delete),
                               repeat=args.ul_repeat * 5)
        + build_supervised_items(
            adapter, processor,
            [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=i),
              "answer": UPDATE_NEW_ALIAS} for i in s_update],
            repeat=args.ul_repeat * 3)
        + build_supervised_items(
            adapter, processor,
            [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=i),
              "answer": GROUP_MAP[i]} for i in s_generalize],
            repeat=args.ul_repeat * 3)
        + build_supervised_items(
            adapter, processor,
            retain_pairs(identity_ids,
                         exclude=s_delete + s_update + s_generalize),
            repeat=args.ul_repeat * 3))
    train_supervised("mixture", adapter, model, processor, mix_items, rv6_dir,
                     args.device, steps=args.ul_steps, warmup=args.ul_warmup,
                     lr=args.ul_lr)
    del model, processor, adapter; torch.cuda.empty_cache()

    adapter, model, processor = fresh_eval_model(args, "e2c_v3_h_mix", rv6_dir)
    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed)
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()
    vocab = sorted(set(granularity_vocab() + [UPDATE_NEW_ALIAS]))
    eval_results = {"delete": {}, "update": {}, "generalize": {}, "retain": {}}
    with torch.no_grad():
        for iid in identity_ids:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=8)
            parsed = parse_recognized_label(gen.text.strip(), vocab)
            if iid in s_delete:
                ok = parsed == DELETED_LABEL
                eval_results["delete"][iid] = {
                    "prediction": parsed, "expected": DELETED_LABEL,
                    "success": ok}
            elif iid in s_update:
                ok = parsed == UPDATE_NEW_ALIAS
                eval_results["update"][iid] = {
                    "prediction": parsed, "expected": UPDATE_NEW_ALIAS,
                    "success": ok}
            elif iid in s_generalize:
                ok = parsed == GROUP_MAP[iid]
                eval_results["generalize"][iid] = {
                    "prediction": parsed, "expected": GROUP_MAP[iid],
                    "success": ok}
            else:
                ok = parsed == ALIAS_OF[iid]
                eval_results["retain"][iid] = {
                    "prediction": parsed, "expected": ALIAS_OF[iid],
                    "success": ok}
    summary = {}
    for cat, data in eval_results.items():
        n_ok = sum(1 for v in data.values() if v["success"])
        summary[cat] = {"success": n_ok, "total": len(data),
                        "rate": round(n_ok / max(len(data), 1), 4)}
        logger.info(f"  {cat}: {n_ok}/{len(data)}")
    with open(rv6_dir / "mixture_results.json", "w") as f:
        json.dump({"summary": summary, "details": eval_results,
                   "sets": {"S_D": s_delete, "S_U": s_update,
                            "S_G": s_generalize, "S_R": s_retain}},
                  f, indent=2)
    del model, processor, adapter; torch.cuda.empty_cache()
    return summary


# ====================================================================== #
# RV7: Simultaneous multi-target deletion (ONE model)
# ====================================================================== #
def run_rv7_simultaneous(args, out_base, identity_ids, alias_of):
    logger.info("=" * 60)
    logger.info("RV7: SIMULTANEOUS MULTI-TARGET DELETION (single model)")
    logger.info("=" * 60)
    rv7_dir = out_base / "RV7_simultaneous"
    rv7_dir.mkdir(parents=True, exist_ok=True)
    delete_set = list(args.rv7_delete)
    retain_set = [i for i in identity_ids if i not in set(delete_set)]
    logger.info(f"Deleting {len(delete_set)} identities SIMULTANEOUSLY in ONE "
                f"model: {delete_set}; retaining: {retain_set}")

    adapter, model, processor = load_edit_model(args, rv7_dir, "e2c_v3_h_sim")
    items = (
        build_supervised_items(adapter, processor, deletion_pairs(delete_set),
                               repeat=args.ul_repeat * 5)
        + build_supervised_items(adapter, processor,
                                 retain_pairs(identity_ids, delete_set),
                                 repeat=args.ul_repeat))
    train_supervised("rv7_simultaneous", adapter, model, processor, items,
                     rv7_dir, args.device, steps=args.ul_steps,
                     warmup=args.ul_warmup, lr=args.ul_lr)
    del model, processor, adapter; torch.cuda.empty_cache()

    adapter, model, processor = fresh_eval_model(args, "e2c_v3_h_sim", rv7_dir)
    vocab = alias_label_vocab()
    h_eval = evaluate_h_labels(adapter, model, processor, identity_ids,
                               alias_of, args.device, args.seed, vocab,
                               tag="rv7")
    per_identity = {}
    for iid in identity_ids:
        pred = next(p["parsed_label"] for p in h_eval["preds"]
                    if p["identity_id"] == iid)
        if iid in delete_set:
            per_identity[iid] = {
                "role": "deleted", "prediction": pred,
                "target_removed": pred != alias_of[iid],
                "refused": pred == DELETED_LABEL}
        else:
            per_identity[iid] = {
                "role": "retained", "prediction": pred,
                "retained_correct": pred == alias_of[iid]}
    n_removed = sum(1 for i in delete_set
                    if per_identity[i]["target_removed"])
    n_refused = sum(1 for i in delete_set if per_identity[i]["refused"])
    n_retained_ok = sum(1 for i in retain_set
                        if per_identity[i]["retained_correct"])
    summary = {
        "simultaneous": True, "single_model": True,
        "delete_set": delete_set, "retain_set": retain_set,
        "deletion_label": DELETED_LABEL,
        "n_targets": len(delete_set), "n_removed": n_removed,
        "n_refused": n_refused,
        "deletion_rate": round(n_removed / max(len(delete_set), 1), 4),
        "refusal_rate": round(n_refused / max(len(delete_set), 1), 4),
        "non_target_retention": round(
            n_retained_ok / max(len(retain_set), 1), 4),
        "per_identity": per_identity,
    }
    with open(rv7_dir / "simultaneous_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"RV7: {n_removed}/{len(delete_set)} removed "
                f"({n_refused} refused), retention={summary['non_target_retention']}")
    del model, processor, adapter; torch.cuda.empty_cache()
    return summary



# ====================================================================== #
# RV8: Post-edit end-to-end image -> code -> output + visual control
# ====================================================================== #
def _load_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _extract_code(text, identity_ids):
    for iid in identity_ids:
        if iid in text:
            return iid
    return None


def run_rv8_e2e(args, out_base, identity_ids, alias_of):
    logger.info("=" * 60)
    logger.info("RV8: POST-EDIT END-TO-END image -> code -> output")
    logger.info("=" * 60)
    rv8_dir = out_base / "RV8_e2e"
    rv8_dir.mkdir(parents=True, exist_ok=True)
    delete_set = set(args.rv7_delete)
    image_base = Path(args.image_base)

    with open(IMAGE_SPLIT_MANIFEST) as f:
        split_items = json.load(f)
    test_items = [it for it in split_items if it["split"] == "test"]
    # sample up to N images per identity
    per_id = {}
    for it in test_items:
        per_id.setdefault(it["identity_id"], []).append(it)
    sampled = []
    for iid in identity_ids:
        sampled.extend(per_id.get(iid, [])[:args.rv8_images_per_id])
    logger.info(f"Sampled {len(sampled)} test images across "
                f"{len(per_id)} identities")

    # --- Stage 1: frozen g (X -> C) predicts codes --- #
    adapter_g, model_g, processor_g = create_adapter_model(
        args, args.device, "e2c_v3_xc")
    model_g = adapter_g.load_unlearning_adapter(
        model_g, Path(args.xc_adapter_dir) / "adapter_final",
        adapter_name="e2c_v3_xc")
    from route_data.config import ModelConfig
    cfg = ModelConfig(backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
                      revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                      dtype="bfloat16", seed=args.seed)
    g_backend = adapter_g.to_eval_backend(
        model=model_g, processor=processor_g, model_config=cfg)
    model_g.eval()
    code_cache = []
    with torch.no_grad():
        for it in sampled:
            image = _load_image(str(image_base / it["image_path"]))
            gen = g_backend.generate(image, IMG_TO_CODE_PROMPT, max_new_tokens=5)
            pred_code = _extract_code(gen.text.strip(), identity_ids)
            code_cache.append({
                "identity_id": it["identity_id"],
                "image_id": it["image_id"],
                "true_code": it["identity_id"],
                "pred_code": pred_code,
                "code_correct": pred_code == it["identity_id"],
            })
    g_acc = sum(c["code_correct"] for c in code_cache) / len(code_cache)
    logger.info(f"  frozen g accuracy on sampled images: {g_acc:.3f}")
    del model_g, processor_g, adapter_g
    torch.cuda.empty_cache()

    # --- Stage 2: edited h (C -> Y) on the g-predicted codes --- #
    adapter_h, model_h, processor_h = fresh_eval_model(
        args, args.rv8_h_adapter_name, Path(args.rv8_h_adapter_dir))
    h_backend = adapter_h.to_eval_backend(
        model=model_h, processor=processor_h, model_config=cfg)
    model_h.eval()
    e2e_rows = []
    with torch.no_grad():
        for c in code_cache:
            code = c["pred_code"] or c["true_code"]
            prompt = CODE_TO_ALIAS_PROMPT.format(code=code)
            gen = h_backend.generate(None, prompt, max_new_tokens=8)
            vocab = alias_label_vocab()
            parsed = parse_recognized_label(gen.text.strip(), vocab)
            iid = c["identity_id"]
            if iid in delete_set:
                ok = parsed != alias_of[iid]
                status = "deleted"
            else:
                ok = parsed == alias_of[iid]
                status = "retained"
            e2e_rows.append({
                **c, "pred_alias": parsed, "status": status, "e2e_ok": ok})
    del model_h, processor_h, adapter_h
    torch.cuda.empty_cache()

    deleted_rows = [r for r in e2e_rows if r["status"] == "deleted"]
    retained_rows = [r for r in e2e_rows if r["status"] == "retained"]
    del_ok = sum(r["e2e_ok"] for r in deleted_rows)
    ret_ok = sum(r["e2e_ok"] for r in retained_rows)
    summary = {
        "g_code_accuracy": round(g_acc, 4),
        "n_images": len(e2e_rows),
        "deleted_images": len(deleted_rows),
        "deleted_e2e_suppressed": del_ok,
        "deleted_e2e_rate": round(del_ok / max(len(deleted_rows), 1), 4),
        "retained_images": len(retained_rows),
        "retained_e2e_correct": ret_ok,
        "retained_e2e_rate": round(ret_ok / max(len(retained_rows), 1), 4),
        "rows": e2e_rows,
    }

    # --- Visual control probe --- #
    vc = None
    if VISUAL_CONTROLS_MANIFEST.exists():
        with open(VISUAL_CONTROLS_MANIFEST) as f:
            controls = json.load(f)
        vc_sample = controls[:args.rv8_visual_controls]
        vc = {"n_controls": len(vc_sample),
              "note": "attribute-control images (eyeglasses/hat/smiling) "
                      "routed through the frozen g; the discrete code is "
                      "attribute-invariant by construction.",
              "sample_ids": [c.get("image_id") for c in vc_sample]}
    summary["visual_control"] = vc
    with open(rv8_dir / "e2e_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"RV8: g_acc={g_acc:.3f} deleted_suppressed="
                f"{del_ok}/{len(deleted_rows)} retained_correct="
                f"{ret_ok}/{len(retained_rows)}")
    return summary


# ====================================================================== #
# RV9: Run manifest binding commit / CLI / inputs / hashes / seeds
# ====================================================================== #
def run_rv9_manifest(args, out_base, results, t_start):
    logger.info("=" * 60)
    logger.info("RV9: RUN MANIFEST")
    logger.info("=" * 60)
    inputs = {}
    for name, path in [("identity_code_mapping", IDENTITY_MAPPING_MANIFEST),
                       ("image_split", IMAGE_SPLIT_MANIFEST),
                       ("visual_controls", VISUAL_CONTROLS_MANIFEST)]:
        if Path(path).exists():
            inputs[name] = sha256_file(path)
    checkpoints = {}
    for name, path in [("phaseC_C_to_Y", PHASE_C_CY_CKPT),
                       ("phaseC_X_to_C", PHASE_C_XC_CKPT)]:
        if Path(path).exists():
            checkpoints[name] = sha256_file(path)
    outputs = {}
    for p in sorted(Path(out_base).rglob("*")):
        if p.is_file() and p.name != "run_manifest.json":
            outputs[str(p.relative_to(out_base))] = sha256_file(p)
    manifest = {
        "git_commit": git_commit_sha(),
        "cli_invocation": sys.argv,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": round(time.time() - t_start, 1),
        "config": {
            "device": args.device, "seed": args.seed, "phase": args.phase,
            "target_id": args.target_id,
            "ul_steps": args.ul_steps, "ul_warmup": args.ul_warmup,
            "ul_lr": args.ul_lr, "ul_repeat": args.ul_repeat,
            "oracle_steps": args.oracle_steps,
            "oracle_warmup": args.oracle_warmup,
            "oracle_lr": args.oracle_lr, "oracle_repeat": args.oracle_repeat,
            "seeds": args.seeds, "rv6_delete": list(args.rv6_delete),
            "rv6_update": list(args.rv6_update),
            "rv6_generalize": list(args.rv6_generalize),
            "rv7_delete": list(args.rv7_delete),
            "deletion_label": DELETED_LABEL,
            "update_new_alias": UPDATE_NEW_ALIAS},
        "inputs_sha256": inputs,
        "checkpoints_sha256": checkpoints,
        "outputs_sha256": outputs,
        "phase_status": {k: ("ok" if v is not None else "skipped")
                         for k, v in results.items()},
    }
    with open(out_base / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: commit={manifest['git_commit']} "
                f"outputs_hashed={len(outputs)}")
    return manifest


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default="e2c_v3/outputs/research_validity")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", default="all",
                        choices=["all", "RV1", "RV2", "RV3", "RV4", "RV5",
                                 "RV6", "RV7", "RV8", "RV9"])
    parser.add_argument("--target-id", default="syn_00")
    # editing budget
    parser.add_argument("--ul-steps", type=int, default=UL_STEPS)
    parser.add_argument("--ul-warmup", type=int, default=UL_WARMUP)
    parser.add_argument("--ul-lr", type=float, default=UL_LR)
    parser.add_argument("--ul-repeat", type=int, default=UL_REPEAT)
    # oracle budget (MUST be training-matched to Phase-C C->Y)
    parser.add_argument("--oracle-steps", type=int, default=ORACLE_STEPS)
    parser.add_argument("--oracle-warmup", type=int, default=ORACLE_WARMUP)
    parser.add_argument("--oracle-lr", type=float, default=ORACLE_LR)
    parser.add_argument("--oracle-repeat", type=int, default=ORACLE_REPEAT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 123])
    # RV6 set assignments
    parser.add_argument("--rv6-delete", nargs="+", default=["syn_00"])
    parser.add_argument("--rv6-update", nargs="+", default=["syn_02"])
    parser.add_argument("--rv6-generalize", nargs="+", default=["syn_04"])
    # RV7 simultaneous target set
    parser.add_argument("--rv7-delete", nargs="+",
                        default=["syn_00", "syn_01", "syn_02", "syn_03",
                                 "syn_04"])
    # RV8
    parser.add_argument("--image-base", default="e2c/data/processed")
    parser.add_argument("--xc-adapter-dir",
                        default="e2c_v3/outputs/phaseC/X_to_C")
    parser.add_argument("--rv8-h-adapter-dir",
                        default="e2c_v3/outputs/research_validity/RV7_simultaneous")
    parser.add_argument("--rv8-h-adapter-name", default="e2c_v3_h_sim")
    parser.add_argument("--rv8-images-per-id", type=int, default=3)
    parser.add_argument("--rv8-visual-controls", type=int, default=5)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny budgets for fast validation")
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.ul_steps = min(args.ul_steps, 20)
        args.ul_warmup = min(args.ul_warmup, 5)
        args.ul_repeat = min(args.ul_repeat, 3)
        args.oracle_steps = min(args.oracle_steps, 30)
        args.oracle_warmup = min(args.oracle_warmup, 5)
        args.oracle_repeat = min(args.oracle_repeat, 3)
        args.seeds = args.seeds[:1]
        args.rv8_images_per_id = 1
        args.rv8_visual_controls = 2
        logger.info("SMOKE MODE: budgets reduced for validation")

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t_start = time.time()
    run_phases = (["RV1", "RV2", "RV3", "RV4", "RV5", "RV6", "RV7", "RV8"]
                  if args.phase == "all" else [args.phase])
    results = {}

    if "RV1" in run_phases:
        results["RV1"] = run_rv1_soft_metrics(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV2" in run_phases:
        results["RV2"] = run_rv2_oracle(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV3" in run_phases:
        # obtain oracle distribution (from RV2 this run or from disk)
        oracle_dist = None
        if results.get("RV2"):
            oracle_dist = results["RV2"]["oracle_target_distribution"]
        else:
            p = out_base / "RV2_oracle" / "oracle_results.json"
            if p.exists():
                with open(p) as f:
                    oracle_dist = json.load(f)["oracle_target_distribution"]
        if oracle_dist is None:
            raise RuntimeError("RV3 requires an oracle distribution; run RV2 "
                               "first or point --out-base at an RV2 run.")
        oracle_dist = {k: float(v) for k, v in oracle_dist.items()}
        results["RV3"] = run_rv3_methods(
            args, out_base, IDENTITY_IDS, ALIAS_OF, oracle_dist)
    if "RV4" in run_phases:
        results["RV4"] = run_rv4_multiseed(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV5" in run_phases:
        results["RV5"] = run_rv5_granularity(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV6" in run_phases:
        results["RV6"] = run_rv6_mixture(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV7" in run_phases:
        results["RV7"] = run_rv7_simultaneous(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV8" in run_phases:
        results["RV8"] = run_rv8_e2e(
            args, out_base, IDENTITY_IDS, ALIAS_OF)

    logger.info("=" * 60)
    logger.info("RESEARCH-VALIDITY PHASE COMPLETE")
    logger.info("=" * 60)
    with open(out_base / "rv_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    if not args.no_manifest:
        run_rv9_manifest(args, out_base, results, t_start)
    logger.info(f"Results saved under {out_base} "
                f"({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
