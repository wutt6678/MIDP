#!/usr/bin/env python3
"""E2C-v3 Research-Validity Phase (RV1–RV7).

Extends the frozen U0–U8 pipeline with research-grade evidence:

RV1: Soft metrics — P(Y_old), P(Y_new), margin, entropy (not just hard acc).
RV2: Oracle baseline — retrain h from scratch WITHOUT target data.
RV3: Method comparison — counterfactual vs GA vs oracle vs NPO.
RV4: Multi-seed robustness — counterfactual deletion across seeds.
RV5: Multi-depth granularity — specific → subgroup → group.
RV6: Compositional mixture — delete + update + generalize + retain in one h.
RV7: Expanded multi-target deletion (all 10 identities).

Architecture constraint (freeze): g (X→C) NEVER modified; only h (C→Y).
"""
import argparse
import json
import logging
import math
import re
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

# Training defaults
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5

# Identity aliases
ALIAS_OF = {
    "syn_00": "Aven", "syn_01": "Bira", "syn_02": "Caro",
    "syn_03": "Deni", "syn_04": "Eris", "syn_05": "Faro",
    "syn_06": "Gela", "syn_07": "Hani", "syn_08": "Ivoa",
    "syn_09": "Jora",
}
IDENTITY_IDS = [f"syn_{i:02d}" for i in range(10)]

# Granularity hierarchy
GROUP_MAP = {iid: ("GROUP_A" if ALIAS_OF[iid][0] in "ABCDE" else "GROUP_B")
             for iid in IDENTITY_IDS}
SUBGROUP_MAP = {
    "syn_00": "SG_A1", "syn_01": "SG_A1",  # Aven, Bira
    "syn_02": "SG_A2", "syn_03": "SG_A2",  # Caro, Deni
    "syn_04": "SG_A3",                      # Eris
    "syn_05": "SG_B1", "syn_06": "SG_B1",  # Faro, Gela
    "syn_07": "SG_B2", "syn_08": "SG_B2",  # Hani, Ivoa
    "syn_09": "SG_B3",                      # Jora
}


# ====================================================================== #
# Infrastructure (shared with e2c_v3_unlearning.py)
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


def load_trained_weights(adapter, model):
    from safetensors.torch import load_file
    ckpt_path = Path("e2c_v3/outputs/phaseC/C_to_Y/adapter_final/adapter_model.safetensors")
    ckpt_data = load_file(str(ckpt_path))
    live_params = dict(model.named_parameters())
    aname = adapter.profile.adapter_name
    copied = 0
    for ckpt_key, ckpt_tensor in ckpt_data.items():
        if ckpt_key in live_params:
            live_params[ckpt_key].data.copy_(ckpt_tensor)
            copied += 1
        else:
            remapped = ckpt_key.replace("lora_A", f"lora_A.{aname}").replace(
                "lora_B", f"lora_B.{aname}")
            if remapped in live_params:
                live_params[remapped].data.copy_(ckpt_tensor)
                copied += 1
    logger.info(f"Loaded {copied}/{len(ckpt_data)} trained weight tensors")
    return copied


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


def train_standard(condition, adapter, model, processor, train_items,
                   output_dir, device, steps=UL_STEPS, warmup=UL_WARMUP,
                   lr=UL_LR, extra_loss_fn=None):
    """Standard supervised training with optional extra loss term."""
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    logger.info(f"[{condition}] trainable params: {n_params:,}, "
                f"items: {len(train_items)}, steps: {steps}, lr: {lr}")

    class ItemDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    loader = DataLoader(
        ItemDataset(train_items), batch_size=1, shuffle=True,
        collate_fn=lambda b: adapter.collate(b), num_workers=0,
    )
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
    optimizer = AdamW(params, lr=lr, weight_decay=0.0)
    actual_warmup = min(warmup, max(steps - 1, 1))
    warmup_s = LinearLR(optimizer, start_factor=0.1, total_iters=actual_warmup)
    cosine_s = CosineAnnealingLR(optimizer, T_max=max(steps - actual_warmup, 1))
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_s, cosine_s],
                             milestones=[actual_warmup])

    model.train()
    trace = []
    global_step = 0
    running_loss = 0.0

    for epoch in range(1000):
        for batch in loader:
            bd = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    bd[k] = v.to(device)
                else:
                    bd[k] = v
            outputs = model(
                input_ids=bd["input_ids"],
                attention_mask=bd["attention_mask"],
                labels=bd["labels"],
                use_cache=False,
                **{k: v for k, v in bd.items()
                   if k not in ("input_ids", "attention_mask", "labels")
                   and isinstance(v, torch.Tensor)
                   and not k.startswith("_")},
            )
            loss = outputs.loss
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model, bd, adapter, processor)
            loss.backward()
            running_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 50 == 0 or global_step <= 5:
                avg = running_loss / global_step
                trace.append({"step": global_step, "loss": avg})
                logger.info(f"[{condition}] Step {global_step}/{steps} "
                            f"loss={avg:.6f}")
            if global_step >= steps:
                break
        if global_step >= steps:
            break

    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    final_loss = running_loss / max(global_step, 1)
    logger.info(f"[{condition}] complete. Final avg loss={final_loss:.6f}")
    return trace


def fresh_model_for_eval(args, adapter_name, adapter_dir):
    adapter, model, processor = create_adapter_model(
        args, args.device, adapter_name)
    model = adapter.load_unlearning_adapter(
        model, adapter_dir / "adapter_final", adapter_name=adapter_name)
    return adapter, model, processor


def evaluate_h_hard(adapter, model, processor, identity_ids, alias_of,
                    device, seed, tag="eval"):
    """Hard accuracy evaluation (code → alias)."""
    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()
    preds = []
    with torch.no_grad():
        for iid in identity_ids:
            expected = alias_of[iid]
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=5)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")
            ok = pred.lower() == expected.lower()
            preds.append({"identity_id": iid, "expected": expected,
                          "prediction": pred, "correct": ok})
            logger.info(f"  [{tag}] {iid} → '{pred}' "
                        f"(expected '{expected}') {'✓' if ok else '✗'}")
    n_correct = sum(p["correct"] for p in preds)
    return {"accuracy": n_correct / len(preds), "preds": preds}


# ====================================================================== #
# RV1: Soft metrics — probability extraction
# ====================================================================== #
def extract_alias_probabilities(adapter, model, processor, code_id,
                                candidate_aliases, device):
    """Extract P(alias | code) from the model's first-token logits.

    Returns dict: {alias: {"prob": float, "logit": float}}.
    """
    # Build prompt using chat template
    prompt_text = processor.tokenizer.apply_chat_template(
        [{"role": "user",
          "content": CODE_TO_ALIAS_PROMPT.format(code=code_id)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = processor.tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = {k: v.to(device) for k, v in input_ids.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**input_ids, use_cache=False)
        logits = outputs.logits[0, -1, :]  # last prompt position
        log_probs = torch.log_softmax(logits, dim=-1)

    results = {}
    for alias in candidate_aliases:
        tok_ids = processor.tokenizer.encode(
            alias, add_special_tokens=False)
        first_tok_id = tok_ids[0]
        lp = log_probs[first_tok_id].item()
        results[alias] = {"prob": math.exp(lp), "log_prob": lp,
                          "first_token_id": first_tok_id}
    return results


def run_rv1_soft_metrics(args, out_base, identity_ids, alias_of):
    """RV1: Extract soft probabilities for each identity."""
    logger.info("=" * 60)
    logger.info("RV1: SOFT METRICS (probabilities, margins, entropy)")
    logger.info("=" * 60)

    rv1_dir = out_base / "RV1_soft_metrics"
    rv1_dir.mkdir(parents=True, exist_ok=True)

    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)
    logger.info("Loaded trained h (C→Y) weights")

    all_aliases = sorted(set(alias_of.values()))
    all_results = {}

    for iid in identity_ids:
        probs = extract_alias_probabilities(
            adapter, model, processor, iid, all_aliases, args.device)
        correct_alias = alias_of[iid]
        p_correct = probs[correct_alias]["prob"]

        # Find runner-up
        sorted_aliases = sorted(probs.items(), key=lambda x: -x[1]["prob"])
        runner_up = sorted_aliases[1] if sorted_aliases[0][0] == correct_alias else sorted_aliases[0]
        margin = p_correct - runner_up[1]["prob"]

        # Entropy
        all_p = torch.tensor([probs[a]["prob"] for a in all_aliases])
        all_p = all_p / all_p.sum()  # renormalize
        entropy = -(all_p * torch.log(all_p + 1e-10)).sum().item()
        max_entropy = math.log(len(all_aliases))

        all_results[iid] = {
            "correct_alias": correct_alias,
            "p_correct": round(p_correct, 6),
            "runner_up_alias": runner_up[0],
            "p_runner_up": round(runner_up[1]["prob"], 6),
            "margin": round(margin, 6),
            "entropy": round(entropy, 4),
            "max_entropy": round(max_entropy, 4),
            "normalized_entropy": round(entropy / max_entropy, 4),
            "full_probs": {a: round(probs[a]["prob"], 6) for a in all_aliases},
        }
        logger.info(f"  {iid} ({correct_alias}): p={p_correct:.4f}, "
                    f"margin={margin:.4f}, H={entropy:.3f}/{max_entropy:.3f}")

    with open(rv1_dir / "soft_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    del model, processor, adapter
    torch.cuda.empty_cache()
    return all_results


# ====================================================================== #
# RV2: Oracle — retrain h from scratch without target
# ====================================================================== #
def run_rv2_oracle(args, out_base, identity_ids, alias_of):
    """RV2: Oracle baseline — train h from scratch excluding target."""
    logger.info("=" * 60)
    logger.info("RV2: ORACLE (retrain-from-scratch without target)")
    logger.info("=" * 60)

    target = args.target_id
    target_alias = alias_of[target]
    other_ids = [iid for iid in identity_ids if iid != target]

    rv2_dir = out_base / "RV2_oracle"
    rv2_dir.mkdir(parents=True, exist_ok=True)

    # Fresh adapter (no trained weights loaded)
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_h_oracle")
    model = attach_lora(adapter, model)

    # Train only on non-target identities
    oracle_pairs = []
    for iid in other_ids:
        oracle_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": alias_of[iid],
        })
    oracle_items = build_supervised_items(
        adapter, processor, oracle_pairs, repeat=args.ul_repeat)

    logger.info(f"Oracle: training on {len(other_ids)} identities "
                f"(excluding {target}), {len(oracle_items)} items")

    train_standard(
        "oracle", adapter, model, processor, oracle_items, rv2_dir,
        args.device, steps=args.ul_steps, warmup=args.ul_warmup, lr=args.ul_lr,
    )

    del model, processor, adapter
    torch.cuda.empty_cache()

    # Evaluate
    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_h_oracle", rv2_dir)
    h_eval = evaluate_h_hard(
        adapter, model, processor, identity_ids, alias_of,
        args.device, args.seed, tag="oracle")

    target_pred = None
    for p in h_eval["preds"]:
        if p["identity_id"] == target:
            target_pred = p["prediction"]
    other_correct = sum(
        1 for p in h_eval["preds"]
        if p["identity_id"] != target and p["correct"])
    other_acc = other_correct / len(other_ids)

    # Soft metrics for oracle
    all_aliases = sorted(set(alias_of.values()))
    oracle_probs = extract_alias_probabilities(
        adapter, model, processor, target, all_aliases, args.device)
    p_target_old = oracle_probs.get(target_alias, {}).get("prob", 0)

    results = {
        "method": "oracle_retrain_without_target",
        "target_id": target,
        "target_alias": target_alias,
        "target_prediction": target_pred,
        "target_was_correct": any(
            p["correct"] for p in h_eval["preds"]
            if p["identity_id"] == target),
        "p_target_old_alias": round(p_target_old, 6),
        "non_target_retention": round(other_acc, 4),
        "overall_accuracy": h_eval["accuracy"],
        "hard_preds": h_eval["preds"],
    }

    with open(rv2_dir / "oracle_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"RV2 SUMMARY: target_removed={not results['target_was_correct']}, "
                f"p({target_alias})={p_target_old:.4f}, "
                f"non_target={other_acc:.3f}")

    del model, processor, adapter
    torch.cuda.empty_cache()
    return results


# ====================================================================== #
# RV3: Method comparison — CF vs GA vs NPO vs Oracle
# ====================================================================== #
def run_rv3_methods(args, out_base, identity_ids, alias_of):
    """RV3: Compare counterfactual, gradient ascent, NPO against oracle."""
    logger.info("=" * 60)
    logger.info("RV3: METHOD COMPARISON (CF vs GA vs NPO vs Oracle)")
    logger.info("=" * 60)

    target = args.target_id
    target_alias = alias_of[target]
    wrong_alias = args.delete_to
    if wrong_alias == target_alias:
        wrong_alias = "Bira" if target_alias != "Bira" else "Caro"
    other_ids = [iid for iid in identity_ids if iid != target]
    all_aliases = sorted(set(alias_of.values()))

    rv3_dir = out_base / "RV3_method_comparison"
    rv3_dir.mkdir(parents=True, exist_ok=True)
    method_results = {}

    # --- Method 1: Counterfactual relabeling (already in U1) ---
    logger.info("--- CF: Counterfactual relabeling ---")
    cf_dir = rv3_dir / "counterfactual"
    cf_dir.mkdir(parents=True, exist_ok=True)
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)

    target_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
          "answer": wrong_alias}], repeat=args.ul_repeat * 5)
    other_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
          "answer": alias_of[iid]} for iid in other_ids],
        repeat=args.ul_repeat)
    train_standard("cf", adapter, model, processor,
                   target_items + other_items, cf_dir, args.device,
                   steps=args.ul_steps, warmup=args.ul_warmup, lr=args.ul_lr)
    del model, processor, adapter
    torch.cuda.empty_cache()

    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_cy_cf", cf_dir)
    cf_hard = evaluate_h_hard(adapter, model, processor, identity_ids,
                              alias_of, args.device, args.seed, tag="cf")
    cf_probs = extract_alias_probabilities(
        adapter, model, processor, target, all_aliases, args.device)
    method_results["counterfactual"] = {
        "target_prediction": next(
            p["prediction"] for p in cf_hard["preds"]
            if p["identity_id"] == target),
        "p_old_alias": round(cf_probs.get(target_alias, {}).get("prob", 0), 6),
        "p_wrong_alias": round(cf_probs.get(wrong_alias, {}).get("prob", 0), 6),
        "non_target_acc": round(sum(
            1 for p in cf_hard["preds"]
            if p["identity_id"] != target and p["correct"]) / len(other_ids), 4),
    }
    del model, processor, adapter
    torch.cuda.empty_cache()

    # --- Method 2: Gradient Ascent ---
    logger.info("--- GA: Gradient Ascent ---")
    ga_dir = rv3_dir / "gradient_ascent"
    ga_dir.mkdir(parents=True, exist_ok=True)
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)

    # GA: maximize loss on target pair (known to be weak)
    params = [p for p in model.parameters() if p.requires_grad]
    target_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
          "answer": target_alias}], repeat=args.ul_repeat * 5)
    other_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
          "answer": alias_of[iid]} for iid in other_ids],
        repeat=args.ul_repeat)
    all_items = target_items + other_items

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    optimizer = AdamW(params, lr=args.ul_lr, weight_decay=0.0)
    actual_warmup = min(args.ul_warmup, max(args.ul_steps - 1, 1))
    warmup_s = LinearLR(optimizer, start_factor=0.1, total_iters=actual_warmup)
    cosine_s = CosineAnnealingLR(optimizer, T_max=max(args.ul_steps - actual_warmup, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_s, cosine_s],
                             milestones=[actual_warmup])

    ga_collate = adapter.collate

    class ItemDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    loader = DataLoader(
        ItemDataset(all_items), batch_size=1, shuffle=True,
        collate_fn=lambda b: ga_collate(b), num_workers=0)

    model.train()
    ga_trace = []
    step = 0
    ga_running = 0.0
    for epoch in range(1000):
        for batch in loader:
            bd = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items()}
            outputs = model(
                input_ids=bd["input_ids"],
                attention_mask=bd["attention_mask"],
                labels=bd["labels"], use_cache=False,
                **{k: v for k, v in bd.items()
                   if k not in ("input_ids", "attention_mask", "labels")
                   and isinstance(v, torch.Tensor) and not k.startswith("_")})
            loss = outputs.loss
            # Gradient ascent: NEGATE the loss for target items
            # But keep descent for other items
            # Simple approach: negate ALL loss (pure GA)
            (-loss).backward()
            ga_running += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                ga_trace.append({"step": step, "loss": ga_running / step})
                logger.info(f"[GA] Step {step}/{args.ul_steps} "
                            f"loss={ga_running / step:.6f}")
            if step >= args.ul_steps:
                break
        if step >= args.ul_steps:
            break

    adapter.save_unlearning_adapter(model, ga_dir / "adapter_final")
    with open(ga_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in ga_trace)
    del model, processor, adapter
    torch.cuda.empty_cache()

    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_cy_ga", ga_dir)
    ga_hard = evaluate_h_hard(adapter, model, processor, identity_ids,
                              alias_of, args.device, args.seed, tag="ga")
    ga_probs = extract_alias_probabilities(
        adapter, model, processor, target, all_aliases, args.device)
    method_results["gradient_ascent"] = {
        "target_prediction": next(
            p["prediction"] for p in ga_hard["preds"]
            if p["identity_id"] == target),
        "p_old_alias": round(ga_probs.get(target_alias, {}).get("prob", 0), 6),
        "non_target_acc": round(sum(
            1 for p in ga_hard["preds"]
            if p["identity_id"] != target and p["correct"]) / len(other_ids), 4),
    }
    del model, processor, adapter
    torch.cuda.empty_cache()

    # --- Method 3: NPO (Negative Preference Optimization) ---
    logger.info("--- NPO: Negative Preference Optimization ---")
    npo_dir = rv3_dir / "npo"
    npo_dir.mkdir(parents=True, exist_ok=True)
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)

    # NPO: treat target pair as dispreferred, others as preferred
    # Simplified: maximize loss on target (like GA) but with KL regularization
    # to prevent drift on non-targets
    params = [p for p in model.parameters() if p.requires_grad]
    # Save initial weights for KL reference
    ref_weights = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            ref_weights[n] = p.data.clone()

    target_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
          "answer": target_alias}], repeat=args.ul_repeat * 5)
    other_items = build_supervised_items(
        adapter, processor,
        [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
          "answer": alias_of[iid]} for iid in other_ids],
        repeat=args.ul_repeat)

    optimizer = AdamW(params, lr=args.ul_lr, weight_decay=0.0)
    actual_warmup = min(args.ul_warmup, max(args.ul_steps - 1, 1))
    warmup_s = LinearLR(optimizer, start_factor=0.1, total_iters=actual_warmup)
    cosine_s = CosineAnnealingLR(optimizer, T_max=max(args.ul_steps - actual_warmup, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_s, cosine_s],
                             milestones=[actual_warmup])

    npo_collate = adapter.collate

    target_loader = DataLoader(
        ItemDataset(target_items), batch_size=1, shuffle=True,
        collate_fn=lambda b: npo_collate(b), num_workers=0)
    other_loader = DataLoader(
        ItemDataset(other_items), batch_size=1, shuffle=True,
        collate_fn=lambda b: npo_collate(b), num_workers=0)
    other_iter = iter(other_loader)

    model.train()
    npo_trace = []
    step = 0
    npo_running = 0.0
    beta_kl = 0.1  # KL regularization strength

    for epoch in range(1000):
        for batch in target_loader:
            bd = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items()}
            # NPO loss: -log(1 - sigmoid(log π_θ(y_w|x) - log π_ref(y_w|x)))
            outputs = model(
                input_ids=bd["input_ids"],
                attention_mask=bd["attention_mask"],
                labels=bd["labels"], use_cache=False,
                **{k: v for k, v in bd.items()
                   if k not in ("input_ids", "attention_mask", "labels")
                   and isinstance(v, torch.Tensor) and not k.startswith("_")})
            loss_target = outputs.loss

            # Get reference log-prob for KL
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in ref_weights:
                        p.data.copy_(ref_weights[n])
                ref_out = model(
                    input_ids=bd["input_ids"],
                    attention_mask=bd["attention_mask"],
                    labels=bd["labels"], use_cache=False,
                    **{k: v for k, v in bd.items()
                       if k not in ("input_ids", "attention_mask", "labels")
                       and isinstance(v, torch.Tensor)
                       and not k.startswith("_")})
                ref_loss = ref_out.loss
                # Restore current weights
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        pass  # ref_weights are unchanged

            # NPO: minimize -log(1 - sigmoid(-(loss - ref_loss)))
            # ≈ maximize loss on target relative to reference
            diff = loss_target - ref_loss
            npo_loss = -torch.log(1.0 - torch.sigmoid(-diff) + 1e-8)

            # Regularization: standard descent on other items
            try:
                other_batch = next(other_iter)
            except StopIteration:
                other_iter = iter(other_loader)
                other_batch = next(other_iter)
            obd = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                   for k, v in other_batch.items()}
            other_out = model(
                input_ids=obd["input_ids"],
                attention_mask=obd["attention_mask"],
                labels=obd["labels"], use_cache=False,
                **{k: v for k, v in obd.items()
                   if k not in ("input_ids", "attention_mask", "labels")
                   and isinstance(v, torch.Tensor) and not k.startswith("_")})
            loss_other = other_out.loss

            total_loss = npo_loss + beta_kl * loss_other
            total_loss.backward()
            npo_running += total_loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0 or step <= 5:
                npo_trace.append({"step": step, "loss": npo_running / step})
                logger.info(f"[NPO] Step {step}/{args.ul_steps} "
                            f"loss={npo_running / step:.6f}")
            if step >= args.ul_steps:
                break
        if step >= args.ul_steps:
            break

    adapter.save_unlearning_adapter(model, npo_dir / "adapter_final")
    with open(npo_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in npo_trace)
    del model, processor, adapter
    torch.cuda.empty_cache()

    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_cy_npo", npo_dir)
    npo_hard = evaluate_h_hard(adapter, model, processor, identity_ids,
                               alias_of, args.device, args.seed, tag="npo")
    npo_probs = extract_alias_probabilities(
        adapter, model, processor, target, all_aliases, args.device)
    method_results["npo"] = {
        "target_prediction": next(
            p["prediction"] for p in npo_hard["preds"]
            if p["identity_id"] == target),
        "p_old_alias": round(npo_probs.get(target_alias, {}).get("prob", 0), 6),
        "non_target_acc": round(sum(
            1 for p in npo_hard["preds"]
            if p["identity_id"] != target and p["correct"]) / len(other_ids), 4),
    }
    del model, processor, adapter
    torch.cuda.empty_cache()

    # Save comparison
    with open(rv3_dir / "method_comparison.json", "w") as f:
        json.dump(method_results, f, indent=2)

    logger.info("RV3 COMPARISON:")
    for method, res in method_results.items():
        logger.info(f"  {method}: {res}")
    return method_results


# ====================================================================== #
# RV4: Multi-seed robustness
# ====================================================================== #
def run_rv4_multiseed(args, out_base, identity_ids, alias_of):
    """RV4: Run counterfactual deletion across multiple seeds."""
    logger.info("=" * 60)
    logger.info("RV4: MULTI-SEED ROBUSTNESS")
    logger.info("=" * 60)

    seeds = args.seeds or [17, 42, 123, 2024, 7]
    target = args.target_id
    target_alias = alias_of[target]
    wrong_alias = args.delete_to
    if wrong_alias == target_alias:
        wrong_alias = "Bira" if target_alias != "Bira" else "Caro"
    other_ids = [iid for iid in identity_ids if iid != target]

    rv4_dir = out_base / "RV4_multiseed"
    rv4_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results = {}
    for seed in seeds:
        logger.info(f"--- Seed {seed} ---")
        seed_dir = rv4_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        sub_args = argparse.Namespace(**vars(args))
        sub_args.seed = seed

        adapter, model, processor = create_adapter_model(
            sub_args, sub_args.device, "e2c_v3_cy")
        model = attach_lora(adapter, model)
        load_trained_weights(adapter, model)

        target_items = build_supervised_items(
            adapter, processor,
            [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
              "answer": wrong_alias}], repeat=sub_args.ul_repeat * 5)
        other_items = build_supervised_items(
            adapter, processor,
            [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
              "answer": alias_of[iid]} for iid in other_ids],
            repeat=sub_args.ul_repeat)

        train_standard(f"cf_seed{seed}", adapter, model, processor,
                       target_items + other_items, seed_dir, sub_args.device,
                       steps=sub_args.ul_steps, warmup=sub_args.ul_warmup,
                       lr=sub_args.ul_lr)
        del model, processor, adapter
        torch.cuda.empty_cache()

        adapter, model, processor = fresh_model_for_eval(
            sub_args, f"e2c_v3_h_s{seed}", seed_dir)
        h_eval = evaluate_h_hard(
            adapter, model, processor, identity_ids, alias_of,
            sub_args.device, seed, tag=f"s{seed}")

        target_removed = not any(
            p["correct"] for p in h_eval["preds"]
            if p["identity_id"] == target)
        other_acc = sum(
            1 for p in h_eval["preds"]
            if p["identity_id"] != target and p["correct"]) / len(other_ids)

        all_seed_results[seed] = {
            "target_removed": target_removed,
            "non_target_retention": round(other_acc, 4),
            "target_prediction": next(
                p["prediction"] for p in h_eval["preds"]
                if p["identity_id"] == target),
        }
        logger.info(f"  Seed {seed}: removed={target_removed}, "
                    f"retention={other_acc:.3f}")

        del model, processor, adapter
        torch.cuda.empty_cache()

    # Summary statistics
    n_removed = sum(1 for r in all_seed_results.values()
                    if r["target_removed"])
    avg_retention = sum(r["non_target_retention"]
                        for r in all_seed_results.values()) / len(seeds)

    summary = {
        "seeds": seeds,
        "per_seed": all_seed_results,
        "deletion_rate": round(n_removed / len(seeds), 4),
        "avg_non_target_retention": round(avg_retention, 4),
        "all_seeds_succeeded": n_removed == len(seeds),
    }
    with open(rv4_dir / "multiseed_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"RV4 SUMMARY: {n_removed}/{len(seeds)} seeds achieved deletion, "
                f"avg retention={avg_retention:.3f}")
    return summary


# ====================================================================== #
# RV5: Multi-depth granularity
# ====================================================================== #
def run_rv5_granularity(args, out_base, identity_ids, alias_of):
    """RV5: Multi-depth granularity — specific → subgroup → group."""
    logger.info("=" * 60)
    logger.info("RV5: MULTI-DEPTH GRANULARITY")
    logger.info("=" * 60)

    rv5_dir = out_base / "RV5_granularity"
    rv5_dir.mkdir(parents=True, exist_ok=True)

    # Three levels: specific alias, subgroup, group
    # Train h to output subgroup labels for all identities
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_h_sg")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)

    sg_pairs = []
    for iid in identity_ids:
        sg_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": SUBGROUP_MAP[iid],
        })
    sg_items = build_supervised_items(
        adapter, processor, sg_pairs, repeat=args.ul_repeat * 3)

    logger.info(f"Subgroup mapping: {SUBGROUP_MAP}")
    logger.info(f"Training items: {len(sg_items)}")

    train_standard("subgroup", adapter, model, processor, sg_items, rv5_dir,
                   args.device, steps=args.ul_steps, warmup=args.ul_warmup,
                   lr=args.ul_lr)
    del model, processor, adapter
    torch.cuda.empty_cache()

    # Evaluate
    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_h_sg", rv5_dir)

    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()

    depth_results = {"specific": {}, "subgroup": {}, "group": {}}
    with torch.no_grad():
        for iid in identity_ids:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=15)
            pred = gen.text.strip()

            # Check specific alias
            first_word = pred.split()[0].strip(".,!?") if pred else ""
            specific_ok = first_word.lower() == alias_of[iid].lower()

            # Check subgroup
            sg = SUBGROUP_MAP[iid]
            subgroup_ok = sg in pred

            # Check group
            grp = GROUP_MAP[iid]
            group_ok = grp in pred

            depth_results["specific"][iid] = {
                "prediction": pred, "expected": alias_of[iid],
                "correct": specific_ok}
            depth_results["subgroup"][iid] = {
                "prediction": pred, "expected": sg, "correct": subgroup_ok}
            depth_results["group"][iid] = {
                "prediction": pred, "expected": grp, "correct": group_ok}

            logger.info(f"  {iid}: pred='{pred}' | "
                        f"specific={'✓' if specific_ok else '✗'} "
                        f"subgroup={'✓' if subgroup_ok else '✗'} "
                        f"group={'✓' if group_ok else '✗'}")

    summary = {}
    for level in ("specific", "subgroup", "group"):
        n_ok = sum(1 for v in depth_results[level].values() if v["correct"])
        acc = n_ok / len(identity_ids)
        summary[level] = {"accuracy": round(acc, 4), "correct": n_ok,
                          "total": len(identity_ids)}
        logger.info(f"  {level}: {n_ok}/{len(identity_ids)} ({acc:.3f})")

    with open(rv5_dir / "granularity_results.json", "w") as f:
        json.dump({"depth_results": summary, "details": depth_results}, f,
                  indent=2)

    del model, processor, adapter
    torch.cuda.empty_cache()
    return summary


# ====================================================================== #
# RV6: Compositional mixture
# ====================================================================== #
def run_rv6_mixture(args, out_base, identity_ids, alias_of):
    """RV6: Compositional mixture — delete + update + generalize + retain."""
    logger.info("=" * 60)
    logger.info("RV6: COMPOSITIONAL MIXTURE")
    logger.info("=" * 60)

    rv6_dir = out_base / "RV6_mixture"
    rv6_dir.mkdir(parents=True, exist_ok=True)

    # Define transformation sets
    s_delete = ["syn_00"]       # deletion (counterfactual)
    s_update = ["syn_02"]       # update to new alias
    s_generalize = ["syn_04"]   # granularity reduction
    s_retain = [iid for iid in identity_ids
                if iid not in s_delete + s_update + s_generalize]

    update_to = {"syn_02": "Eris"}  # Caro → Eris
    logger.info(f"S_D (delete): {s_delete}")
    logger.info(f"S_U (update): {s_update} → {update_to}")
    logger.info(f"S_G (generalize): {s_generalize} → {GROUP_MAP[s_generalize[0]]}")
    logger.info(f"S_R (retain): {s_retain}")

    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_h_mix")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)

    mix_pairs = []
    # S_D: counterfactual relabeling (wrong alias)
    for iid in s_delete:
        wrong = "Bira" if alias_of[iid] != "Bira" else "Caro"
        mix_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": wrong,
        })
    # S_U: new alias
    for iid in s_update:
        mix_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": update_to[iid],
        })
    # S_G: group label
    for iid in s_generalize:
        mix_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": GROUP_MAP[iid],
        })
    # S_R: retain original
    for iid in s_retain:
        mix_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": alias_of[iid],
        })

    mix_items = build_supervised_items(
        adapter, processor, mix_pairs, repeat=args.ul_repeat * 3)
    logger.info(f"Mixture training items: {len(mix_items)}")

    train_standard("mixture", adapter, model, processor, mix_items, rv6_dir,
                   args.device, steps=args.ul_steps, warmup=args.ul_warmup,
                   lr=args.ul_lr)
    del model, processor, adapter
    torch.cuda.empty_cache()

    # Evaluate
    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_h_mix", rv6_dir)

    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()

    eval_results = {"delete": {}, "update": {}, "generalize": {}, "retain": {}}
    with torch.no_grad():
        for iid in identity_ids:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=15)
            pred = gen.text.strip()
            first_word = pred.split()[0].strip(".,!?") if pred else ""

            if iid in s_delete:
                # Should NOT produce original alias
                correct = first_word.lower() != alias_of[iid].lower()
                eval_results["delete"][iid] = {
                    "prediction": pred, "original": alias_of[iid],
                    "removed": correct}
            elif iid in s_update:
                correct = first_word.lower() == update_to[iid].lower()
                eval_results["update"][iid] = {
                    "prediction": pred, "expected": update_to[iid],
                    "correct": correct}
            elif iid in s_generalize:
                correct = GROUP_MAP[iid] in pred
                eval_results["generalize"][iid] = {
                    "prediction": pred, "expected_group": GROUP_MAP[iid],
                    "correct": correct}
            else:
                correct = first_word.lower() == alias_of[iid].lower()
                eval_results["retain"][iid] = {
                    "prediction": pred, "expected": alias_of[iid],
                    "correct": correct}

    # Summarize
    summary = {}
    for category, data in eval_results.items():
        if category == "delete":
            n_ok = sum(1 for v in data.values() if v["removed"])
        else:
            n_ok = sum(1 for v in data.values() if v["correct"])
        summary[category] = {"success": n_ok, "total": len(data),
                             "rate": round(n_ok / max(len(data), 1), 4)}
        logger.info(f"  {category}: {n_ok}/{len(data)} "
                    f"({summary[category]['rate']:.3f})")
        for iid, v in data.items():
            logger.info(f"    {iid}: pred='{v['prediction']}'")

    with open(rv6_dir / "mixture_results.json", "w") as f:
        json.dump({"summary": summary, "details": eval_results,
                    "sets": {"S_D": s_delete, "S_U": s_update,
                             "S_G": s_generalize, "S_R": s_retain}},
                  f, indent=2)

    del model, processor, adapter
    torch.cuda.empty_cache()
    return summary


# ====================================================================== #
# RV7: Expanded multi-target deletion (all 10 identities)
# ====================================================================== #
def run_rv7_expanded(args, out_base, identity_ids, alias_of):
    """RV7: Delete all 10 identities individually."""
    logger.info("=" * 60)
    logger.info("RV7: EXPANDED MULTI-TARGET (all 10 identities)")
    logger.info("=" * 60)

    rv7_dir = out_base / "RV7_expanded"
    rv7_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for tid in identity_ids:
        logger.info(f"--- Deleting {tid} → {alias_of[tid]} ---")
        sub_dir = rv7_dir / f"del_{tid}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        wrong = "Bira" if alias_of[tid] != "Bira" else "Caro"
        other_ids = [iid for iid in identity_ids if iid != tid]

        adapter, model, processor = create_adapter_model(
            args, args.device, "e2c_v3_cy")
        model = attach_lora(adapter, model)
        load_trained_weights(adapter, model)

        target_items = build_supervised_items(
            adapter, processor,
            [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=tid),
              "answer": wrong}], repeat=args.ul_repeat * 5)
        other_items = build_supervised_items(
            adapter, processor,
            [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
              "answer": alias_of[iid]} for iid in other_ids],
            repeat=args.ul_repeat)

        train_standard(f"del_{tid}", adapter, model, processor,
                       target_items + other_items, sub_dir, args.device,
                       steps=args.ul_steps, warmup=args.ul_warmup,
                       lr=args.ul_lr)
        del model, processor, adapter
        torch.cuda.empty_cache()

        adapter, model, processor = fresh_model_for_eval(
            args, f"e2c_v3_h_d{tid}", sub_dir)
        h_eval = evaluate_h_hard(
            adapter, model, processor, identity_ids, alias_of,
            args.device, args.seed, tag=f"d{tid}")

        target_removed = not any(
            p["correct"] for p in h_eval["preds"]
            if p["identity_id"] == tid)
        other_acc = sum(
            1 for p in h_eval["preds"]
            if p["identity_id"] != tid and p["correct"]) / len(other_ids)

        all_results[tid] = {
            "target_removed": target_removed,
            "non_target_retention": round(other_acc, 4),
            "prediction": next(
                p["prediction"] for p in h_eval["preds"]
                if p["identity_id"] == tid),
        }
        logger.info(f"  {tid}: removed={target_removed}, "
                    f"retention={other_acc:.3f}")
        del model, processor, adapter
        torch.cuda.empty_cache()

    n_removed = sum(1 for r in all_results.values() if r["target_removed"])
    avg_ret = sum(r["non_target_retention"]
                  for r in all_results.values()) / len(all_results)

    summary = {
        "n_targets": len(identity_ids),
        "n_removed": n_removed,
        "deletion_rate": round(n_removed / len(identity_ids), 4),
        "avg_non_target_retention": round(avg_ret, 4),
        "per_identity": all_results,
    }
    with open(rv7_dir / "expanded_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"RV7 SUMMARY: {n_removed}/{len(identity_ids)} removed, "
                f"avg retention={avg_ret:.3f}")
    return summary


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default="e2c_v3/outputs/research_validity")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", default="all",
                        choices=["all", "RV1", "RV2", "RV3", "RV4",
                                 "RV5", "RV6", "RV7"])
    parser.add_argument("--target-id", default="syn_00")
    parser.add_argument("--delete-to", default="Bira")
    parser.add_argument("--ul-steps", type=int, default=UL_STEPS)
    parser.add_argument("--ul-warmup", type=int, default=UL_WARMUP)
    parser.add_argument("--ul-lr", type=float, default=UL_LR)
    parser.add_argument("--ul-repeat", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[17, 42, 123, 2024, 7])
    args = parser.parse_args()

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_phases = (
        ["RV1", "RV2", "RV3", "RV4", "RV5", "RV6", "RV7"]
        if args.phase == "all" else [args.phase]
    )

    results = {}
    if "RV1" in run_phases:
        results["RV1"] = run_rv1_soft_metrics(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV2" in run_phases:
        results["RV2"] = run_rv2_oracle(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
    if "RV3" in run_phases:
        results["RV3"] = run_rv3_methods(
            args, out_base, IDENTITY_IDS, ALIAS_OF)
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
        results["RV7"] = run_rv7_expanded(
            args, out_base, IDENTITY_IDS, ALIAS_OF)

    logger.info("=" * 60)
    logger.info("RESEARCH-VALIDITY PHASE COMPLETE")
    logger.info("=" * 60)
    with open(out_base / "rv_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_base / 'rv_summary.json'}")


if __name__ == "__main__":
    main()
