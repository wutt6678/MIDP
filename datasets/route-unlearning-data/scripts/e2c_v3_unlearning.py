#!/usr/bin/env python3
"""E2C-v3 Controlled Unlearning (U1–U8).

Key insight: gradient ascent FAILS because the trained C→Y adapter has
loss ≈ 9e-6 (near-zero).  Instead we use COUNTERFACTUAL RELABELING:
train the target pair with a WRONG answer, which gives a large,
informative gradient that overwrites the association.

U0: Frozen route-establishment benchmark (already complete).
U1: Association deletion — C_p → Y_p becomes C_p → Y_wrong.
U2: Measurement — target removal + preservation + non-target retention.
U3: Association update — C_p → Y_old becomes C_p → Y_new.
U4: Granularity reduction — C_p → A_specific becomes C_p → A_general.
U5: Method comparison.
U6: Route-localization / collateral-effect analysis.
U7: Multi-identity extension.
U8: Hierarchical setting.

Architecture constraint (freeze):
  g (X→C) is NEVER modified.
  Only h (C→Y) is modified during unlearning.
"""
import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image
from typing import Any

import torch
import torch.nn as tnn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_unlearning")

SCOPE_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"

# Training defaults
UL_STEPS = 500
UL_WARMUP = 50
UL_LR = 2e-5
COARSEN_STEPS = 2000
COARSEN_WARMUP = 200
COARSEN_LR = 2e-5

# Default target
DEFAULT_TARGET = "syn_00"
DEFAULT_UPDATE_TO = "Bira"
# Wrong alias for deletion (different from target's alias)
DEFAULT_DELETE_TO = "Bira"  # syn_00 is Aven, we relabel to Bira
# Granularity groups
GROUP_A_LETTERS = set("ABCDE")
GROUP_B_LETTERS = set("FGHIJ")

# All 10 identities and aliases
IDENTITY_IDS = [f"syn_{i:02d}" for i in range(10)]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def create_adapter_model(args, device, adapter_name):
    from route_data.models.trainable.qwen35 import Qwen35Adapter
    from route_data.models.trainable.base import ModelFamilyProfile
    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name=adapter_name,
        trust_remote_code=True, dtype="bfloat16",
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
    """Load trained C→Y weights into fresh LoRA."""
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


def train_standard(condition, adapter, model, processor,
                   train_items, output_dir, device,
                   steps=UL_STEPS, warmup=UL_WARMUP, lr=UL_LR):
    """Standard supervised training (gradient descent only)."""
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    logger.info(f"[{condition}] trainable params: {n_params:,}")
    logger.info(f"[{condition}] training items: {len(train_items)}, "
                f"steps: {steps}, lr: {lr}")

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
        CosineAnnealingLR, LinearLR, SequentialLR,
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
        for e in trace:
            f.write(json.dumps(e) + "\n")
    final_loss = running_loss / max(global_step, 1)
    logger.info(f"[{condition}] complete. Final avg loss={final_loss:.6f}")
    return trace


def evaluate_h_adapter(adapter, model, processor, identity_ids, alias_of,
                       device, seed, tag="eval"):
    """Evaluate h adapter: code → alias accuracy for all 10 identities."""
    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()

    results = {}
    preds = []
    with torch.no_grad():
        for iid in identity_ids:
            expected = alias_of[iid]
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=5)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")
            ok = pred.lower() == expected.lower()
            results[iid] = {
                "expected": expected, "prediction": pred, "correct": ok,
            }
            preds.append({
                "identity_id": iid, "code": iid,
                "expected": expected, "prediction": pred, "correct": ok,
            })
            logger.info(f"  [{tag}] {iid} → '{pred}' "
                        f"(expected '{expected}') {'✓' if ok else '✗'}")

    n_correct = sum(1 for p in preds if p["correct"])
    acc = n_correct / len(preds)
    logger.info(f"  [{tag}] accuracy: {n_correct}/{len(preds)} ({acc:.3f})")
    return {"accuracy": acc, "correct": n_correct, "total": len(preds),
            "preds": preds, "per_identity": results}


def evaluate_x_to_c_preservation(args, eval_items, image_base,
                                  identity_ids, seed):
    """Evaluate g adapter (X→C) — should be unchanged since g is frozen."""
    from route_data.config import ModelConfig
    xc_dir = Path("e2c_v3/outputs/phaseC/X_to_C")

    xc_adapter, xc_model, xc_processor = create_adapter_model(
        args, args.device, "e2c_v3_xc")
    xc_model = xc_adapter.load_unlearning_adapter(
        xc_model, xc_dir / "adapter_final",
        adapter_name="e2c_v3_xc")

    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    backend = xc_adapter.to_eval_backend(
        model=xc_model, processor=xc_processor, model_config=config)
    xc_model.eval()

    by_split = defaultdict(lambda: {"correct": 0, "total": 0})
    with torch.no_grad():
        for item in eval_items:
            image = load_image(str(image_base / item["image_path"]))
            gen = backend.generate(image, IMG_TO_CODE_PROMPT, max_new_tokens=5)
            pred = gen.text.strip()
            expected = item["code_id"]
            correct = expected in pred
            by_split[item["split"]]["correct"] += int(correct)
            by_split[item["split"]]["total"] += 1

    results = {}
    for split in ("train", "validation", "test"):
        d = by_split[split]
        acc = d["correct"] / max(d["total"], 1)
        results[f"{split}_acc"] = acc
        results[f"{split}_correct"] = d["correct"]
        results[f"{split}_total"] = d["total"]
        logger.info(f"  X→C preservation ({split}): "
                    f"{d['correct']}/{d['total']} ({acc:.3f})")

    del xc_model, xc_processor, xc_adapter
    torch.cuda.empty_cache()
    return results


def run_intervention(adapter, model, processor, identity_ids, alias_of,
                     device, seed, target_id, tag="intervention"):
    """Run causal intervention on unlearned h adapter."""
    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()

    records = []
    with torch.no_grad():
        for jid in identity_ids:
            target_j = alias_of[jid]
            prompt = CODE_TO_ALIAS_PROMPT.format(code=jid)
            gen = backend.generate(None, prompt, max_new_tokens=5)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")
            follows = pred.lower() == target_j.lower()
            records.append({
                "presented_code": jid, "code_target": target_j,
                "prediction": pred, "follows_target": follows,
                "is_target_identity": (jid == target_id),
            })

    n_follow = sum(1 for r in records if r["follows_target"])
    agreement = n_follow / len(records)
    target_rec = [r for r in records if r["is_target_identity"]]
    target_follows = target_rec[0]["follows_target"] if target_rec else None
    non_target = [r for r in records if not r["is_target_identity"]]
    nt_follow = sum(1 for r in non_target if r["follows_target"])
    nt_agreement = nt_follow / max(len(non_target), 1)

    logger.info(f"  [{tag}] Agreement: {n_follow}/{len(records)} "
                f"({agreement:.3f}), target_follows={target_follows}, "
                f"non_target_agreement={nt_agreement:.3f}")

    return {
        "agreement": agreement, "follow": n_follow,
        "total": len(records), "target_follows": target_follows,
        "non_target_agreement": nt_agreement, "records": records,
    }


def fresh_model_for_eval(args, adapter_name, adapter_dir):
    """Load model + adapter for evaluation."""
    adapter, model, processor = create_adapter_model(
        args, args.device, adapter_name)
    model = adapter.load_unlearning_adapter(
        model, adapter_dir / "adapter_final",
        adapter_name=adapter_name)
    return adapter, model, processor


# ====================================================================== #
# U1+U2: Deletion via counterfactual relabeling
# ====================================================================== #
def run_u1_u2(args, out_base, identity_ids, alias_of, eval_items, image_base):
    """U1: Delete C_p → Y_p via relabeling to wrong answer.
    U2: Measure deletion effects."""
    logger.info("=" * 60)
    logger.info("U1+U2: ASSOCIATION DELETION (counterfactual relabeling)")
    logger.info("=" * 60)

    target = args.target_id
    target_alias = alias_of[target]
    wrong_alias = args.delete_to
    # Ensure wrong alias is different from target alias
    if wrong_alias == target_alias:
        # Pick first alias that's different
        for iid in identity_ids:
            if alias_of[iid] != target_alias:
                wrong_alias = alias_of[iid]
                break
    other_ids = [iid for iid in identity_ids if iid != target]

    ul_dir = out_base / "U1_deletion"
    ul_dir.mkdir(parents=True, exist_ok=True)

    # Load trained h adapter
    logger.info("Loading trained h adapter (C→Y)...")
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)
    logger.info("Loaded trained h weights")

    # Build training items:
    # Target pair with WRONG answer (counterfactual relabeling)
    target_pairs = [{
        "prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
        "answer": wrong_alias,
    }]
    target_items = build_supervised_items(
        adapter, processor, target_pairs, repeat=args.ul_repeat * 5)

    # Preserve other identities
    other_pairs = []
    for iid in other_ids:
        other_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": alias_of[iid],
        })
    other_items = build_supervised_items(
        adapter, processor, other_pairs, repeat=args.ul_repeat)

    all_items = target_items + other_items
    logger.info(f"Target: {target} → {target_alias} RELABELED to {wrong_alias}")
    logger.info(f"Target items: {len(target_items)}, "
                f"Other items: {len(other_items)}, "
                f"Total: {len(all_items)}")

    # Train
    train_standard(
        "deletion", adapter, model, processor,
        all_items, ul_dir, args.device,
        steps=args.ul_steps, warmup=args.ul_warmup, lr=args.ul_lr,
    )

    del model, processor, adapter
    torch.cuda.empty_cache()

    # ================================================================== #
    # U2: Measurement
    # ================================================================== #
    logger.info("=" * 60)
    logger.info("U2: MEASURING DELETION EFFECTS")
    logger.info("=" * 60)

    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_h_deleted", ul_dir)

    # 1. Target removal
    logger.info("--- Target removal ---")
    h_eval = evaluate_h_adapter(
        adapter, model, processor, identity_ids, alias_of,
        args.device, args.seed, tag="deleted")

    target_result = h_eval["per_identity"][target]
    target_removed = not target_result["correct"]

    # 2. Non-target retention
    other_correct = sum(
        1 for iid in other_ids
        if h_eval["per_identity"][iid]["correct"]
    )
    other_acc = other_correct / len(other_ids)

    # 3. X→C preservation
    logger.info("--- X→C preservation (g frozen) ---")
    xc_pres = evaluate_x_to_c_preservation(
        args, eval_items, image_base, identity_ids, args.seed)

    # 4. Causal intervention
    logger.info("--- Causal intervention on unlearned h ---")
    int_results = run_intervention(
        adapter, model, processor, identity_ids, alias_of,
        args.device, args.seed, target, tag="deleted-intervention")

    results = {
        "method": "deletion_counterfactual",
        "target_id": target,
        "target_alias": target_alias,
        "wrong_alias": wrong_alias,
        "training_steps": args.ul_steps,
        "training_lr": args.ul_lr,
        "target_removed": target_removed,
        "target_prediction": target_result["prediction"],
        "target_expected": target_result["expected"],
        "non_target_retention": other_acc,
        "non_target_correct": other_correct,
        "non_target_total": len(other_ids),
        "overall_accuracy": h_eval["accuracy"],
        "xc_preservation": xc_pres,
        "intervention": {
            "agreement": int_results["agreement"],
            "target_follows": int_results["target_follows"],
            "non_target_agreement": int_results["non_target_agreement"],
        },
        "h_eval_details": h_eval["preds"],
        "gate_target_removed": target_removed,
        "gate_non_target_retained": other_acc >= 0.90,
        "gate_xc_preserved": xc_pres.get("test_acc", 0) >= 0.90,
    }

    with open(ul_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"U1+U2 SUMMARY: target_removed={target_removed}, "
                f"non_target_retention={other_acc:.3f}, "
                f"xc_preserved={xc_pres.get('test_acc', 0):.3f}")

    del model, processor, adapter
    torch.cuda.empty_cache()
    return results


# ====================================================================== #
# U3: Update
# ====================================================================== #
def run_u3(args, out_base, identity_ids, alias_of, eval_items, image_base):
    """U3: Update C_p → Y_old to C_p → Y_new."""
    logger.info("=" * 60)
    logger.info("U3: ASSOCIATION UPDATE")
    logger.info("=" * 60)

    target = args.target_id
    old_alias = alias_of[target]
    new_alias = args.update_to
    other_ids = [iid for iid in identity_ids if iid != target]

    ul_dir = out_base / "U3_update"
    ul_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target: {target} → {old_alias} becomes {target} → {new_alias}")

    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)

    # Target pair with NEW alias
    target_pairs = [{
        "prompt": CODE_TO_ALIAS_PROMPT.format(code=target),
        "answer": new_alias,
    }]
    target_items = build_supervised_items(
        adapter, processor, target_pairs, repeat=args.ul_repeat * 5)

    # Preserve other identities
    other_pairs = []
    for iid in other_ids:
        other_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": alias_of[iid],
        })
    other_items = build_supervised_items(
        adapter, processor, other_pairs, repeat=args.ul_repeat)

    all_items = target_items + other_items
    logger.info(f"Target items: {len(target_items)}, "
                f"Other items: {len(other_items)}, "
                f"Total: {len(all_items)}")

    train_standard(
        "update", adapter, model, processor,
        all_items, ul_dir, args.device,
        steps=args.ul_steps, warmup=args.ul_warmup, lr=args.ul_lr,
    )

    del model, processor, adapter
    torch.cuda.empty_cache()

    # Evaluate
    logger.info("--- Evaluating updated h ---")
    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_h_updated", ul_dir)

    h_eval = evaluate_h_adapter(
        adapter, model, processor, identity_ids, alias_of,
        args.device, args.seed, tag="updated")

    target_pred = h_eval["per_identity"][target]["prediction"]
    follows_new = target_pred.lower() == new_alias.lower()
    follows_old = target_pred.lower() == old_alias.lower()

    other_correct = sum(
        1 for iid in other_ids
        if h_eval["per_identity"][iid]["correct"]
    )
    other_acc = other_correct / len(other_ids)

    xc_pres = evaluate_x_to_c_preservation(
        args, eval_items, image_base, identity_ids, args.seed)
    int_results = run_intervention(
        adapter, model, processor, identity_ids, alias_of,
        args.device, args.seed, target, tag="updated-intervention")

    results = {
        "method": "update",
        "target_id": target,
        "old_alias": old_alias, "new_alias": new_alias,
        "training_steps": args.ul_steps,
        "target_follows_new": follows_new,
        "target_follows_old": follows_old,
        "target_prediction": target_pred,
        "non_target_retention": other_acc,
        "non_target_correct": other_correct,
        "non_target_total": len(other_ids),
        "xc_preservation": xc_pres,
        "intervention": {
            "agreement": int_results["agreement"],
            "target_follows": int_results["target_follows"],
            "non_target_agreement": int_results["non_target_agreement"],
        },
        "h_eval_details": h_eval["preds"],
        "gate_updated_to_new": follows_new,
        "gate_not_following_old": not follows_old,
        "gate_non_target_retained": other_acc >= 0.90,
    }

    with open(ul_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"U3 SUMMARY: follows_new={follows_new}, "
                f"follows_old={follows_old}, "
                f"non_target={other_acc:.3f}")

    del model, processor, adapter
    torch.cuda.empty_cache()
    return results


# ====================================================================== #
# U4: Granularity Reduction
# ====================================================================== #
def run_u4(args, out_base, identity_ids, alias_of, eval_items, image_base):
    """U4: Granularity reduction — C_p → A_specific becomes C_p → A_general."""
    logger.info("=" * 60)
    logger.info("U4: GRANULARITY REDUCTION")
    logger.info("=" * 60)

    ul_dir = out_base / "U4_granularity"
    ul_dir.mkdir(parents=True, exist_ok=True)

    group_map = {}
    for iid in identity_ids:
        alias = alias_of[iid]
        first_letter = alias[0].upper()
        if first_letter in GROUP_A_LETTERS:
            group_map[iid] = "GROUP_A"
        else:
            group_map[iid] = "GROUP_B"

    logger.info(f"Group mapping: {group_map}")

    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_h_coarse")
    model = attach_lora(adapter, model)

    coarse_pairs = []
    for iid in identity_ids:
        coarse_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": group_map[iid],
        })

    sup_items = build_supervised_items(
        adapter, processor, coarse_pairs, repeat=50)

    logger.info(f"Coarsened training items: {len(sup_items)}")

    train_standard(
        "coarse", adapter, model, processor,
        sup_items, ul_dir, args.device,
        steps=COARSEN_STEPS, warmup=COARSEN_WARMUP, lr=COARSEN_LR,
    )

    del model, processor, adapter
    torch.cuda.empty_cache()

    # Evaluate
    logger.info("--- Evaluating coarsened h ---")
    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_h_coarse", ul_dir)

    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()

    coarse_preds = []
    with torch.no_grad():
        for iid in identity_ids:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=10)
            pred = gen.text.strip()
            expected_group = group_map[iid]
            group_correct = expected_group in pred
            coarse_preds.append({
                "identity_id": iid,
                "expected_group": expected_group,
                "prediction": pred,
                "group_correct": group_correct,
            })
            logger.info(f"  [coarse] {iid} → '{pred}' "
                        f"(expected '{expected_group}') "
                        f"{'✓' if group_correct else '✗'}")

    n_group_correct = sum(1 for p in coarse_preds if p["group_correct"])
    group_acc = n_group_correct / len(coarse_preds)

    # Check specific alias retention
    specific_alias_correct = 0
    with torch.no_grad():
        for iid in identity_ids:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=5)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")
            orig_alias = alias_of[iid]
            if pred.lower() == orig_alias.lower():
                specific_alias_correct += 1

    specific_acc = specific_alias_correct / len(identity_ids)

    xc_pres = evaluate_x_to_c_preservation(
        args, eval_items, image_base, identity_ids, args.seed)

    results = {
        "method": "granularity_reduction",
        "group_map": group_map,
        "group_accuracy": group_acc,
        "group_correct": n_group_correct,
        "group_total": len(identity_ids),
        "specific_alias_retained": specific_acc,
        "specific_alias_correct": specific_alias_correct,
        "xc_preservation": xc_pres,
        "coarse_preds": coarse_preds,
        "gate_group_accuracy": group_acc >= 0.80,
        "gate_specific_lost": specific_acc <= 0.20,
    }

    with open(ul_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"U4 SUMMARY: group_acc={group_acc:.3f}, "
                f"specific_retained={specific_acc:.3f}")

    del model, processor, adapter
    torch.cuda.empty_cache()
    return results


# ====================================================================== #
# U5+U6: Comparison + Collateral Analysis
# ====================================================================== #
def run_u5_u6(out_base, results_u1, results_u3, results_u4):
    """U5: Compare methods. U6: Collateral-effect analysis."""
    logger.info("=" * 60)
    logger.info("U5+U6: COMPARISON + COLLATERAL ANALYSIS")
    logger.info("=" * 60)

    comparison = {
        "deletion": {
            "target_removed": results_u1.get("target_removed"),
            "target_prediction": results_u1.get("target_prediction", ""),
            "non_target_retention": results_u1.get("non_target_retention", 0),
            "xc_preserved": results_u1.get(
                "xc_preservation", {}).get("test_acc", 0),
            "intervention_target_follows": results_u1.get(
                "intervention", {}).get("target_follows"),
        },
        "update": {
            "follows_new": results_u3.get("target_follows_new"),
            "follows_old": results_u3.get("target_follows_old"),
            "target_prediction": results_u3.get("target_prediction", ""),
            "non_target_retention": results_u3.get("non_target_retention", 0),
            "xc_preserved": results_u3.get(
                "xc_preservation", {}).get("test_acc", 0),
        },
        "granularity_reduction": {
            "group_accuracy": results_u4.get("group_accuracy", 0),
            "specific_alias_retained": results_u4.get(
                "specific_alias_retained", 0),
            "xc_preserved": results_u4.get(
                "xc_preservation", {}).get("test_acc", 0),
        },
    }

    collateral = {
        "deletion_non_target_impact": 1.0 - results_u1.get(
            "non_target_retention", 1.0),
        "update_non_target_impact": 1.0 - results_u3.get(
            "non_target_retention", 1.0),
        "xc_impact_deletion": 1.0 - results_u1.get(
            "xc_preservation", {}).get("test_acc", 1.0),
        "xc_impact_update": 1.0 - results_u3.get(
            "xc_preservation", {}).get("test_acc", 1.0),
        "visual_impact": 0.0,
    }

    report = {
        "comparison": comparison,
        "collateral_analysis": collateral,
        "summary": {
            "deletion_success": (
                results_u1.get("target_removed", False)
                and results_u1.get("non_target_retention", 0) >= 0.90
            ),
            "update_success": (
                results_u3.get("target_follows_new", False)
                and results_u3.get("non_target_retention", 0) >= 0.90
            ),
            "granularity_success": (
                results_u4.get("group_accuracy", 0) >= 0.80
                and results_u4.get("specific_alias_retained", 1.0) <= 0.20
            ),
        },
    }

    report_dir = out_base / "U5_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("U5+U6 Comparison:")
    for method, data in comparison.items():
        logger.info(f"  {method}: {data}")
    logger.info(f"Collateral: {collateral}")

    return report


# ====================================================================== #
# U7: Multi-identity extension
# ====================================================================== #
def run_u7(args, out_base, identity_ids, alias_of, eval_items, image_base):
    """U7: Run deletion on multiple identities to test robustness."""
    logger.info("=" * 60)
    logger.info("U7: MULTI-IDENTITY DELETION")
    logger.info("=" * 60)

    u7_dir = out_base / "U7_multi_identity"
    u7_dir.mkdir(parents=True, exist_ok=True)

    # Test deletion on 3 additional identities
    test_ids = ["syn_01", "syn_05", "syn_09"]
    all_results = {}

    for tid in test_ids:
        logger.info(f"--- U7: Deleting {tid} → {alias_of[tid]} ---")
        # Create a temporary args copy
        sub_args = argparse.Namespace(**vars(args))
        sub_args.target_id = tid
        sub_args.delete_to = alias_of.get("syn_00", "Aven")
        if sub_args.delete_to == alias_of[tid]:
            sub_args.delete_to = alias_of.get("syn_01", "Bira")
            if sub_args.delete_to == alias_of[tid]:
                sub_args.delete_to = alias_of.get("syn_02", "Caro")

        sub_dir = u7_dir / f"deletion_{tid}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Load trained h adapter
        adapter, model, processor = create_adapter_model(
            sub_args, sub_args.device, "e2c_v3_cy")
        model = attach_lora(adapter, model)
        load_trained_weights(adapter, model)

        target_alias = alias_of[tid]
        wrong_alias = sub_args.delete_to
        other_ids = [iid for iid in identity_ids if iid != tid]

        target_pairs = [{
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=tid),
            "answer": wrong_alias,
        }]
        target_items = build_supervised_items(
            adapter, processor, target_pairs, repeat=args.ul_repeat * 5)
        other_pairs = [{"prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
                        "answer": alias_of[iid]} for iid in other_ids]
        other_items = build_supervised_items(
            adapter, processor, other_pairs, repeat=args.ul_repeat)
        all_items = target_items + other_items

        train_standard(
            f"deletion_{tid}", adapter, model, processor,
            all_items, sub_dir, sub_args.device,
            steps=args.ul_steps, warmup=args.ul_warmup, lr=args.ul_lr,
        )

        del model, processor, adapter
        torch.cuda.empty_cache()

        # Evaluate
        adapter, model, processor = fresh_model_for_eval(
            sub_args, f"e2c_v3_h_del_{tid}", sub_dir)

        h_eval = evaluate_h_adapter(
            adapter, model, processor, identity_ids, alias_of,
            sub_args.device, sub_args.seed, tag=f"del_{tid}")

        target_removed = not h_eval["per_identity"][tid]["correct"]
        other_correct = sum(
            1 for iid in other_ids
            if h_eval["per_identity"][iid]["correct"])
        other_acc = other_correct / len(other_ids)

        all_results[tid] = {
            "target_removed": target_removed,
            "target_prediction": h_eval["per_identity"][tid]["prediction"],
            "non_target_retention": other_acc,
            "overall_accuracy": h_eval["accuracy"],
        }

        logger.info(f"U7 {tid}: removed={target_removed}, "
                    f"non_target={other_acc:.3f}")

        del model, processor, adapter
        torch.cuda.empty_cache()

    with open(u7_dir / "multi_identity_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    n_removed = sum(1 for r in all_results.values() if r["target_removed"])
    avg_retention = sum(r["non_target_retention"]
                        for r in all_results.values()) / len(all_results)
    logger.info(f"U7 SUMMARY: {n_removed}/{len(test_ids)} targets removed, "
                f"avg non-target retention={avg_retention:.3f}")

    return all_results


# ====================================================================== #
# U8: Hierarchical setting
# ====================================================================== #
def run_u8(args, out_base, identity_ids, alias_of, eval_items, image_base):
    """U8: Hierarchical — delete group association, test individual impact."""
    logger.info("=" * 60)
    logger.info("U8: HIERARCHICAL UNLEARNING")
    logger.info("=" * 60)

    u8_dir = out_base / "U8_hierarchical"
    u8_dir.mkdir(parents=True, exist_ok=True)

    # Define groups
    group_map = {}
    for iid in identity_ids:
        alias = alias_of[iid]
        first_letter = alias[0].upper()
        group_map[iid] = "GROUP_A" if first_letter in GROUP_A_LETTERS else "GROUP_B"

    # Train h to output group labels for GROUP_A only, keep GROUP_B specific
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_h_hier")
    model = attach_lora(adapter, model)
    load_trained_weights(adapter, model)

    # GROUP_A identities → group label
    group_a_ids = [iid for iid in identity_ids if group_map[iid] == "GROUP_A"]
    group_b_ids = [iid for iid in identity_ids if group_map[iid] == "GROUP_B"]

    hier_pairs = []
    for iid in group_a_ids:
        hier_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": "GROUP_A",
        })
    for iid in group_b_ids:
        hier_pairs.append({
            "prompt": CODE_TO_ALIAS_PROMPT.format(code=iid),
            "answer": alias_of[iid],
        })

    hier_items = build_supervised_items(
        adapter, processor, hier_pairs, repeat=args.ul_repeat * 3)

    logger.info(f"Group A (→GROUP_A): {group_a_ids}")
    logger.info(f"Group B (→specific): {group_b_ids}")
    logger.info(f"Hierarchical training items: {len(hier_items)}")

    train_standard(
        "hierarchical", adapter, model, processor,
        hier_items, u8_dir, args.device,
        steps=args.ul_steps, warmup=args.ul_warmup, lr=args.ul_lr,
    )

    del model, processor, adapter
    torch.cuda.empty_cache()

    # Evaluate
    adapter, model, processor = fresh_model_for_eval(
        args, "e2c_v3_h_hier", u8_dir)

    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()

    hier_results = {"group_a": {}, "group_b": {}}
    with torch.no_grad():
        for iid in identity_ids:
            prompt = CODE_TO_ALIAS_PROMPT.format(code=iid)
            gen = backend.generate(None, prompt, max_new_tokens=10)
            pred = gen.text.strip()
            expected = group_map[iid] if group_map[iid] == "GROUP_A" and iid in group_a_ids else alias_of[iid]
            correct = expected in pred
            group_key = "group_a" if iid in group_a_ids else "group_b"
            hier_results[group_key][iid] = {
                "prediction": pred, "expected": expected, "correct": correct,
            }
            logger.info(f"  [hier] {iid} → '{pred}' "
                        f"(expected '{expected}') {'✓' if correct else '✗'}")

    ga_correct = sum(1 for v in hier_results["group_a"].values() if v["correct"])
    ga_total = len(hier_results["group_a"])
    gb_correct = sum(1 for v in hier_results["group_b"].values() if v["correct"])
    gb_total = len(hier_results["group_b"])

    xc_pres = evaluate_x_to_c_preservation(
        args, eval_items, image_base, identity_ids, args.seed)

    results = {
        "method": "hierarchical",
        "group_a_accuracy": ga_correct / max(ga_total, 1),
        "group_a_correct": ga_correct, "group_a_total": ga_total,
        "group_b_accuracy": gb_correct / max(gb_total, 1),
        "group_b_correct": gb_correct, "group_b_total": gb_total,
        "xc_preservation": xc_pres,
        "details": hier_results,
        "gate_group_a_coarsened": (ga_correct / max(ga_total, 1)) >= 0.80,
        "gate_group_b_preserved": (gb_correct / max(gb_total, 1)) >= 0.80,
    }

    with open(u8_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"U8 SUMMARY: group_a_acc={ga_correct}/{ga_total}, "
                f"group_b_acc={gb_correct}/{gb_total}")

    del model, processor, adapter
    torch.cuda.empty_cache()
    return results


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default="e2c_v3/outputs/unlearning")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", default="all",
                        choices=["all", "U1", "U3", "U4", "U5", "U7", "U8"],
                        help="Run specific phase or all")
    parser.add_argument("--target-id", default=DEFAULT_TARGET)
    parser.add_argument("--update-to", default=DEFAULT_UPDATE_TO)
    parser.add_argument("--delete-to", default=DEFAULT_DELETE_TO)
    parser.add_argument("--ul-steps", type=int, default=UL_STEPS)
    parser.add_argument("--ul-warmup", type=int, default=UL_WARMUP)
    parser.add_argument("--ul-lr", type=float, default=UL_LR)
    parser.add_argument("--ul-repeat", type=int, default=50)
    args = parser.parse_args()

    out_base = Path(args.out_base)
    image_base = Path(args.image_base_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Load identity info
    mapping = json.load(open("e2c_v3/manifests/identity_code_mapping.json"))
    identity_ids = [m["identity_id"] for m in mapping["mappings"]]
    identity_to_alias = mapping["identity_to_alias"]
    alias_of = identity_to_alias

    split_manifest = json.load(open("e2c_v2/manifests/e2c_image_split.json"))
    eval_items = []
    for e in split_manifest:
        if e["identity_id"] in identity_ids:
            eval_items.append({
                "identity_id": e["identity_id"],
                "image_id": e["image_id"],
                "image_path": e["image_path"],
                "split": e["split"],
                "code_id": e["identity_id"],
                "correct_alias": alias_of[e["identity_id"]],
            })

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_phases = (
        ["U1", "U3", "U4", "U5", "U7", "U8"] if args.phase == "all"
        else [args.phase]
    )

    results = {}

    if "U1" in run_phases:
        results["U1_U2"] = run_u1_u2(
            args, out_base, identity_ids, alias_of, eval_items, image_base)

    if "U3" in run_phases:
        results["U3"] = run_u3(
            args, out_base, identity_ids, alias_of, eval_items, image_base)

    if "U4" in run_phases:
        results["U4"] = run_u4(
            args, out_base, identity_ids, alias_of, eval_items, image_base)

    if "U5" in run_phases:
        if all(k in results for k in ["U1_U2", "U3", "U4"]):
            results["U5_U6"] = run_u5_u6(
                out_base, results["U1_U2"], results["U3"], results["U4"])
        else:
            logger.warning("U5 requires U1, U3, U4 results. Skipping.")

    if "U7" in run_phases:
        results["U7"] = run_u7(
            args, out_base, identity_ids, alias_of, eval_items, image_base)

    if "U8" in run_phases:
        results["U8"] = run_u8(
            args, out_base, identity_ids, alias_of, eval_items, image_base)

    # Final summary
    logger.info("=" * 60)
    logger.info("UNLEARNING EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    with open(out_base / "unlearning_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_base / 'unlearning_summary.json'}")


if __name__ == "__main__":
    main()
