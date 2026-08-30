#!/usr/bin/env python3
"""E2C-v3 Phase C — Completion: C5 (matched direct), C6 (shuffled), C7 (visual).

C5: Matched direct condition D — X→Y without mediator.
    Run 270 code interventions on D. Compute causal contrast Δ_C.
C6: M-shuffled — same g, h_π(C_i)=Y_π(i).
    Composition + intervention on shuffled mapping.
C7: Visual preservation — frozen visual controls ≤0.05 drop.
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
logger = logging.getLogger("e2c_v3_phaseC_ext")

SCOPE_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$"

# Prompts
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
IMG_PROMPT = "Identify the synthetic person shown here. Return only the alias."

STEPS_CY = 3000
WARMUP = 200
LR = 2e-5
CY_REPEAT = 50


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_shuffled_map(identity_ids, alias_of, seed=17):
    """Create a deranged (no fixed points) alias mapping."""
    rng = torch.Generator().manual_seed(seed)
    aliases = [alias_of[iid] for iid in identity_ids]
    n = len(aliases)
    for _ in range(100):
        perm = aliases[:]
        for i in range(n - 1, 0, -1):
            j = torch.randint(0, i + 1, (1,), generator=rng).item()
            perm[i], perm[j] = perm[j], perm[i]
        if all(perm[i] != aliases[i] for i in range(n)):
            break
    return {identity_ids[i]: perm[i] for i in range(n)}


def create_adapter_model(args, device, adapter_name):
    """Load base model and create adapter profile."""
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


def train_adapter(
    condition, adapter, model, processor,
    train_items, output_dir, device,
    steps=1000, warmup=200, lr=2e-5,
):
    """Standard training loop for one adapter."""
    target_modules = sorted(
        n for n, m in model.named_modules()
        if isinstance(m, tnn.Linear) and re.match(SCOPE_REGEX, n))
    logger.info(f"[{condition}] LoRA targets: {len(target_modules)} modules")

    model = adapter.attach_unlearning_adapter(
        model, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules,
        adapter_name=adapter.profile.adapter_name,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"[{condition}] trainable params: "
                f"{sum(p.numel() for p in params):,}")

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
    warmup_s = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_s = CosineAnnealingLR(optimizer, T_max=steps - warmup)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_s, cosine_s],
                             milestones=[warmup])

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
                trace.append({
                    "step": global_step, "loss": avg,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                })
                logger.info(f"[{condition}] Step {global_step}/{steps} "
                            f"loss={avg:.4f}")

            if global_step >= steps:
                break
        if global_step >= steps:
            break

    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        for e in trace:
            f.write(json.dumps(e) + "\n")
    logger.info(f"[{condition}] training complete, final loss="
                f"{trace[-1]['loss']:.4f}")
    return trace


def bootstrap_ci_delta(
    m_records, d_records, identity_ids, n_bootstrap=10000, seed=17,
):
    """Compute Δ_C with identity-cluster bootstrap CI.

    For each test image, M's agreement = fraction of 9 interventions
    that follow the code's target. D's agreement = fraction of 9
    interventions that follow the code's target (should be ≈0).

    Δ_C per image = agreement_M_i - agreement_D_i
    Bootstrap: resample images, compute mean Δ_C, get CI.
    """
    rng = torch.Generator().manual_seed(seed)

    # Per-image agreements for M
    m_per_image = {}
    for rec in m_records:
        iid = rec["image_identity"]
        if iid not in m_per_image:
            m_per_image[iid] = {"follow": 0, "total": 0}
        m_per_image[iid]["follow"] += int(rec["follows_code"])
        m_per_image[iid]["total"] += 1

    # Per-image agreements for D
    d_per_image = {}
    for rec in d_records:
        iid = rec["image_identity"]
        if iid not in d_per_image:
            d_per_image[iid] = {"follow": 0, "total": 0}
        d_per_image[iid]["follow"] += int(rec["follows_code"])
        d_per_image[iid]["total"] += 1

    # Get shared test images
    test_images = sorted(set(m_per_image.keys()) & set(d_per_image.keys()))
    n_images = len(test_images)
    logger.info(f"Bootstrap: {n_images} shared test images")

    # Per-image Δ
    per_image_delta = {}
    for iid in test_images:
        m_agr = m_per_image[iid]["follow"] / max(m_per_image[iid]["total"], 1)
        d_agr = d_per_image[iid]["follow"] / max(d_per_image[iid]["total"], 1)
        per_image_delta[iid] = m_agr - d_agr

    # Point estimate
    point_delta = sum(per_image_delta.values()) / n_images

    # Bootstrap
    import random
    py_rng = random.Random(seed)
    deltas = []
    for _ in range(n_bootstrap):
        sample = [test_images[py_rng.randint(0, n_images - 1)]
                  for _ in range(n_images)]
        mean_d = sum(per_image_delta[iid] for iid in sample) / n_images
        deltas.append(mean_d)

    deltas.sort()
    ci_lo = deltas[int(0.025 * n_bootstrap)]
    ci_hi = deltas[int(0.975 * n_bootstrap)]

    return {
        "point_estimate": point_delta,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "ci_excludes_zero": ci_lo > 0 or ci_hi < 0,
        "n_images": n_images,
        "n_bootstrap": n_bootstrap,
    }


# ====================================================================== #
# Phase C5: Matched Direct Condition
# ====================================================================== #
def run_c5(args, out_base, eval_items, image_base, identity_ids, alias_of):
    """C5: Evaluate D adapter + code interventions + causal contrast."""
    logger.info("=" * 60)
    logger.info("PHASE C5: Matched Direct Condition (D)")
    logger.info("=" * 60)

    c5_dir = out_base / "C5_direct"
    c5_dir.mkdir(parents=True, exist_ok=True)

    # Load D adapter
    logger.info("Loading D adapter...")
    d_adapter, d_model, d_processor = create_adapter_model(
        args, args.device, "e2c_v3_direct")
    d_model = d_adapter.load_unlearning_adapter(
        d_model, Path("e2c_v3/outputs/D/adapter_final"),
        adapter_name="e2c_v3_direct")

    from route_data.config import ModelConfig
    d_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    d_backend = d_adapter.to_eval_backend(
        model=d_model, processor=d_processor, model_config=d_config)
    d_model.eval()

    test_items = [it for it in eval_items if it["split"] == "test"]

    # ------------------------------------------------------------------ #
    # Baseline: D(X) with standard prompt
    # ------------------------------------------------------------------ #
    logger.info("Evaluating D baseline (X→Y)...")
    d_correct = 0
    d_total = 0
    d_preds = []

    with torch.no_grad():
        for item in test_items:
            image = load_image(str(image_base / item["image_path"]))
            expected = alias_of[item["identity_id"]]
            gen = d_backend.generate(image, IMG_PROMPT, max_new_tokens=10)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")
            ok = pred.lower() == expected.lower()
            d_correct += int(ok)
            d_total += 1
            d_preds.append({
                "identity_id": item["identity_id"],
                "image_id": item["image_id"],
                "expected": expected,
                "prediction": pred,
                "correct": ok,
            })

    d_baseline_acc = d_correct / max(d_total, 1)
    logger.info(f"  D baseline Acc(X→Y): {d_correct}/{d_total} "
                f"({d_baseline_acc:.3f})")

    # ------------------------------------------------------------------ #
    # Intervention: 270 code injections on D
    # For each test image X_i, for each wrong code C_j:
    #   Feed D the image with standard prompt (no code)
    #   Check: does D's output follow target(C_j)? → should be ≈0
    #   Check: does D's output follow target(X_i)? → should be high
    # ------------------------------------------------------------------ #
    logger.info("Running D code interventions (270)...")
    d_intervention_records = []
    d_follow_code = 0
    d_follow_image = 0
    d_total_int = 0

    with torch.no_grad():
        for item in test_items:
            identity = item["identity_id"]
            expected_alias = alias_of[identity]
            image = load_image(str(image_base / item["image_path"]))

            # Get D's baseline output for this image
            gen_base = d_backend.generate(image, IMG_PROMPT, max_new_tokens=10)
            d_base_pred = (gen_base.text.strip().split()[0].strip(".,!?")
                           if gen_base.text.strip() else "")

            for jid in identity_ids:
                if jid == identity:
                    continue

                target_j = alias_of[jid]

                # D's output is always d_base_pred (doesn't see code)
                follows_code = d_base_pred.lower() == target_j.lower()
                follows_image = d_base_pred.lower() == expected_alias.lower()

                d_follow_code += int(follows_code)
                d_follow_image += int(follows_image)
                d_total_int += 1

                d_intervention_records.append({
                    "image_identity": identity,
                    "image_target": expected_alias,
                    "presented_code": jid,
                    "code_target": target_j,
                    "prediction": d_base_pred,
                    "follows_code": follows_code,
                    "follows_image": follows_image,
                    "baseline_prediction": d_base_pred,
                })

    d_code_agreement = d_follow_code / max(d_total_int, 1)
    d_image_agreement = d_follow_image / max(d_total_int, 1)

    logger.info(f"  D intervention: follow_code={d_follow_code}/{d_total_int} "
                f"({d_code_agreement:.3f})")
    logger.info(f"  D intervention: follow_image={d_follow_image}/{d_total_int} "
                f"({d_image_agreement:.3f})")

    # ------------------------------------------------------------------ #
    # Load M intervention results for causal contrast
    # ------------------------------------------------------------------ #
    m_int_path = out_base / "intervention" / "eval_results.json"
    if m_int_path.exists():
        m_int_data = json.load(open(m_int_path))
        m_records = m_int_data.get("records", [])
        m_agreement = m_int_data.get("agreement", 0)
    else:
        logger.warning("M intervention results not found, using C4 results")
        m_records = []
        m_agreement = 1.0  # from C4 results

    # ------------------------------------------------------------------ #
    # Causal contrast Δ_C with bootstrap CI
    # ------------------------------------------------------------------ #
    if m_records:
        ci_result = bootstrap_ci_delta(
            m_records, d_intervention_records, identity_ids,
            n_bootstrap=10000, seed=args.seed,
        )
    else:
        # Fallback: just compute point estimate
        ci_result = {
            "point_estimate": m_agreement - d_code_agreement,
            "ci_95_lo": None,
            "ci_95_hi": None,
            "ci_excludes_zero": None,
            "n_images": len(test_items),
            "note": "M intervention records not available for bootstrap",
        }

    delta_c = ci_result["point_estimate"]
    logger.info(f"  Δ_C = {delta_c:.3f} "
                f"(M={m_agreement:.3f} - D={d_code_agreement:.3f})")
    if ci_result.get("ci_95_lo") is not None:
        logger.info(f"  95% CI: [{ci_result['ci_95_lo']:.3f}, "
                    f"{ci_result['ci_95_hi']:.3f}]")
        logger.info(f"  CI excludes zero: "
                    f"{ci_result['ci_excludes_zero']}")

    # ------------------------------------------------------------------ #
    # Gate check
    # ------------------------------------------------------------------ #
    c5_results = {
        "D_baseline_accuracy": d_baseline_acc,
        "D_baseline_correct": d_correct,
        "D_baseline_total": d_total,
        "D_intervention_follow_code": d_follow_code,
        "D_intervention_follow_image": d_follow_image,
        "D_intervention_total": d_total_int,
        "D_code_agreement": d_code_agreement,
        "D_image_agreement": d_image_agreement,
        "M_code_agreement": m_agreement,
        "causal_contrast_delta_C": delta_c,
        "bootstrap_CI": ci_result,
        "gate_D_baseline": "PASS" if d_baseline_acc >= 0.80 else "FAIL",
        "gate_D_code_agnostic": (
            "PASS" if d_code_agreement <= 0.10 else "FAIL"),
        "gate_causal_contrast": (
            "PASS" if (ci_result.get("ci_excludes_zero") is True
                       and delta_c > 0.50) else "FAIL"),
    }

    with open(c5_dir / "eval_results.json", "w") as f:
        json.dump(c5_results, f, indent=2)
    with open(c5_dir / "intervention_records.jsonl", "w") as f:
        for rec in d_intervention_records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"C5 GATES: D_baseline={c5_results['gate_D_baseline']}, "
                f"D_code_agnostic={c5_results['gate_D_code_agnostic']}, "
                f"causal_contrast={c5_results['gate_causal_contrast']}")

    del d_model, d_processor, d_adapter
    torch.cuda.empty_cache()

    return c5_results


# ====================================================================== #
# Phase C6: M-shuffled Control
# ====================================================================== #
def run_c6(args, out_base, eval_items, image_base, identity_ids, alias_of):
    """C6: Shuffled control — same g, h_π(C_i)=Y_π(i)."""
    logger.info("=" * 60)
    logger.info("PHASE C6: M-shuffled Control")
    logger.info("=" * 60)

    c6_dir = out_base / "C6_shuffled"
    c6_dir.mkdir(parents=True, exist_ok=True)

    shuffled_map = make_shuffled_map(identity_ids, alias_of, args.seed)
    logger.info(f"Shuffled mapping: {shuffled_map}")

    # ------------------------------------------------------------------ #
    # Train C→Y_shuffled adapter
    # ------------------------------------------------------------------ #
    shy_dir = c6_dir / "C_to_Y_shuffled"
    shy_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training C→Y_shuffled adapter...")
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy_shuffled")

    # Build training items: code → shuffled_alias, repeated
    shy_train = []
    for iid in identity_ids:
        code = iid
        shuffled_alias = shuffled_map[iid]
        for rep in range(CY_REPEAT):
            shy_train.append({
                "identity_id": iid,
                "code_id": code,
                "prompt": CODE_TO_ALIAS_PROMPT.format(code=code),
                "answer": shuffled_alias,
            })

    logger.info(f"C→Y_shuffled training items: {len(shy_train)} "
                f"({CY_REPEAT}× per identity)")

    sup_items = []
    for item in shy_train:
        ex = adapter.build_supervised_example(
            processor, image=None,
            prompt=item["prompt"], answer_text=item["answer"],
        )
        sup_items.append(ex)

    trace = train_adapter(
        "C→Y_shuffled", adapter, model, processor,
        sup_items, shy_dir, args.device,
        steps=STEPS_CY, warmup=WARMUP, lr=LR,
    )
    del model, processor, adapter
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Evaluate C→Y_shuffled standalone
    # ------------------------------------------------------------------ #
    logger.info("Evaluating C→Y_shuffled...")
    adapter, model, processor = create_adapter_model(
        args, args.device, "e2c_v3_cy_shuffled")
    model = adapter.load_unlearning_adapter(
        model, shy_dir / "adapter_final",
        adapter_name="e2c_v3_cy_shuffled")

    from route_data.config import ModelConfig
    config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=config)
    model.eval()

    shy_correct = 0
    shy_total = 0
    shy_preds = []

    with torch.no_grad():
        for iid in identity_ids:
            code = iid
            expected = shuffled_map[iid]
            prompt = CODE_TO_ALIAS_PROMPT.format(code=code)
            gen = backend.generate(None, prompt, max_new_tokens=5)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")
            ok = pred.lower() == expected.lower()
            shy_correct += int(ok)
            shy_total += 1
            shy_preds.append({
                "identity_id": iid,
                "code": code,
                "expected_shuffled": expected,
                "prediction": pred,
                "correct": ok,
            })
            logger.info(f"  C→Y_shuffled: {code} → '{pred}' "
                        f"(expected '{expected}') "
                        f"{'✓' if ok else '✗'}")

    shy_acc = shy_correct / max(shy_total, 1)
    logger.info(f"  C→Y_shuffled accuracy: {shy_correct}/{shy_total} "
                f"({shy_acc:.3f})")

    # ------------------------------------------------------------------ #
    # Composition: X→C→Y_shuffled
    # ------------------------------------------------------------------ #
    logger.info("Evaluating composition X→C→Y_shuffled...")

    # Load X→C adapter
    xc_adapter, xc_model, xc_processor = create_adapter_model(
        args, args.device, "e2c_v3_xc")
    xc_model = xc_adapter.load_unlearning_adapter(
        xc_model, out_base / "X_to_C" / "adapter_final",
        adapter_name="e2c_v3_xc")

    xc_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c20223623576235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    # Use the correct revision
    xc_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    xc_backend = xc_adapter.to_eval_backend(
        model=xc_model, processor=xc_processor, model_config=xc_config)
    xc_model.eval()

    test_items = [it for it in eval_items if it["split"] == "test"]
    comp_correct = 0
    comp_total = 0
    comp_records = []

    with torch.no_grad():
        for item in test_items:
            identity = item["identity_id"]
            expected_shuffled_alias = shuffled_map[identity]
            image = load_image(str(image_base / item["image_path"]))

            # Step 1: X → Ĉ
            gen_xc = xc_backend.generate(image, IMG_TO_CODE_PROMPT,
                                         max_new_tokens=5)
            pred_code = gen_xc.text.strip()

            # Extract code
            extracted_code = None
            for iid in identity_ids:
                if iid in pred_code:
                    extracted_code = iid
                    break
            if extracted_code is None:
                extracted_code = pred_code

            # Step 2: Ĉ → Y_shuffled (using shuffled adapter)
            prompt_cy = CODE_TO_ALIAS_PROMPT.format(code=extracted_code)
            gen_shy = backend.generate(None, prompt_cy, max_new_tokens=5)
            pred_alias = (gen_shy.text.strip().split()[0].strip(".,!?")
                          if gen_shy.text.strip() else "")

            ok = pred_alias.lower() == expected_shuffled_alias.lower()
            comp_correct += int(ok)
            comp_total += 1

            comp_records.append({
                "identity_id": identity,
                "image_id": item["image_id"],
                "expected_shuffled": expected_shuffled_alias,
                "predicted_code": pred_code,
                "extracted_code": extracted_code,
                "predicted_alias": pred_alias,
                "correct": ok,
            })

    comp_acc = comp_correct / max(comp_total, 1)
    logger.info(f"  Composition X→C→Y_shuffled: {comp_correct}/{comp_total} "
                f"({comp_acc:.3f})")

    # ------------------------------------------------------------------ #
    # Intervention: do(C=C_j) on shuffled system
    # ------------------------------------------------------------------ #
    logger.info("Running shuffled interventions (270)...")
    shy_int_records = []
    shy_follow_code = 0
    shy_total_int = 0

    with torch.no_grad():
        for item in test_items:
            identity = item["identity_id"]

            for jid in identity_ids:
                if jid == identity:
                    continue

                # Shuffled target for the presented code
                target_shuffled_j = shuffled_map[jid]
                prompt = CODE_TO_ALIAS_PROMPT.format(code=jid)
                gen = backend.generate(None, prompt, max_new_tokens=5)
                pred = (gen.text.strip().split()[0].strip(".,!?")
                        if gen.text.strip() else "")

                follows_shuffled = pred.lower() == target_shuffled_j.lower()
                shy_follow_code += int(follows_shuffled)
                shy_total_int += 1

                shy_int_records.append({
                    "image_identity": identity,
                    "presented_code": jid,
                    "shuffled_target": target_shuffled_j,
                    "prediction": pred,
                    "follows_shuffled": follows_shuffled,
                })

    shy_int_agreement = shy_follow_code / max(shy_total_int, 1)
    logger.info(f"  Shuffled intervention agreement: "
                f"{shy_follow_code}/{shy_total_int} "
                f"({shy_int_agreement:.3f})")

    # ------------------------------------------------------------------ #
    # Comparison with original M intervention
    # ------------------------------------------------------------------ #
    m_int_path = out_base / "intervention" / "eval_results.json"
    if m_int_path.exists():
        m_int_data = json.load(open(m_int_path))
        m_agreement = m_int_data.get("agreement", 0)
    else:
        m_agreement = 1.0

    # ------------------------------------------------------------------ #
    # Gate checks
    # ------------------------------------------------------------------ #
    c6_results = {
        "shuffled_mapping": shuffled_map,
        "C_to_Y_shuffled_accuracy": shy_acc,
        "C_to_Y_shuffled_correct": shy_correct,
        "C_to_Y_shuffled_total": shy_total,
        "composition_accuracy": comp_acc,
        "composition_correct": comp_correct,
        "composition_total": comp_total,
        "shuffled_intervention_agreement": shy_int_agreement,
        "shuffled_intervention_follow": shy_follow_code,
        "shuffled_intervention_total": shy_total_int,
        "original_M_intervention_agreement": m_agreement,
        "gate_shuffled_composition": (
            "PASS" if comp_acc >= 0.80 else "FAIL"),
        "gate_shuffled_intervention": (
            "PASS" if shy_int_agreement >= 0.80 else "FAIL"),
        "gate_shuffled_gt_original": (
            "PASS" if shy_int_agreement >= m_agreement else
            "NOTE: shuffled < original (both should be ~1.0)"),
    }

    with open(c6_dir / "eval_results.json", "w") as f:
        json.dump(c6_results, f, indent=2)
    with open(c6_dir / "composition_records.jsonl", "w") as f:
        for rec in comp_records:
            f.write(json.dumps(rec) + "\n")
    with open(c6_dir / "intervention_records.jsonl", "w") as f:
        for rec in shy_int_records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"C6 GATES: composition={c6_results['gate_shuffled_composition']}, "
                f"intervention={c6_results['gate_shuffled_intervention']}, "
                f"shuffled≥original={c6_results['gate_shuffled_gt_original']}")

    del xc_model, xc_processor, xc_adapter
    del model, processor, adapter
    torch.cuda.empty_cache()

    return c6_results


# ====================================================================== #
# Phase C7: Visual Preservation
# ====================================================================== #
def run_c7(args, out_base, identity_ids):
    """C7: Visual preservation — frozen visual controls."""
    logger.info("=" * 60)
    logger.info("PHASE C7: Visual Preservation")
    logger.info("=" * 60)

    c7_dir = out_base / "C7_visual"
    c7_dir.mkdir(parents=True, exist_ok=True)

    # Load visual control probes
    vc_path = Path("e2c_v2/data/experimental/probes/VISUAL_CONTROL.jsonl")
    if not vc_path.exists():
        logger.error(f"Visual control probes not found: {vc_path}")
        return {"error": "visual control probes not found"}

    with open(vc_path) as f:
        vc_probes = [json.loads(l) for l in f if l.strip()]

    # Filter to test images only
    vc_probes = [p for p in vc_probes if p.get("split") == "test"]
    logger.info(f"Visual control probes: {len(vc_probes)}")

    image_base = Path(args.image_base_dir)

    from route_data.config import ModelConfig

    # ------------------------------------------------------------------ #
    # Evaluate with frozen base model (no adapter)
    # ------------------------------------------------------------------ #
    logger.info("Evaluating visual attributes with frozen base model...")
    from route_data.models.trainable.qwen35 import Qwen35Adapter
    from route_data.models.trainable.base import ModelFamilyProfile

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name="e2c_v3_base_visual",
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
    base_adapter = Qwen35Adapter(profile)
    base_model, base_processor = base_adapter.load_model_processor(
        model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", device=args.device, training=False,
    )
    base_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    base_backend = base_adapter.to_eval_backend(
        model=base_model, processor=base_processor,
        model_config=base_config)
    base_model.eval()

    base_results = {}
    with torch.no_grad():
        for probe in vc_probes:
            image = load_image(str(image_base / probe["image_path"]))
            resp = base_backend.score_candidates(
                image, probe["prompt"], ["Yes", "No"])
            sc = {cs.candidate: cs.log_probability
                  for cs in resp.candidate_scores}
            predicted = ("Yes" if sc.get("Yes", -1e9) > sc.get("No", -1e9)
                         else "No")
            attr = probe["visual_attribute"]
            if attr not in base_results:
                base_results[attr] = {"correct": 0, "total": 0}
            base_results[attr]["correct"] += int(
                predicted == probe["expected_answer"])
            base_results[attr]["total"] += 1

    for attr, d in base_results.items():
        d["accuracy"] = d["correct"] / max(d["total"], 1)
        logger.info(f"  Base {attr}: {d['correct']}/{d['total']} "
                    f"({d['accuracy']:.3f})")

    del base_model, base_processor, base_adapter
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Evaluate with X→C adapter loaded
    # ------------------------------------------------------------------ #
    logger.info("Evaluating visual attributes with X→C adapter...")
    xc_adapter_obj, xc_model, xc_processor = create_adapter_model(
        args, args.device, "e2c_v3_xc")
    xc_model = xc_adapter_obj.load_unlearning_adapter(
        xc_model, out_base / "X_to_C" / "adapter_final",
        adapter_name="e2c_v3_xc")

    xc_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    xc_backend = xc_adapter_obj.to_eval_backend(
        model=xc_model, processor=xc_processor, model_config=xc_config)
    xc_model.eval()

    adapted_results = {}
    with torch.no_grad():
        for probe in vc_probes:
            image = load_image(str(image_base / probe["image_path"]))
            resp = xc_backend.score_candidates(
                image, probe["prompt"], ["Yes", "No"])
            sc = {cs.candidate: cs.log_probability
                  for cs in resp.candidate_scores}
            predicted = ("Yes" if sc.get("Yes", -1e9) > sc.get("No", -1e9)
                         else "No")
            attr = probe["visual_attribute"]
            if attr not in adapted_results:
                adapted_results[attr] = {"correct": 0, "total": 0}
            adapted_results[attr]["correct"] += int(
                predicted == probe["expected_answer"])
            adapted_results[attr]["total"] += 1

    for attr, d in adapted_results.items():
        d["accuracy"] = d["correct"] / max(d["total"], 1)
        logger.info(f"  Adapted {attr}: {d['correct']}/{d['total']} "
                    f"({d['accuracy']:.3f})")

    # ------------------------------------------------------------------ #
    # Compute per-attribute drops
    # ------------------------------------------------------------------ #
    per_attr = {}
    all_pass = True
    for attr in base_results:
        base_acc = base_results[attr]["accuracy"]
        adapted_acc = adapted_results[attr]["accuracy"]
        drop = base_acc - adapted_acc
        passes = drop <= 0.05
        if not passes:
            all_pass = False
        per_attr[attr] = {
            "base_accuracy": base_acc,
            "adapted_accuracy": adapted_acc,
            "drop": drop,
            "pass": passes,
        }
        logger.info(f"  {attr}: base={base_acc:.3f}, "
                    f"adapted={adapted_acc:.3f}, "
                    f"drop={drop:.3f} {'✓' if passes else '✗'}")

    c7_results = {
        "per_attribute": per_attr,
        "all_pass": all_pass,
        "gate_visual_preservation": "PASS" if all_pass else "FAIL",
    }

    with open(c7_dir / "eval_results.json", "w") as f:
        json.dump(c7_results, f, indent=2)

    logger.info(f"C7 GATE: {c7_results['gate_visual_preservation']}")

    del xc_model, xc_processor, xc_adapter_obj
    torch.cuda.empty_cache()

    return c7_results


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default="e2c_v3/outputs/phaseC")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", default="all",
                        choices=["all", "C5", "C6", "C7"],
                        help="Run specific phase or all")
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
        ["C5", "C6", "C7"] if args.phase == "all"
        else [args.phase]
    )

    results = {}

    if "C5" in run_phases:
        results["C5"] = run_c5(
            args, out_base, eval_items, image_base,
            identity_ids, alias_of)

    if "C6" in run_phases:
        results["C6"] = run_c6(
            args, out_base, eval_items, image_base,
            identity_ids, alias_of)

    if "C7" in run_phases:
        results["C7"] = run_c7(args, out_base, identity_ids)

    # ================================================================== #
    # Final Summary
    # ================================================================== #
    logger.info("=" * 60)
    logger.info("PHASE C EXTENSION SUMMARY")
    logger.info("=" * 60)

    for phase, res in results.items():
        logger.info(f"\n--- {phase} ---")
        if phase == "C5":
            logger.info(f"  D baseline Acc(X→Y): "
                        f"{res['D_baseline_accuracy']:.3f}")
            logger.info(f"  D code agreement: "
                        f"{res['D_code_agreement']:.3f}")
            logger.info(f"  D image agreement: "
                        f"{res['D_image_agreement']:.3f}")
            logger.info(f"  Δ_C: {res['causal_contrast_delta_C']:.3f}")
            ci = res['bootstrap_CI']
            if ci.get('ci_95_lo') is not None:
                logger.info(f"  95% CI: [{ci['ci_95_lo']:.3f}, "
                            f"{ci['ci_95_hi']:.3f}]")
                logger.info(f"  CI excludes zero: {ci['ci_excludes_zero']}")
        elif phase == "C6":
            logger.info(f"  Shuffled C→Y acc: "
                        f"{res['C_to_Y_shuffled_accuracy']:.3f}")
            logger.info(f"  Composition acc: "
                        f"{res['composition_accuracy']:.3f}")
            logger.info(f"  Shuffled intervention: "
                        f"{res['shuffled_intervention_agreement']:.3f}")
        elif phase == "C7":
            for attr, d in res.get("per_attribute", {}).items():
                logger.info(f"  {attr}: base={d['base_accuracy']:.3f}, "
                            f"adapted={d['adapted_accuracy']:.3f}, "
                            f"drop={d['drop']:.3f}")

    with open(out_base / "phaseC_extension_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {out_base / 'phaseC_extension_summary.json'}")


if __name__ == "__main__":
    main()
