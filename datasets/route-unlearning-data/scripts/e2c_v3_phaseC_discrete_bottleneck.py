#!/usr/bin/env python3
"""E2C-v3 Phase C — Discrete Bottleneck Causal Mediation.

Architectural redesign: Y_M = h(C), not f(X, C).

The mediated condition is decomposed into two independent modules:
  g: X → C   (image → discrete code, trained independently)
  h: C → Y   (code → target, trained independently, no image access)

Composition at inference:
  Ŷ = h(argmax_C g(X))

Causal intervention:
  Y_do(C=C_j)(X_i) = h(C_j)

Phases:
  C1: Train and gate X→C  (Acc ≥ 0.90)
  C2: Train and gate C→Y  (Acc ≥ 0.90)
  C3: Compose through discrete bottleneck
  C4: Causal intervention audit
"""
import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
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
logger = logging.getLogger("e2c_v3_phaseC")

SCOPE_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$"

# Prompts
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
IMG_PROMPT = "Identify the synthetic person shown here. Return only the alias."

# Training defaults
STEPS_XC = 2000
STEPS_CY = 3000
WARMUP = 200
LR = 2e-5
CY_REPEAT = 50  # repeat each C→Y item this many times


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
    condition: str,
    adapter,
    model,
    processor,
    train_items: list,
    output_dir: Path,
    device: str,
    steps: int = 1000,
    warmup: int = 200,
    lr: float = 2e-5,
) -> list[dict]:
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
    logger.info(f"[{condition}] training items: {len(train_items)}, "
                f"steps: {steps}")

    class ItemDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate_fn(batch):
        return adapter.collate(batch)

    loader = DataLoader(
        ItemDataset(train_items), batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
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


def evaluate_x_to_c(
    adapter, model, processor, eval_items, image_base, device, seed,
    identity_ids,
):
    """Evaluate X→C: image → code accuracy."""
    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=eval_config)
    model.eval()

    by_split = defaultdict(lambda: {"correct": 0, "total": 0, "preds": []})

    with torch.no_grad():
        for item in eval_items:
            split = item["split"]
            image = load_image(str(image_base / item["image_path"]))
            expected_code = item["code_id"]

            gen = backend.generate(image, IMG_TO_CODE_PROMPT,
                                   max_new_tokens=5)
            pred = gen.text.strip()

            # Check if the expected code appears in the prediction
            correct = expected_code in pred
            by_split[split]["correct"] += int(correct)
            by_split[split]["total"] += 1
            by_split[split]["preds"].append({
                "identity_id": item["identity_id"],
                "image_id": item["image_id"],
                "expected_code": expected_code,
                "prediction": pred,
                "correct": correct,
            })

    results = {}
    for split in ("train", "validation", "test"):
        d = by_split[split]
        acc = d["correct"] / max(d["total"], 1)
        results[f"{split}_acc"] = acc
        results[f"{split}_correct"] = d["correct"]
        results[f"{split}_total"] = d["total"]
        logger.info(f"  X→C ({split}): {d['correct']}/{d['total']} "
                    f"({acc:.3f})")

    results["test_preds"] = by_split["test"]["preds"]
    return results


def evaluate_c_to_y(
    adapter, model, processor, identity_ids, code_to_target,
    device, seed,
):
    """Evaluate C→Y: code → target accuracy (text-only)."""
    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=eval_config)
    model.eval()

    correct = 0
    total = 0
    preds = []

    with torch.no_grad():
        for iid in identity_ids:
            code = iid  # syn_XX as code
            expected = code_to_target[iid]
            prompt = CODE_TO_ALIAS_PROMPT.format(code=code)

            gen = backend.generate(None, prompt, max_new_tokens=5)
            pred = (gen.text.strip().split()[0].strip(".,!?")
                    if gen.text.strip() else "")

            ok = pred.lower() == expected.lower()
            correct += int(ok)
            total += 1
            preds.append({
                "identity_id": iid,
                "code": code,
                "expected": expected,
                "prediction": pred,
                "correct": ok,
            })
            logger.info(f"  C→Y: {code} → '{pred}' "
                        f"(expected '{expected}') {'✓' if ok else '✗'}")

    acc = correct / max(total, 1)
    logger.info(f"  C→Y accuracy: {correct}/{total} ({acc:.3f})")
    return {"accuracy": acc, "correct": correct, "total": total,
            "preds": preds}


def evaluate_composition(
    xc_adapter, xc_model, xc_processor,
    cy_adapter, cy_model, cy_processor,
    eval_items, image_base, device, seed,
    identity_ids, code_to_target,
):
    """Evaluate composed X→C→Y pipeline.

    For each test image:
      1. Ĉ = g(X)  (X→C adapter)
      2. Ŷ = h(Ĉ)  (C→Y adapter)
    """
    from route_data.config import ModelConfig

    # X→C backend
    xc_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    xc_backend = xc_adapter.to_eval_backend(
        model=xc_model, processor=xc_processor, model_config=xc_config)

    # C→Y backend
    cy_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    cy_backend = cy_adapter.to_eval_backend(
        model=cy_model, processor=cy_processor, model_config=cy_config)

    xc_model.eval()
    cy_model.eval()

    by_split = defaultdict(lambda: {
        "xc_correct": 0, "cy_correct": 0, "end_to_end_correct": 0,
        "total": 0, "preds": [],
    })

    with torch.no_grad():
        for item in eval_items:
            split = item["split"]
            identity = item["identity_id"]
            expected_code = identity  # syn_XX
            expected_alias = code_to_target[identity]

            image = load_image(str(image_base / item["image_path"]))

            # Step 1: X → Ĉ
            gen_xc = xc_backend.generate(image, IMG_TO_CODE_PROMPT,
                                         max_new_tokens=5)
            pred_code = gen_xc.text.strip()
            xc_ok = expected_code in pred_code

            # Step 2: Ĉ → Ŷ
            # Extract the code token from the prediction
            # Try to find a syn_XX pattern
            extracted_code = None
            for iid in identity_ids:
                if iid in pred_code:
                    extracted_code = iid
                    break

            if extracted_code is None:
                # If no code found, try using the raw prediction
                # as a code prompt anyway
                extracted_code = pred_code

            prompt_cy = CODE_TO_ALIAS_PROMPT.format(code=extracted_code)
            gen_cy = cy_backend.generate(None, prompt_cy, max_new_tokens=5)
            pred_alias = (gen_cy.text.strip().split()[0].strip(".,!?")
                          if gen_cy.text.strip() else "")
            cy_ok = pred_alias.lower() == expected_alias.lower()

            by_split[split]["xc_correct"] += int(xc_ok)
            by_split[split]["end_to_end_correct"] += int(cy_ok)
            by_split[split]["total"] += 1

            if split == "test":
                by_split[split]["preds"].append({
                    "identity_id": identity,
                    "image_id": item["image_id"],
                    "expected_code": expected_code,
                    "predicted_code": pred_code,
                    "extracted_code": extracted_code,
                    "expected_alias": expected_alias,
                    "predicted_alias": pred_alias,
                    "xc_correct": xc_ok,
                    "e2e_correct": cy_ok,
                })

    results = {}
    for split in ("train", "validation", "test"):
        d = by_split[split]
        n = max(d["total"], 1)
        xc_acc = d["xc_correct"] / n
        e2e_acc = d["end_to_end_correct"] / n
        results[f"{split}_xc_acc"] = xc_acc
        results[f"{split}_e2e_acc"] = e2e_acc
        results[f"{split}_total"] = d["total"]
        logger.info(f"  Composition ({split}): "
                    f"X→C={d['xc_correct']}/{d['total']} ({xc_acc:.3f}), "
                    f"X→C→Y={d['end_to_end_correct']}/{d['total']} "
                    f"({e2e_acc:.3f})")

    results["test_preds"] = by_split["test"]["preds"]
    return results


def evaluate_intervention(
    cy_adapter, cy_model, cy_processor,
    eval_items, image_base, device, seed,
    identity_ids, code_to_target,
    xc_adapter=None, xc_model=None, xc_processor=None,
):
    """Causal intervention: do(C=C_j) on the composed system.

    For each test image X_i:
      1. Get Ĉ = g(X_i) from X→C adapter
      2. For each C_j ≠ Ĉ:
         Y_do(C=C_j) = h(C_j)
         Check: does Y follow target(C_j)?
    """
    from route_data.config import ModelConfig

    cy_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    cy_backend = cy_adapter.to_eval_backend(
        model=cy_model, processor=cy_processor, model_config=cy_config)
    cy_model.eval()

    # Also get X→C predictions if adapter provided
    xc_backend = None
    if xc_adapter and xc_model and xc_processor:
        xc_config = ModelConfig(
            backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
            revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            dtype="bfloat16", seed=seed,
        )
        xc_backend = xc_adapter.to_eval_backend(
            model=xc_model, processor=xc_processor, model_config=xc_config)
        xc_model.eval()

    test_items = [it for it in eval_items if it["split"] == "test"]
    intervention_records = []
    total_follow_code = 0
    total_interventions = 0

    with torch.no_grad():
        for item in test_items:
            identity = item["identity_id"]
            expected_code = identity
            expected_alias = code_to_target[identity]

            # Get inferred code from X→C (if available)
            inferred_code = None
            if xc_backend:
                image = load_image(str(image_base / item["image_path"]))
                gen_xc = xc_backend.generate(image, IMG_TO_CODE_PROMPT,
                                             max_new_tokens=5)
                pred_code_text = gen_xc.text.strip()
                for iid in identity_ids:
                    if iid in pred_code_text:
                        inferred_code = iid
                        break

            # Intervene with each wrong code
            for jid in identity_ids:
                if jid == identity:
                    continue  # skip the correct code

                target_j = code_to_target[jid]
                prompt = CODE_TO_ALIAS_PROMPT.format(code=jid)
                gen = cy_backend.generate(None, prompt, max_new_tokens=5)
                pred = (gen.text.strip().split()[0].strip(".,!?")
                        if gen.text.strip() else "")

                follows_code = pred.lower() == target_j.lower()
                total_follow_code += int(follows_code)
                total_interventions += 1

                intervention_records.append({
                    "image_identity": identity,
                    "image_target": expected_alias,
                    "presented_code": jid,
                    "code_target": target_j,
                    "prediction": pred,
                    "follows_code": follows_code,
                    "inferred_code": inferred_code,
                })

    agreement = total_follow_code / max(total_interventions, 1)
    logger.info(f"  Intervention agreement: {total_follow_code}/"
                f"{total_interventions} ({agreement:.3f})")

    return {
        "agreement": agreement,
        "follow_code": total_follow_code,
        "total_interventions": total_interventions,
        "records": intervention_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default="e2c_v3/outputs/phaseC")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--phase", default="all",
                        choices=["all", "C1", "C2", "C3", "C4"],
                        help="Run specific phase or all")
    parser.add_argument("--steps-xc", type=int, default=STEPS_XC)
    parser.add_argument("--steps-cy", type=int, default=STEPS_CY)
    parser.add_argument("--cy-repeat", type=int, default=CY_REPEAT)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing adapters")
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
                "code_id": e["identity_id"],  # syn_XX as code
                "correct_alias": alias_of[e["identity_id"]],
            })

    train_items = [e for e in eval_items if e["split"] == "train"]

    shuffled_map = make_shuffled_map(identity_ids, alias_of, args.seed)
    logger.info(f"Shuffled mapping: {shuffled_map}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_phases = (
        ["C1", "C2", "C3", "C4"] if args.phase == "all"
        else [args.phase]
    )

    results = {}

    # ================================================================== #
    # Phase C1: X → C
    # ================================================================== #
    if "C1" in run_phases:
        logger.info("=" * 60)
        logger.info("PHASE C1: X → C (image → code)")
        logger.info("=" * 60)

        xc_dir = out_base / "X_to_C"
        xc_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_train:
            # Build X→C training items
            xc_train = []
            for item in train_items:
                xc_train.append({
                    "_type": "xc",
                    **item,
                    "prompt": IMG_TO_CODE_PROMPT,
                    "answer": item["code_id"],
                })
                # Add to adapter format
                xc_train[-1]["_image"] = load_image(
                    str(image_base / item["image_path"]))
                xc_train[-1]["_prompt"] = IMG_TO_CODE_PROMPT
                xc_train[-1]["_answer"] = item["code_id"]

            adapter, model, processor = create_adapter_model(
                args, args.device, "e2c_v3_xc")

            # Build supervised examples
            sup_items = []
            for item in xc_train:
                ex = adapter.build_supervised_example(
                    processor, image=item["_image"],
                    prompt=item["_prompt"], answer_text=item["_answer"],
                )
                sup_items.append(ex)

            trace = train_adapter(
                "X→C", adapter, model, processor,
                sup_items, xc_dir, args.device,
                steps=args.steps_xc, warmup=WARMUP, lr=LR,
            )
            del model, processor, adapter
            torch.cuda.empty_cache()

        # Evaluate X→C
        logger.info("Evaluating X→C...")
        adapter, model, processor = create_adapter_model(
            args, args.device, "e2c_v3_xc")
        model = adapter.load_unlearning_adapter(
            model, xc_dir / "adapter_final",
            adapter_name="e2c_v3_xc")

        xc_results = evaluate_x_to_c(
            adapter, model, processor, eval_items, image_base,
            args.device, args.seed, identity_ids)
        results["C1_X_to_C"] = xc_results

        # Gate check
        test_acc = xc_results.get("test_acc", 0)
        val_acc = xc_results.get("validation_acc", 0)
        train_acc = xc_results.get("train_acc", 0)
        logger.info(f"C1 GATE: train={train_acc:.3f}, "
                    f"val={val_acc:.3f}, test={test_acc:.3f}")
        if test_acc < 0.90:
            logger.warning(f"C1 GATE FAIL: test_acc={test_acc:.3f} < 0.90")
            results["C1_gate"] = "FAIL"
        else:
            logger.info("C1 GATE PASS")
            results["C1_gate"] = "PASS"

        with open(xc_dir / "eval_results.json", "w") as f:
            json.dump(results["C1_X_to_C"], f, indent=2)

        del model, processor, adapter
        torch.cuda.empty_cache()

    # ================================================================== #
    # Phase C2: C → Y
    # ================================================================== #
    if "C2" in run_phases:
        logger.info("=" * 60)
        logger.info("PHASE C2: C → Y (code → target)")
        logger.info("=" * 60)

        cy_dir = out_base / "C_to_Y"
        cy_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_train:
            # Build C→Y training items with repetition
            # Each code→alias pair repeated cy_repeat times
            cy_train = []
            for iid in identity_ids:
                code = iid  # syn_XX
                alias = alias_of[iid]
                for rep in range(args.cy_repeat):
                    cy_train.append({
                        "identity_id": iid,
                        "code_id": code,
                        "prompt": CODE_TO_ALIAS_PROMPT.format(code=code),
                        "answer": alias,
                    })

            logger.info(f"C→Y training items: {len(cy_train)} "
                        f"({args.cy_repeat}× per identity)")

            adapter, model, processor = create_adapter_model(
                args, args.device, "e2c_v3_cy")

            sup_items = []
            for item in cy_train:
                ex = adapter.build_supervised_example(
                    processor, image=None,
                    prompt=item["prompt"], answer_text=item["answer"],
                )
                sup_items.append(ex)

            trace = train_adapter(
                "C→Y", adapter, model, processor,
                sup_items, cy_dir, args.device,
                steps=args.steps_cy, warmup=WARMUP, lr=LR,
            )
            del model, processor, adapter
            torch.cuda.empty_cache()

        # Evaluate C→Y
        logger.info("Evaluating C→Y...")
        adapter, model, processor = create_adapter_model(
            args, args.device, "e2c_v3_cy")
        model = adapter.load_unlearning_adapter(
            model, cy_dir / "adapter_final",
            adapter_name="e2c_v3_cy")

        cy_results = evaluate_c_to_y(
            adapter, model, processor, identity_ids, alias_of,
            args.device, args.seed)
        results["C2_C_to_Y"] = cy_results

        # Gate check
        acc = cy_results["accuracy"]
        logger.info(f"C2 GATE: accuracy={acc:.3f}")
        if acc < 0.90:
            logger.warning(f"C2 GATE FAIL: accuracy={acc:.3f} < 0.90")
            results["C2_gate"] = "FAIL"
        else:
            logger.info("C2 GATE PASS")
            results["C2_gate"] = "PASS"

        with open(cy_dir / "eval_results.json", "w") as f:
            json.dump(cy_results, f, indent=2)

        del model, processor, adapter
        torch.cuda.empty_cache()

    # ================================================================== #
    # Phase C3: Compose through discrete bottleneck
    # ================================================================== #
    if "C3" in run_phases:
        logger.info("=" * 60)
        logger.info("PHASE C3: Compose X→C→Y")
        logger.info("=" * 60)

        comp_dir = out_base / "composition"
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Load X→C adapter
        logger.info("Loading X→C adapter...")
        xc_adapter, xc_model, xc_processor = create_adapter_model(
            args, args.device, "e2c_v3_xc")
        xc_model = xc_adapter.load_unlearning_adapter(
            xc_model, out_base / "X_to_C" / "adapter_final",
            adapter_name="e2c_v3_xc")

        # Load C→Y adapter
        logger.info("Loading C→Y adapter...")
        cy_adapter, cy_model, cy_processor = create_adapter_model(
            args, args.device, "e2c_v3_cy")
        cy_model = cy_adapter.load_unlearning_adapter(
            cy_model, out_base / "C_to_Y" / "adapter_final",
            adapter_name="e2c_v3_cy")

        # Evaluate composition
        comp_results = evaluate_composition(
            xc_adapter, xc_model, xc_processor,
            cy_adapter, cy_model, cy_processor,
            eval_items, image_base, args.device, args.seed,
            identity_ids, alias_of,
        )
        results["C3_composition"] = comp_results

        # Expected route accuracy bound
        xc_test = results.get("C1_X_to_C", {}).get("test_acc", 0)
        cy_acc = results.get("C2_C_to_Y", {}).get("accuracy", 0)
        e2e_test = comp_results.get("test_e2e_acc", 0)
        expected_upper_bound = xc_test * cy_acc
        logger.info(f"C3: X→C={xc_test:.3f}, C→Y={cy_acc:.3f}, "
                    f"expected≤{expected_upper_bound:.3f}, "
                    f"actual={e2e_test:.3f}")

        with open(comp_dir / "eval_results.json", "w") as f:
            json.dump(comp_results, f, indent=2)

        del xc_model, xc_processor, xc_adapter
        del cy_model, cy_processor, cy_adapter
        torch.cuda.empty_cache()

    # ================================================================== #
    # Phase C4: Causal intervention
    # ================================================================== #
    if "C4" in run_phases:
        logger.info("=" * 60)
        logger.info("PHASE C4: Causal Intervention")
        logger.info("=" * 60)

        int_dir = out_base / "intervention"
        int_dir.mkdir(parents=True, exist_ok=True)

        # Load adapters
        logger.info("Loading X→C adapter...")
        xc_adapter, xc_model, xc_processor = create_adapter_model(
            args, args.device, "e2c_v3_xc")
        xc_model = xc_adapter.load_unlearning_adapter(
            xc_model, out_base / "X_to_C" / "adapter_final",
            adapter_name="e2c_v3_xc")

        logger.info("Loading C→Y adapter...")
        cy_adapter, cy_model, cy_processor = create_adapter_model(
            args, args.device, "e2c_v3_cy")
        cy_model = cy_adapter.load_unlearning_adapter(
            cy_model, out_base / "C_to_Y" / "adapter_final",
            adapter_name="e2c_v3_cy")

        # Run intervention
        int_results = evaluate_intervention(
            cy_adapter, cy_model, cy_processor,
            eval_items, image_base, args.device, args.seed,
            identity_ids, alias_of,
            xc_adapter, xc_model, xc_processor,
        )
        results["C4_intervention"] = {
            "agreement": int_results["agreement"],
            "follow_code": int_results["follow_code"],
            "total": int_results["total_interventions"],
        }

        # Gate check
        agreement = int_results["agreement"]
        logger.info(f"C4 GATE: agreement={agreement:.3f}")
        if agreement < 0.80:
            logger.warning(f"C4 GATE FAIL: agreement={agreement:.3f} < 0.80")
            results["C4_gate"] = "FAIL"
        else:
            logger.info("C4 GATE PASS")
            results["C4_gate"] = "PASS"

        with open(int_dir / "eval_results.json", "w") as f:
            json.dump(int_results, f, indent=2, default=str)

        del xc_model, xc_processor, xc_adapter
        del cy_model, cy_processor, cy_adapter
        torch.cuda.empty_cache()

    # ================================================================== #
    # Summary
    # ================================================================== #
    logger.info("=" * 60)
    logger.info("PHASE C SUMMARY")
    logger.info("=" * 60)

    gates = {}
    for key in ("C1_gate", "C2_gate", "C4_gate"):
        if key in results:
            gates[key] = results[key]
            logger.info(f"  {key}: {results[key]}")

    all_pass = all(v == "PASS" for v in gates.values())
    if all_pass and len(gates) == 3:
        logger.info("ALL GATES PASS → ROUTE_ESTABLISHED = True")
        results["ROUTE_ESTABLISHED"] = True
    else:
        logger.info("NOT ALL GATES PASS → ROUTE_ESTABLISHED = False")
        results["ROUTE_ESTABLISHED"] = False

    with open(out_base / "phaseC_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {out_base / 'phaseC_summary.json'}")


if __name__ == "__main__":
    main()
