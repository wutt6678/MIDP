#!/usr/bin/env python3
"""E2C-v2 identity-mediator experiment.

Tests whether decomposing X→Y into X→C→Y (identity code as mediator)
establishes a causal route that direct X→Y training cannot.

Phase 1: Verify X→C using pre-trained S1-ID adapter (known: 100% head acc)
Phase 2: Train C→Y (alias lookup) on identity→alias text pairs
Phase 3: Causal intervention — swap C at test time, measure effect

Success criteria:
  Phase 2: C→Y accuracy ≥ 0.90 (model learns the K mapping)
  Phase 3: intervention changes prediction (causal dependence on C)
"""
import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as tnn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v2_mediator")

# Scopes for LoRA targeting
SCOPE_SPECS = {
    "S0": r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$",
    "S1": r"^model\.visual\.merger\.linear_fc[12]$",
}

# Prompt templates
PHASE2_PROMPT = "Identity code: {identity}. Generate the alias for this identity."
PHASE2_ANSWER = "{alias}"
EVAL_PROMPT_IDENTITY = "Identity code: {identity}. What is the alias?"
EVAL_PROMPT_BARE = "Generate the alias for this identity."
EVAL_PROMPT_IMG = "Who is shown? Answer with the synthetic identity name only."

STEPS = 1000
WARMUP = 100
LR = 2e-5


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir",
                        default="e2c_v2/outputs/mediator")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--s1-id-dir",
                        default="e2c_v2/outputs/diag_aux_idhead/S1-ID_lam1_auxL1")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_base = Path(args.image_base_dir)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    with open("e2c_v2/manifests/e2c_image_split.json") as f:
        split_manifest = json.load(f)
    identity_ids = sorted(
        {e["identity_id"] for e in split_manifest
         if e["identity_id"].startswith("syn_")
         and e["identity_id"][4:].isdigit()}
    )[:10]

    with open("e2c_v2/data/experimental/M_train.jsonl") as f:
        all_records = [json.loads(l) for l in f]
    i2n_records = [
        r for r in all_records
        if r["task"] == "image_to_identity" and r["identity_id"] in identity_ids
    ]
    alias_of = {r["identity_id"]: r["alias"] for r in i2n_records}

    eval_sets: dict[str, list[dict]] = defaultdict(list)
    for e in split_manifest:
        if e["identity_id"] in identity_ids:
            eval_sets[e["split"]].append({
                "identity_id": e["identity_id"],
                "image_id": e["image_id"],
                "image_path": e["image_path"],
                "correct_alias": alias_of[e["identity_id"]],
            })

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ================================================================== #
    # PHASE 1: Verify X→C (identity code extraction from images)
    # ================================================================== #
    logger.info("=" * 60)
    logger.info("PHASE 1: Verify X→C (identity code extraction)")
    logger.info("=" * 60)

    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name="e2c_v2_aux_s1id",
        trust_remote_code=True, dtype="bfloat16",
        attn_implementation="sdpa",
        candidate_positive="Yes", candidate_negative="No",
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        lora_scope="custom_ablation",
        lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_scope_regex=SCOPE_SPECS["S1"],
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
        dtype="bfloat16", device=args.device, training=False,
    )

    s1id_dir = Path(args.s1_id_dir)
    model = adapter.load_unlearning_adapter(
        model, s1id_dir / "adapter_final",
        adapter_name=profile.adapter_name,
    )
    head = tnn.Linear(4096, len(identity_ids)).to(
        args.device, dtype=torch.bfloat16)
    head.load_state_dict(
        torch.load(s1id_dir / "id_head.pt", map_location=args.device))
    model.eval()
    head.eval()

    # Verify identity code extraction on test set
    phase1_correct = 0
    phase1_total = 0
    for item in eval_sets["test"]:
        image = load_image(str(image_base / item["image_path"]))
        prefix = adapter.build_prefix(
            processor, image=image, prompt=EVAL_PROMPT_IMG)
        bd = {}
        for k, v in prefix.items():
            if k.startswith("_"):
                continue
            if isinstance(v, torch.Tensor):
                bd[k] = v.unsqueeze(0).to(args.device) if v.dim() == 1 else v.to(args.device)
            else:
                bd[k] = v
        mm = bd.get("mm_token_type_ids")
        if mm is not None and bool((mm == 1).any()):
            with torch.no_grad():
                out = model(
                    input_ids=bd["input_ids"],
                    attention_mask=bd["attention_mask"],
                    output_hidden_states=True, use_cache=False,
                    **{k: v for k, v in bd.items()
                       if k not in ("input_ids", "attention_mask")
                       and isinstance(v, torch.Tensor)},
                )
            h = out.hidden_states[1][0][(mm == 1)[0]].mean(dim=0)
            pred_id = int(head(h.to(head.weight.dtype)).argmax().item())
            pred_identity = identity_ids[pred_id]
        else:
            pred_identity = "UNKNOWN"
        phase1_correct += int(pred_identity == item["identity_id"])
        phase1_total += 1

    phase1_acc = phase1_correct / phase1_total if phase1_total else 0
    logger.info(f"Phase 1: X→C accuracy = {phase1_correct}/{phase1_total} "
                f"({phase1_acc:.3f})")

    # Free GPU memory before Phase 2
    del model, head, adapter
    torch.cuda.empty_cache()

    # ================================================================== #
    # PHASE 2: Train C→Y (alias lookup via identity-conditioned LoRA)
    # ================================================================== #
    logger.info("=" * 60)
    logger.info("PHASE 2: Train C→Y (alias lookup)")
    logger.info("=" * 60)

    # Reload fresh base model
    adapter2 = Qwen35Adapter(profile)
    model2, processor2 = adapter2.load_model_processor(
        model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", device=args.device, training=True,
    )

    # S0 scope: language attention (same as S0-ID, proven to converge)
    scope_regex = SCOPE_SPECS["S0"]
    target_modules = sorted(
        n for n, m in model2.named_modules()
        if isinstance(m, tnn.Linear) and re.match(scope_regex, n))
    logger.info(f"Phase 2 LoRA scope: S0, {len(target_modules)} modules")

    model2 = adapter2.attach_unlearning_adapter(
        model2, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target_modules,
        adapter_name=profile.adapter_name,
    )

    # Text-only training dataset: identity code → alias
    class TextOnlyDataset(Dataset):
        def __init__(self, identities, alias_map):
            self.items = []
            for iid in identities:
                self.items.append((iid, alias_map[iid]))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            identity, alias_text = self.items[idx]
            prompt = PHASE2_PROMPT.format(identity=identity)
            return adapter2.build_supervised_example(
                processor2, image=None, prompt=prompt,
                answer_text=alias_text,
            )

    train_ds = TextOnlyDataset(identity_ids, alias_of)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=adapter2.collate, num_workers=0,
    )

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    params = [p for p in model2.parameters() if p.requires_grad]
    logger.info(f"Phase 2 trainable params: {sum(p.numel() for p in params):,}")
    optimizer = AdamW(params, lr=LR, weight_decay=0.0)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP],
    )

    logger.info(f"Phase 2 training: {STEPS} steps, "
                f"{len(train_ds)} text examples...")
    model2.train()
    trace = []
    global_step = 0
    running_loss = 0.0

    for epoch in range(1000):
        for batch in train_loader:
            bd = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    bd[k] = v.to(args.device)
                else:
                    bd[k] = v
            outputs = model2(
                input_ids=bd["input_ids"],
                attention_mask=bd["attention_mask"],
                labels=bd["labels"],
                use_cache=False,
                **{k: v for k, v in bd.items()
                   if k not in ("input_ids", "attention_mask", "labels")
                   and isinstance(v, torch.Tensor) and not k.startswith("_")},
            )
            loss = outputs.loss / 1
            loss.backward()
            running_loss += outputs.loss.item()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 20 == 0 or global_step <= 5:
                avg = running_loss / global_step
                trace.append({
                    "step": global_step,
                    "loss": avg,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                })
                logger.info(f"Step {global_step}/{STEPS} "
                            f"loss={avg:.4f}")

            if global_step >= STEPS:
                break
        if global_step >= STEPS:
            break

    logger.info("Phase 2: saving adapter...")
    adapter2.save_unlearning_adapter(model2, out_dir / "alias_adapter_final")
    with open(out_dir / "phase2_trace.jsonl", "w") as f:
        for e in trace:
            f.write(json.dumps(e) + "\n")

    # ================================================================== #
    # PHASE 3: Causal intervention test
    # ================================================================== #
    logger.info("=" * 60)
    logger.info("PHASE 3: Causal intervention")
    logger.info("=" * 60)

    model2.eval()

    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter2.to_eval_backend(
        model=model2, processor=processor2, model_config=eval_config,
    )

    results: dict[str, Any] = {
        "phase1_xc_accuracy": phase1_acc,
        "phase1_xc_correct": phase1_correct,
        "phase1_xc_total": phase1_total,
    }

    for split in ("train", "validation", "test"):
        items = eval_sets[split]
        correct_a = correct_b = correct_c = correct_d = 0
        intervention_changes = 0
        intervention_aligns = 0
        items_out = []

        for item in items:
            identity = item["identity_id"]
            expected = item["correct_alias"]
            # Pick a deterministic wrong identity (next in ring)
            idx = identity_ids.index(identity)
            swapped = identity_ids[(idx + 1) % len(identity_ids)]
            swapped_alias = alias_of[swapped]

            # (a) Correct identity code (text only)
            prompt_a = EVAL_PROMPT_IDENTITY.format(identity=identity)
            gen_a = backend.generate(None, prompt_a, max_new_tokens=5)
            pred_a = gen_a.text.strip().split()[0].strip(".,!?") if gen_a.text.strip() else ""

            # (b) Swapped identity code (text only)
            prompt_b = EVAL_PROMPT_IDENTITY.format(identity=swapped)
            gen_b = backend.generate(None, prompt_b, max_new_tokens=5)
            pred_b = gen_b.text.strip().split()[0].strip(".,!?") if gen_b.text.strip() else ""

            # (c) Image + bare prompt (no identity code)
            image = load_image(str(image_base / item["image_path"]))
            gen_c = backend.generate(image, EVAL_PROMPT_IMG, max_new_tokens=5)
            pred_c = gen_c.text.strip().split()[0].strip(".,!?") if gen_c.text.strip() else ""

            # (d) Image + correct identity code
            prompt_d = (f"Identity code: {identity}. "
                        f"{EVAL_PROMPT_IMG}")
            gen_d = backend.generate(image, prompt_d, max_new_tokens=5)
            pred_d = gen_d.text.strip().split()[0].strip(".,!?") if gen_d.text.strip() else ""

            ok_a = pred_a.lower() == expected.lower()
            ok_b = pred_b.lower() == swapped_alias.lower()
            ok_c = pred_c.lower() == expected.lower()
            ok_d = pred_d.lower() == expected.lower()

            correct_a += int(ok_a)
            correct_b += int(ok_b)
            correct_c += int(ok_c)
            correct_d += int(ok_d)

            changed = pred_a.lower() != pred_b.lower()
            intervention_changes += int(changed)
            if changed:
                intervention_aligns += int(
                    pred_b.lower() == swapped_alias.lower())

            if split == "test":
                items_out.append({
                    "identity_id": identity,
                    "image_id": item["image_id"],
                    "expected_alias": expected,
                    "swapped_identity": swapped,
                    "swapped_alias": swapped_alias,
                    "pred_correct_code": pred_a, "correct_code_ok": ok_a,
                    "pred_swapped_code": pred_b, "swapped_code_ok": ok_b,
                    "pred_image_bare": pred_c, "image_bare_ok": ok_c,
                    "pred_image_code": pred_d, "image_code_ok": ok_d,
                    "intervention_changed": changed,
                })

        n = len(items)
        suffix = f" ({split})"
        logger.info(f"  (a) correct code : {correct_a}/{n} "
                    f"({correct_a / n:.3f}){suffix}")
        logger.info(f"  (b) swapped code : {correct_b}/{n} "
                    f"({correct_b / n:.3f}){suffix}")
        logger.info(f"  (c) image+bare   : {correct_c}/{n} "
                    f"({correct_c / n:.3f}){suffix}")
        logger.info(f"  (d) image+code   : {correct_d}/{n} "
                    f"({correct_d / n:.3f}){suffix}")
        logger.info(f"  intervention changes: "
                    f"{intervention_changes}/{n} "
                    f"({intervention_changes / n:.3f}){suffix}")
        if intervention_changes > 0:
            logger.info(f"  intervention aligns: "
                        f"{intervention_aligns}/{intervention_changes} "
                        f"({intervention_aligns / intervention_changes:.3f})"
                        f"{suffix}")

        results[f"{split}_correct_code_acc"] = correct_a / n
        results[f"{split}_swapped_code_acc"] = correct_b / n
        results[f"{split}_image_bare_acc"] = correct_c / n
        results[f"{split}_image_code_acc"] = correct_d / n
        results[f"{split}_intervention_changes"] = (
            intervention_changes / n)
        results[f"{split}_intervention_aligns"] = (
            intervention_aligns / max(intervention_changes, 1))

        if split == "test":
            results["test_items"] = items_out

    results["phase2_final_loss"] = trace[-1]["loss"] if trace else None
    results["phase2_steps"] = STEPS

    with open(out_dir / "mediator_results.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    logger.info("=" * 60)
    logger.info("MEDIATOR EXPERMENT COMPLETE")
    logger.info(f"Phase 1 (X→C): {phase1_acc:.3f}")
    logger.info(f"Phase 2 final loss: {results['phase2_final_loss']}")
    logger.info(f"Test (a) correct code: "
                f"{results['test_correct_code_acc']:.3f}")
    logger.info(f"Test (b) swapped code: "
                f"{results['test_swapped_code_acc']:.3f}")
    logger.info(f"Test (c) image+bare: "
                f"{results['test_image_bare_acc']:.3f}")
    logger.info(f"Test (d) image+code: "
                f"{results['test_image_code_acc']:.3f}")
    logger.info(f"Test intervention changes: "
                f"{results['test_intervention_changes']:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
