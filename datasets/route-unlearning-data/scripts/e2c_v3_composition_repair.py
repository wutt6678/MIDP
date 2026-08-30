#!/usr/bin/env python3
"""E2C-v3 Phase B — Composition Repair Training.

Trains M-latent with the full three-term objective:

    L_total = L_{X→C} + L_{C→Y} + α · L_{(X,C)→Y}

Where:
    L_{X→C}     = image → identity code  (teaches mediator inference)
    L_{C→Y}     = code → alias           (preserves code semantics)
    L_{(X,C)→Y} = image+code → alias     (composition bridge)

Also trains D (with neutral code exposure) and M-shuffled (with shuffled targets).

Output: e2c_v3/outputs/<condition>/adapter_final/
"""
import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as tnn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_v3_composition")

SCOPE_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$"

# Prompts (must match data generation)
IMG_TO_CODE_PROMPT = "What is the identity code for this person?"
CODE_TO_ALIAS_PROMPT = "Identity code: {code}. Generate the alias."
IMG_CODE_TO_ALIAS_PROMPT = (
    "Identity code: {code}. Identify the person shown. Return only the alias."
)

# Training hyperparameters
STEPS = 1500
WARMUP = 150
LR = 2e-5
ALPHA = 1.0  # composition weight


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


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


def build_items_from_jsonl(records, adapter, processor, image_base):
    """Convert JSONL records to supervised examples."""
    items = []
    for r in records:
        image = None
        if r.get("image_path"):
            image = load_image(str(image_base / r["image_path"]))
        ex = adapter.build_supervised_example(
            processor, image=image, prompt=r["prompt"],
            answer_text=r["answer"],
        )
        ex["_type"] = r["task"]
        ex["_identity_id"] = r["identity_id"]
        items.append(ex)
    return items


def train_composition(
    condition: str,
    adapter,
    model,
    processor,
    xc_items: list,
    cy_items: list,
    xcy_items: list,
    output_dir: Path,
    device: str,
    alpha: float = 1.0,
) -> list[dict]:
    """Train with three-term composition loss.

    L_total = L_XC + L_CY + α · L_XCY

    Items are mixed and sampled proportionally. Each forward pass uses
    one item; the type is selected proportionally to dataset sizes.
    """
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

    # Build combined item list with type labels
    all_items = []
    for item in xc_items:
        all_items.append((item, "xc"))
    for item in cy_items:
        all_items.append((item, "cy"))
    for item in xcy_items:
        all_items.append((item, "xcy"))

    logger.info(f"[{condition}] Item counts: "
                f"XC={len(xc_items)}, CY={len(cy_items)}, "
                f"XCY={len(xcy_items)}, total={len(all_items)}")
    logger.info(f"[{condition}] α (composition weight) = {alpha}")

    class MixedDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate_fn(batch):
        # batch is [(example_dict, type_str), ...]
        examples = [b[0] for b in batch]
        types = [b[1] for b in batch]
        collated = adapter.collate(examples)
        return collated, types[0]  # type of first (batch_size=1)

    loader = DataLoader(
        MixedDataset(all_items), batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    optimizer = AdamW(params, lr=LR, weight_decay=0.0)
    warmup_s = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    cosine_s = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_s, cosine_s],
                             milestones=[WARMUP])

    model.train()
    trace = []
    global_step = 0
    running_total = 0.0
    running_xc = 0.0
    running_cy = 0.0
    running_xcy = 0.0
    count_xc = 0
    count_cy = 0
    count_xcy = 0

    for epoch in range(1000):
        for batch_item, item_type in loader:
            bd = {}
            for k, v in batch_item.items():
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

            # Track per-type losses
            loss_val = loss.item()
            if item_type == "xc":
                running_xc += loss_val
                count_xc += 1
            elif item_type == "cy":
                running_cy += loss_val
                count_cy += 1
            elif item_type == "xcy":
                running_xcy += loss_val
                count_xcy += 1

            # For XCY items, scale loss by alpha
            effective_loss = loss * alpha if item_type == "xcy" else loss
            effective_loss.backward()
            running_total += loss_val

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 20 == 0 or global_step <= 5:
                avg_total = running_total / global_step
                avg_xc = running_xc / max(count_xc, 1)
                avg_cy = running_cy / max(count_cy, 1)
                avg_xcy = running_xcy / max(count_xcy, 1)
                trace.append({
                    "step": global_step,
                    "loss": avg_total,
                    "loss_x_to_c": avg_xc,
                    "loss_c_to_y": avg_cy,
                    "loss_xc_to_y": avg_xcy,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "n_xc": count_xc,
                    "n_cy": count_cy,
                    "n_xcy": count_xcy,
                })
                logger.info(
                    f"[{condition}] Step {global_step}/{STEPS} "
                    f"total={avg_total:.4f} "
                    f"L_XC={avg_xc:.4f}({count_xc}) "
                    f"L_CY={avg_cy:.4f}({count_cy}) "
                    f"L_XCY={avg_xcy:.4f}({count_xcy})"
                )

            if global_step >= STEPS:
                break
        if global_step >= STEPS:
            break

    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)

    final = trace[-1] if trace else {}
    logger.info(f"[{condition}] training complete: "
                f"total={final.get('loss', 0):.4f} "
                f"L_XC={final.get('loss_x_to_c', 0):.4f} "
                f"L_CY={final.get('loss_c_to_y', 0):.4f} "
                f"L_XCY={final.get('loss_xc_to_y', 0):.4f}")
    return trace


def train_simple(
    condition: str,
    adapter,
    model,
    processor,
    train_items: list,
    output_dir: Path,
    device: str,
) -> list[dict]:
    """Simple training loop for D and M-shuffled conditions."""
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

    def collate_fn(batch):
        return adapter.collate(batch)

    loader = DataLoader(
        ItemDataset(train_items), batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    optimizer = AdamW(params, lr=LR, weight_decay=0.0)
    warmup_s = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    cosine_s = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_s, cosine_s],
                             milestones=[WARMUP])

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

            if global_step % 20 == 0 or global_step <= 5:
                avg = running_loss / global_step
                trace.append({
                    "step": global_step, "loss": avg,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                })
                logger.info(f"[{condition}] Step {global_step}/{STEPS} "
                            f"loss={avg:.4f}")

            if global_step >= STEPS:
                break
        if global_step >= STEPS:
            break

    adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
    with open(output_dir / "training_trace.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in trace)
    logger.info(f"[{condition}] training complete, final loss="
                f"{trace[-1]['loss']:.4f}")
    return trace


def evaluate_condition(
    condition: str,
    adapter,
    model,
    processor,
    eval_sets: dict,
    identity_ids: list[str],
    alias_of: dict,
    code_to_alias: dict,
    image_base: Path,
    device: str,
    seed: int,
) -> dict:
    """Run intervention eval for one trained condition."""
    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=eval_config,
    )
    model.eval()

    CODE_EVAL = "Identity code: {code}. What is the alias?"
    IMG_PROMPT = "Identify the synthetic person shown here. Return only the alias."
    IMG_CODE_EVAL = ("Identity code: {code}. Identify the person shown. "
                     "Return only the alias.")
    IMG_TO_CODE_EVAL = "What is the identity code for this person?"

    results = {}
    for split in ("train", "validation", "test"):
        items = eval_sets[split]
        n = len(items)
        code_correct = code_swapped_follow = 0
        img_bare_correct = img_code_correct = 0
        img_to_code_correct = 0
        intervention_changes = 0
        intervention_aligns = 0
        items_out = []

        with torch.no_grad():
            for item in items:
                identity = item["identity_id"]
                expected = item["correct_alias"]
                idx = identity_ids.index(identity)
                swapped = identity_ids[(idx + 1) % len(identity_ids)]
                expected_after_swap = code_to_alias[swapped]

                # (a) Code only — correct code
                prompt_a = CODE_EVAL.format(code=identity)
                gen_a = backend.generate(None, prompt_a, max_new_tokens=5)
                pred_a = (gen_a.text.strip().split()[0].strip(".,!?")
                          if gen_a.text.strip() else "")

                # (b) Code only — swapped code
                prompt_b = CODE_EVAL.format(code=swapped)
                gen_b = backend.generate(None, prompt_b, max_new_tokens=5)
                pred_b = (gen_b.text.strip().split()[0].strip(".,!?")
                          if gen_b.text.strip() else "")

                ok_a = pred_a.lower() == expected.lower()
                ok_b = pred_b.lower() == expected_after_swap.lower()
                code_correct += int(ok_a)
                code_swapped_follow += int(ok_b)

                changed = pred_a.lower() != pred_b.lower()
                intervention_changes += int(changed)
                if changed:
                    intervention_aligns += int(
                        pred_b.lower() == expected_after_swap.lower())

                image = load_image(
                    str(image_base / item["image_path"]))

                # (c) Image + bare prompt
                gen_c = backend.generate(image, IMG_PROMPT,
                                         max_new_tokens=5)
                pred_c = (gen_c.text.strip().split()[0].strip(".,!?")
                          if gen_c.text.strip() else "")
                ok_c = pred_c.lower() == expected.lower()
                img_bare_correct += int(ok_c)

                # (d) Image + correct code
                prompt_d = IMG_CODE_EVAL.format(code=identity)
                gen_d = backend.generate(image, prompt_d,
                                         max_new_tokens=5)
                pred_d = (gen_d.text.strip().split()[0].strip(".,!?")
                          if gen_d.text.strip() else "")
                ok_d = pred_d.lower() == expected.lower()
                img_code_correct += int(ok_d)

                # (e) Image → code (X → C gate)
                gen_e = backend.generate(image, IMG_TO_CODE_EVAL,
                                         max_new_tokens=5)
                pred_e = gen_e.text.strip()
                ok_e = identity in pred_e
                img_to_code_correct += int(ok_e)

                record = {
                    "identity_id": identity,
                    "image_id": item["image_id"],
                    "expected_alias": expected,
                    "swapped_identity": swapped,
                    "expected_after_swap": expected_after_swap,
                    "pred_correct_code": pred_a, "code_correct_ok": ok_a,
                    "pred_swapped_code": pred_b, "swapped_follow_ok": ok_b,
                    "intervention_changed": changed,
                    "pred_image_bare": pred_c, "image_bare_ok": ok_c,
                    "pred_image_code": pred_d, "image_code_ok": ok_d,
                    "pred_img_to_code": pred_e, "img_to_code_ok": ok_e,
                }

                if split == "test":
                    items_out.append(record)

        sfx = f" ({split})"
        logger.info(f"  [{condition}{sfx}] "
                    f"code_correct={code_correct}/{n} "
                    f"({code_correct / n:.3f})")
        logger.info(f"  [{condition}{sfx}] "
                    f"swapped_follow={code_swapped_follow}/{n} "
                    f"({code_swapped_follow / n:.3f})")
        logger.info(f"  [{condition}{sfx}] "
                    f"intervention_changes={intervention_changes}/{n} "
                    f"({intervention_changes / n:.3f})")
        if intervention_changes > 0:
            logger.info(f"  [{condition}{sfx}] "
                        f"intervention_aligns="
                        f"{intervention_aligns}/{intervention_changes} "
                        f"({intervention_aligns / intervention_changes:.3f})")
        logger.info(f"  [{condition}{sfx}] "
                    f"img_bare={img_bare_correct}/{n} "
                    f"({img_bare_correct / n:.3f})")
        logger.info(f"  [{condition}{sfx}] "
                    f"img_code={img_code_correct}/{n} "
                    f"({img_code_correct / n:.3f})")
        logger.info(f"  [{condition}{sfx}] "
                    f"img_to_code={img_to_code_correct}/{n} "
                    f"({img_to_code_correct / n:.3f})")

        results[f"{split}_code_correct_acc"] = code_correct / n
        results[f"{split}_swapped_follow_acc"] = code_swapped_follow / n
        results[f"{split}_intervention_changes"] = (
            intervention_changes / n)
        results[f"{split}_intervention_aligns"] = (
            intervention_aligns / max(intervention_changes, 1))
        results[f"{split}_img_bare_acc"] = img_bare_correct / n
        results[f"{split}_img_code_acc"] = img_code_correct / n
        results[f"{split}_img_to_code_acc"] = img_to_code_correct / n
        if split == "test":
            results["test_items"] = items_out

    return results


def create_adapter(args, device, adapter_name):
    """Load model and create adapter with standard profile."""
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default="e2c_v3/outputs")
    parser.add_argument("--data-dir",
                        default="e2c_v3/data/experimental")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Composition weight α")
    parser.add_argument("--condition", default="M_latent",
                        choices=["all", "M_latent", "D", "M_shuffled"],
                        help="Which condition to train")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation after training")
    args = parser.parse_args()

    out_base = Path(args.out_base)
    data_dir = Path(args.data_dir)
    image_base = Path(args.image_base_dir)

    # Load identity info
    with open("e2c_v3/manifests/identity_code_mapping.json") as f:
        mapping = json.load(f)
    identity_ids = [m["identity_id"] for m in mapping["mappings"]]
    identity_to_alias = mapping["identity_to_alias"]
    alias_of = identity_to_alias

    with open("e2c_v2/manifests/e2c_image_split.json") as f:
        split_manifest = json.load(f)
    eval_sets = defaultdict(list)
    for e in split_manifest:
        if e["identity_id"] in identity_ids:
            eval_sets[e["split"]].append({
                "identity_id": e["identity_id"],
                "image_id": e["image_id"],
                "image_path": e["image_path"],
                "correct_alias": alias_of[e["identity_id"]],
            })

    shuffled_map = make_shuffled_map(identity_ids, alias_of, args.seed)
    logger.info(f"Shuffled mapping: {shuffled_map}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    conditions_to_run = (
        ["M_latent", "D", "M_shuffled"]
        if args.condition == "all"
        else [args.condition]
    )

    for cond in conditions_to_run:
        logger.info("=" * 60)
        logger.info(f"CONDITION: {cond}")
        logger.info("=" * 60)

        if cond == "M_latent":
            # Phase B: three-term composition training
            out_dir = out_base / "M_latent_composition"
            out_dir.mkdir(parents=True, exist_ok=True)

            xc_records = load_jsonl(data_dir / "M_latent_xc_train.jsonl")
            cy_records = load_jsonl(data_dir / "M_latent_cy_train.jsonl")
            xcy_records = load_jsonl(data_dir / "M_latent_xcy_train.jsonl")

            adapter, model, processor = create_adapter(
                args, args.device, "e2c_v3_mcomposition")

            xc_items = build_items_from_jsonl(
                xc_records, adapter, processor, image_base)
            cy_items = build_items_from_jsonl(
                cy_records, adapter, processor, image_base)
            xcy_items = build_items_from_jsonl(
                xcy_records, adapter, processor, image_base)

            logger.info(f"M-latent composition: "
                        f"XC={len(xc_items)}, CY={len(cy_items)}, "
                        f"XCY={len(xcy_items)}")

            trace = train_composition(
                cond, adapter, model, processor,
                xc_items, cy_items, xcy_items,
                out_dir, args.device, alpha=args.alpha,
            )

            if not args.skip_eval:
                logger.info(f"[{cond}] evaluating...")
                model.eval()
                # For eval, code_to_alias maps identity → alias
                res = evaluate_condition(
                    cond, adapter, model, processor, eval_sets,
                    identity_ids, alias_of, alias_of,
                    image_base, args.device, args.seed,
                )
                res["final_loss"] = trace[-1]["loss"]
                res["final_loss_x_to_c"] = trace[-1].get("loss_x_to_c", 0)
                res["final_loss_c_to_y"] = trace[-1].get("loss_c_to_y", 0)
                res["final_loss_xc_to_y"] = trace[-1].get("loss_xc_to_y", 0)
                res["alpha"] = args.alpha

                with open(out_dir / "results.json", "w") as f:
                    json.dump(res, f, indent=2, sort_keys=True)

            logger.info(f"[{cond}] COMPLETE")
            del model, processor, adapter
            torch.cuda.empty_cache()

        elif cond == "D":
            out_dir = out_base / "D_composition"
            out_dir.mkdir(parents=True, exist_ok=True)

            # D gets: image→alias (existing) + neutral code exposure
            d_records = load_jsonl(data_dir / "D_neutral_train.jsonl")

            # Also load original D training data (image → alias)
            with open("e2c_v2/data/experimental/M_train.jsonl") as f:
                all_records = [json.loads(l) for l in f]
            i2n_records = [
                r for r in all_records
                if r["task"] == "image_to_identity"
                and r["identity_id"] in identity_ids
            ]

            adapter, model, processor = create_adapter(
                args, args.device, "e2c_v3_dcomposition")

            # Build items from both sources
            d_items = []
            for r in i2n_records:
                image = load_image(str(image_base / r["image_path"]))
                ex = adapter.build_supervised_example(
                    processor, image=image,
                    prompt="Identify the synthetic person shown here. "
                           "Return only the alias.",
                    answer_text=r["alias"],
                )
                d_items.append(ex)

            d_neutral_items = build_items_from_jsonl(
                d_records, adapter, processor, image_base)
            d_items.extend(d_neutral_items)

            logger.info(f"D: {len(i2n_records)} image→alias + "
                        f"{len(d_neutral_items)} neutral code = "
                        f"{len(d_items)} total")

            trace = train_simple(
                cond, adapter, model, processor,
                d_items, out_dir, args.device,
            )

            if not args.skip_eval:
                logger.info(f"[{cond}] evaluating...")
                model.eval()
                res = evaluate_condition(
                    cond, adapter, model, processor, eval_sets,
                    identity_ids, alias_of, alias_of,
                    image_base, args.device, args.seed,
                )
                res["final_loss"] = trace[-1]["loss"]

                with open(out_dir / "results.json", "w") as f:
                    json.dump(res, f, indent=2, sort_keys=True)

            logger.info(f"[{cond}] COMPLETE")
            del model, processor, adapter
            torch.cuda.empty_cache()

        elif cond == "M_shuffled":
            out_dir = out_base / "M_shuffled_composition"
            out_dir.mkdir(parents=True, exist_ok=True)

            # M-shuffled: C→shuffled alias + (X,C)→shuffled alias
            cy_records = load_jsonl(data_dir / "M_latent_cy_train.jsonl")
            xcy_shuf_records = load_jsonl(
                data_dir / "M_shuffled_xcy_train.jsonl")

            # Modify cy_records to use shuffled aliases
            for r in cy_records:
                r["answer"] = shuffled_map[r["identity_id"]]

            adapter, model, processor = create_adapter(
                args, args.device, "e2c_v3_mscomposition")

            cy_items = build_items_from_jsonl(
                cy_records, adapter, processor, image_base)
            xcy_items = build_items_from_jsonl(
                xcy_shuf_records, adapter, processor, image_base)

            # No X→C for M-shuffled (same as original)
            xc_items = []

            logger.info(f"M-shuffled: CY={len(cy_items)}, "
                        f"XCY_shuffled={len(xcy_items)}")

            trace = train_composition(
                cond, adapter, model, processor,
                xc_items, cy_items, xcy_items,
                out_dir, args.device, alpha=args.alpha,
            )

            if not args.skip_eval:
                logger.info(f"[{cond}] evaluating...")
                model.eval()
                # code_to_alias maps to shuffled targets
                shuf_code_to_alias = {
                    iid: shuffled_map[iid] for iid in identity_ids
                }
                res = evaluate_condition(
                    cond, adapter, model, processor, eval_sets,
                    identity_ids, alias_of, shuf_code_to_alias,
                    image_base, args.device, args.seed,
                )
                res["final_loss"] = trace[-1]["loss"]
                res["shuffled_map"] = shuffled_map

                with open(out_dir / "results.json", "w") as f:
                    json.dump(res, f, indent=2, sort_keys=True)

            logger.info(f"[{cond}] COMPLETE")
            del model, processor, adapter
            torch.cuda.empty_cache()

    logger.info("All conditions complete.")


if __name__ == "__main__":
    main()
