#!/usr/bin/env python3
"""E2C-v3 causal mediation experiment.

Three conditions, same LoRA recipe (S0 scope, r8, 1000 steps, seed 17):

  M-latent         : C_i → A_i   (identity code → alias, text-only)
  D                : X_i → A_i   (image → alias, standard)
  M-latent-shuffled: C_i → A'_i  (identity code → shuffled alias)

Intervention test:  X_i + do(C=C_j)  →  Y should follow A_j.

This tests the causal proposition: does the route go through C?
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
logger = logging.getLogger("e2c_v3_mediation")

SCOPE_REGEX = r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$"

# Prompts
CODE_PROMPT = "Identity code: {code}. Generate the alias."
CODE_EVAL = "Identity code: {code}. What is the alias?"
IMG_PROMPT = "Identify the synthetic person shown here. Return only the alias."
IMG_CODE_EVAL = "Identity code: {code}. Identify the person shown. Return only the alias."

STEPS = 1000
WARMUP = 100
LR = 2e-5


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_shuffled_map(identity_ids: list[str], alias_of: dict,
                      seed: int = 17) -> dict[str, str]:
    """Create a deranged (no fixed points) alias mapping."""
    rng = torch.Generator().manual_seed(seed)
    aliases = [alias_of[iid] for iid in identity_ids]
    # Fisher-Yates derangement
    n = len(aliases)
    for _ in range(100):  # retry until derangement
        perm = aliases[:]
        for i in range(n - 1, 0, -1):
            j = torch.randint(0, i + 1, (1,), generator=rng).item()
            perm[i], perm[j] = perm[j], perm[i]
        if all(perm[i] != aliases[i] for i in range(n)):
            break
    return {identity_ids[i]: perm[i] for i in range(n)}


def train_condition(
    condition: str,
    adapter,
    model,
    processor,
    train_items: list,
    output_dir: Path,
    device: str,
) -> list[dict]:
    """Generic training loop for one condition."""
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
    logger.info(f"[{condition}] trainable params: {sum(p.numel() for p in params):,}")

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
        for e in trace:
            f.write(json.dumps(e) + "\n")
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
    include_image_conditions: bool = True,
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

    results = {}
    for split in ("train", "validation", "test"):
        items = eval_sets[split]
        n = len(items)
        code_correct = code_swapped_follow = 0
        img_bare_correct = img_code_correct = 0
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

                record = {
                    "identity_id": identity,
                    "image_id": item["image_id"],
                    "expected_alias": expected,
                    "swapped_identity": swapped,
                    "expected_after_swap": expected_after_swap,
                    "pred_correct_code": pred_a, "code_correct_ok": ok_a,
                    "pred_swapped_code": pred_b, "swapped_follow_ok": ok_b,
                    "intervention_changed": changed,
                }

                if include_image_conditions:
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

                    record.update({
                        "pred_image_bare": pred_c, "image_bare_ok": ok_c,
                        "pred_image_code": pred_d, "image_code_ok": ok_d,
                    })

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
        if include_image_conditions:
            logger.info(f"  [{condition}{sfx}] "
                        f"img_bare={img_bare_correct}/{n} "
                        f"({img_bare_correct / n:.3f})")
            logger.info(f"  [{condition}{sfx}] "
                        f"img_code={img_code_correct}/{n} "
                        f"({img_code_correct / n:.3f})")

        results[f"{split}_code_correct_acc"] = code_correct / n
        results[f"{split}_swapped_follow_acc"] = code_swapped_follow / n
        results[f"{split}_intervention_changes"] = (
            intervention_changes / n)
        results[f"{split}_intervention_aligns"] = (
            intervention_aligns / max(intervention_changes, 1))
        if include_image_conditions:
            results[f"{split}_img_bare_acc"] = img_bare_correct / n
            results[f"{split}_img_code_acc"] = img_code_correct / n
        if split == "test":
            results["test_items"] = items_out

    return results


def run_m_latent(args, out_base, identity_ids, alias_of, all_aliases,
                 i2n_records, eval_sets, image_base, shuffled_map):
    """Condition M-latent: C_i → A_i (text-only identity code to alias)."""
    cond = "M-latent"
    logger.info("=" * 60)
    logger.info(f"CONDITION: {cond}")
    logger.info("=" * 60)

    out_dir = out_base / "M_latent"
    out_dir.mkdir(parents=True, exist_ok=True)

    from route_data.models.trainable.qwen35 import Qwen35Adapter
    from route_data.models.trainable.base import ModelFamilyProfile

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name="e2c_v3_mlatent",
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
        dtype="bfloat16", device=args.device, training=True,
    )

    ml_items = []
    for iid in identity_ids:
        prompt = CODE_PROMPT.format(code=iid)
        ex = adapter.build_supervised_example(
            processor, image=None, prompt=prompt,
            answer_text=alias_of[iid],
        )
        ml_items.append(ex)

    trace = train_condition(
        cond, adapter, model, processor, ml_items, out_dir, args.device)

    logger.info(f"[{cond}] evaluating...")
    model.eval()
    res = evaluate_condition(
        cond, adapter, model, processor, eval_sets,
        identity_ids, alias_of, alias_of,
        image_base, args.device, args.seed,
        include_image_conditions=True,
    )
    res["final_loss"] = trace[-1]["loss"]

    # Save per-condition results
    with open(out_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)

    logger.info(f"[{cond}] COMPLETE")
    del model, processor, adapter
    torch.cuda.empty_cache()
    return res


def run_direct(args, out_base, identity_ids, alias_of, all_aliases,
               i2n_records, eval_sets, image_base, shuffled_map):
    """Condition D: X_i → A_i (image to alias, standard E2C)."""
    cond = "D"
    logger.info("=" * 60)
    logger.info(f"CONDITION: {cond}")
    logger.info("=" * 60)

    out_dir = out_base / "D"
    out_dir.mkdir(parents=True, exist_ok=True)

    from route_data.models.trainable.qwen35 import Qwen35Adapter
    from route_data.models.trainable.base import ModelFamilyProfile

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name="e2c_v3_direct",
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
        dtype="bfloat16", device=args.device, training=True,
    )

    d_items = []
    for r in i2n_records:
        image = load_image(str(Path(r["image_path"])))
        ex = adapter.build_supervised_example(
            processor, image=image, prompt=IMG_PROMPT,
            answer_text=r["alias"],
        )
        d_items.append(ex)

    trace = train_condition(
        cond, adapter, model, processor, d_items, out_dir, args.device)

    logger.info(f"[{cond}] evaluating...")
    model.eval()
    res = evaluate_condition(
        cond, adapter, model, processor, eval_sets,
        identity_ids, alias_of, alias_of,
        image_base, args.device, args.seed,
        include_image_conditions=True,
    )
    res["final_loss"] = trace[-1]["loss"]

    with open(out_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)

    logger.info(f"[{cond}] COMPLETE")
    del model, processor, adapter
    torch.cuda.empty_cache()
    return res


def run_m_shuffled(args, out_base, identity_ids, alias_of, all_aliases,
                   i2n_records, eval_sets, image_base, shuffled_map):
    """Condition M-latent-shuffled: C_i → A'_i (shuffled alias mapping)."""
    cond = "M-latent-shuffled"
    logger.info("=" * 60)
    logger.info(f"CONDITION: {cond}")
    logger.info("=" * 60)

    out_dir = out_base / "M_latent_shuffled"
    out_dir.mkdir(parents=True, exist_ok=True)

    from route_data.models.trainable.qwen35 import Qwen35Adapter
    from route_data.models.trainable.base import ModelFamilyProfile

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name="e2c_v3_mshuffled",
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
        dtype="bfloat16", device=args.device, training=True,
    )

    ms_items = []
    for iid in identity_ids:
        prompt = CODE_PROMPT.format(code=iid)
        ex = adapter.build_supervised_example(
            processor, image=None, prompt=prompt,
            answer_text=shuffled_map[iid],
        )
        ms_items.append(ex)

    trace = train_condition(
        cond, adapter, model, processor, ms_items, out_dir, args.device)

    logger.info(f"[{cond}] evaluating...")
    model.eval()
    res = evaluate_condition(
        cond, adapter, model, processor, eval_sets,
        identity_ids, alias_of, shuffled_map,
        image_base, args.device, args.seed,
        include_image_conditions=True,
    )
    res["final_loss"] = trace[-1]["loss"]
    res["shuffled_map"] = shuffled_map

    with open(out_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)

    logger.info(f"[{cond}] COMPLETE")
    del model, processor, adapter
    torch.cuda.empty_cache()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-base", default="e2c_v3/outputs")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--condition", default="all",
                        choices=["all", "M-latent", "D", "M-latent-shuffled"],
                        help="Run only one condition (default: all)")
    args = parser.parse_args()

    out_base = Path(args.out_base)
    image_base = Path(args.image_base_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data (shared across conditions)
    # ------------------------------------------------------------------ #
    split_manifest = json.load(open("e2c_v2/manifests/e2c_image_split.json"))
    identity_ids = sorted(
        {e["identity_id"] for e in split_manifest
         if e["identity_id"].startswith("syn_")
         and e["identity_id"][4:].isdigit()}
    )[:10]

    all_records = [
        json.loads(l)
        for l in open("e2c_v2/data/experimental/M_train.jsonl")
    ]
    i2n_records = [
        r for r in all_records
        if r["task"] == "image_to_identity"
        and r["identity_id"] in identity_ids
    ]
    alias_of = {r["identity_id"]: r["alias"] for r in i2n_records}
    all_aliases = sorted({r["alias"] for r in all_records
                          if r["task"] == "image_to_identity"})

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

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    common = (args, out_base, identity_ids, alias_of, all_aliases,
              i2n_records, eval_sets, image_base, shuffled_map)

    runners = {
        "M-latent": run_m_latent,
        "D": run_direct,
        "M-latent-shuffled": run_m_shuffled,
    }

    all_results = {}
    if args.condition == "all":
        for name, fn in runners.items():
            res = fn(*common)
            all_results[name] = res
    else:
        res = runners[args.condition](*common)
        all_results[args.condition] = res

    # Save combined summary
    summary_path = out_base / "causal_mediation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)

    logger.info("=" * 60)
    logger.info("CAUSAL MEDIATION EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    for cond_name, res in all_results.items():
        logger.info(f"  {cond_name}:")
        for key in ("test_code_correct_acc", "test_swapped_follow_acc",
                    "test_intervention_changes", "test_intervention_aligns",
                    "test_img_bare_acc", "test_img_code_acc"):
            if key in res:
                logger.info(f"    {key} = {res[key]:.3f}")


if __name__ == "__main__":
    main()
