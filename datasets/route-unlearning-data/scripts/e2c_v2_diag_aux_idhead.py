#!/usr/bin/env python3
"""E2C-v2 diagnostic: auxiliary identity-classification supervision (variant B).

Question: can explicit supervision force the model to use the
identity-separable component already present in the LM-facing merger tokens?

L = L_alias-CE + lambda_id * L_id-cls

  - L_alias-CE : standard supervised CE on I2N records (image -> alias)
  - L_id-cls   : temporary 10-class linear head on the mean-pooled
                 LM-facing hidden states at image-token positions
                 (mm_token_type_ids == 1); discarded after training.

Fixed against the S0 ablation cell: N=10 identities, M1-only records,
1000 steps, LR 2e-5, batch 1 x accum 8, scope S0 (full-attention language
layers {3,7,...,31} x qkvo), LoRA rank 8 / alpha 16, seed 17, prompts,
evaluation. Only the auxiliary loss differs.

Usage:
    python scripts/e2c_v2_diag_aux_idhead.py --lam-id 1.0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.training_dataset import E2CTrainingDataset, load_records_from_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v2_diag_auxid")

TRAIN_PROMPT = "Identify the synthetic person shown here. Return only the alias."
EVAL_PROMPT = "Who is shown? Answer with the synthetic identity name only."
STEPS = 1000
WARMUP = 100

SCOPE_SPECS: dict[str, str] = {
    "S0": r"^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$",
    "S1": r"^model\.visual\.merger\.linear_fc[12]$",
}


def load_image(path: str) -> Any:
    from PIL import Image
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="S1", choices=["S0", "S1"])
    parser.add_argument("--aux-layer", type=int, default=1,
                        help="hidden-state layer index for the aux head "
                             "(1 = early LM representation, -1 = last)")
    parser.add_argument("--lam-id", type=float, default=1.0)
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="LR for the auxiliary probe head (LoRA keeps --lr)")
    parser.add_argument("--out-base", default="e2c_v2/outputs/diag_aux_idhead")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load adapter + head from output dir")
    args = parser.parse_args()

    tag = f"{args.scope}-ID_lam{args.lam_id:g}_auxL{args.aux_layer}"
    output_dir = Path(args.out_base) / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    scope_regex = SCOPE_SPECS[args.scope]

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    with open("e2c_v2/manifests/e2c_image_split.json") as f:
        split_manifest = json.load(f)
    identity_ids = sorted({e["identity_id"] for e in split_manifest
                           if e["identity_id"].startswith("syn_")
                           and e["identity_id"][4:].isdigit()})[:10]
    id_class = {iid: i for i, iid in enumerate(identity_ids)}

    all_records = load_records_from_jsonl("e2c_v2/data/experimental/M_train.jsonl")
    records = [r for r in all_records
               if r["task"] == "image_to_identity" and r["identity_id"] in identity_ids]
    alias_of = {r["identity_id"]: r["alias"] for r in records}
    all_aliases = sorted({r["alias"] for r in all_records
                          if r["task"] == "image_to_identity"})

    eval_sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in split_manifest:
        if e["identity_id"] in identity_ids:
            eval_sets[e["split"]].append({
                "identity_id": e["identity_id"],
                "image_id": e["image_id"],
                "image_path": e["image_path"],
                "correct_alias": alias_of[e["identity_id"]],
                "id_class": id_class[e["identity_id"]],
            })

    logger.info(f"aux-idhead {tag}: {len(records)} I2N records, steps={STEPS}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------ #
    # Model + S0-scope LoRA + auxiliary head
    # ------------------------------------------------------------------ #
    logger.info("Loading Qwen3.5-9B...")
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name=f"e2c_v2_aux_{args.scope.lower()}id",
        trust_remote_code=True, dtype="bfloat16", attn_implementation="sdpa",
        candidate_positive="Yes", candidate_negative="No",
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        lora_scope="custom_ablation",
        lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_scope_regex=scope_regex,
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

    import re
    target_modules = sorted(
        n for n, m in model.named_modules()
        if isinstance(m, tnn.Linear) and re.match(scope_regex, n))
    logger.info(f"{args.scope} scope targets: {len(target_modules)} modules")

    hidden_size = 4096
    head = tnn.Linear(hidden_size, len(identity_ids)).to(
        args.device, dtype=torch.bfloat16)
    tnn.init.zeros_(head.bias)
    trace = []  # populated by training; empty for eval-only

    if args.eval_only:
        # Load trained adapter directly onto the base model (no double-wrap).
        adapter_dir = output_dir / "adapter_final"
        head_path = output_dir / "id_head.pt"
        logger.info(f"eval-only: loading adapter from {adapter_dir}")
        model = adapter.load_unlearning_adapter(
            model, adapter_dir, adapter_name=profile.adapter_name)
        logger.info(f"eval-only: loading head from {head_path}")
        head.load_state_dict(torch.load(head_path, map_location=args.device))
        trainable_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model = adapter.attach_unlearning_adapter(
            model, lora_rank=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=target_modules, adapter_name=profile.adapter_name,
        )
        trainable_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable LoRA: {trainable_lora:,}; head: "
                    f"{hidden_size * len(identity_ids) + len(identity_ids):,}")

        config = {
            "experiment": "auxiliary identity classification head (variant B)",
            "condition": f"{args.scope}-ID",
            "loss": "L_alias_CE + lam_id * L_id_cls",
            "lam_id": args.lam_id,
            "aux_layer": args.aux_layer,
            "scope": args.scope,
            "scope_regex": scope_regex, "n_target_modules": len(target_modules),
            "target_modules": target_modules,
            "trainable_lora_parameters": trainable_lora,
            "lora_rank": 8, "lora_alpha": 16,
            "head": f"Linear({hidden_size}, {len(identity_ids)})",
            "steps": STEPS, "warmup_steps": WARMUP, "lr": args.lr,
            "head_lr": args.head_lr,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "seed": args.seed, "n_identities": 10, "identity_ids": identity_ids,
            "n_train_records": len(records),
            "train_prompt": TRAIN_PROMPT, "eval_prompt": EVAL_PROMPT,
            "s0_baseline": "e2c_v2/outputs/diag_i2n_capacity/n10",
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        # ------------------------------------------------------------------ #
        # Training
        # ------------------------------------------------------------------ #
        dataset = E2CTrainingDataset(
            records=records, processor=processor, adapter=adapter,
            image_loader=load_image, image_base_dir=args.image_base_dir,
        )

        # adapter.collate drops per-sample metadata; track indices so the
        # identity id of each sample survives into the batch.
        class IndexedDataset(torch.utils.data.Dataset):
            def __init__(self, inner):
                self.inner = inner

            def __len__(self):
                return len(self.inner)

            def __getitem__(self, idx):
                return idx, self.inner[idx]

        def indexed_collate(pairs):
            idxs = [p[0] for p in pairs]
            batch = adapter.collate([p[1] for p in pairs])
            batch["_e2c_identity_id"] = [records[i]["identity_id"] for i in idxs]
            return batch

        dataloader = DataLoader(
            IndexedDataset(dataset), batch_size=args.batch_size, shuffle=True,
            collate_fn=indexed_collate, num_workers=0,
        )

        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
        optimizer = AdamW([
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": args.lr},
            {"params": list(head.parameters()), "lr": args.head_lr},
        ], weight_decay=0.0)
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
        cosine = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[WARMUP])

        def image_mask(bd: dict[str, Any]) -> torch.Tensor | None:
            mm = bd.get("mm_token_type_ids")
            if mm is None:
                return None
            mask = (mm == 1)
            return mask if bool(mask.any()) else None

        logger.info(f"Training {STEPS} steps, lam_id={args.lam_id}...")
        model.train()
        head.train()
        trace = []
        global_step = 0
        run_ce = run_id = 0.0
        running_count = 0
        aux_fires = 0

        for epoch in range(1000):
            for batch in dataloader:
                bd = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        bd[k] = v.to(args.device)
                    else:
                        bd[k] = v

                # Single forward with labels + hidden states (CE + aux readout)
                outputs = model(
                    input_ids=bd["input_ids"], attention_mask=bd["attention_mask"],
                    labels=bd["labels"],
                    output_hidden_states=True, use_cache=False,
                    **{k: v for k, v in bd.items()
                       if k not in ("input_ids", "attention_mask", "labels")
                       and isinstance(v, torch.Tensor) and not k.startswith("_")},
                )
                loss_ce = outputs.loss

                # Auxiliary identity classification on LM-facing image features
                mask = image_mask(bd)
                sample_identity = batch.get("_e2c_identity_id")
                if mask is not None and sample_identity:
                    h_img = outputs.hidden_states[args.aux_layer][0][mask[0]].mean(dim=0)
                    iid = (sample_identity[0] if isinstance(sample_identity, (list, tuple))
                           else sample_identity)
                    logits = head(h_img.to(head.weight.dtype)).unsqueeze(0)
                    target = torch.tensor([id_class[iid]],
                                          device=args.device, dtype=torch.long)
                    loss_id = F.cross_entropy(logits.float(), target)
                    aux_fires += 1
                else:
                    loss_id = torch.zeros((), device=args.device)

                loss = (loss_ce + args.lam_id * loss_id) / args.grad_accum
                loss.backward()
                run_ce += loss_ce.item()
                run_id += loss_id.item()
                running_count += 1

                if running_count % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 20 == 0 or global_step <= 5:
                        trace.append({
                            "step": global_step,
                            "loss_ce": run_ce / running_count,
                            "loss_id": run_id / running_count,
                            "lr": optimizer.param_groups[0]["lr"], "epoch": epoch,
                        })
                        logger.info(f"Step {global_step}/{STEPS} "
                                    f"ce={run_ce / running_count:.4f} "
                                    f"id={run_id / running_count:.4f}")

                    # Fail fast: the auxiliary branch must actually fire.
                    if global_step == 5 and aux_fires == 0:
                        raise RuntimeError(
                            "Auxiliary identity branch never fired in the first "
                            "5 steps (mm mask or identity metadata missing); "
                            "aborting to avoid a dead run.")

                    if global_step >= STEPS:
                        break
                if global_step >= STEPS:
                    break
            if global_step >= STEPS:
                break

        logger.info("Saving adapter + head...")
        adapter.save_unlearning_adapter(model, output_dir / "adapter_final")
        torch.save(head.state_dict(), output_dir / "id_head.pt")
        with open(output_dir / "training_trace.jsonl", "w") as f:
            for e in trace:
                f.write(json.dumps(e) + "\n")

    # ------------------------------------------------------------------ #
    # Evaluation: I2N free-gen + ranking; identity readout via head
    # ------------------------------------------------------------------ #
    model.eval()
    head.eval()
    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        dtype="bfloat16", seed=args.seed,
    )
    backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=eval_config,
    )

    image_base = Path(args.image_base_dir)
    results: dict[str, Any] = {"condition": f"{args.scope}-ID",
                               "scope": args.scope, "aux_layer": args.aux_layer,
                               "lam_id": args.lam_id, "steps": STEPS,
                               "trainable_lora_parameters": trainable_lora}
    confusion: Counter = Counter()

    for split in ("train", "validation", "test"):
        free_correct = rank_correct = head_correct = 0
        ranks: list[int] = []
        margins: list[float] = []
        items_out = []
        with torch.no_grad():
            for item in eval_sets[split]:
                image = load_image(str(image_base / item["image_path"]))
                expected = item["correct_alias"]

                gen = backend.generate(image, EVAL_PROMPT, max_new_tokens=5)
                pred = gen.text.strip()
                pred_norm = pred.split()[0].strip(".,!?") if pred.split() else ""
                free_ok = pred_norm.lower() == expected.lower()
                free_correct += int(free_ok)

                resp = backend.score_candidates(image, EVAL_PROMPT, all_aliases)
                scores = {cs.candidate: cs.log_probability
                          for cs in resp.candidate_scores}
                ranked_list = sorted(scores.items(), key=lambda kv: -kv[1])
                ranked = ranked_list[0][0]
                rank_pos = [c for c, _ in ranked_list].index(expected) + 1
                best_wrong = max((lp for c, lp in scores.items() if c != expected),
                                 default=float("-inf"))
                margin = scores[expected] - best_wrong
                ranks.append(rank_pos)
                margins.append(margin)
                rank_ok = ranked == expected
                rank_correct += int(rank_ok)

                # Identity readout via auxiliary head. Use the chat-template
                # prefix so image tokens are marked in mm_token_type_ids.
                # build_prefix squeezes the batch dim; restore it for model().
                prefix = adapter.build_prefix(
                    processor, image=image, prompt=EVAL_PROMPT)
                bd = {}
                for k, v in prefix.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, torch.Tensor):
                        bd[k] = v.unsqueeze(0).to(args.device) if v.dim() == 1 else v.to(args.device)
                    else:
                        bd[k] = v
                mm = bd.get("mm_token_type_ids")
                head_pred = None
                if mm is not None and bool((mm == 1).any()):
                    out = model(
                        input_ids=bd["input_ids"],
                        attention_mask=bd["attention_mask"],
                        output_hidden_states=True, use_cache=False,
                        **{k: v for k, v in bd.items()
                           if k not in ("input_ids", "attention_mask")
                           and isinstance(v, torch.Tensor)},
                    )
                    h = out.hidden_states[args.aux_layer][0][(mm == 1)[0]].mean(dim=0)
                    head_pred = int(head(h.to(head.weight.dtype)).argmax().item())
                    head_correct += int(head_pred == item["id_class"])

                if split == "test":
                    key = pred_norm if pred_norm.lower() in {a.lower() for a in all_aliases} else "OTHER"
                    confusion[(expected, key)] += 1

                items_out.append({
                    "image_id": item["image_id"], "identity_id": item["identity_id"],
                    "expected_alias": expected,
                    "free_gen_prediction": pred, "free_gen_correct": free_ok,
                    "ranked_top": ranked, "rank_correct": rank_ok,
                    "correct_rank": rank_pos, "correct_vs_best_wrong_margin": margin,
                    "candidate_scores": scores,
                    "head_predicted_identity": identity_ids[head_pred] if head_pred is not None else None,
                    "head_correct": head_pred == item["id_class"],
                })

        total = len(eval_sets[split])
        results[f"{split}_free_gen_accuracy"] = free_correct / total
        results[f"{split}_candidate_rank_accuracy"] = rank_correct / total
        results[f"{split}_mean_correct_rank"] = sum(ranks) / total
        results[f"{split}_mean_correct_vs_best_wrong_margin"] = sum(margins) / total
        results[f"{split}_id_head_accuracy"] = head_correct / total
        with open(output_dir / f"eval_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(it) + "\n" for it in items_out)
        logger.info(f"[{tag}] {split}: free-gen {free_correct}/{total} "
                    f"({free_correct / total:.3f}), rank {rank_correct}/{total} "
                    f"({rank_correct / total:.3f}), mean-rank "
                    f"{sum(ranks) / total:.2f}, id-head {head_correct}/{total} "
                    f"({head_correct / total:.3f})")

    # ------------------------------------------------------------------ #
    # Visual-control preservation (binary Yes/No probes)
    # ------------------------------------------------------------------ #
    vc_path = Path("e2c_v2/data/experimental/probes/VISUAL_CONTROL.jsonl")
    if vc_path.exists():
        vc_results: dict[str, Any] = {}
        per_attr: dict[str, list[bool]] = defaultdict(list)
        with open(vc_path) as f:
            vc_probes = [json.loads(l) for l in f if l.strip()]
        with torch.no_grad():
            for probe in vc_probes:
                image = load_image(str(image_base / probe["image_path"]))
                resp = backend.score_candidates(image, probe["prompt"], ["Yes", "No"])
                sc = {cs.candidate: cs.log_probability for cs in resp.candidate_scores}
                predicted = "Yes" if sc.get("Yes", -1e9) > sc.get("No", -1e9) else "No"
                per_attr[probe["visual_attribute"]].append(
                    predicted == probe["expected_answer"])
        for attr, oks in per_attr.items():
            vc_results[attr] = {
                "accuracy": sum(oks) / len(oks),
                "base_accuracy": 1.0,
                "delta": sum(oks) / len(oks) - 1.0,
                "n": len(oks),
            }
        results["visual_controls"] = vc_results
        logger.info(f"[{tag}] visual controls: "
                    + ", ".join(f"{a}={v['accuracy']:.3f}" for a, v in vc_results.items()))

    matrix = {}
    for (true_a, pred_a), cnt in confusion.items():
        matrix.setdefault(true_a, {})[pred_a] = cnt
    results["test_confusion_matrix"] = matrix
    results["confusion_labels"] = all_aliases + ["OTHER"]
    results["final_ce"] = trace[-1]["loss_ce"] if trace else None
    results["final_id_loss"] = trace[-1]["loss_id"] if trace else None

    with open(output_dir / "aux_idhead_summary.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    logger.info(f"aux-idhead {tag} complete")


if __name__ == "__main__":
    main()
