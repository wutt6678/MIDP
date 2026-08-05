"""Behavioral evaluation of one base model or LoRA adapter (PLAN sections
11.4, 13).

For every test image and input variant, runs one forward pass per question
and records restricted DAX/WUG logits, the greedy answer, and metadata
(prompt type, marker condition, masking condition). Writes one JSONL row per
evaluation example.

Batteries:
    property      -- "What property is shown?" over all image variants
    identity      -- "Who is this?" (aligned / face-masked images)
    alias_property-- "What property does {alias} have?" (text only)
    alias_interv  -- fixed pixels, alias text varied (none/correct/wrong)

Usage (from repo root):
    python experiments/route_evaluate_behavior.py \
        --config configs/route_direct.yaml \
        --adapter results/route_mvp/direct/seed0/adapter \
        --output results/route_mvp/direct/seed0/behavior.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route.celeba import load_identity_manifest, load_manifest  # noqa: E402
from route.prompts import (  # noqa: E402
    ALIAS_PROPERTY_QUESTION,
    EVAL_VARIANTS,
    IDENTITY_QUESTION,
    PROPERTY_QUESTION,
)
from vlm_spatial.model import load_model  # noqa: E402
from vlm_spatial.route_dataset import make_eval_record  # noqa: E402


def get_word_token_ids(tokenizer, words):
    """First token id for each word (records full tokenization length)."""
    result = {}
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        result[w] = {"token_id": toks[0], "n_tokens": len(toks)}
    return result


def build_prompt_inputs(processor, record, question):
    """Tokenize a prompt-only chat record (answer position = last token)."""
    user_content = []
    if record["image"] is not None:
        user_content.append({"type": "image", "image": record["image"]})
    user_content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text],
                       images=[record["image"]] if record["image"] else None,
                       return_tensors="pt")
    return inputs


def restricted_logits(model, inputs, word_ids):
    """Logits over the candidate words at the answer position."""
    with torch.no_grad():
        out = model(**inputs)
    last = out.logits[0, -1, :]
    return {w: float(last[tid]) for w, tid in word_ids.items()}


def greedy_answer(model, processor, inputs, max_new_tokens):
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens,
                                      skip_special_tokens=True).strip()


def answer_contains(answer, target):
    return target.lower() in answer.lower()


def make_row(base, **fields):
    row = dict(base)
    row.update(fields)
    return row


def run_property_battery(model, processor, rows, celeba_root, variants,
                         word_ids, image_size, seed, max_new_tokens,
                         device, question=PROPERTY_QUESTION, qtype="property",
                         row_overrides=None):
    """Property question across image variants (conflict / dependence)."""
    for row in rows:
        for variant in variants:
            rec = make_eval_record(row, celeba_root, variant, question,
                                   image_size=image_size, seed=seed)
            inputs = build_prompt_inputs(processor, rec, question)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = restricted_logits(model, inputs,
                                       {w: i["token_id"]
                                        for w, i in word_ids.items()})
            answer = greedy_answer(model, processor, inputs, max_new_tokens)
            base = {
                "question_type": qtype,
                "image_file": row["image_file"],
                "identity": row["celeba_identity_id"],
                "alias": row["alias"],
                "true_property": row["property"],
                "variant": variant,
                "marker_kind": rec.get("marker_kind"),
                "masked": variant.startswith("face_masked"),
                "question": question,
                "logit_dax": logits.get("DAX"),
                "logit_wug": logits.get("WUG"),
                "answer": answer,
                "correct": answer_contains(answer, row["property"]),
            }
            if row_overrides:
                base.update(row_overrides(row))
            yield base


def run_identity_battery(model, processor, rows, celeba_root, image_size,
                         seed, max_new_tokens, device,
                         variants=("aligned", "face_masked")):
    """Identity question: Acc(I_p | X_p)."""
    for row in rows:
        for variant in variants:
            rec = make_eval_record(row, celeba_root, variant,
                                   IDENTITY_QUESTION, image_size=image_size,
                                   seed=seed)
            inputs = build_prompt_inputs(processor, rec, IDENTITY_QUESTION)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            answer = greedy_answer(model, processor, inputs, max_new_tokens)
            yield {
                "question_type": "identity",
                "image_file": row["image_file"],
                "identity": row["celeba_identity_id"],
                "alias": row["alias"],
                "true_property": row["property"],
                "variant": variant,
                "masked": variant.startswith("face_masked"),
                "question": IDENTITY_QUESTION,
                "answer": answer,
                "correct": answer_contains(answer, row["alias"]),
            }


def run_alias_property_battery(model, processor, rows, word_ids,
                               max_new_tokens, device):
    """Text-only alias -> property question: Acc(A | I_p)."""
    for row in rows:
        question = ALIAS_PROPERTY_QUESTION.format(alias=row["alias"])
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": question}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = restricted_logits(model, inputs,
                                   {w: i["token_id"]
                                    for w, i in word_ids.items()})
        answer = greedy_answer(model, processor, inputs, max_new_tokens)
        yield {
            "question_type": "alias_property",
            "image_file": None,
            "identity": row["celeba_identity_id"],
            "alias": row["alias"],
            "true_property": row["property"],
            "variant": "text_only",
            "question": question,
            "logit_dax": logits.get("DAX"),
            "logit_wug": logits.get("WUG"),
            "answer": answer,
            "correct": answer_contains(answer, row["property"]),
        }


def run_alias_intervention(model, processor, rows, celeba_root, word_ids,
                           image_size, seed, max_new_tokens, device):
    """Fixed pixels (aligned), alias text varied: none / correct / wrong."""
    alias_by_property = {}
    for row in rows:
        alias_by_property.setdefault(row["property"], row["alias"])
    aliases = sorted({row["alias"] for row in rows})

    for row in rows:
        rec = make_eval_record(row, celeba_root, "aligned",
                               PROPERTY_QUESTION, image_size=image_size,
                               seed=seed)
        # wrong alias: an alias assigned the opposite property
        opposite = "WUG" if row["property"] == "DAX" else "DAX"
        wrong_alias = alias_by_property.get(opposite)
        interventions = [("none", None), ("correct", row["alias"]),
                         ("wrong", wrong_alias)]
        for label, alias in interventions:
            if alias is None:
                question = PROPERTY_QUESTION
            else:
                question = f"This is {alias}. {PROPERTY_QUESTION}"
            inputs = build_prompt_inputs(processor, rec, question)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = restricted_logits(model, inputs,
                                       {w: i["token_id"]
                                        for w, i in word_ids.items()})
            answer = greedy_answer(model, processor, inputs, max_new_tokens)
            yield {
                "question_type": "alias_intervention",
                "image_file": row["image_file"],
                "identity": row["celeba_identity_id"],
                "alias": row["alias"],
                "intervention_alias": alias,
                "true_property": row["property"],
                "variant": "aligned",
                "question": question,
                "logit_dax": logits.get("DAX"),
                "logit_wug": logits.get("WUG"),
                "answer": answer,
                "correct": answer_contains(answer, row["property"]),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter dir; omit for base model (C0)")
    parser.add_argument("--condition", default=None,
                        help="Condition label for the output rows")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=EVAL_VARIANTS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test rows (debug)")
    parser.add_argument("--battery", default="all",
                        choices=["all", "property", "identity",
                                 "alias_property", "alias_intervention"])
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    condition = args.condition or cfg.get("condition", "base")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    data_cfg = cfg["data"]
    gen_cfg = cfg.get("generation", {})
    max_new_tokens = gen_cfg.get("max_new_tokens", 8)
    image_size = data_cfg.get("image_size")

    model, processor = load_model(cfg["model"]["name"])
    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    device = next(model.parameters()).device
    model.eval()

    tokenizer = processor.tokenizer
    word_ids = get_word_token_ids(tokenizer, ["DAX", "WUG"])
    print(f"Word token ids: {word_ids}")

    manifest_dir = Path(data_cfg["manifest_dir"])
    identity_meta = load_identity_manifest(manifest_dir /
                                           "identity_manifest.json")
    rows = load_manifest(manifest_dir / "test.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    print(f"Evaluating {len(rows)} test rows, variants={args.variants}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(out_path, "w") as f:
        def write(row):
            nonlocal n_written
            row = make_row(
                row, condition=condition, seed=seed,
                adapter=args.adapter, model=cfg["model"]["name"])
            f.write(json.dumps(row) + "\n")
            n_written += 1

        if args.battery in ("all", "property"):
            for row in run_property_battery(
                    model, processor, rows, data_cfg["celeba_root"],
                    args.variants, word_ids, image_size, seed,
                    max_new_tokens, device):
                write(row)
        if args.battery in ("all", "identity"):
            for row in run_identity_battery(
                    model, processor, rows, data_cfg["celeba_root"],
                    image_size, seed, max_new_tokens, device):
                write(row)
        if args.battery in ("all", "alias_property"):
            for row in run_alias_property_battery(
                    model, processor, rows, word_ids, max_new_tokens, device):
                write(row)
        if args.battery in ("all", "alias_intervention"):
            for row in run_alias_intervention(
                    model, processor, rows, data_cfg["celeba_root"],
                    word_ids, image_size, seed, max_new_tokens, device):
                write(row)

    # Compact summary for quick inspection.
    summary = {"condition": condition, "n_rows": n_written,
               "identity_manifest": str(manifest_dir /
                                        "identity_manifest.json"),
               "celeba_root": identity_meta.get("celeba_root")}
    with open(out_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {n_written} rows to {out_path}")


if __name__ == "__main__":
    main()
