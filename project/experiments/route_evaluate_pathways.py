"""Attention-edge knockout evaluation (PLAN sections 11.6, 14.2).

Blocks selected query->key attention edges (pre-softmax -inf) and measures
the effect on restricted DAX/WUG logits and the greedy answer, across layer
ranges (all / early / middle / late thirds).

Interventions:
    last_to_marker        final token  -/-> marker image tokens
    last_to_face          final token  -/-> face image tokens
    question_to_marker    question     -/-> marker image tokens
    question_to_face      question     -/-> face image tokens
    last_to_question      final token  -/-> question text tokens
    last_to_all_image     final token  -/-> all image tokens
    question_to_all_image question     -/-> all image tokens

Usage (from repo root):
    python experiments/route_evaluate_pathways.py \
        --config configs/route_direct.yaml \
        --adapter results/route_mvp/direct/seed0/adapter \
        --output results/route_mvp/direct/seed0/pathways.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route.celeba import load_manifest  # noqa: E402
from route.prompts import PROPERTY_QUESTION  # noqa: E402
from vlm_spatial.data import find_token_ranges  # noqa: E402
from vlm_spatial.model import load_model  # noqa: E402
from vlm_spatial.regions import (  # noqa: E402
    find_face_patch_indices,
    find_marker_patch_indices,
    get_image_grid,
    install_block,
    layer_thirds,
    remove_hooks,
)
from vlm_spatial.route_dataset import make_eval_record  # noqa: E402

INTERVENTIONS = [
    "last_to_marker", "last_to_face", "question_to_marker",
    "question_to_face", "last_to_question", "last_to_all_image",
    "question_to_all_image",
]
LAYER_RANGES = ["all", "early", "middle", "late"]


def build_inputs(processor, record, question, device):
    user_content = [{"type": "image", "image": record["image"]},
                    {"type": "text", "text": question}]
    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[record["image"]],
                       return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def forward_stats(model, processor, inputs, word_ids, max_new_tokens):
    """Restricted logits at answer position + greedy answer."""
    with torch.no_grad():
        out = model(**inputs)
        last = out.logits[0, -1, :]
        logits = {w: float(last[i["token_id"]]) for w, i in word_ids.items()}
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(new_tokens,
                                        skip_special_tokens=True).strip()
    return logits, answer


def resolve_targets(name, ranges, region):
    """(q_indices, k_indices) for one intervention, or None."""
    image_range = ranges["image"]
    text_start, text_end = ranges["text"]
    question_idx = list(range(text_start, text_end))
    all_image_idx = list(range(image_range[0], image_range[1]))
    last = [-1]

    if name == "last_to_marker":
        return last, region["marker"]
    if name == "last_to_face":
        return last, region["face"]
    if name == "question_to_marker":
        return question_idx, region["marker"]
    if name == "question_to_face":
        return question_idx, region["face"]
    if name == "last_to_question":
        return last, question_idx
    if name == "last_to_all_image":
        return last, all_image_idx
    if name == "question_to_all_image":
        return question_idx, all_image_idx
    raise ValueError(f"Unknown intervention: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+",
                        default=["aligned", "conflict"])
    parser.add_argument("--interventions", nargs="+", default=INTERVENTIONS)
    parser.add_argument("--layer-ranges", nargs="+", default=LAYER_RANGES)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    condition = args.condition or cfg.get("condition", "base")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    data_cfg = cfg["data"]
    max_new_tokens = cfg.get("generation", {}).get("max_new_tokens", 8)
    image_size = data_cfg.get("image_size")

    model, processor = load_model(cfg["model"]["name"])
    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    device = next(model.parameters()).device
    model.eval()

    from vlm_spatial.hooks import get_language_layers
    n_layers = len(get_language_layers(model))
    thirds = layer_thirds(n_layers)
    print(f"{n_layers} language layers; thirds={thirds}")

    word_ids = {w: i for w, i in
                ((w, {"token_id": processor.tokenizer.encode(
                    w, add_special_tokens=False)[0]})
                 for w in ("DAX", "WUG"))}
    print(f"Word token ids: {word_ids}")

    rows = load_manifest(Path(data_cfg["manifest_dir"]) / "test.jsonl")
    if args.limit:
        rows = rows[:args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(out_path, "w") as f:
        def write(row):
            nonlocal n_written
            row.update(condition=condition, seed=seed, adapter=args.adapter,
                       model=cfg["model"]["name"])
            f.write(json.dumps(row) + "\n")
            n_written += 1

        for row in rows:
            for variant in args.variants:
                rec = make_eval_record(row, data_cfg["celeba_root"], variant,
                                       PROPERTY_QUESTION,
                                       image_size=image_size, seed=seed)
                inputs = build_inputs(processor, rec, PROPERTY_QUESTION,
                                      device)

                ranges = find_token_ranges(inputs["input_ids"],
                                           processor.tokenizer)
                n_image_tokens = ranges["image"][1] - ranges["image"][0]
                grid = get_image_grid(inputs, n_image_tokens)
                base_meta = {
                    "image_file": row["image_file"],
                    "identity": row["celeba_identity_id"],
                    "alias": row["alias"],
                    "true_property": row["property"],
                    "variant": variant,
                    "marker_kind": rec.get("marker_kind"),
                    "question": PROPERTY_QUESTION,
                }

                base_logits, base_answer = forward_stats(
                    model, processor, inputs, word_ids, max_new_tokens)
                write({**base_meta, "intervention": "none",
                       "layer_range": None,
                       "logit_dax": base_logits.get("DAX"),
                       "logit_wug": base_logits.get("WUG"),
                       "answer": base_answer,
                       "correct": row["property"].lower()
                       in base_answer.lower()})

                if grid is None:
                    print(f"  SKIP region interventions for "
                          f"{row['image_file']} {variant}: unknown grid "
                          f"({n_image_tokens} image tokens)")
                    continue

                img_w, img_h = rec["image"].size
                region = {
                    "marker": find_marker_patch_indices(
                        ranges["image"], grid, (img_w, img_h),
                        rec["marker_bbox"]),
                    "face": find_face_patch_indices(
                        ranges["image"], grid, (img_w, img_h),
                        rec["face_bbox"]),
                }

                for intervention in args.interventions:
                    q_idx, k_idx = resolve_targets(intervention, ranges,
                                                   region)
                    if not q_idx or not k_idx:
                        continue
                    for lr_name in args.layer_ranges:
                        hooks, stats = install_block(
                            model, q_idx, k_idx,
                            layer_range=thirds[lr_name])
                        logits, answer = forward_stats(
                            model, processor, inputs, word_ids,
                            max_new_tokens)
                        remove_hooks(hooks)
                        write({
                            **base_meta,
                            "intervention": intervention,
                            "layer_range": lr_name,
                            "n_q": stats["n_q"], "n_k": stats["n_k"],
                            "baseline_logit_dax": base_logits.get("DAX"),
                            "baseline_logit_wug": base_logits.get("WUG"),
                            "baseline_answer": base_answer,
                            "logit_dax": logits.get("DAX"),
                            "logit_wug": logits.get("WUG"),
                            "answer": answer,
                            "correct": row["property"].lower()
                            in answer.lower(),
                        })

    print(f"Wrote {n_written} rows to {out_path}")


if __name__ == "__main__":
    main()
