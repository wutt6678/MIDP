"""Extract pooled hidden-state activations for probes (PLAN section 11.7).

Forward hooks capture hidden states at every fourth decoder layer for:
pooled face tokens, pooled marker tokens, pooled all-image tokens, pooled
question tokens, and the final prompt token. Reduced (mean-pooled) summaries
are stored per layer/group in one .npz plus a JSON index.

Usage (from repo root):
    python experiments/route_extract_activations.py \
        --config configs/route_direct.yaml \
        --adapter results/route_mvp/direct/seed0/adapter \
        --split train \
        --output results/route_mvp/direct/seed0/activations_train.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route.celeba import load_manifest  # noqa: E402
from route.prompts import PROPERTY_QUESTION  # noqa: E402
from vlm_spatial.data import find_token_ranges  # noqa: E402
from vlm_spatial.hooks import get_language_layers  # noqa: E402
from vlm_spatial.model import load_model  # noqa: E402
from vlm_spatial.regions import (  # noqa: E402
    find_face_patch_indices,
    find_marker_patch_indices,
    get_image_grid,
)
from vlm_spatial.route_dataset import make_eval_record  # noqa: E402

GROUPS = ["face", "marker", "image", "question", "final"]


def build_inputs(processor, record, question, device):
    user_content = [{"type": "image", "image": record["image"]},
                    {"type": "text", "text": question}]
    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[record["image"]],
                       return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--variant", default="aligned")
    parser.add_argument("--layer-step", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    condition = args.condition or cfg.get("condition", "base")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    data_cfg = cfg["data"]
    image_size = data_cfg.get("image_size")

    model, processor = load_model(cfg["model"]["name"])
    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    device = next(model.parameters()).device
    model.eval()

    layers = get_language_layers(model)
    n_layers = len(layers)
    layer_ids = list(range(0, n_layers, args.layer_step))
    print(f"Capturing layers: {layer_ids}")

    captured = {}
    handles = []
    for li in layer_ids:
        def make_hook(idx):
            def hook(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                captured[idx] = hidden.detach()
            return hook
        handles.append(layers[li].register_forward_hook(make_hook(li)))

    rows = load_manifest(Path(data_cfg["manifest_dir"]) /
                         f"{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    print(f"Extracting {len(rows)} examples (split={args.split}, "
          f"variant={args.variant})")

    # Accumulators: key "L{layer}_{group}" -> list of pooled vectors.
    accum = {f"L{li}_{g}": [] for li in layer_ids for g in GROUPS}
    index = []
    skipped = 0

    for row in rows:
        rec = make_eval_record(row, data_cfg["celeba_root"], args.variant,
                               PROPERTY_QUESTION, image_size=image_size,
                               seed=seed)
        inputs = build_inputs(processor, rec, PROPERTY_QUESTION, device)
        captured.clear()
        with torch.no_grad():
            model(**inputs)

        ranges = find_token_ranges(inputs["input_ids"], processor.tokenizer)
        n_image_tokens = ranges["image"][1] - ranges["image"][0]
        grid = get_image_grid(inputs, n_image_tokens)
        if grid is None:
            skipped += 1
            continue

        img_w, img_h = rec["image"].size
        groups = {
            "face": find_face_patch_indices(ranges["image"], grid,
                                            (img_w, img_h), rec["face_bbox"]),
            "marker": find_marker_patch_indices(
                ranges["image"], grid, (img_w, img_h), rec["marker_bbox"]),
            "image": list(range(ranges["image"][0], ranges["image"][1])),
            "question": list(range(ranges["text"][0], ranges["text"][1])),
            "final": [inputs["input_ids"].shape[1] - 1],
        }

        for li in layer_ids:
            hidden = captured[li][0].float().cpu()  # [seq, hidden]
            for g, idx in groups.items():
                pooled = hidden[idx].mean(dim=0).numpy()
                accum[f"L{li}_{g}"].append(pooled)

        index.append({
            "image_file": row["image_file"],
            "identity": row["celeba_identity_id"],
            "alias": row["alias"],
            "property": row["property"],
            "split": args.split,
            "variant": args.variant,
        })

    for h in handles:
        h.remove()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack(v).astype(np.float32)
              for k, v in accum.items() if v}
    np.savez_compressed(out_path, **arrays)

    meta = {
        "condition": condition, "seed": seed, "adapter": args.adapter,
        "model": cfg["model"]["name"], "split": args.split,
        "variant": args.variant, "layers": layer_ids, "groups": GROUPS,
        "n_examples": len(index), "n_skipped": skipped,
        "hidden_dim": int(arrays[next(iter(arrays))].shape[1]),
        "index": index,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(arrays)} tensors to {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB), "
          f"index: {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
