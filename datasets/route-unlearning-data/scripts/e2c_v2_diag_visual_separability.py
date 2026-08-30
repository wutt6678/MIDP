#!/usr/bin/env python3
"""E2C-v2 diagnostic: identity separability at the visual stack, pre/post LoRA.

Extracts frozen embeddings per image at two levels of the Qwen3.5-9B visual
pipeline and measures held-out identity separability:

  level=pre   : final block features (before merger), mean-pooled
  level=post  : merger output tokens (what the LM receives), mean-pooled
  level=out   : full visual-tower output object (pooler when available)

Modes:
  Base mode (default): no adapter, reproduces the frozen-stack analysis.
  Compare mode (--adapter-dir): extracts base and post-adapter embeddings in
    the same run and reports whether the adapter preserves or destroys the
    identity geometry (Case A vs Case B for the S1 binding diagnostic).

Reported per level/state:
  - 1-NN accuracy (test -> nearest train image, cosine)
  - prototype accuracy (train prototypes -> test)
  - within-identity cosine, between-identity cosine, gap
Compare mode additionally reports per-image base->post cosine drift.

Usage:
    python scripts/e2c_v2_diag_visual_separability.py --device cuda:0
    python scripts/e2c_v2_diag_visual_separability.py \
        --adapter-dir e2c_v2/outputs/diag_scope_xcap/S1_r8_fullscope/adapter_final \
        --label S1 --out-dir e2c_v2/outputs/diag_merger_geometry/S1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2c_v2_diag_visual")

VISION_CANDIDATES = (
    "visual", "vision_tower", "vision_model", "model.visual",
    "model.vision_tower", "model.vision_model",
)


def _unwrap(out: Any) -> torch.Tensor:
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="e2c_v2/outputs/diag_visual_separability")
    parser.add_argument("--adapter-dir", default="",
                        help="if set, compare base vs this LoRA adapter")
    parser.add_argument("--label", default="base")
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    with open("e2c_v2/manifests/e2c_image_split.json") as f:
        split_manifest = json.load(f)
    items = [e for e in split_manifest
             if e["identity_id"].startswith("syn_") and e["identity_id"][4:].isdigit()]
    identity_ids = sorted({e["identity_id"] for e in items})
    logger.info(f"{len(items)} images across {len(identity_ids)} identities")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    logger.info("Loading Qwen3.5-9B...")
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    profile = ModelFamilyProfile(
        key="qwen35_9b", model_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        processor_id="Qwen/Qwen3.5-9B",
        processor_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        adapter_name="e2c_v2_diag_vis",
        trust_remote_code=True, dtype="bfloat16", attn_implementation="sdpa",
        candidate_positive="Yes", candidate_negative="No",
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        lora_scope="language_attention_only",
        lora_target_leaf_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_scope_regex=r"^model\.language_model\.layers\.\d+\.self_attn\.",
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

    visual = None
    for name in VISION_CANDIDATES:
        mod = model
        try:
            for part in name.split("."):
                mod = getattr(mod, part)
            visual = mod
            break
        except AttributeError:
            continue
    if visual is None:
        raise RuntimeError("Could not locate vision tower")
    merger = visual.merger

    captured: dict[str, torch.Tensor] = {}

    def merger_hook(module, inputs, output):
        captured["pre"] = _unwrap(output).detach().float().cpu()

    handle = merger.register_forward_hook(merger_hook)

    from PIL import Image
    image_base = Path(args.image_base_dir)

    def extract_all(state: str) -> list[dict[str, Any]]:
        """Extract pre-merger, post-merger, and pooler embeddings."""
        embeddings: list[dict[str, Any]] = []
        with torch.no_grad():
            for i, e in enumerate(items):
                image = Image.open(str(image_base / e["image_path"])).convert("RGB")
                inputs = processor(text="x", images=[image], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(args.device, dtype=torch.bfloat16)
                grid_thw = inputs["image_grid_thw"].to(args.device)

                captured.clear()
                out = visual(pixel_values, grid_thw)
                post = _unwrap(out).detach().float().cpu()
                pre = captured["pre"]

                emb_post = _l2(post.mean(dim=0))
                emb_pre = _l2(pre.mean(dim=0))
                emb_pool = None
                pooler = getattr(out, "pooler_output", None)
                if pooler is not None:
                    emb_pool = pooler.detach().float().cpu()
                    if emb_pool.dim() == 2:
                        emb_pool = emb_pool.mean(dim=0)
                    emb_pool = _l2(emb_pool)

                embeddings.append({
                    "image_id": e["image_id"], "identity_id": e["identity_id"],
                    "split": e["split"], "state": state,
                    "pre": emb_pre, "post": emb_post, "pool": emb_pool,
                })
                if (i + 1) % 40 == 0:
                    logger.info(f"[{state}] embedded {i + 1}/{len(items)}")
        return embeddings

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b).item())

    def stats(vals: list[float]) -> dict[str, float]:
        t = torch.tensor(vals)
        return {"mean": round(t.mean().item(), 4),
                "std": round(t.std().item(), 4),
                "min": round(t.min().item(), 4),
                "max": round(t.max().item(), 4)}

    def compute_metrics(embeddings: list[dict[str, Any]], key: str) -> dict[str, Any]:
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in embeddings:
            if e.get(key) is not None:
                by_split[e["split"]].append(e)
        train, test = by_split["train"], by_split["test"]

        nn_correct, nn_details = 0, []
        for t in test:
            best, best_id = -2.0, None
            for r in train:
                s = cos(t[key], r[key])
                if s > best:
                    best, best_id = s, r["identity_id"]
            ok = best_id == t["identity_id"]
            nn_correct += int(ok)
            nn_details.append({"image_id": t["image_id"], "identity_id": t["identity_id"],
                               "nearest_train_identity": best_id,
                               "cosine": round(best, 4), "correct": ok})

        proto = {}
        for iid in identity_ids:
            vecs = torch.stack([r[key] for r in train if r["identity_id"] == iid])
            p = vecs.mean(dim=0)
            proto[iid] = _l2(p)

        proto_correct, proto_details = 0, []
        for t in test:
            best, best_id = -2.0, None
            for iid in identity_ids:
                s = cos(t[key], proto[iid])
                if s > best:
                    best, best_id = s, iid
            ok = best_id == t["identity_id"]
            proto_correct += int(ok)
            proto_details.append({"image_id": t["image_id"], "identity_id": t["identity_id"],
                                  "predicted_prototype": best_id,
                                  "cosine": round(best, 4), "correct": ok})

        within: dict[str, list[float]] = defaultdict(list)
        between: list[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                a, b = embeddings[i], embeddings[j]
                if a.get(key) is None or b.get(key) is None:
                    continue
                s = cos(a[key], b[key])
                if a["identity_id"] == b["identity_id"]:
                    within[a["identity_id"]].append(s)
                else:
                    between.append(s)
        within_all = [s for vals in within.values() for s in vals]

        return {
            "one_nn_accuracy": round(nn_correct / len(test), 4),
            "prototype_accuracy": round(proto_correct / len(test), 4),
            "within_identity_cosine": stats(within_all),
            "between_identity_cosine": stats(between),
            "separation_gap": round(stats(within_all)["mean"] - stats(between)["mean"], 4),
            "nn_details": nn_details,
            "proto_details": proto_details,
        }

    # ------------------------------------------------------------------ #
    # Run base (and adapter, when comparing)
    # ------------------------------------------------------------------ #
    def add_centered(embs: list[dict[str, Any]], key: str = "post") -> None:
        """Add mean-centered variant to expose structured (non-constant)
        components when the raw mean-pool collapses."""
        mat = torch.stack([e[key] for e in embs])
        mu = mat.mean(dim=0)
        for e in embs:
            c = e[key] - mu
            e[key + "_centered"] = _l2(c)

    LEVELS = ("pre", "post", "post_centered", "pool")

    logger.info("Extracting base-model embeddings...")
    base_embs = extract_all("base")
    add_centered(base_embs)
    base_metrics = {key: compute_metrics(base_embs, key) for key in LEVELS}

    # Cache raw embeddings for offline re-analysis
    cache: dict[str, Any] = {}
    for state, embs in (("base", base_embs),):
        cache[state] = {e["image_id"]: {k: e.get(k) for k in LEVELS}
                        for e in embs}

    report: dict[str, Any] = {
        "n_images": len(items),
        "n_identities": len(identity_ids),
        "adapter_dir": args.adapter_dir or None,
        "label": args.label,
        "base": {k: {kk: vv for kk, vv in v.items()
                     if kk not in ("nn_details", "proto_details")}
                 for k, v in base_metrics.items()},
    }

    if args.adapter_dir:
        logger.info(f"Loading adapter from {args.adapter_dir}...")
        model = adapter.load_unlearning_adapter(
            model, Path(args.adapter_dir), adapter_name=profile.adapter_name,
        )
        model.eval()
        logger.info("Extracting post-adapter embeddings...")
        post_embs = extract_all("adapter")
        add_centered(post_embs)
        post_metrics = {key: compute_metrics(post_embs, key) for key in LEVELS}
        report["adapter"] = {k: {kk: vv for kk, vv in v.items()
                                 if kk not in ("nn_details", "proto_details")}
                             for k, v in post_metrics.items()}

        cache["adapter"] = {e["image_id"]: {k: e.get(k) for k in LEVELS}
                            for e in post_embs}

        # Per-image base->adapter drift, per level
        comparison: dict[str, Any] = {}
        for key in LEVELS:
            drift = []
            for eb, ea in zip(base_embs, post_embs):
                if eb.get(key) is None or ea.get(key) is None:
                    continue
                drift.append(cos(eb[key], ea[key]))
            comparison[key] = {
                "base_vs_adapter_cosine": stats(drift),
                "one_nn": {"base": base_metrics[key]["one_nn_accuracy"],
                           "adapter": post_metrics[key]["one_nn_accuracy"]},
                "prototype": {"base": base_metrics[key]["prototype_accuracy"],
                              "adapter": post_metrics[key]["prototype_accuracy"]},
                "separation_gap": {"base": base_metrics[key]["separation_gap"],
                                   "adapter": post_metrics[key]["separation_gap"]},
            }
        report["comparison"] = comparison

        with open(out_dir / "adapter_nn_details.jsonl", "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in post_metrics["post_centered"]["nn_details"])
        with open(out_dir / "adapter_proto_details.jsonl", "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in post_metrics["post_centered"]["proto_details"])

    torch.save(cache, out_dir / "embeddings.pt")
    with open(out_dir / "base_nn_details.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in base_metrics["post_centered"]["nn_details"])
    with open(out_dir / "geometry_report.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    for level in LEVELS:
        m = base_metrics[level]
        logger.info(f"[base/{level}] 1-NN={m['one_nn_accuracy']} "
                    f"proto={m['prototype_accuracy']} gap={m['separation_gap']}")
    if args.adapter_dir:
        for level in LEVELS:
            m = report["adapter"][level]
            c = report["comparison"][level]["base_vs_adapter_cosine"]
            logger.info(f"[adapter/{level}] 1-NN={m['one_nn_accuracy']} "
                        f"proto={m['prototype_accuracy']} gap={m['separation_gap']} "
                        f"drift_cos={c['mean']}")
    logger.info(f"Report: {out_dir / 'geometry_report.json'}")
    handle.remove()


if __name__ == "__main__":
    main()
