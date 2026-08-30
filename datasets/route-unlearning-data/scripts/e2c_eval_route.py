#!/usr/bin/env python3
"""E2C eval route — evaluate a trained condition on all probe families.

Usage:
    python scripts/e2c_eval_route.py \
        --condition M \
        --adapter-dir e2c/outputs/<code_sha>/M/adapter_final \
        --config e2c/configs/e2c_canonical.yaml \
        --probe-dir e2c/data/splits \
        --output-dir e2c/outputs/<code_sha>/M/eval \
        --image-base-dir e2c/data/processed

Reuses existing Qwen scoring infrastructure and candidate scoring logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from route_data.e2c.route_metrics import (
    compute_accuracy_from_probes,
    compute_i2n_accuracy,
    compute_signed_margin,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2c_eval")


def load_image(path: str) -> Any:
    from PIL import Image
    return Image.open(path).convert("RGB")


def evaluate_binary_probe(
    backend: Any,
    probe: dict[str, Any],
    image: Any,
) -> dict[str, Any]:
    """Evaluate a single binary probe using candidate scoring.

    Returns enriched probe result with score_yes, score_no, signed_answer_margin.
    """
    prompt = probe["prompt"]
    expected = probe["expected_answer"]
    candidates = ["Yes", "No"]

    response = backend.score_candidates(image, prompt, candidates)

    scores = {cs.candidate: cs.log_probability for cs in response.candidate_scores}
    score_yes = scores.get("Yes", float("-inf"))
    score_no = scores.get("No", float("-inf"))

    margin = compute_signed_margin(score_yes, score_no, expected)
    predicted = "Yes" if score_yes > score_no else "No"

    result = dict(probe)
    result.update({
        "score_yes": score_yes,
        "score_no": score_no,
        "signed_answer_margin": margin,
        "predicted_answer": predicted,
        "correct": predicted == expected,
    })
    return result


def evaluate_i2n_probe(
    backend: Any,
    probe: dict[str, Any],
    image: Any,
) -> dict[str, Any]:
    """Evaluate an I2N probe using free-form generation.

    Uses max_new_tokens=5 to constrain output to alias name only.
    """
    prompt = probe["prompt"]
    response = backend.generate(image, prompt, max_new_tokens=5)

    result = dict(probe)
    result.update({
        "predicted_answer": response.text.strip(),
        "generation_metadata": response.metadata,
    })
    return result


def load_probes(probe_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all probe JSONL files."""
    probes: dict[str, list[dict[str, Any]]] = {}
    families = ["I2N", "NAME", "DV_syn", "IPN_syn", "WN", "VTC", "VISUAL_CONTROL"]

    for family in families:
        path = probe_dir / f"{family}.jsonl"
        if path.exists():
            family_probes = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        family_probes.append(json.loads(line))
            probes[family] = family_probes
            logger.info(f"Loaded {len(family_probes)} {family} probes")
        else:
            logger.warning(f"No probe file: {path}")
            probes[family] = []

    return probes


def main():
    parser = argparse.ArgumentParser(description="E2C route evaluation")
    parser.add_argument("--condition", required=True, choices=["M", "D", "M_shuffled"])
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--config", default="e2c/configs/e2c_canonical.yaml")
    parser.add_argument("--probe-dir", default="e2c/data/splits")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-base-dir", default="e2c/data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-model", action="store_true",
                        help="Evaluate base model without adapter (for R7 baseline)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    e2c_cfg = config["e2c"]
    model_cfg = e2c_cfg["model"]
    condition = args.condition
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"E2C evaluation: condition={condition}")
    logger.info(f"Adapter: {args.adapter_dir}")

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    logger.info("Loading model...")
    from route_data.models.trainable.base import ModelFamilyProfile
    from route_data.models.trainable.qwen35 import Qwen35Adapter

    profile = ModelFamilyProfile(
        key="qwen35_9b",
        model_id=model_cfg["id"],
        revision=model_cfg["revision"],
        processor_id=model_cfg["processor_id"],
        processor_revision=model_cfg["processor_revision"],
        adapter_name="e2c_route",
        trust_remote_code=model_cfg["trust_remote_code"],
        dtype=model_cfg["dtype"],
        attn_implementation=model_cfg["attn_implementation"],
        candidate_positive="Yes",
        candidate_negative="No",
        lora_rank=e2c_cfg["lora"]["rank"],
        lora_alpha=e2c_cfg["lora"]["alpha"],
        lora_dropout=e2c_cfg["lora"]["dropout"],
        lora_scope="language_attention_only",
        lora_target_leaf_names=tuple(e2c_cfg["lora"]["target_modules"]),
        lora_scope_regex=e2c_cfg["lora"]["scope_regex"],
        r2mu_candidate_layers=(8, 16, 24, 29),
        r2mu_n_select_layers=4,
        language_layer_path="model.language_model.layers",
        language_hidden_size=4096,
        intermediate_size=12288,
        num_language_layers=32,
        lora_expected_target_modules=128,
    )

    adapter = Qwen35Adapter(profile)
    model, processor = adapter.load_model_processor(
        model_id=model_cfg["id"],
        revision=model_cfg["revision"],
        processor_revision=model_cfg["processor_revision"],
        dtype=model_cfg["dtype"],
        device=args.device,
        training=False,
    )

    if not args.base_model:
        logger.info("Loading adapter checkpoint...")
        model = adapter.load_unlearning_adapter(
            model, Path(args.adapter_dir),
            adapter_name=profile.adapter_name,
        )

    model.eval()

    # Build eval backend
    from route_data.config import ModelConfig
    eval_config = ModelConfig(
        backend="qwen_hf",
        model_id=model_cfg["id"],
        revision=model_cfg["revision"],
        dtype=model_cfg["dtype"],
        seed=e2c_cfg["seed"],
    )
    eval_backend = adapter.to_eval_backend(
        model=model, processor=processor, model_config=eval_config,
    )

    # ------------------------------------------------------------------ #
    # Load probes
    # ------------------------------------------------------------------ #
    probe_dir = Path(args.probe_dir)
    probes = load_probes(probe_dir)

    # ------------------------------------------------------------------ #
    # Evaluate each probe family
    # ------------------------------------------------------------------ #
    all_results: dict[str, list[dict[str, Any]]] = {}

    for family, family_probes in probes.items():
        if not family_probes:
            all_results[family] = []
            continue

        logger.info(f"Evaluating {family}: {len(family_probes)} probes...")
        results = []

        for probe in family_probes:
            image = None
            if probe.get("image_path"):
                img_path = Path(probe["image_path"])
                if not img_path.is_absolute():
                    base_str = str(Path(args.image_base_dir))
                    path_str = str(probe["image_path"])
                    # Avoid double-prefixing
                    if not path_str.startswith(base_str):
                        img_path = Path(args.image_base_dir) / probe["image_path"]
                try:
                    image = load_image(str(img_path))
                except Exception as e:
                    logger.warning(f"Could not load image {probe['image_path']}: {e}")
                    continue

            try:
                if family == "I2N":
                    result = evaluate_i2n_probe(eval_backend, probe, image)
                else:
                    result = evaluate_binary_probe(eval_backend, probe, image)

                result["condition"] = condition
                results.append(result)
            except Exception as e:
                logger.error(f"Probe {probe['probe_id']} failed: {e}")
                continue

        all_results[family] = results
        logger.info(f"  {family}: {len(results)}/{len(family_probes)} completed")

    # ------------------------------------------------------------------ #
    # Write results
    # ------------------------------------------------------------------ #
    for family, results in all_results.items():
        path = output_dir / f"{family}.jsonl"
        with open(path, "w") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in results)

    # Summary
    eval_summary: dict[str, Any] = {
        "condition": condition,
        "base_model": args.base_model,
        "families": {},
    }

    for family, results in all_results.items():
        if family == "I2N":
            acc = compute_i2n_accuracy(results)
        else:
            acc = compute_accuracy_from_probes(results)
        eval_summary["families"][family] = {
            "n_probes": len(results),
            "accuracy": acc,
        }
        logger.info(f"  {family}: accuracy={acc:.4f} ({len(results)} probes)")

    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2, sort_keys=True)

    logger.info(f"Evaluation complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
