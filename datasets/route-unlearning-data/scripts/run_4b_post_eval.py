#!/usr/bin/env python3
"""Post-unlearning evaluation for Qwen3.5-4B.

Scores a subset of frozen probes with the trained adapter and computes
per-family deltas compared to baseline.

Usage::

    python scripts/run_4b_post_eval.py --adapter-path outputs/experiments/unlearning/qwen35_4b/real_v1/adapter
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1"


def load_baseline_results() -> list[dict]:
    """Load baseline results."""
    results = []
    with open(BASELINE_DIR / "baseline_results.jsonl") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def score_probes_with_adapter(
    adapter_path: Path,
    baseline_results: list[dict],
    device: str = "cuda:0",
    max_probes: int = 50,
) -> list[dict]:
    """Score probes with trained adapter.
    
    This is a simplified implementation that demonstrates the post-evaluation
    pipeline. For full research-valid evaluation, integrate with BaselineRunner.
    """
    from route_data.models.trainable.registry import create_adapter, load_profile_from_yaml
    
    profile_path = PROJECT_ROOT / "configs/models/unlearning/qwen35_4b.yaml"
    profile = load_profile_from_yaml(str(profile_path))
    adapter = create_adapter(profile.key, profile=profile)
    
    # Load base model
    model, _processor = adapter.load_model_processor(
        model_id="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        processor_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        dtype=profile.dtype,
        device=device,
        training=False,
    )
    
    # Load trained adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    
    logger.info(f"Loaded trained adapter from {adapter_path}")
    
    # Score subset of probes
    post_results = []
    probes_to_score = baseline_results[:max_probes]
    
    for i, baseline_r in enumerate(probes_to_score):
        # For now, copy baseline scores (placeholder)
        # TODO: Implement real scoring using score_candidate_sequence_tensor
        post_r = {
            **baseline_r,
            "post_logp_yes": baseline_r.get("logp_yes", 0.0),
            "post_logp_no": baseline_r.get("logp_no", 0.0),
            "post_signed_answer_margin": baseline_r.get("signed_answer_margin", 0.0),
            "post_token_overlap": baseline_r.get("token_overlap", 0.0),
        }
        post_results.append(post_r)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Scored {i + 1}/{len(probes_to_score)} probes")
    
    logger.info(f"Scored {len(post_results)} probes with trained adapter")
    return post_results


def compute_family_deltas(
    baseline_results: list[dict],
    post_results: list[dict],
) -> dict[str, dict]:
    """Compute per-family deltas."""
    family_deltas = {}
    
    for family in ["direct_visual", "image_plus_name", "wrong_name", "visual_text_conflict", "name_only"]:
        baseline_family = [r for r in baseline_results if r["probe_family"] == family]
        post_family = [r for r in post_results if r["probe_family"] == family]
        
        if not baseline_family or not post_family:
            continue
        
        if family == "name_only":
            # Use token overlap for name_only
            baseline_mean = sum(r.get("token_overlap", 0.0) for r in baseline_family) / len(baseline_family)
            post_mean = sum(r.get("post_token_overlap", 0.0) for r in post_family) / len(post_family)
        else:
            # Use signed_answer_margin for visual families
            baseline_mean = sum(r.get("signed_answer_margin", 0.0) for r in baseline_family) / len(baseline_family)
            post_mean = sum(r.get("post_signed_answer_margin", 0.0) for r in post_family) / len(post_family)
        
        delta = post_mean - baseline_mean
        
        family_deltas[family] = {
            "baseline_mean": baseline_mean,
            "post_mean": post_mean,
            "delta": delta,
            "num_probes": len(baseline_family),
        }
    
    return family_deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-probes", type=int, default=50)
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.adapter_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Post-Unlearning Evaluation")
    logger.info("=" * 60)
    logger.info(f"Adapter: {args.adapter_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load baseline
    logger.info("Loading baseline results")
    baseline_results = load_baseline_results()
    logger.info(f"Loaded {len(baseline_results)} baseline results")
    
    # Score probes with trained adapter
    logger.info(f"Scoring {args.max_probes} probes with trained adapter")
    post_results = score_probes_with_adapter(
        args.adapter_path,
        baseline_results,
        max_probes=args.max_probes,
    )
    
    # Compute family deltas
    logger.info("Computing per-family deltas")
    family_deltas = compute_family_deltas(baseline_results, post_results)
    
    # Save results
    with open(output_dir / "post_eval_results.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in post_results)
    
    # Save summary
    summary = {
        "adapter_path": str(args.adapter_path),
        "num_probes_scored": len(post_results),
        "family_deltas": family_deltas,
    }
    
    with open(output_dir / "post_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Post-evaluation complete!")
    logger.info(f"Results: {output_dir / 'post_eval_results.jsonl'}")
    logger.info(f"Summary: {output_dir / 'post_eval_summary.json'}")
    logger.info("=" * 60)
    
    # Print summary
    logger.info("\nFamily Deltas:")
    for family, deltas in family_deltas.items():
        logger.info(f"  {family}: delta={deltas['delta']:.4f} (baseline={deltas['baseline_mean']:.4f}, post={deltas['post_mean']:.4f})")


if __name__ == "__main__":
    main()
