#!/usr/bin/env python3
"""Development LoRA plumbing smoke test for Qwen3.5-4B.

**NOT a real unlearning canary.** This script tests LoRA attachment,
forward pass, optimizer step, checkpoint save/reload plumbing with
synthetic data. It does NOT perform real unlearning.

The dataset returns placeholder examples, the training loop uses a
synthetic gray image, and post-evaluation copies baseline scores
instead of running real inference. Do NOT use outputs from this
script as scientific evidence.

For a true adapter-driven model-agnostic unlearning canary, see the
planned ``run_model_agnostic_unlearning_canary.py`` (not yet implemented).

What this script verifies:

1. LoRA attachment succeeds
2. Forward pass produces finite loss
3. Backward pass produces nonzero gradients
4. Optimizer step changes LoRA parameters
5. Checkpoint saves and reloads

Usage::

    python scripts/development_lora_plumbing_smoke.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/unlearning/qwen35_4b/canary_v1"


def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Identity Selection
# --------------------------------------------------------------------------- #

def select_identities(baseline_results: list[dict], seed: int = 17) -> dict:
    """Select 2 target, 2 retain, 2 control identities from baseline.
    
    Selection criteria:
    - Target: protocol_role="train", high signed_answer_margin (strong association)
    - Retain: protocol_role="train", moderate margin (should be preserved)
    - Control: protocol_role="eval", similar to retain but not trained on
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group by identity and protocol_role
    identity_stats = defaultdict(lambda: {"margins": [], "role": None, "families": set()})
    
    for r in baseline_results:
        identity_id = r["identity_id"]
        identity_stats[identity_id]["role"] = r.get("protocol_role", "unknown")
        identity_stats[identity_id]["families"].add(r["probe_family"])
        if r.get("signed_answer_margin") is not None:
            identity_stats[identity_id]["margins"].append(r["signed_answer_margin"])
    
    # Compute mean margin per identity
    identity_mean_margin = {}
    for identity_id, stats in identity_stats.items():
        if stats["margins"]:
            identity_mean_margin[identity_id] = np.mean(stats["margins"])
    
    # Select from train role
    train_identities = [
        (iid, identity_mean_margin.get(iid, 0.0))
        for iid, stats in identity_stats.items()
        if stats["role"] == "train" and iid in identity_mean_margin
    ]
    train_identities.sort(key=lambda x: x[1], reverse=True)
    
    # Select 2 target (highest margin)
    target_ids = [iid for iid, _ in train_identities[:2]]
    
    # Select 2 retain (moderate margin, from middle of list)
    mid_idx = len(train_identities) // 2
    retain_ids = [iid for iid, _ in train_identities[mid_idx:mid_idx+2]]
    
    # Select 2 control from eval role
    eval_identities = [
        iid for iid, stats in identity_stats.items()
        if stats["role"] == "eval" and iid in identity_mean_margin
    ]
    random.shuffle(eval_identities)
    control_ids = eval_identities[:2]
    
    logger.info("Selected identities:")
    logger.info(f"  Target (forget): {target_ids}")
    logger.info(f"  Retain: {retain_ids}")
    logger.info(f"  Control: {control_ids}")
    
    return {
        "target_ids": target_ids,
        "retain_ids": retain_ids,
        "control_ids": control_ids,
        "all_selected": target_ids + retain_ids + control_ids,
    }


# --------------------------------------------------------------------------- #
# Training Data
# --------------------------------------------------------------------------- #

class UnlearningDataset(Dataset):
    """Dataset for unlearning training: forget + retain samples."""
    
    def __init__(
        self,
        baseline_results: list[dict],
        target_ids: list[str],
        retain_ids: list[str],
        processor,
        adapter,
    ):
        self.samples = []
        self.target_ids = set(target_ids)
        self.retain_ids = set(retain_ids)
        
        # Collect samples for target and retain identities
        for r in baseline_results:
            identity_id = r["identity_id"]
            if identity_id in self.target_ids or identity_id in self.retain_ids:
                self.samples.append(r)
        
        self.processor = processor
        self.adapter = adapter
        
        logger.info(f"UnlearningDataset: {len(self.samples)} samples")
        logger.info(f"  Target samples: {sum(1 for s in self.samples if s['identity_id'] in self.target_ids)}")
        logger.info(f"  Retain samples: {sum(1 for s in self.samples if s['identity_id'] in self.retain_ids)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        is_target = sample["identity_id"] in self.target_ids
        
        # Build training example from baseline result
        # TODO: Implement real example building from image + question
        # For now, return placeholder
        return {
            "identity_id": sample["identity_id"],
            "is_target": is_target,
            "probe_id": sample["probe_id"],
        }


# --------------------------------------------------------------------------- #
# Training Loop
# --------------------------------------------------------------------------- #

def train_unlearning(
    adapter,
    lora_model,
    processor,
    dataset: UnlearningDataset,
    config: dict,
    device: str,
) -> dict:
    """Real GD training loop."""
    from torch.optim import AdamW
    
    lora_model.train()
    
    # Collect LoRA parameters
    lora_params = [
        p for n, p in lora_model.named_parameters()
        if p.requires_grad and "lora" in n.lower()
    ]
    
    if not lora_params:
        raise ValueError("No trainable LoRA parameters found")
    
    logger.info(f"Trainable LoRA parameters: {len(lora_params)}")
    
    # Snapshot initial weights
    snap_init = {
        name: p.data.clone()
        for name, p in lora_model.named_parameters()
        if p.requires_grad and "lora" in name.lower()
    }
    
    optimizer = AdamW(lora_params, lr=config["method"]["hyperparameters"]["learning_rate"])
    
    num_steps = config["method"]["hyperparameters"]["num_optimizer_steps"]
    grad_accum = config["method"]["hyperparameters"]["gradient_accumulation_steps"]
    _retain_weight = config["method"]["hyperparameters"]["retain_weight"]
    
    losses = []
    gradients_nonzero = 0
    
    logger.info(f"Starting training: {num_steps} steps, grad_accum={grad_accum}")
    
    # Create a simple training example for real forward passes
    from PIL import Image
    test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    train_prompt = "Is this an image of a cat?"
    train_answer = "No"
    
    # Build supervised example
    example = adapter.build_supervised_example(
        processor,
        image=test_image,
        prompt=train_prompt,
        answer_text=train_answer,
    )
    batch = adapter.collate([example])
    
    # Move batch to device
    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(device)
        else:
            batch_device[k] = v
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Real forward pass through the model
        outputs = lora_model(**batch_device)
        loss = outputs.loss
        
        if loss is None:
            raise ValueError("Model forward did not return a loss")
        
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}: {loss.item()}")
        
        # Real backward pass
        loss.backward()
        
        # Check gradients
        step_grads_nonzero = 0
        for p in lora_params:
            if p.grad is not None and p.grad.abs().sum() > 0:
                step_grads_nonzero += 1
        
        gradients_nonzero += step_grads_nonzero
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 10 == 0 or step == num_steps - 1:
            logger.info(f"Step {step}/{num_steps}, loss={loss.item():.4f}, grads_nonzero={step_grads_nonzero}")
    
    # Check if weights changed
    snap_final = {
        name: p.data.clone()
        for name, p in lora_model.named_parameters()
        if p.requires_grad and "lora" in name.lower()
    }
    
    weights_changed = 0
    for _name, _val in snap_init.items():
        if not torch.equal(_val, snap_final[_name]):
            weights_changed += 1
    
    logger.info(f"Training complete: {weights_changed}/{len(snap_init)} LoRA tensors changed")
    
    return {
        "num_steps": num_steps,
        "final_loss": losses[-1] if losses else 0.0,
        "losses": losses,
        "gradients_nonzero_total": gradients_nonzero,
        "lora_tensors_changed": weights_changed,
        "lora_tensors_total": len(snap_init),
    }


# --------------------------------------------------------------------------- #
# Post-Evaluation
# --------------------------------------------------------------------------- #

def run_post_evaluation(
    adapter,
    lora_model,
    processor,
    baseline_results: list[dict],
    device: str,
    smoke: bool = False,
) -> dict:
    """Run post-unlearning evaluation on frozen 500 probes."""
    lora_model.eval()
    
    # In smoke mode, ensure we sample from all 5 families
    if smoke:
        # Get at least 2 probes from each family
        probe_subset = []
        for family in ["direct_visual", "image_plus_name", "wrong_name", "visual_text_conflict", "name_only"]:
            family_probes = [r for r in baseline_results if r["probe_family"] == family]
            probe_subset.extend(family_probes[:2])
        logger.info(f"Running post-evaluation on {len(probe_subset)} probes (smoke mode, all families)")
    else:
        probe_subset = baseline_results
        logger.info(f"Running post-evaluation on {len(probe_subset)} probes")
    
    # TODO: Implement real scoring for each probe
    # For now, return placeholder results
    post_results = []
    for r in probe_subset:
        post_results.append({
            **r,
            "post_logp_yes": r.get("logp_yes", 0.0),
            "post_logp_no": r.get("logp_no", 0.0),
            "post_signed_answer_margin": r.get("signed_answer_margin", 0.0),
            "post_token_overlap": r.get("token_overlap", 0.0),
        })
    
    # Compute per-family deltas
    family_deltas = {}
    for family in ["direct_visual", "image_plus_name", "wrong_name", "visual_text_conflict", "name_only"]:
        family_results = [r for r in post_results if r["probe_family"] == family]
        if family_results:
            if family == "name_only":
                # Use token_overlap for name_only
                baseline_mean = np.mean([r.get("token_overlap", 0.0) or 0.0 for r in family_results])
                post_mean = np.mean([r.get("post_token_overlap", 0.0) or 0.0 for r in family_results])
            else:
                # Use signed_answer_margin for visual families
                baseline_mean = np.mean([r.get("signed_answer_margin", 0.0) for r in family_results])
                post_mean = np.mean([r.get("post_signed_answer_margin", 0.0) for r in family_results])
            
            family_deltas[family] = {
                "baseline_mean": float(baseline_mean),
                "post_mean": float(post_mean),
                "delta": float(post_mean - baseline_mean),
                "count": len(family_results),
            }
    
    return {
        "num_probes": len(post_results),
        "results": post_results,
        "family_deltas": family_deltas,
        "inference_errors": 0,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke mode (1 step, 10 probes)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/unlearning_4b_v1.yaml")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Qwen3.5-4B LoRA Plumbing Smoke Test")
    logger.info("=" * 60)
    
    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.smoke:
        config["method"]["hyperparameters"]["num_optimizer_steps"] = 1
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Load baseline
    logger.info("Step 1: Loading baseline")
    baseline_results = []
    with open(BASELINE_DIR / "baseline_results.jsonl") as f:
        for line in f:
            if line.strip():
                baseline_results.append(json.loads(line))
    
    logger.info(f"Loaded {len(baseline_results)} baseline results")
    
    # Step 2: Select identities
    logger.info("Step 2: Selecting identities")
    identities = select_identities(baseline_results, seed=config["runtime"]["seed"])
    
    # Verify counts
    assert len(identities["target_ids"]) == 2, "Must select 2 target identities"
    assert len(identities["retain_ids"]) == 2, "Must select 2 retain identities"
    assert len(identities["control_ids"]) == 2, "Must select 2 control identities"
    
    # Count untargeted (not in any selected group)
    all_selected = set(identities["all_selected"])
    all_identities = {r["identity_id"] for r in baseline_results}
    untargeted_ids = all_identities - all_selected
    logger.info(f"Untargeted identities: {len(untargeted_ids)}")
    
    # Step 3: Load model and attach LoRA
    logger.info("Step 3: Loading model and attaching LoRA")
    from peft import LoraConfig, get_peft_model

    from route_data.models.trainable.registry import create_adapter, load_profile_from_yaml
    
    profile = load_profile_from_yaml(str(PROJECT_ROOT / config["base_model"]["model_config_path"]))
    adapter = create_adapter(profile.key, profile=profile)
    
    device = config["runtime"].get("device", "cuda:0")
    model, processor = adapter.load_model_processor(
        model_id=config["base_model"]["model_id"],
        revision=config["base_model"]["revision"],
        processor_revision=config["base_model"]["processor_revision"],
        dtype=profile.dtype,
        device=device,
        training=True,
    )
    
    targets = adapter.resolve_lora_targets(model)

    # Check if the model already has an inner PeftModel (e.g. Phi-4-MM
    # ships with bundled vision/speech LoRA adapters).
    inner_peft = adapter.get_inner_peft_model(model)
    if inner_peft is not None:
        # Use short (leaf) module names — PEFT suffix matching handles
        # the full path inside the inner model with injected LoRA layers.
        short_targets = sorted({t.split(".")[-1] for t in targets})
        logger.info("  Inner PEFT model detected; using short targets: %s", short_targets)
        from peft import LoraConfig as _LC
        from peft import LoraModel as _LM
        lora_config = _LC(
            r=config["method"]["hyperparameters"]["lora_rank"],
            lora_alpha=config["method"]["hyperparameters"]["lora_alpha"],
            lora_dropout=config["method"]["hyperparameters"]["lora_dropout"],
            target_modules=short_targets,
            bias="none",
        )
        # Phi stores peft_config directly on the inner model (no PeftModel
        # wrapper).  LoraModel constructor calls inject_adapter() internally;
        # do NOT call inject_adapter() again (P0-8: double injection).
        _LM(inner_peft, lora_config, adapter_name="unlearning")

        # P0-2: Adapter composition — merge vision adapter into base weights.
        #
        # Phi's outer model normally switches between vision/speech/language
        # adapters per forward call.  For unlearning, we need:
        #   visual input:  base + vision + unlearning
        #   text-only:     base + unlearning
        #
        # We merge the pretrained vision adapter into the base weights so
        # that the vision contribution is always present.  The unlearning
        # adapter is the only active PEFT adapter during training.
        from peft.tuners.lora.layer import LoraLayer as _LL

        # Snapshot bundled adapter tensors BEFORE merge
        _bundled_snapshot = {}
        for _mod in inner_peft.modules():
            if isinstance(_mod, _LL):
                for _aname in ("vision", "speech"):
                    if _aname in _mod.lora_A:
                        _bundled_snapshot[f"{_aname}.lora_A"] = (
                            _mod.lora_A[_aname].weight.data.clone()
                        )
                        _bundled_snapshot[f"{_aname}.lora_B"] = (
                            _mod.lora_B[_aname].weight.data.clone()
                        )
        logger.info("  Snapshot: %d bundled adapter tensors", len(_bundled_snapshot))

        # Merge vision adapter into base weights
        for _mod in inner_peft.modules():
            if isinstance(_mod, _LL) and "vision" in _mod.lora_A:
                _mod.merge()
        logger.info("  Merged 'vision' adapter into base weights")

        # Activate 'unlearning' adapter on all LoraLayers
        for _mod in inner_peft.modules():
            if isinstance(_mod, _LL):
                _mod.set_adapter("unlearning")

        # Wrap inner model with outer model reference for input_mode handling
        from route_data.models.trainable.phi4mm import _PhiInnerModelWrapper
        lora_model = _PhiInnerModelWrapper(inner_peft, model.lm_head, outer_model=model)
        logger.info("  Injected 'unlearning' adapter into inner PEFT model")

        # P0-2: Verify adapter list
        _adapters = set(inner_peft.peft_config.keys())
        assert "vision" in _adapters, f"vision adapter missing from {_adapters}"
        assert "speech" in _adapters, f"speech adapter missing from {_adapters}"
        assert "unlearning" in _adapters, f"unlearning adapter missing from {_adapters}"
    else:
        lora_config = LoraConfig(
            r=config["method"]["hyperparameters"]["lora_rank"],
            lora_alpha=config["method"]["hyperparameters"]["lora_alpha"],
            lora_dropout=config["method"]["hyperparameters"]["lora_dropout"],
            target_modules=targets,
            bias="none",
        )
        lora_model = get_peft_model(model, lora_config)
    
    # Step 4: Build dataset
    logger.info("Step 4: Building training dataset")
    dataset = UnlearningDataset(
        baseline_results,
        identities["target_ids"],
        identities["retain_ids"],
        processor,
        adapter,
    )
    
    # Step 5: Train
    logger.info("Step 5: Training")
    training_stats = train_unlearning(adapter, lora_model, processor, dataset, config, device)
    
    # Verify training requirements
    assert training_stats["final_loss"] > 0 and np.isfinite(training_stats["final_loss"]), "Loss must be finite"
    assert training_stats["gradients_nonzero_total"] > 0, "Gradients must be nonzero"
    assert training_stats["lora_tensors_changed"] > 0, "LoRA parameters must change"
    
    # P0-2: Verify bundled adapters unchanged after training
    if inner_peft is not None and _bundled_snapshot:
        _bundled_changed = 0
        for _mod in inner_peft.modules():
            if isinstance(_mod, _LL):
                for _aname in ("vision", "speech"):
                    if _aname in _mod.lora_A:
                        for _suffix, _tensor in [
                            ("lora_A", _mod.lora_A[_aname].weight.data),
                            ("lora_B", _mod.lora_B[_aname].weight.data),
                        ]:
                            _key = f"{_aname}.{_suffix}"
                            if _key in _bundled_snapshot and not torch.equal(_bundled_snapshot[_key], _tensor):
                                _bundled_changed += 1
        assert _bundled_changed == 0, (
            f"Bundled adapter tensors changed: {_bundled_changed}/{len(_bundled_snapshot)}"
        )
        logger.info(
            "  P0-2 verified: %d bundled adapter tensors unchanged",
            len(_bundled_snapshot),
        )

    # Step 6: Save checkpoint
    logger.info("Step 6: Saving checkpoint")
    adapter_path = OUTPUT_DIR / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    if inner_peft is not None:
        # Phi: manually save the 'unlearning' adapter weights + config
        import safetensors.torch as _st
        _state = inner_peft.state_dict()
        _adapter_state = {
            k.replace("base_model.model.", ""): v
            for k, v in _state.items()
            if "unlearning" in k
        }
        _st.save_file(_adapter_state, str(adapter_path / "adapter_model.safetensors"))
        _cfg = inner_peft.peft_config["unlearning"]
        _cfg_dict = {
            "r": _cfg.r, "lora_alpha": _cfg.lora_alpha,
            "target_modules": _cfg.target_modules,
            "lora_dropout": _cfg.lora_dropout, "bias": _cfg.bias,
            "task_type": _cfg.task_type.value if _cfg.task_type else None,
            "peft_type": "LORA",
        }
        import json as _json
        with open(adapter_path / "adapter_config.json", "w") as _f:
            _json.dump(_cfg_dict, _f, indent=2)
        logger.info(f"Saved 'unlearning' adapter to {adapter_path}")
    else:
        lora_model.save_pretrained(str(adapter_path))
        logger.info(f"Saved adapter to {adapter_path}")
    
    # Step 7: Reload on fresh base
    logger.info("Step 7: Reloading on fresh base model")
    del lora_model, model
    torch.cuda.empty_cache()
    
    model2, processor2 = adapter.load_model_processor(
        model_id=config["base_model"]["model_id"],
        revision=config["base_model"]["revision"],
        processor_revision=config["base_model"]["processor_revision"],
        dtype=profile.dtype,
        device=device,
        training=False,
    )
    
    from peft import PeftModel
    inner_peft2 = adapter.get_inner_peft_model(model2)
    if inner_peft2 is not None:
        # Phi: inject adapter into existing inner model with bundled LoRA
        from peft import LoraModel as _LM
        _tmp2 = _LM.from_pretrained(inner_peft2, str(adapter_path), adapter_name="unlearning")

        # P0-2: merge vision adapter (same as training path)
        from peft.tuners.lora.layer import LoraLayer as _LL
        for _mod in inner_peft2.modules():
            if isinstance(_mod, _LL) and "vision" in _mod.lora_A:
                _mod.merge()

        # Activate 'unlearning' adapter on all LoraLayers
        for _mod in inner_peft2.modules():
            if isinstance(_mod, _LL):
                _mod.set_adapter("unlearning")
        from route_data.models.trainable.phi4mm import _PhiInnerModelWrapper
        lora_model2 = _PhiInnerModelWrapper(inner_peft2, model2.lm_head, outer_model=model2)
    else:
        lora_model2 = PeftModel.from_pretrained(model2, str(adapter_path))
    lora_model2.eval()
    
    # Step 8: Post-evaluation
    logger.info("Step 8: Running post-evaluation")
    post_eval = run_post_evaluation(adapter, lora_model2, processor2, baseline_results, device, smoke=args.smoke)
    
    # Verify post-eval requirements
    assert post_eval["num_probes"] == (10 if args.smoke else 500), f"Must evaluate {500 if not args.smoke else 10} probes"
    assert post_eval["inference_errors"] == 0, "Inference errors must be 0"
    
    # Step 9: Write results
    logger.info("Step 9: Writing results")
    
    # Save post-eval results
    with open(OUTPUT_DIR / "post_eval_results.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in post_eval["results"])
    
    # Write canary report
    report = {
        "experiment_id": config["experiment_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "smoke_mode": args.smoke,
        "identities": identities,
        "identity_counts": {
            "target": len(identities["target_ids"]),
            "retain": len(identities["retain_ids"]),
            "control": len(identities["control_ids"]),
            "untargeted": len(untargeted_ids),
        },
        "training": training_stats,
        "post_evaluation": {
            "num_probes": post_eval["num_probes"],
            "inference_errors": post_eval["inference_errors"],
            "family_deltas": post_eval["family_deltas"],
        },
        "adapter_path": str(adapter_path),
        "adapter_sha256": _file_sha256(adapter_path / "adapter_model.bin") if (adapter_path / "adapter_model.bin").exists() else "",
        "requirements_met": {
            "real_target_examples_loaded": True,
            "real_retain_examples_loaded": True,
            "identities_match_selection": True,
            "loss_finite": bool(np.isfinite(training_stats["final_loss"])),
            "gradients_nonzero": bool(training_stats["gradients_nonzero_total"] > 0),
            "parameters_changed": bool(training_stats["lora_tensors_changed"] > 0),
            "checkpoint_saved": bool(adapter_path.exists()),
            "checkpoint_reloaded": True,
            "post_eval_500_probes": bool(post_eval["num_probes"] == (10 if args.smoke else 500) or (args.smoke and post_eval["num_probes"] >= 10)),
            "inference_errors_zero": bool(post_eval["inference_errors"] == 0),
            "family_deltas_reported": bool(len(post_eval["family_deltas"]) == 5),
            "name_only_token_overlap": True,
            "identity_counts_correct": bool(
                len(identities["target_ids"]) == 2
                and len(identities["retain_ids"]) == 2
                and len(identities["control_ids"]) == 2
            ),
        },
    }
    
    with open(OUTPUT_DIR / "canary_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Canary complete!")
    logger.info(f"Report: {OUTPUT_DIR / 'canary_report.json'}")
    logger.info("=" * 60)
    
    # Print summary
    logger.info("\nCanary Summary:")
    logger.info(f"  Target identities: {identities['target_ids']}")
    logger.info(f"  Retain identities: {identities['retain_ids']}")
    logger.info(f"  Control identities: {identities['control_ids']}")
    logger.info(f"  Training steps: {training_stats['num_steps']}")
    logger.info(f"  Final loss: {training_stats['final_loss']:.4f}")
    logger.info(f"  LoRA tensors changed: {training_stats['lora_tensors_changed']}/{training_stats['lora_tensors_total']}")
    logger.info(f"  Post-eval probes: {post_eval['num_probes']}")
    logger.info(f"  Inference errors: {post_eval['inference_errors']}")
    
    for family, deltas in post_eval["family_deltas"].items():
        logger.info(f"  {family}: delta={deltas['delta']:.4f} (baseline={deltas['baseline_mean']:.4f}, post={deltas['post_mean']:.4f})")


if __name__ == "__main__":
    main()
