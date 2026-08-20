# Qwen3.5-4B Unlearning Implementation Status

## What's Complete

### Baseline Phase ✓
- **500-probe baseline evaluation** completed
- **128-token name_only budget** implemented and verified
- **Family-specific cache keys** for selective reruns
- **Manifest with binding file** for self-referential hash fix
- **Round-trip LoRA test** with real training and weight verification
- **All 12 resolver tests** passing
- **1282 unit tests** passing

### Unlearning Infrastructure ✓
- **Experiment config**: `configs/experiments/unlearning_4b_v1.yaml`
- **Canary script**: `scripts/run_4b_unlearning_canary.py` (567 lines)
- **Shell wrapper**: `scripts/r16_4b_canary.sh`
- **Identity selection**: Real selection from baseline results (2 target, 2 retain, 2 control)
- **Dataset structure**: UnlearningDataset with forget/retain samples
- **Training loop skeleton**: Real GD with AdamW, gradient tracking
- **Post-evaluation skeleton**: 500-probe evaluation with per-family deltas
- **Canary report**: All 15 requirements tracked

## What Needs Implementation

### 1. Real Training Loss Functions

The canary has placeholder loss computation. To make it real, integrate the existing loss functions from `src/route_data/eval/unlearning_harness.py`:

```python
from route_data.eval.unlearning_harness import (
    compute_forget_loss,
    compute_retain_loss,
    ForgetDataset,
    RetainDataset,
)
```

**Forget Loss** (`compute_forget_loss`):
- Computes candidate margin: `M = logP(correct) - logP(wrong)`
- Uses shared `score_candidate_sequence_tensor` from scoring.py
- Minimizing reduces the margin for target identities
- Requires: prefix extraction, Yes/No token scoring

**Retain Loss** (`compute_retain_loss`):
- KL divergence to frozen reference model on assistant tokens
- Or standard LM loss with masked labels
- Preserves general capability for retain identities

**Total Loss**:
```python
total_loss = forget_loss + retain_weight * retain_loss
```

### 2. Real Dataset Building

Replace placeholder `UnlearningDataset` with real datasets:

```python
# Build forget dataset from target identities
forget_samples = [r for r in baseline_results if r["identity_id"] in target_ids]
forget_dataset = ForgetDataset(forget_samples, processor)

# Build retain dataset from retain identities
retain_samples = [r for r in baseline_results if r["identity_id"] in retain_ids]
retain_dataset = RetainDataset(retain_samples, processor)
```

**Challenge**: Baseline results don't have `image_uri` - they have `image_sha256`. Need to map back to original images in the processed dataset.

### 3. Real Post-Evaluation Scoring

Replace placeholder scoring with real probe evaluation:

```python
from route_data.eval.baseline_runner import BaselineRunner

# Reuse baseline runner for post-eval
runner = BaselineRunner(
    model_config=profile,
    adapter=adapter,
    output_dir=OUTPUT_DIR / "post_eval",
)
post_results = runner.run_all_probes(probes)
```

**Challenge**: Need to load the trained adapter on a fresh base model and run the full 500-probe evaluation using the same infrastructure as the baseline.

### 4. Image Loading

Baseline results reference images by SHA-256. Need to:
1. Load `fiubench_processed.jsonl` which has `image_uri`
2. Map `image_sha256` → `image_uri`
3. Load images via PIL for training

```python
processed_dataset = []
with open("outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl") as f:
    for line in f:
        processed_dataset.append(json.loads(line))

# Build SHA-256 → URI mapping
sha_to_uri = {compute_sha256(item["image_uri"]): item["image_uri"] for item in processed_dataset}
```

## Commands to Run

### Option A: Smoke Canary (Infrastructure Test)

```bash
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data
bash scripts/r16_4b_canary.sh --smoke
```

**What it does**:
- Selects real identities from baseline
- Builds placeholder dataset
- Runs 1 training step with placeholder loss
- Evaluates 10 probes with placeholder scoring
- Verifies all 15 requirements (with placeholder data)

**Expected output**:
```
outputs/experiments/unlearning/qwen35_4b/canary_v1/
├── adapter/                          # Trained LoRA (placeholder)
├── post_eval_results.jsonl           # 10 results (placeholder)
├── canary_report.json                # All requirements met (placeholder)
└── canary.log
```

### Option B: Full Production (Requires Real Loss Implementation)

To run a real unlearning experiment:

1. **Implement real loss functions** in `run_4b_unlearning_canary.py`:
   - Replace placeholder loss with `compute_forget_loss` and `compute_retain_loss`
   - Implement real dataset building with image loading
   - Implement real post-evaluation scoring

2. **Run the canary**:
   ```bash
   bash scripts/r16_4b_canary.sh
   ```

3. **Verify requirements**:
   - Check `canary_report.json` for all 15 requirements
   - Review per-family deltas for DV collapse
   - Verify identity counts (2/2/2/94)

### Option C: Use Existing Unlearning Harness

The existing `UnlearningTrainer` in `unlearning_harness.py` is fully implemented. To use it:

```python
from route_data.eval.unlearning_harness import (
    UnlearningConfig,
    UnlearningTrainer,
    load_base_model,
    apply_lora,
)

# Load config
config = UnlearningConfig(
    model_id="Qwen/Qwen3.5-4B",
    model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    forget_identity_ids=target_ids,
    retain_identity_ids=retain_ids,
    processed_dataset_path="outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl",
    num_optimizer_steps=50,
)

# Load model
model, processor = load_base_model(config)
lora_model = apply_lora(model, config)

# Build datasets
forget_dataset = ForgetDataset(forget_samples, processor)
retain_dataset = RetainDataset(retain_samples, processor)

# Train
trainer = UnlearningTrainer(config, lora_model, processor, forget_dataset, retain_dataset)
trainer.train()
```

## Verification Checklist

Before expanding to full experiment, verify:

- [ ] **Identity selection**: 2 target, 2 retain, 2 control from baseline
- [ ] **Dataset building**: Real images loaded, forget/retain samples correct
- [ ] **Loss computation**: Forget loss reduces margin, retain loss preserves capability
- [ ] **Training**: Loss finite, gradients nonzero, parameters change
- [ ] **Checkpoint**: Adapter saved and reloads on fresh base
- [ ] **Post-eval**: 500/500 probes evaluated, 0 inference errors
- [ ] **Per-family deltas**: DV/IPN/WN/VTC/name_only reported separately
- [ ] **Identity counts**: 2/2/2/94 (target/retain/control/untargeted)
- [ ] **Baseline validation**: Model/revision/profile identity passes
- [ ] **No DV collapse**: Visual accuracy maintained within gate (0.98)

## Next Steps

1. **Decide approach**:
   - **Quick canary**: Run smoke test with placeholder loss to verify infrastructure
   - **Real implementation**: Integrate real loss functions from `unlearning_harness.py`
   - **Use existing harness**: Leverage `UnlearningTrainer` directly

2. **If real implementation**:
   - Implement image loading (SHA-256 → URI mapping)
   - Integrate `compute_forget_loss` and `compute_retain_loss`
   - Implement real post-evaluation using `BaselineRunner`
   - Run full 50-step canary

3. **Verify requirements**:
   - Check all 15 requirements in `canary_report.json`
   - Review per-family deltas
   - Ensure no DV collapse

4. **Document results**:
   - Training loss trajectory
   - Identity-level changes
   - Per-family preservation
   - GO/NO-GO decision

## Files Reference

**Baseline artifacts**:
- `outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_results.jsonl` (500 probes)
- `outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_manifest.json`
- `outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_binding.json`

**Unlearning infrastructure**:
- `scripts/run_4b_unlearning_canary.py` (canary script)
- `scripts/r16_4b_canary.sh` (shell wrapper)
- `configs/experiments/unlearning_4b_v1.yaml` (experiment config)

**Existing unlearning code**:
- `src/route_data/eval/unlearning_harness.py` (full trainer with real loss)
- `src/route_data/eval/baseline_runner.py` (500-probe evaluator)
- `src/route_data/models/scoring.py` (candidate scoring)

**Dataset**:
- `outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_processed.jsonl` (training data)
- `outputs/full_fiubench/Qwen_Qwen3.5-9B/fiubench/fiubench_route_conflict_eval.jsonl` (500 probes)
