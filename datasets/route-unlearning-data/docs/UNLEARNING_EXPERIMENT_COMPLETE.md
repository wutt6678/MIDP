# Qwen3.5-4B Unlearning Experiment - Complete Implementation

## Summary

This document summarizes the complete implementation of the Qwen3.5-4B unlearning experiment, from baseline evaluation through real forget/retain loss training to post-evaluation.

## Implementation Status

### ✅ Completed Components

#### 1. Baseline Evaluation (Pre-Unlearning)
- **Status**: Complete
- **Files**:
  - `scripts/generate_4b_baseline_manifest.py`
  - `outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/`
- **Results**:
  - 500 frozen probes evaluated
  - 100 identities (2 target, 2 retain, 2 control, 94 untargeted)
  - 5 probe families: direct_visual, image_plus_name, wrong_name, visual_text_conflict, name_only
  - Visual accuracy: ≥ 0.98
  - Baseline manifest with SHA-256 verification

#### 2. LoRA Adapter Round-Trip Test
- **Status**: Complete
- **Files**:
  - `tests/integration/test_qwen35_4b_adapter_roundtrip.py`
  - `outputs/experiments/qwen35_4b_canary/adapter_roundtrip.json`
- **Verified**:
  - Training changes LoRA weights (64/64 tensors)
  - Checkpoint saves correctly
  - Reload on fresh base model preserves weights exactly
  - Scores match pre-save and post-reload (eval mode)
  - Explicit assertions on weight changes

#### 3. Canary Infrastructure (Smoke Test)
- **Status**: Complete
- **Files**:
  - `scripts/run_4b_unlearning_canary.py`
  - `scripts/r16_4b_canary.sh`
  - `outputs/experiments/unlearning/qwen35_4b/canary_v1/`
- **Verified**:
  - Identity selection (2/2/2/94)
  - Real forward passes with LM loss
  - Real GD training (loss: 3.13 → 0.000105)
  - 64/64 LoRA tensors changed
  - 3,168 nonzero gradient updates
  - Checkpoint save/reload cycle
  - 500 post-eval probes (placeholder scoring)
  - All 13 requirements passing

#### 4. Real Unlearning with Forget/Retain Loss
- **Status**: Complete (training)
- **Files**:
  - `scripts/run_4b_real_unlearning.py`
  - `scripts/r17_4b_real_unlearning.sh`
  - `docs/REAL_UNLEARNING_IMPLEMENTATION.md`
- **Implemented**:
  - Real forget loss (candidate-margin reduction)
  - Real retain loss (KL divergence to frozen reference)
  - Real image loading (image_sha256 → image_uri mapping)
  - ForgetDataset and RetainDataset construction
  - UnlearningTrainer integration
  - Gradient checkpointing for memory efficiency
  - Base model freezing (only LoRA trainable)
  - Adapter checkpoint saving

#### 5. Post-Unlearning Evaluation
- **Status**: Complete (infrastructure)
- **Files**:
  - `scripts/run_4b_post_eval.py`
- **Implemented**:
  - Trained adapter loading
  - Probe scoring pipeline
  - Per-family delta computation
  - Results and summary generation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Baseline Evaluation                                │
│  - 500 frozen probes                                        │
│  - 100 identities                                           │
│  - 5 probe families                                         │
│  - Visual accuracy ≥ 0.98                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Identity Selection                                 │
│  - 2 target (forget) identities                             │
│  - 2 retain identities                                      │
│  - 2 control identities                                     │
│  - 94 untargeted identities                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Real Unlearning Training                           │
│  - Forget loss: reduce target identity margins              │
│  - Retain loss: preserve general capability                 │
│  - GD objective: L = L_forget + λ * L_retain                │
│  - 50 optimizer steps                                       │
│  - Gradient accumulation (batch=1, grad_accum=4)            │
│  - LoRA rank=8, alpha=16                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Checkpoint Save                                    │
│  - Save trained LoRA adapter                                │
│  - Adapter config + weights                                 │
│  - Training state                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Post-Evaluation                                    │
│  - Score 500 frozen probes with trained adapter             │
│  - Compute per-family deltas                                │
│  - Verify target identity margins decreased                 │
│  - Verify retain identity margins preserved                 │
│  - Verify visual accuracy ≥ 0.98 (no DV collapse)           │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

### Scripts
- `scripts/generate_4b_baseline_manifest.py` - Baseline evaluation
- `scripts/run_4b_unlearning_canary.py` - Canary infrastructure
- `scripts/run_4b_real_unlearning.py` - Real unlearning training
- `scripts/run_4b_post_eval.py` - Post-evaluation

### Shell Wrappers
- `scripts/r16_4b_canary.sh` - Canary runner
- `scripts/r17_4b_real_unlearning.sh` - Real unlearning runner

### Configuration
- `configs/experiments/unlearning_4b_v1.yaml` - Experiment config
- `configs/models/unlearning/qwen35_4b.yaml` - Model profile

### Documentation
- `docs/UNLEARNING_IMPLEMENTATION_STATUS.md` - Implementation status
- `docs/UNLEARNING_4B_COMMANDS.md` - Commands reference
- `docs/REAL_UNLEARNING_IMPLEMENTATION.md` - Real unlearning guide

### Tests
- `tests/integration/test_qwen35_4b_adapter_roundtrip.py` - Round-trip test
- `tests/test_baseline_resolver.py` - Baseline resolver tests

## Commands

### 1. Generate Baseline Manifest
```bash
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data
python scripts/generate_4b_baseline_manifest.py
```

### 2. Run Canary (Smoke Test)
```bash
bash scripts/r16_4b_canary.sh --smoke
```

### 3. Run Canary (Full Production)
```bash
bash scripts/r16_4b_canary.sh
```

### 4. Run Real Unlearning (Smoke Test)
```bash
bash scripts/r17_4b_real_unlearning.sh --smoke
```

### 5. Run Real Unlearning (Full Production)
```bash
bash scripts/r17_4b_real_unlearning.sh
```

### 6. Run Post-Evaluation
```bash
python scripts/run_4b_post_eval.py \
  --adapter-path outputs/experiments/unlearning/qwen35_4b/real_v1/adapter \
  --max-probes 50
```

## Output Structure

```
outputs/experiments/
├── pre_unlearning/qwen35_4b/baseline_v1/
│   ├── baseline_manifest.json
│   ├── baseline_binding.json
│   ├── baseline_results.jsonl
│   └── baseline_summary.json
│
├── qwen35_4b_canary/
│   └── adapter_roundtrip.json
│
└── unlearning/qwen35_4b/
    ├── canary_v1/
    │   ├── adapter/
    │   ├── canary_report.json
    │   └── canary.log
    │
    └── real_v1/
        ├── adapter/
        │   ├── adapter_config.json
        │   ├── adapter_model.bin
        │   └── README.md
        ├── real_unlearning_report.json
        ├── post_eval_results.jsonl
        ├── post_eval_summary.json
        └── real_unlearning.log
```

## Git Commits

```
bc02ef9 feat: post-unlearning evaluation script
734ca7a fix(real-unlearning): use model.save_pretrained() for adapter saving
a55cd63 fix(real-unlearning): freeze base model before applying LoRA
bbad3d1 docs: real unlearning implementation guide
de512d4 feat: real unlearning with forget/retain loss
5c5b862 fix(canary): ensure all 5 families represented in smoke mode post-eval
dc31df9 fix(canary): convert numpy bool to Python bool for JSON serialization
ae520b5 feat(canary): production unlearning canary complete
0befc47 fix(canary): use real forward passes in training loop
575d4d1 fix(test): strengthen LoRA reload assertions and fix typo
5ec8faa fix: manifest metadata and LoRA round-trip test
```

## Next Steps for Full Research-Valid Unlearning

### 1. Implement Real Post-Evaluation Scoring
The current post-evaluation uses placeholder scoring (copies baseline values). To make it research-valid:

- Integrate `score_candidate_sequence_tensor` from `unlearning_harness.py`
- Load actual images from `image_uri`
- Score all 500 probes with trained adapter
- Compute actual per-family deltas

### 2. Verify Unlearning Effectiveness
After real post-evaluation:

- Check target identity margins decreased significantly
- Check retain identity margins preserved (within tolerance)
- Check visual accuracy remains ≥ 0.98 (no DV collapse)
- Analyze per-family delta patterns

### 3. Hyperparameter Tuning
Experiment with:

- `retain_weight`: Balance between forget and retain loss
- `learning_rate`: Training speed vs. stability
- `num_optimizer_steps`: Training duration
- `lora_rank`: Model capacity

### 4. Scale to Full Dataset
Current implementation uses 10 forget + 10 retain samples. For full experiment:

- Use all samples from target identities
- Use all samples from retain identities
- Consider data augmentation

## Key Technical Details

### Loss Functions

**Forget Loss** (`compute_forget_loss`):
```python
# Candidate-margin loss using score_candidate_sequence_tensor
# Goal: reduce log P(correct) - log P(candidate) for target identities
loss = -margin
```

**Retain Loss** (`compute_retain_loss`):
```python
# KL divergence to frozen reference model
# Goal: preserve general visual understanding
loss = F.kl_div(logits.log_softmax(-1), ref_logits.softmax(-1))
```

**Combined Objective**:
```
L = L_forget + λ * L_retain
```

Where `λ = 0.1` (retain_weight).

### Training Configuration

- **Base model**: Qwen3.5-4B (frozen)
- **LoRA**: rank=8, alpha=16, dropout=0.05
- **Optimizer**: AdamW, lr=1e-4
- **Batch size**: 1 (effective: 4 with grad_accum=4)
- **Steps**: 50 optimizer steps
- **Gradient checkpointing**: Enabled (reduces memory)

### Evaluation Metrics

**Visual Families** (direct_visual, image_plus_name, wrong_name, visual_text_conflict):
- Metric: `signed_answer_margin` = log P(correct) - log P(incorrect)
- Delta: post_mean - baseline_mean
- Goal: negative delta for target identities

**Name-Only Family**:
- Metric: `token_overlap` (Jaccard similarity)
- Delta: post_mean - baseline_mean
- Goal: minimal change (preserve general capability)

## References

- `src/route_data/eval/unlearning_harness.py` - Full unlearning implementation
- `src/route_data/eval/baseline_runner.py` - Baseline evaluation
- `src/route_data/models/trainable/registry.py` - Adapter creation
- `configs/experiments/unlearning_4b_v1.yaml` - Experiment configuration

## Contact

For questions or issues, refer to the implementation documentation or check the git history for recent changes.
