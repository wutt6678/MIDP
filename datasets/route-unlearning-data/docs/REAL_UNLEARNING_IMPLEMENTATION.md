# Real Unlearning Implementation

This document describes the research-valid unlearning implementation for Qwen3.5-4B.

## Overview

The real unlearning experiment uses the full `UnlearningTrainer` infrastructure from `unlearning_harness.py` to perform targeted identity unlearning with:

- **Real forget loss**: Candidate-margin reduction using `compute_forget_loss()`
- **Real retain loss**: KL divergence to frozen reference using `compute_retain_loss()`
- **Real image loading**: Maps `image_sha256` → `image_uri` from processed dataset
- **Real datasets**: `ForgetDataset` and `RetainDataset` with actual images
- **Real training**: Full GD optimization with gradient accumulation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  1. Load Baseline Results (500 probes)                  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  2. Select Identities                                    │
│     - 2 target (forget) identities                       │
│     - 2 retain identities                                │
│     - 2 control identities                               │
│     - 94 untargeted identities                           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  3. Load Processed Dataset                               │
│     - Map image_sha256 → image_uri                       │
│     - Load actual images from disk                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  4. Build Training Datasets                              │
│     - ForgetDataset: 10 samples from 2 target identities │
│     - RetainDataset: 10 samples from 2 retain identities │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  5. Load Model + LoRA                                    │
│     - Qwen3.5-4B base model (frozen)                     │
│     - LoRA adapter (trainable)                           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  6. Train with Real Loss                                 │
│     - Forget loss: reduce target identity margins        │
│     - Retain loss: preserve general capability           │
│     - GD objective: L = L_forget + λ * L_retain          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  7. Save Adapter Checkpoint                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  8. Post-Evaluation (TODO)                               │
│     - Score 500 frozen probes with trained adapter       │
│     - Compute per-family deltas                          │
└─────────────────────────────────────────────────────────┘
```

## Loss Functions

### Forget Loss

```python
def compute_forget_loss(model, batch, adapter, candidate_token_ids):
    """Candidate-margin loss to reduce target identity associations."""
    # Score candidate sequence using model
    logits = score_candidate_sequence_tensor(model, batch, candidate_token_ids)
    
    # Compute margin: log P(correct) - log P(candidate)
    # Goal: reduce this margin for target identities
    loss = -margin
    return loss
```

**Purpose**: Reduce the model's confidence in correct answers for target identities.

### Retain Loss

```python
def compute_retain_loss(model, reference_model, batch, adapter):
    """KL divergence to frozen reference model."""
    # Get model logits
    logits = model(**batch).logits
    
    # Get reference logits (frozen)
    with torch.no_grad():
        ref_logits = reference_model(**batch).logits
    
    # KL divergence: preserve general capability
    loss = F.kl_div(logits.log_softmax(-1), ref_logits.softmax(-1))
    return loss
```

**Purpose**: Prevent catastrophic forgetting of general visual understanding.

### Combined Objective

```
L = L_forget + λ * L_retain
```

Where `λ` (retain_weight) controls the trade-off between unlearning and preservation.

## Configuration

```yaml
# configs/experiments/unlearning_4b_v1.yaml

method:
  hyperparameters:
    learning_rate: 1.0e-4
    num_optimizer_steps: 50
    retain_weight: 0.1
    train_batch_size: 1
    gradient_accumulation_steps: 4
    lora_rank: 8
    lora_alpha: 16
    lora_dropout: 0.05
```

## Running the Experiment

### Smoke Test (1 step)

```bash
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data
bash scripts/r17_4b_real_unlearning.sh --smoke
```

**Expected output:**
- 1 training step
- Adapter checkpoint saved
- Report: `outputs/experiments/unlearning/qwen35_4b/real_v1/real_unlearning_report.json`

### Full Production Run (50 steps)

```bash
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data
bash scripts/r17_4b_real_unlearning.sh
```

**Expected output:**
- 50 training steps
- Loss should decrease over time
- Adapter checkpoint saved
- Report with training statistics

## Output Files

```
outputs/experiments/unlearning/qwen35_4b/real_v1/
├── adapter/                          # Trained LoRA checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
├── real_unlearning_report.json       # Training statistics
└── real_unlearning.log               # Execution log
```

## Next Steps

After real training completes:

1. **Implement post-evaluation scoring**:
   - Use `BaselineRunner` to score all 500 probes
   - Compute actual per-family deltas
   - Verify target identity margins decreased
   - Verify retain identity margins preserved

2. **Analyze results**:
   - Check loss curves
   - Verify gradient flow
   - Check for DV collapse (visual accuracy should remain ≥ 0.98)

3. **Commit results**:
   - Add adapter checkpoint to git
   - Commit report with training evidence

## Status

✅ **Implemented:**
- Identity selection (2/2/2/94)
- Image loading (image_sha256 → image_uri)
- ForgetDataset and RetainDataset construction
- UnlearningTrainer integration
- Real forget/retain loss
- Adapter checkpoint saving

⏳ **TODO:**
- Real post-evaluation scoring (currently placeholder)
- Per-family delta computation
- DV collapse verification

## References

- `src/route_data/eval/unlearning_harness.py`: Full implementation
- `src/route_data/models/trainable/registry.py`: Adapter creation
- `configs/experiments/unlearning_4b_v1.yaml`: Experiment configuration
