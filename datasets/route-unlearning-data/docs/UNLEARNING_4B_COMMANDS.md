# Qwen3.5-4B Unlearning Experiment

This document describes the commands to run the Qwen3.5-4B unlearning experiment.

## Prerequisites

1. **Baseline completed**: The pre-unlearning baseline must be complete:
   ```
   outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/
   ├── baseline_results.jsonl
   ├── baseline_manifest.json
   └── baseline_binding.json
   ```

2. **Conda environment**: `midp-qwen35` must be available:
   ```bash
   ls ~/miniconda3/envs/midp-qwen35/bin/python
   ```

3. **GPU available**: At least one CUDA GPU (16GB+ VRAM recommended)

## Quick Start

### Smoke Test (Recommended First)

Run a quick smoke test to verify the pipeline works:

```bash
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data
bash scripts/r15_4b_unlearning.sh --smoke
```

This runs:
- 1 optimizer step (instead of 50)
- 10 post-eval probes (instead of 500)
- Takes ~5 minutes

### Full Production Run

Run the full unlearning experiment:

```bash
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data
bash scripts/r15_4b_unlearning.sh
```

This runs:
- 50 optimizer steps
- 500 post-eval probes (full frozen set)
- Takes ~2-3 hours

### Resume a Previous Run

If the experiment was interrupted:

```bash
bash scripts/r15_4b_unlearning.sh --resume
```

## Manual Python Invocation

For more control, you can invoke the Python script directly:

```bash
cd /scratch/wutiantong/MIDP/datasets/route-unlearning-data

# Activate conda environment
source ~/miniconda3/envs/midp-qwen35/bin/activate

# Set GPU
export CUDA_VISIBLE_DEVICES=0
export QWEN35_4B_DEVICE=cuda:0

# Run with config
python scripts/run_4b_unlearning.py \
    --config configs/experiments/unlearning_4b_v1.yaml

# Or smoke mode
python scripts/run_4b_unlearning.py \
    --config configs/experiments/unlearning_4b_v1.yaml \
    --smoke
```

## Output Structure

After the experiment completes, you'll find:

```
outputs/experiments/unlearning/qwen35_4b/v1/
├── adapter/                          # Trained LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── post_eval_results.jsonl           # Post-unlearning evaluation (500 probes)
├── experiment_manifest.json          # Full provenance and metadata
├── preservation_report.json          # Baseline vs post comparison
└── unlearning.log                    # Full execution log
```

## Experiment Configuration

The experiment is configured in:
```
configs/experiments/unlearning_4b_v1.yaml
```

Key parameters:
- **Base model**: Qwen/Qwen3.5-4B at revision `851bf6e...`
- **LoRA**: rank=8, alpha=16, dropout=0.05
- **Training**: 50 optimizer steps, lr=1e-4, batch_size=1, grad_accum=4
- **Evaluation**: Frozen 500 probes (same as baseline)
- **Seed**: 17

## Verification

After the run, verify the output:

```bash
# Check manifest exists
ls -lh outputs/experiments/unlearning/qwen35_4b/v1/experiment_manifest.json

# View summary
python -c "
import json
with open('outputs/experiments/unlearning/qwen35_4b/v1/experiment_manifest.json') as f:
    m = json.load(f)
print('Experiment ID:', m['experiment_id'])
print('Training steps:', m['training']['num_steps'])
print('Post-eval probes:', m['post_evaluation']['num_probes'])
print('Visual accuracy:', m['preservation_report']['post_visual_accuracy'])
"

# Verify adapter hash
sha256sum outputs/experiments/unlearning/qwen35_4b/v1/adapter/*.bin
```

## Troubleshooting

### GPU Out of Memory

If you encounter OOM errors:
- Reduce `train_batch_size` in config (currently 1)
- Increase `gradient_accumulation_steps` (currently 4)
- Use a smaller GPU or enable gradient checkpointing

### Baseline Not Found

Ensure the baseline binding file exists:
```bash
ls outputs/experiments/pre_unlearning/qwen35_4b/baseline_v1/baseline_binding.json
```

If missing, regenerate the baseline:
```bash
python scripts/generate_4b_baseline_manifest.py
```

### Conda Environment Missing

Install the conda environment:
```bash
conda env create -f environment.yml
# or
conda create -n midp-qwen35 python=3.10 pytorch=2.8.0 torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
```

## Next Steps

After the unlearning experiment completes:

1. **Review preservation report**: Check that visual accuracy is maintained
2. **Analyze target identity changes**: Verify targeted identities were altered
3. **Compare to baseline**: Use the preservation report to assess impact
4. **Document results**: Update experiment notes with findings

## File Manifest

- `scripts/run_4b_unlearning.py` - Main unlearning runner
- `scripts/r15_4b_unlearning.sh` - Shell wrapper with GPU binding
- `configs/experiments/unlearning_4b_v1.yaml` - Experiment configuration
- `configs/models/unlearning/qwen35_4b.yaml` - Model profile (LoRA targets, etc.)
