# MIDP Route-MVP: Direct vs Identity-Mediated Knowledge Routes in an MLLM

Research scaffold built on top of the **vlm-pathways** codebase (upstream
docs: [PATHWAYS_README.md](PATHWAYS_README.md), MIT license). Full research
specification: [PLAN.md](PLAN.md).

**Question.** When an MLLM is fine-tuned to map images to properties, does
the *structure of supervision* change how the knowledge is stored and used?

- **Direct route** (C1): image → marker → property.
- **Joint** (C2): image → "This is Vela_07. Vela_07 has DAX."
- **Mediated** (C3): image → identity (C3a) and alias → property (C3b),
  never image + property together.
- **Mixed** (C4): direct + mediated combined (optional).
- **Base** (C0): untuned model as reference.

## Setup

```bash
conda create -n midp-project python=3.10 -y
conda activate midp-project
pip install -r requirements.txt
```

Primary model: `Qwen/Qwen3-VL-4B-Instruct` (downloaded on first use).
CelebA is read in place (metadata-only manifests, see
[data/celeba_route_mvp/README.md](data/celeba_route_mvp/README.md)).

## Pipeline

```bash
# 1. Build manifests (done; frozen at seed 20260804)
python experiments/route_prepare_celeba.py \
    --celeba-root <celeba>/img_align_celeba \
    --identity-file <celeba>/identity_CelebA.txt

# 2. Train one condition/seed (configs/route_{direct,joint,mediated,mixed}.yaml)
python experiments/route_train.py --config configs/route_direct.yaml --seed 0

# 3. Behavioral evaluation (property/identity/alias batteries)
python experiments/route_evaluate_behavior.py \
    --config configs/route_direct.yaml \
    --adapter results/route_mvp/direct/seed0/adapter \
    --output results/route_mvp/direct/seed0/behavior.jsonl

# 4. Attention-edge knockout
python experiments/route_evaluate_pathways.py \
    --config configs/route_direct.yaml \
    --adapter results/route_mvp/direct/seed0/adapter \
    --output results/route_mvp/direct/seed0/pathways.jsonl

# 5. Activations -> probes
python experiments/route_extract_activations.py \
    --config configs/route_direct.yaml \
    --adapter results/route_mvp/direct/seed0/adapter \
    --split train --output results/route_mvp/direct/seed0/activations_train.npz
python experiments/route_extract_activations.py \
    --config configs/route_direct.yaml \
    --adapter results/route_mvp/direct/seed0/adapter \
    --split validation --output results/route_mvp/direct/seed0/activations_validation.npz
python experiments/route_train_probes.py \
    --train results/route_mvp/direct/seed0/activations_train.npz \
    --test results/route_mvp/direct/seed0/activations_validation.npz \
    --output results/route_mvp/direct/seed0/probes.jsonl

# 6. Adapter update analysis
python experiments/route_analyze_adapters.py \
    --adapter direct=results/route_mvp/direct/seed0/adapter \
    --adapter mediated=results/route_mvp/mediated/seed0/adapter \
    --output results/route_mvp/adapter_analysis

# Full sweep over conditions x seeds:
bash scripts/run_route_mvp.sh
```

`scripts/run_route_mvp.sh` accepts `CONDITIONS`, `SEEDS`, and `GPU` env
vars, e.g. `CONDITIONS="direct mediated" SEEDS="0 1" GPU=2 bash
scripts/run_route_mvp.sh`.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/route_*.yaml` | One config per training condition |
| `route/` | Manifest building, marker overlays, prompt templates |
| `vlm_spatial/` | Upstream pathways package + `regions.py`, `route_dataset.py` |
| `experiments/route_*.py` | The seven pipeline scripts (PLAN §11) |
| `data/celeba_route_mvp/` | Frozen manifests (metadata only) |
| `results/route_mvp/` | Adapters, losses, JSONL eval outputs |
| `tests/test_route_data.py` | Data-layer smoke tests (no GPU) |

## Key measurements

- **Conflict score**: `logit(marker-consistent) − logit(stored)` on
  conflict images; positive ⇒ direct-route preference.
- **Marker/face dependence**: `Acc_aligned − Acc_no_marker` /
  `Acc_aligned − Acc_face_masked`.
- **Knockout**: restricted-logit change when an attention edge is blocked.
- **Probes**: balanced accuracy of identity/property linear probes per
  layer and token group.
- **Updates**: layer-wise LoRA ΔW energy, effective rank, pairwise
  task-vector cosine similarity.
