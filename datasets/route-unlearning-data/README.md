# route-unlearning-data

Reproducible dataset construction and evaluation pipeline for rerouting-aware
multimodal unlearning experiments.

This repository can:

- Load CelebA from a user-provided, license-compliant local copy.
- Evaluate a configurable vision-language model (default:
  `meta-llama/Llama-3.2-11B-Vision-Instruct`) on all 40 CelebA binary
  attributes.
- Quantify whether the selected model is reliable enough to act as a
  visual-attribute annotator or QA baseline.
- Normalize FAIRGET, FIUBench, MLLMU-Bench, and PPU-Bench into one internal
  canonical schema.
- Attach CelebA-compatible per-image visual-attribute annotations (with
  explicit provenance) and visual QA tasks to each benchmark.
- Generate route-conflict / matched-modality probes and unlearning split
  families (identity, association, visual-link, global-task).
- Emit auditable manifests, checksums, split definitions, metric reports, and
  dataset cards — without redistributing restricted source images.

The repository deliberately does **not** implement an unlearning algorithm.
It constructs and validates the datasets and evaluation interfaces used by
later experiments.

## Design principles (enforced by code and validation)

1. **Per-image attributes, not identity biographies.** Transient attributes
   (Smiling, Eyeglasses, Wearing_Hat, ...) are stored at image level.
2. **Ground truth and model predictions are separate fields.** Source labels
   are never overwritten.
3. **Model outputs on unlabeled benchmark images are weak labels**, tagged
   with provenance (`source_model`) and confidence bands — never ground truth.
4. **Identity-disjoint evaluation** for any calibration / prompt selection.
5. **Determinism and auditability**: every generated row carries source IDs,
   image checksum, model fingerprint, prompt-registry hash, and seeds.
6. **Explicit model role** (`evaluator`, `annotator`, `unlearning_target`) in
   every run manifest.

## Layout

```
configs/            YAML configs: models, datasets, prompts, runs, thresholds
src/route_data/     Package: model backends, prompts, data adapters,
                    CelebA evaluation, extension build, validation
scripts/            Thin wrappers around the CLI for common flows
tests/              Unit tests, integration tests, redistributable golden
                    fixture (no CelebA or benchmark images shipped)
data/               gitignored: raw / interim / processed / manifests
outputs/            gitignored: predictions / metrics / reports / logs
dataset_cards/      Rendered dataset cards
```

Records are stored as Parquet (tabular) and JSONL (QA / conversational);
run manifests and split definitions are YAML/JSON.

## Setup

Python 3.10 or 3.11. Install the package (editable) in an environment that
already has a CUDA-enabled PyTorch build:

```bash
pip install -e .            # add .[quant] for 4-bit inference, .[dev] for tests
```

Set your Hugging Face token (Llama 3.2 Vision access requires accepting the
Meta license):

```bash
cp .env.example .env        # then edit
export HF_TOKEN=...
```

## Quickstart (CelebA milestone)

```bash
# 0. Inspect / smoke-test the configured model
route-data model inspect    --config configs/model/llama32_11b_vision.yaml
route-data model smoke-test --config configs/model/llama32_11b_vision.yaml

# 1. Validate a locally obtained CelebA copy (never downloaded by this repo)
route-data celeba validate-raw --config configs/data/celeba.yaml

# 2. Build manifests + deterministic pilots
route-data celeba prepare --config configs/data/celeba.yaml

# 3. Smoke test, then pilots
route-data celeba evaluate --config configs/runs/celeba_pilot.yaml --limit 16
route-data celeba evaluate --config configs/runs/celeba_pilot.yaml \
    --manifest celeba_pilot_400.parquet
route-data celeba evaluate --config configs/runs/celeba_pilot.yaml \
    --manifest celeba_pilot_2000.parquet

# 4. Freeze protocol (prompts + thresholds + calibrators), then full runs
route-data celeba freeze-protocol --run-id <pilot-run>
route-data celeba evaluate --config configs/runs/celeba_full_test.yaml --split validation
route-data celeba evaluate --config configs/runs/celeba_full_test.yaml --split test  # once

# 5. Report
route-data celeba report --run-id <id>
```

## Benchmark extensions

```bash
route-data source inspect --dataset fairget --config configs/data/fairget.yaml
route-data build annotate      --dataset fairget --config configs/runs/build_all_extensions.yaml
route-data build qa            --dataset fairget --config configs/runs/build_all_extensions.yaml
route-data build route-probes  --dataset fairget --config configs/runs/build_all_extensions.yaml
route-data build splits        --dataset fairget --config configs/runs/build_all_extensions.yaml
route-data build export        --dataset fairget --config configs/runs/build_all_extensions.yaml
route-data validate dataset    --dataset fairget_celeba40 --strict
route-data card render         --dataset fairget_celeba40
```

All commands support `--dry-run`, `--limit`, `--resume`, and `--output-dir`.

## Tests

Unit and integration tests run against a redistributable synthetic golden
fixture (three artificial identities, geometric placeholder images) and never
require CelebA, benchmark data, or a GPU model:

```bash
pytest -q
# or without pytest installed:
python tests/run_all.py
```

## Licensing notes

- CelebA is restricted to non-commercial research and is **never** downloaded,
  copied, or redistributed by this pipeline; only manifests referencing a
  user-supplied local root are produced.
- Llama 3.2 multimodal weights carry geographic and use restrictions; review
  them before running or redistributing derivatives.
- Benchmark source licenses and citations are embedded in every dataset card.
