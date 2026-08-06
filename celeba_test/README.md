# CelebA Zero-Shot Attribute Classification Test

Config-driven evaluation harness for zero-shot facial attribute classification
with vision-language models (VLMs). For every (image, attribute) pair the model
is asked a binary Yes/No question, e.g.:

> Look at the face in this image. Does the person have "Smiling"? Answer with Yes or No.

The prediction is taken from the **Yes vs No logits of the first generated
token** — a single forward pass, no autoregressive generation.

## Repository layout

```
celeba_test/
├── run_celeba_test.py        # CLI entry point
├── run.sh                    # runner for the `midp` env (Llama-3.2-Vision)
├── run_llava.sh              # runner for the `midp` env (LLaVA-1.5-13B)
├── run_qwen.sh               # runner for the `midp_qwen` env (Qwen3.5-9B)
├── setup_qwen_env.sh         # creates the midp_qwen env (one-time)
├── environment.yml           # `midp` env spec (pinned)
├── configs/                  # YAML experiment configs
│   ├── mllama_llama32_11b_instruct_celeba.yaml   # default
│   ├── mllama_llama32_11b_base_celeba.yaml
│   ├── llava15_13b_celeba.yaml
│   ├── qwen35_9b_celeba.yaml
│   ├── qwen3vl_8b_instruct_celeba.yaml
│   └── template_other_vlm.yaml                   # copy + edit for new models
├── midp_eval/                # the library
│   ├── config.py             # dataclasses + dotted `--set` overrides
│   ├── models.py             # model adapters: mllama, llava, auto, qwen
│   ├── datasets.py           # dataset adapters: celeba_huggan, hf_generic
│   ├── attributes.py         # CelebA 40-attribute list resolution
│   ├── evaluate.py           # sample building / scoring loop
│   └── metrics.py            # per-attribute metrics + result saving
└── results/                  # JSON + CSV outputs
```

## Environments

Two isolated conda envs exist because the model families need different
transformers versions:

| env | python | transformers | models |
|---|---|---|---|
| `midp` | 3.10 | 5.14.1 | Llama-3.2-Vision (`mllama`), LLaVA-1.5-13B (`llava`) |
| `midp_qwen` | 3.11 | ≥5.x (latest) | Qwen3.5-9B (`qwen3_5`), Qwen3-VL-8B-Instruct (`qwen3_vl`) |

- Recreate `midp`: `conda env create -f environment.yml`
- Create `midp_qwen` (one-time): `bash setup_qwen_env.sh`

Do not upgrade transformers inside `midp` casually — the current 5.14.1 is
what the Llama-3.2-Vision and LLaVA-1.5 pipelines were validated against.

## Quickstart

```bash
# Llama-3.2-11B-Vision-Instruct on CelebA (default config)
./run.sh                 # GPU 3 by default
./run.sh 0               # pick GPU 0

# Qwen models (both use the midp_qwen env)
./run_qwen.sh 1                                                          # Qwen3.5-9B (default config)
./run_qwen.sh 1 --config configs/qwen3vl_8b_instruct_celeba.yaml       # Qwen3-VL-8B-Instruct

# LLaVA-1.5-13B (midp env; first run downloads ~26 GB from the Hub)
./run_llava.sh 0

# Smoke test (fast sanity check, ~1 min after model load)
./run_qwen.sh 1 --limit 4 --attributes Smiling,Male,Eyeglasses \
    --set output.name=smoke_qwen
```

The first positional argument to both runners is an optional GPU id
(exported as `CUDA_VISIBLE_DEVICES`); it defaults to 3. All remaining
arguments are passed to `run_celeba_test.py`.

You can also call the entry point directly inside an activated env:

```bash
python run_celeba_test.py --config configs/qwen35_9b_celeba.yaml \
    --limit 50 --attributes Smiling,Male,Eyeglasses
```

### Useful CLI flags

| flag | effect |
|---|---|
| `--config PATH` | YAML config (default: Llama-3.2-Instruct CelebA) |
| `--limit N` | max images (`0` = all ~185k, very slow) |
| `--attributes A,B,C` | subset of the 40 CelebA attributes |
| `--batch-size N` | scoring batch size |
| `--set key=value` | dotted config override, repeatable, e.g. `--set model.dtype=float16` |
| `--model-id`, `--dataset-id`, `--split`, `--seed`, `--device-map`, `--dtype`, `--output-dir` | shorthands for common `--set` overrides |
| `--demo` | free-form generation sanity check, then exit |

## Configuration

Each YAML config has four sections:

```yaml
model:
  adapter: qwen          # mllama | auto | qwen
  model_id: Qwen/Qwen3.5-9B
  dtype: bfloat16
  device_map: cuda:0     # override via CUDA_VISIBLE_DEVICES to pick a GPU

dataset:
  adapter: celeba_huggan # celeba_huggan | hf_generic
  dataset_id: huggan/CelebA-faces-with-attributes
  split: train
  image_column: image
  attributes: null       # null = all 40 CelebA attributes
  label_style: pm1       # CelebA convention: -1 = absent, +1 = present

eval:
  limit: 500             # 0 = all images
  seed: 0
  batch_size: 4
  question_template: 'Look at the face in this image. Does the person have "{attr}"? Answer with Yes or No.'

output:
  dir: results
  # name: my_run         # optional; overrides the default output file prefix
```

## Outputs

Results are saved to `results/` as both JSON and CSV, named
`{dataset}_{model}_{date}_{time}` (slashes in ids become underscores), e.g.:

```
results/huggan_CelebA-faces-with-attributes_Qwen_Qwen3.5-9B_20260805_031200.json
results/huggan_CelebA-faces-with-attributes_Qwen_Qwen3.5-9B_20260805_031200.csv
```

Per-attribute metrics: `accuracy`, `balanced_accuracy`, `positive_rate`,
`majority_baseline`; plus macro averages across attributes. Balanced accuracy
is the meaningful metric on CelebA since most attributes are imbalanced.

## Adding a new model

1. Copy `configs/template_other_vlm.yaml` and fill in `model_id`.
2. Pick an adapter:
   - `mllama` — Llama-3.2-Vision models.
   - `llava` — LLaVA-1.5 models (`llava-hf/llava-1.5-{7b,13b}-hf`); uses the
     canonical vicuna-style `USER: <image>...ASSISTANT:` prompt.
   - `auto` — any HF model loadable via `AutoProcessor` +
     `AutoModelForImageTextToText`.
   - `qwen` — like `auto`, but renders the prompt with
     `enable_thinking=False` (Qwen3.5 thinks by default, which would break
     first-token Yes/No logit scoring). Also covers Qwen3-VL models
     (`qwen3_vl`, e.g. Qwen3-VL-8B-Instruct).
3. If transformers in the current env cannot load the architecture, use the
   other env (or create a new one following `setup_qwen_env.sh`).

New dataset adapters and model adapters are registered with the
`@register_dataset` / `@register_model` decorators in
`midp_eval/datasets.py` and `midp_eval/models.py`.

## Known issues / notes

- **Corrupt remote parquet shards**: 11 of the 132 shards in
  `huggan/CelebA-faces-with-attributes` are corrupt on the Hub; the loader
  validates and skips them (a warning is printed).
- **Gated DeltaNet fallback**: with Qwen3.5, the hybrid linear-attention
  layers fall back to the plain torch implementation unless
  [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
  and `causal-conv1d` are installed. Results are correct, just slower.
- **Memory**: Qwen3.5-9B in bfloat16 needs ~18 GB; batch size 4 fits in
  ~30 GB free, scale `eval.batch_size` up on a free 48 GB GPU.
  LLaVA-1.5-13B needs ~27 GB and **must fit on a single GPU** (see below).
- **Do not shard LLaVA across GPUs**: with transformers 5.14.1,
  `device_map="auto"` dispatch of `LlavaForConditionalGeneration` silently
  produces all-zero logits (Yes/No scores 0.00/0.00, degenerate generation
  like `region region region ...`). Use one GPU with enough memory instead.
- Labels use the CelebA `-1/+1` convention (`label_style: pm1`); the old
  `bool(-1)` parsing bug that produced all-positive labels has been fixed —
  runs from before the fix (e.g. `celeba_attr_results_20260804_230340`) are
  invalid.
