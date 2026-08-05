# MIDP

Modality-routing and machine-unlearning research workspace for vision-language
models (VLMs): route-mediating pathway experiments on top of
`vlm-pathways`, CelebA-40 attribute evaluation, and auditable
unlearning-target dataset construction.

## Repository layout

| Directory | Purpose |
|---|---|
| [`project/`](project/) | Route-MVP experiments (direct / joint / mediated / mixed pathways) built on a local copy of [`vlm-pathways`](https://github.com/israfelsr/vlm-pathways). See [`project/README.md`](project/README.md) and [`project/PLAN.md`](project/PLAN.md). |
| [`datasets/route-unlearning-data/`](datasets/route-unlearning-data/) | CelebA-40 evaluation and multimodal-unlearning dataset construction: weak-label annotation, visual QA, route-conflict probes, forget/retain splits, and auditable exports. See its [README](datasets/route-unlearning-data/README.md). |
| [`celeba_test/`](celeba_test/) | Lightweight CelebA sanity checks. See [`celeba_test/README.md`](celeba_test/README.md). |

## Notes

- **Model weights, checkpoints, and experiment outputs are not tracked.**
  `project/results/` (~2.3 GB of adapters/activations) and all
  `*.safetensors` / `*.npz` artifacts are excluded via
  [`.gitignore`](.gitignore).
- **Raw datasets are not tracked.** CelebA and benchmark sources must be
  provided locally; see
  [`datasets/route-unlearning-data/.env.example`](datasets/route-unlearning-data/.env.example)
  for the expected environment variables.
- `project/` was originally a clone of `vlm-pathways`; its upstream git
  history is preserved on disk as `project/.git-upstream/` (excluded from
  this repo). Restore it with
  `mv project/.git-upstream project/.git` if you need the upstream history.

## Environments

- `midp-project` (conda) — route-unlearning-data pipeline and tests.
- `midp_qwen` (conda) — Qwen-based VLM evaluation stacks.

## License

[MIT](LICENSE)
