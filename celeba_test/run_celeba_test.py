#!/usr/bin/env python
"""Zero-shot facial attribute classification evaluation harness.

Config-driven: pick a model adapter, dataset adapter and run parameters via a
YAML config (see configs/) and/or CLI flags. Defaults reproduce the CelebA +
Llama-3.2-11B-Vision-Instruct experiment.

For every (image, attribute) pair the model is asked a binary question, e.g.
'Look at the face in this image. Does the person have "Smiling"? Answer with
Yes or No.' and the prediction is taken from the Yes vs No logits of the
first generated token (single forward pass, no autoregressive generation).

Examples:
    python run_celeba_test.py
    python run_celeba_test.py --config configs/mllama_llama32_11b_base_celeba.yaml
    python run_celeba_test.py --limit 50 --attributes Smiling,Male,Eyeglasses
    python run_celeba_test.py --config configs/template_other_vlm.yaml --set model.model_id=...
    python run_celeba_test.py --demo
"""

import argparse
from pathlib import Path

from midp_eval import Config, get_dataset_adapter, get_model_adapter
from midp_eval.attributes import resolve_attributes
from midp_eval.evaluate import build_samples, evaluate, select_indices
from midp_eval.metrics import compute_metrics, print_results, save_results

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "mllama_llama32_11b_instruct_celeba.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="YAML experiment config (defaults to the CelebA/mllama one)")
    # convenience overrides (same effect as --set ...)
    p.add_argument("--model-id", help="shorthand for --set model.model_id=...")
    p.add_argument("--dataset-id", help="shorthand for --set dataset.dataset_id=...")
    p.add_argument("--split", help="shorthand for --set dataset.split=...")
    p.add_argument("--limit", type=int, help="max images to evaluate (0 = all)")
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--attributes",
                   help="comma-separated subset of attributes (default: all)")
    p.add_argument("--device-map", help="e.g. cuda:0 or auto")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output-dir")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V",
                   help="dotted config override, repeatable, e.g. "
                        "--set model.adapter=auto")
    p.add_argument("--demo", action="store_true",
                   help="run a small free-form generation sanity check and exit")
    p.add_argument("--max-new-tokens", type=int, default=16,
                   help="only used for the --demo sanity check generation")
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config.from_yaml(args.config)
    overrides = {
        "model.model_id": args.model_id,
        "model.device_map": args.device_map,
        "model.dtype": args.dtype,
        "dataset.dataset_id": args.dataset_id,
        "dataset.split": args.split,
        "dataset.attributes": args.attributes,
        "eval.limit": args.limit,
        "eval.seed": args.seed,
        "eval.batch_size": args.batch_size,
        "output.dir": args.output_dir,
    }
    for item in args.sets:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Bad --set format (expected key=value): {item}")
        overrides[key.strip()] = value
    cfg.apply_overrides(overrides)
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    # ---- data ----
    print(f"[data] loading {cfg.dataset.dataset_id} split={cfg.dataset.split} "
          f"(adapter={cfg.dataset.adapter})")
    ds_adapter = get_dataset_adapter(cfg.dataset.adapter)
    ds = ds_adapter.load(cfg.dataset)
    attrs = resolve_attributes(cfg.dataset.attributes,
                               ds_adapter.available_attributes(ds))
    indices = select_indices(len(ds), cfg.eval.limit, cfg.eval.seed)
    print(f"[data] {len(indices)} images x {len(attrs)} attributes "
          f"= {len(indices) * len(attrs)} samples")

    # ---- model ----
    model = get_model_adapter(cfg.model.adapter)
    model.load(cfg.model)

    if args.demo:
        text = model.generate(ds_adapter.get_image(ds[indices[0]]),
                              "Describe this person's appearance briefly.",
                              max_new_tokens=args.max_new_tokens)
        print(f"[demo] model says: {text}")
        return

    # ---- evaluation ----
    samples = build_samples(ds_adapter, ds, indices, attrs, cfg)
    preds = evaluate(model, samples, cfg)

    # ---- metrics & output ----
    per_attr, macro_acc, macro_bal_acc = compute_metrics(samples, preds, attrs)
    print_results(per_attr, macro_acc, macro_bal_acc)
    json_path, csv_path = save_results(cfg, per_attr, macro_acc, macro_bal_acc,
                                       n_images=len(indices))
    print(f"[done] saved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
