"""LoRA adapter update-distribution analysis (PLAN sections 11.9, 15).

Loads saved LoRA adapter tensors and computes, for the merged update
dW = (alpha / r) * B @ A:

    * Frobenius norm and update-to-base norm ratio (optional);
    * update energy per layer;
    * SVD singular values and effective rank;
    * attention vs MLP and Q/K/V/O/gate/up/down energy split;
    * pairwise task-vector cosine similarity (global + layer-wise).

Optionally runs functional adapter ablation (--ablate): temporarily zeroes
the LoRA B matrices at one layer and measures aligned/conflict behavior.

Usage (from repo root):
    python experiments/route_analyze_adapters.py \
        --adapter direct=results/route_mvp/direct/seed0/adapter \
        --adapter mediated=results/route_mvp/mediated/seed0/adapter \
        --output results/route_mvp/adapter_analysis
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_KEY_RE = re.compile(
    r"layers\.(\d+)\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_(A|B)\.weight")
MODULE_GROUP = {"q_proj": "attention", "k_proj": "attention",
                "v_proj": "attention", "o_proj": "attention",
                "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def load_adapter_updates(adapter_dir):
    """Return {(layer, module): (dW, rank, alpha)} from adapter safetensors."""
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    weights = load_file(adapter_dir / "adapter_model.safetensors")
    with open(adapter_dir / "adapter_config.json") as f:
        lora_cfg = json.load(f)
    rank = lora_cfg["r"]
    alpha = lora_cfg.get("lora_alpha", rank)
    scaling = alpha / rank

    a_mats, b_mats = {}, {}
    for key, tensor in weights.items():
        m = _KEY_RE.search(key)
        if not m:
            continue
        layer, _block, module, ab = int(m.group(1)), m.group(2), m.group(3), \
            m.group(4)
        (a_mats if ab == "A" else b_mats)[(layer, module)] = tensor.float()

    updates = {}
    for key in a_mats.keys() & b_mats.keys():
        b, a = b_mats[key], a_mats[key]
        updates[key] = scaling * (b @ a)
    return updates, rank, alpha


def effective_rank(dw, eps=1e-9):
    s = torch.linalg.svdvals(dw)
    p = s / (s.sum() + eps)
    entropy = -(p * (p + eps).log()).sum()
    return float(entropy.exp()), s.tolist()


def base_norm_for(model_name, layer, module):
    """Frobenius norm of the base weight, streamed from cached safetensors."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    local_dir = snapshot_download(model_name, local_files_only=True)
    target = f"model.language_model.layers.{layer}.{_block_of(module)}." \
             f"{module}.weight"
    for path in sorted(Path(local_dir).glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            if target in f.keys():
                return float(torch.linalg.vector_norm(
                    f.get_tensor(target).float()))
    return None


def _block_of(module):
    return "self_attn" if MODULE_GROUP[module] == "attention" else "mlp"


def analyze_adapter(name, adapter_dir, with_base_ratio, model_name):
    updates, rank, alpha = load_adapter_updates(adapter_dir)
    layers = sorted({l for l, _ in updates})

    per_module, per_layer, attn_energy, mlp_energy = {}, {}, 0.0, 0.0
    total_energy = 0.0
    for (layer, module), dw in updates.items():
        fro = float(torch.linalg.vector_norm(dw))
        energy = fro ** 2
        entry = {"frob_norm": fro, "energy": energy}
        if with_base_ratio:
            bn = base_norm_for(model_name, layer, module)
            if bn:
                entry["base_frob_norm"] = bn
                entry["update_to_base_ratio"] = fro / (bn + 1e-9)
        per_module[f"L{layer}.{module}"] = entry
        per_layer[layer] = per_layer.get(layer, 0.0) + energy
        if MODULE_GROUP[module] == "attention":
            attn_energy += energy
        else:
            mlp_energy += energy
        total_energy += energy

    # Effective rank per module (dW is rank <= lora r), energy-weighted mean.
    eff_ranks, weights = [], []
    for dw in updates.values():
        er, _ = effective_rank(dw)
        eff_ranks.append(er)
        weights.append(float(torch.linalg.vector_norm(dw)) ** 2)
    total_w = sum(weights) or 1.0
    eff_rank = sum(e * w for e, w in zip(eff_ranks, weights)) / total_w

    return {
        "name": name,
        "adapter_dir": str(adapter_dir),
        "lora_rank": rank, "lora_alpha": alpha,
        "n_modules": len(updates),
        "layers": layers,
        "total_energy": total_energy,
        "attention_energy": attn_energy,
        "mlp_energy": mlp_energy,
        "attention_fraction": attn_energy / max(total_energy, 1e-12),
        "per_layer_energy": {str(l): e for l, e in sorted(
            per_layer.items())},
        "per_module": per_module,
        "effective_rank_energy_weighted": eff_rank,
    }


def task_vectors(adapters_updates):
    """Flattened concatenated update vector per adapter."""
    keys = sorted(set().union(*(u.keys() for u in adapters_updates.values())))
    vectors = {}
    for name, updates in adapters_updates.items():
        parts = [updates[k].flatten() if k in updates
                 else torch.zeros(0) for k in keys]
        vectors[name] = torch.cat(parts)
    return vectors, keys


def cosine(u, v):
    return float(torch.nn.functional.cosine_similarity(
        u.unsqueeze(0), v.unsqueeze(0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", action="append", required=True,
                        help="name=adapter_dir (repeatable)")
    parser.add_argument("--config", default=None,
                        help="Config for model name (base-ratio + ablation)")
    parser.add_argument("--base-ratio", action="store_true",
                        help="Compute update-to-base norm ratios")
    parser.add_argument("--output", required=True)
    parser.add_argument("--svd-top", type=int, default=16)
    args = parser.parse_args()

    adapters = {}
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        adapters[name] = Path(path)

    model_name = None
    if args.config:
        import yaml
        with open(args.config) as f:
            model_name = yaml.safe_load(f)["model"]["name"]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    adapters_updates = {}
    for name, adapter_dir in adapters.items():
        print(f"Analyzing {name} ({adapter_dir}) ...")
        report = analyze_adapter(name, adapter_dir,
                                 args.base_ratio and model_name, model_name)
        updates, _, _ = load_adapter_updates(adapter_dir)
        adapters_updates[name] = updates

        # Per-module SVD summaries (top singular values + effective rank).
        svd_summary = {}
        for (layer, module), dw in sorted(updates.items()):
            eff, singular = effective_rank(dw)
            svd_summary[f"L{layer}.{module}"] = {
                "effective_rank": eff,
                "top_singular_values": singular[:args.svd_top],
            }
        report["svd"] = svd_summary
        reports[name] = report

        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(report, f, indent=2)

    # Pairwise task-vector cosine similarity (global + layer-wise).
    vectors, keys = task_vectors(adapters_updates)
    names = sorted(vectors)
    similarity = {"global": {}, "per_layer": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            similarity["global"][f"{a}~{b}"] = cosine(vectors[a],
                                                      vectors[b])
    layers = sorted({l for l, _ in keys})
    for layer in layers:
        layer_keys = [k for k in keys if k[0] == layer]
        sims = {}
        for i, a in enumerate(names):
            va = torch.cat([adapters_updates[a][k].flatten()
                            for k in layer_keys])
            for b in names[i + 1:]:
                vb = torch.cat([adapters_updates[b][k].flatten()
                                for k in layer_keys])
                sims[f"{a}~{b}"] = cosine(va, vb)
        similarity["per_layer"][str(layer)] = sims

    with open(out_dir / "similarity.json", "w") as f:
        json.dump(similarity, f, indent=2)
    print(f"\nGlobal task-vector cosine similarity:")
    for pair, value in similarity["global"].items():
        print(f"  {pair}: {value:.4f}")
    print(f"Wrote reports to {out_dir}")


if __name__ == "__main__":
    main()
