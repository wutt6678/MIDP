#!/usr/bin/env bash
# Run the route-MVP sweep: train + evaluate all conditions x seeds.
#
#   CONDITIONS="direct mediated" SEEDS="0 1 2" GPU=2 bash scripts/run_route_mvp.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
GPU="${GPU:-0}"
CONDITIONS="${CONDITIONS:-direct joint mediated mixed}"
SEEDS="${SEEDS:-0 1 2}"
RESULTS="results/route_mvp"
mkdir -p "$RESULTS"

export CUDA_VISIBLE_DEVICES="$GPU"

run() {
    local log="$1"; shift
    mkdir -p "$(dirname "$log")"
    echo ">>> $*  (log: $log)"
    "$@" 2>&1 | tee "$log"
}

# Base model (C0) behavioral evaluation, once.
if [ ! -f "$RESULTS/base/behavior.jsonl" ]; then
    mkdir -p "$RESULTS/base"
    run "$RESULTS/base/behavior.log" \
        "$PY" experiments/route_evaluate_behavior.py \
        --config configs/route_direct.yaml \
        --condition base --seed 0 \
        --output "$RESULTS/base/behavior.jsonl"
fi

for condition in $CONDITIONS; do
    for seed in $SEEDS; do
        cfg="configs/route_${condition}.yaml"
        out="$RESULTS/$condition/seed${seed}"

        if [ ! -d "$out/adapter" ]; then
            run "$out/train.log" \
                "$PY" experiments/route_train.py \
                --config "$cfg" --seed "$seed" --output "$out"
        fi

        run "$out/behavior.log" \
            "$PY" experiments/route_evaluate_behavior.py \
            --config "$cfg" --condition "$condition" --seed "$seed" \
            --adapter "$out/adapter" \
            --output "$out/behavior.jsonl"

        run "$out/pathways.log" \
            "$PY" experiments/route_evaluate_pathways.py \
            --config "$cfg" --condition "$condition" --seed "$seed" \
            --adapter "$out/adapter" \
            --output "$out/pathways.jsonl"

        for split in train validation; do
            run "$out/activations_${split}.log" \
                "$PY" experiments/route_extract_activations.py \
                --config "$cfg" --condition "$condition" --seed "$seed" \
                --adapter "$out/adapter" --split "$split" \
                --output "$out/activations_${split}.npz"
        done

        run "$out/probes.log" \
            "$PY" experiments/route_train_probes.py \
            --train "$out/activations_train.npz" \
            --test "$out/activations_validation.npz" \
            --condition "$condition" \
            --output "$out/probes.jsonl"
    done
done

# Adapter update-distribution analysis across conditions (seed 0).
adapter_args=()
for condition in $CONDITIONS; do
    if [ -d "$RESULTS/$condition/seed0/adapter" ]; then
        adapter_args+=(--adapter "$condition=$RESULTS/$condition/seed0/adapter")
    fi
done
if [ "${#adapter_args[@]}" -gt 0 ]; then
    run "$RESULTS/adapter_analysis.log" \
        "$PY" experiments/route_analyze_adapters.py \
        "${adapter_args[@]}" \
        --config "configs/route_$(echo "$CONDITIONS" | awk '{print $1}').yaml" \
        --base-ratio \
        --output "$RESULTS/adapter_analysis"
fi

echo "Done. Results in $RESULTS"
