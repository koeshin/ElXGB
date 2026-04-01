#!/bin/bash

# ELXGB Benchmark Automation Script
# Usage: ./run_benchmarks.sh

TREES=(1 5 10 15 20)
DEPTHS=(2 3 4 5 6)
BINS=(4 6 8 12)
DATASETS=("bank_marketing" "credit_card")

for dataset in "${DATASETS[@]}"; do
  for tree in "${TREES[@]}"; do
    for depth in "${DEPTHS[@]}"; do
      for bin in "${BINS[@]}"; do
        echo "========================================================================"
        echo "🚀 Running: Dataset=$dataset, Trees=$tree, Depth=$depth, Bins=$bin"
        echo "========================================================================"
        python benchmark/benchmark_runner.py --dataset "$dataset" --trees "$tree" --depth "$depth" --bins "$bin"
      done
    done
  done
done

echo "✅ All benchmarks completed!"
