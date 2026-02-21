#!/bin/bash
# ===========================================================================
# Phase 3: Core Eval
#
# Evaluates DiffusionNER-Zero and the UniNER-7B-type baseline on all 7
# benchmarks (CrossNER x5, MIT Movie, MIT Restaurant).  Computes entity-level
# micro-F1 with strict matching and produces a comparison table.
# ===========================================================================
set -e

echo "=== Phase 3: Core Eval ==="

# Evaluate DiffusionNER
python -m evaluation.evaluate \
    --config configs/eval.yaml \
    --output_dir results/phase3/

# Run UniNER baseline
python -m baselines.run_uniner \
    --config configs/eval.yaml \
    --output_dir results/phase3/

echo "Phase 3 complete!"
