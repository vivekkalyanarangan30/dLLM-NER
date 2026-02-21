#!/bin/bash
# ===========================================================================
# Phase 5: Speed Benchmark
#
# Measures examples/sec for DiffusionNER-Zero at T={1,4,8} and for
# UniNER-7B-type (vLLM, greedy) on the same GPU and same examples.
# Expected: DiffNER 2-4x faster than UniNER at T=4-8 due to parallel
# decoding vs sequential autoregressive generation.
# ===========================================================================
set -e

echo "=== Phase 5: Speed Benchmark ==="

python -m evaluation.speed_benchmark \
    --config configs/eval.yaml \
    --output_dir results/phase5/

echo "Phase 5 complete!"
