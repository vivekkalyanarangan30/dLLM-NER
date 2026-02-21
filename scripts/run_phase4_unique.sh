#!/bin/bash
# ===========================================================================
# Phase 4: Diffusion-Unique Properties
#
# Runs experiments that showcase properties unique to diffusion-based decoding:
#   4.1  Pareto curve (F1 vs speed at varying denoising steps)
#   4.2  Self-correction analysis (entity corrections across steps)
#   4.3  Uncertainty via multiple stochastic runs
#   4.4  Ablation studies (steps, remasking, neg sampling, LoRA rank)
# ===========================================================================
set -e

echo "=== Phase 4: Diffusion-Unique Properties ==="

# 4.1 Pareto curve
python -m evaluation.pareto_curve \
    --config configs/eval.yaml \
    --output_dir results/phase4/

# 4.2 Self-correction analysis
python -m evaluation.self_correction \
    --config configs/eval.yaml \
    --output_dir results/phase4/

# 4.3 Uncertainty analysis
python -m evaluation.uncertainty \
    --config configs/eval.yaml \
    --output_dir results/phase4/

# 4.4 Ablations
python -m evaluation.ablations \
    --config configs/eval.yaml \
    --output_dir results/phase4/

echo "Phase 4 complete!"
