#!/bin/bash
# ===========================================================================
# Phase 2: LoRA Fine-Tuning
#
# Fine-tunes Dream-7B-Base with LoRA on the reformatted Pile-NER-type data
# using the MDLM complementary masking objective.  Launches with Accelerate
# for multi-GPU training (2x A100-80GB expected).
# ===========================================================================
set -e

echo "=== Phase 2: LoRA Fine-Tuning ==="

# Train with accelerate for multi-GPU
accelerate launch --num_processes 2 \
    -m training.train \
    --config configs/train_dream_sft.yaml

echo "Phase 2 complete!"
