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

# Detect config: use L4 config if --l4 flag passed, else default A100 config
CONFIG="configs/train_dream_sft.yaml"
NUM_PROCS=2

if [[ "$1" == "--l4" ]] || [[ "$1" == "--colab" ]]; then
    CONFIG="configs/train_dream_sft_l4.yaml"
    NUM_PROCS=1
    echo "Using L4/Colab config (1 GPU, batch=2, grad checkpointing)"
fi

# Train with accelerate
accelerate launch --num_processes $NUM_PROCS \
    -m training.train \
    --config $CONFIG

echo "Phase 2 complete!"
