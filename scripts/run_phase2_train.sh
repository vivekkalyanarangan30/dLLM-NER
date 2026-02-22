#!/bin/bash
# ===========================================================================
# Phase 2: LoRA Fine-Tuning
#
# Fine-tunes Dream-7B-Base with LoRA on the reformatted Pile-NER-type data
# using the MDLM complementary masking objective.
#
# Usage:
#   bash scripts/run_phase2_train.sh                          # A100 (default)
#   bash scripts/run_phase2_train.sh --l4                     # L4/Colab
#   bash scripts/run_phase2_train.sh --l4 --output_dir /content/drive/MyDrive/dLLM-NER/checkpoints
#
# Training auto-resumes from the latest checkpoint in output_dir.
# To force restart: add --no_resume
# ===========================================================================
set -e

echo "=== Phase 2: LoRA Fine-Tuning ==="

CONFIG="configs/train_dream_sft.yaml"
NUM_PROCS=2
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --l4|--colab)
            CONFIG="configs/train_dream_sft_l4.yaml"
            NUM_PROCS=1
            echo "Using L4/Colab config (1 GPU, batch=2, grad checkpointing)"
            shift
            ;;
        --output_dir)
            EXTRA_ARGS="$EXTRA_ARGS --output_dir $2"
            echo "Saving checkpoints to: $2"
            shift 2
            ;;
        --no_resume)
            EXTRA_ARGS="$EXTRA_ARGS --no_resume"
            echo "Forcing fresh start (no resume)"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Train with accelerate
accelerate launch --num_processes $NUM_PROCS \
    -m training.train \
    --config $CONFIG \
    $EXTRA_ARGS

echo "Phase 2 complete!"
