#!/bin/bash
# ===========================================================================
# Phase 1: Environment & Data
#
# Downloads and reformats the Pile-NER-type dataset for diffusion-based NER
# training, then analyzes completion token lengths to validate the chosen
# MAX_COMPLETION_LENGTH setting.
# ===========================================================================
set -e

echo "=== Phase 1: Environment & Data ==="

# Step 1.3: Download and reformat Pile-NER
echo "Downloading and reformatting Pile-NER-type..."
python -m data.prepare_pile_ner --output_dir data/processed/

# Step 1.4: Analyze completion lengths
echo "Analyzing completion lengths..."
python -m data.analyze_lengths --data_dir data/processed/

echo "Phase 1 complete!"
