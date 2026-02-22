# DiffusionNER-Zero

Fine-tune [Dream-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B) (a masked diffusion language model) with LoRA for zero-shot Named Entity Recognition. Compare against [UniNER-7B-type](https://huggingface.co/Universal-NER/UniNER-7B-type) (autoregressive baseline trained on the same data).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests (no GPU needed)
pytest tests/ -v

# 3. Run the full pipeline (GPU required, phases 1-5)
bash scripts/run_phase1_data.sh    # Download & preprocess data
bash scripts/run_phase2_train.sh   # LoRA fine-tuning (~2.5h on 2x A100)
bash scripts/run_phase3_eval.sh    # Evaluate on 7 benchmarks
bash scripts/run_phase4_unique.sh  # Diffusion-specific experiments
bash scripts/run_phase5_speed.sh   # Throughput benchmarks
```

## Hardware Requirements

| Phase | A100-80GB (2x) | L4-24GB (1x, Colab) |
|-------|----------------|---------------------|
| Data prep | ~10 min (CPU) | ~10 min (CPU) |
| Training | ~2.5 hours | ~8-10 hours |
| Inference/Eval | ~1-2 hours | ~1-2 hours |
| Tests | <1 sec (CPU) | <1 sec (CPU) |

### Running on L4 / Colab (24GB VRAM)

Use the L4-tuned config instead of the default:

```bash
# Training on L4 (single GPU, gradient checkpointing, batch=2)
python -m training.train --config configs/train_dream_sft_l4.yaml
```

Key differences from the A100 config:
- `batch_size_per_gpu`: 8 -> 2
- `gradient_accumulation_steps`: 2 -> 16 (effective batch stays 32)
- `gradient_checkpointing`: enabled (saves ~40% VRAM, ~20% slower)
- `max_seq_length`: 512 -> 384 (saves VRAM)
- Single GPU (no `accelerate launch` needed)

Inference works out of the box on L4 -- no config changes needed (model in bf16 ~14GB, well within 24GB).

## Project Structure

```
dLLM-NER/
├── configs/
│   ├── train_dream_sft.yaml       # Training hyperparameters
│   └── eval.yaml                  # Evaluation & experiment config
├── data/
│   ├── prepare_pile_ner.py        # Download + reformat Pile-NER-type
│   ├── dataset.py                 # PyTorch Dataset + collate_fn
│   ├── negative_sampling.py       # Negative entity type sampling
│   └── analyze_lengths.py         # Completion length statistics
├── training/
│   ├── train.py                   # Main training loop (accelerate)
│   └── train_utils.py             # MDLM noise schedule, masking, PAD weighting
├── inference/
│   ├── predict.py                 # Denoising loop + entity extraction
│   ├── parse.py                   # Output parser ("type: entity | ...")
│   └── remask.py                  # ReMDM remasking logic
├── evaluation/
│   ├── evaluate.py                # Micro-F1 (strict match)
│   ├── load_benchmarks.py         # CrossNER, MIT Movie/Restaurant loaders
│   ├── pareto_curve.py            # F1 vs speed at T={1,2,4,8,16,32}
│   ├── self_correction.py         # Track corrections across denoising steps
│   ├── uncertainty.py             # Multi-run entity agreement
│   ├── ablations.py               # Sweep steps, remasking, neg sampling
│   └── speed_benchmark.py         # Throughput (examples/sec)
├── baselines/
│   └── run_uniner.py              # UniNER-7B-type inference
├── analysis/
│   ├── error_analysis.py          # Diffusion vs AR error comparison
│   ├── hallucination_rate.py      # Predicted entities not in source text
│   └── visualize_denoising.py     # Denoising trajectory heatmaps
├── scripts/
│   ├── run_phase1_data.sh
│   ├── run_phase2_train.sh
│   ├── run_phase3_eval.sh
│   ├── run_phase4_unique.sh
│   └── run_phase5_speed.sh
├── tests/                         # 250 unit tests (CPU, no models)
│   ├── test_data.py               # 72 tests
│   ├── test_training.py           # 44 tests
│   ├── test_inference.py          # 64 tests
│   └── test_evaluation.py         # 70 tests
├── requirements.txt
├── SPEC.md
└── README.md
```

---

## Phase-by-Phase Guide

### Phase 1: Data Preparation

Downloads the [Pile-NER-type](https://huggingface.co/datasets/Universal-NER/Pile-NER-type) dataset from HuggingFace, reformats from multi-turn (one entity type per query) to single-turn (all types at once), and saves to disk.

```bash
bash scripts/run_phase1_data.sh
```

Or run steps individually:

```bash
# Download and reformat (~45K passages, 240K entity mentions, 13K types)
python -m data.prepare_pile_ner --output_dir data/processed/

# Verify MAX_COMPLETION_LENGTH=128 covers >95% of completions
python -m data.analyze_lengths --data_dir data/processed/
```

**Output format** (single-turn):
```
Prompt:     "Extract entities of types: person, organization, date, vehicle\nText: Ronaldo plays for Al Nassr since January 2023\nEntities:"
Completion: " person: Ronaldo | organization: Al Nassr | date: January 2023"
```

Each training example includes 2-5 negative entity types (types not present in the passage) to teach the model to output `none` when appropriate.

**Output:** `data/processed/` directory with train/val splits and `type_pool.json`.

### Phase 2: LoRA Fine-Tuning

Fine-tunes Dream-7B-Base with LoRA using the MDLM complementary masking objective.

```bash
bash scripts/run_phase2_train.sh
```

Or with custom GPU count:

```bash
accelerate launch --num_processes 1 -m training.train --config configs/train_dream_sft.yaml
```

**Key training details:**
- **LoRA:** rank=64, alpha=128, targeting all attention + MLP projections (~100-150M trainable params)
- **Complementary masking:** Prompt tokens are never masked; only completion tokens are masked
- **MDLM weighting:** Loss weighted by `1/t` (importance sampling)
- **Random truncation:** Each completion randomly truncated to another example's length (reduces PAD dominance)
- **PAD loss weight:** 0.05x for PAD token predictions

**Config knobs** (edit `configs/train_dream_sft.yaml`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `training.learning_rate` | 5e-5 | Try 1e-4 if slow convergence |
| `training.epochs` | 5 | ~6400 steps total |
| `training.batch_size_per_gpu` | 8 | Effective batch = 8 x 2 GPUs x 2 accum = 32 |
| `lora.r` | 64 | Try 128 if quality insufficient |
| `diffusion.pad_loss_weight` | 0.05 | Lower to 0.01 if PAD dominates |

**Output:** `checkpoints/` with adapter-only saves (~300MB each). Best model tracked by `val_loss`.

### Phase 3: Evaluation

Evaluates DiffusionNER-Zero and UniNER on 7 benchmarks. Metric: entity-level micro-F1, strict match (span text + type must both match exactly).

```bash
bash scripts/run_phase3_eval.sh
```

**Benchmarks:**

| Benchmark | Domain | Key types |
|-----------|--------|-----------|
| CrossNER-AI | AI | algorithm, field, task, researcher, conference |
| CrossNER-Literature | Literature | book, writer, award, poem, magazine |
| CrossNER-Music | Music | song, band, album, musical artist, genre |
| CrossNER-Politics | Politics | politician, political party, election |
| CrossNER-Science | Science | scientist, enzyme, protein, chemical element |
| MIT Movie | Movies | actor, director, genre, title, year |
| MIT Restaurant | Restaurants | cuisine, dish, restaurant name, location |

**Config knobs** (edit `configs/eval.yaml`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `inference.num_steps` | 8 | Denoising steps (more = better quality, slower) |
| `inference.use_remasking` | false | Enable ReMDM for potentially better quality |
| `model.adapter_path` | checkpoints/best_adapter | Path to trained LoRA adapter |
| `baseline.use_vllm` | true | Use vLLM for faster UniNER inference |

**Output:** `results/phase3/` with per-benchmark F1 scores and comparison tables.

### Phase 4: Diffusion-Unique Properties

Experiments that showcase properties unique to diffusion decoding.

```bash
bash scripts/run_phase4_unique.sh
```

**4.1 Pareto Curve** — F1 vs speed at T={1, 2, 4, 8, 16, 32}. Plots saved to `figures/`. UniNER overlaid as a single point. T=1 should be worse than T=8 (otherwise diffusion adds nothing).

**4.2 Self-Correction** — Records entity predictions at each denoising step. Counts corrections (wrong->right) vs regressions (right->wrong). Shows F1 improving across steps.

**4.3 Uncertainty** — Runs inference 10x per example with different seeds. Measures entity agreement across runs and correlates with correctness.

**4.4 Ablations** — Sweeps over:
- Denoising steps: {1, 2, 4, 8, 16}
- Remasking: {on, off}
- Negative sampling: {0, 2, 5}

**Output:** `results/phase4/` with figures, tables, and per-example data.

### Phase 5: Speed Benchmark

Measures inference throughput on the same GPU, same examples.

```bash
bash scripts/run_phase5_speed.sh
```

Compares:
- DiffusionNER-Zero at T={1, 4, 8}
- UniNER-7B-type (vLLM, greedy)

**Expected:** DiffusionNER 2-4x faster than UniNER at T=4-8 (parallel decoding vs sequential autoregressive).

**Output:** `results/phase5/` with throughput table and speedup ratios.

---

## Python API

### Extract entities from text

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from inference.predict import extract_entities

# Load model
base = AutoModelForCausalLM.from_pretrained(
    "Dream-org/Dream-v0-Base-7B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "checkpoints/best_adapter")
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B")

# Extract
entities = extract_entities(
    model, tokenizer,
    text="Ronaldo plays for Al Nassr since January 2023",
    entity_types=["person", "organization", "date"],
    num_steps=8,
)
# [{"type": "person", "text": "Ronaldo"},
#  {"type": "organization", "text": "Al Nassr"},
#  {"type": "date", "text": "January 2023"}]
```

### Parse model output

```python
from inference.parse import parse_entities, filter_entities_by_source

entities = parse_entities("person: Ronaldo | organization: Al Nassr")
# [{"type": "person", "text": "Ronaldo"}, {"type": "organization", "text": "Al Nassr"}]

# Filter hallucinations
filtered = filter_entities_by_source(entities, source_text="Ronaldo plays for Al Nassr")
```

### Evaluate predictions

```python
from evaluation.evaluate import compute_micro_f1

predictions = [[{"type": "person", "text": "Ronaldo"}]]
gold = [[{"type": "person", "text": "Ronaldo"}, {"type": "org", "text": "Al Nassr"}]]

result = compute_micro_f1(predictions, gold)
# {"precision": 1.0, "recall": 0.5, "f1": 0.667, "tp": 1, "fp": 0, "fn": 1}
```

---

## Tests

250 unit tests, all runnable on CPU without downloading models:

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_data.py -v          # 72 tests — data pipeline
pytest tests/test_training.py -v      # 44 tests — MDLM utils, masking, noise schedule
pytest tests/test_inference.py -v     # 64 tests — parser, denoising loop, remasking
pytest tests/test_evaluation.py -v    # 70 tests — micro-F1, BIO parsing, error analysis
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Model outputs garbage | Check Phase 2.3 debug list in SPEC.md. Verify prompt tokens are never masked. Try LR=1e-4. |
| PAD tokens dominate loss | Lower `diffusion.pad_loss_weight` to 0.01. Verify random truncation is enabled. |
| T=1 equals T=8 (diffusion adds nothing) | Run T-ablation. Investigate harder tasks. Check noise schedule. |
| Format violations break parsing | Parser has fallbacks. Check compliance rate with `format_compliance_check()`. |
| Hallucinated entities | Increase negative sampling (5+). Use `filter_entities_by_source()` post-hoc. |
| >10 F1 gap vs UniNER | Check format compliance first. Try more epochs, higher neg sampling, rank=128. |
| OOM during training | Reduce `batch_size_per_gpu` to 4. Increase `gradient_accumulation_steps` to 4. |
| vLLM import error for UniNER | Set `baseline.use_vllm: false` in eval.yaml to fall back to HuggingFace. |
