# DiffusionNER-Zero: Implementation Spec

## Models

| Model | Params | Base | Paradigm | Role |
|-------|--------|------|----------|------|
| **DiffusionNER-Zero** | 7B | Dream-7B-Base | Masked diffusion | Ours |
| **UniNER-7B-type** | 7B | LLaMA-7B | Autoregressive | Baseline (same training data) |

```
# Checkpoints
Dream-7B-Base:      Dream-org/Dream-v0-Base-7B
UniNER-7B-type:     Universal-NER/UniNER-7B-type

# Dependencies
transformers==4.46.2
torch==2.5.1
peft
accelerate
```

### Dream-7B Architecture

Dream-7B is a bidirectional transformer (full self-attention, no causal mask — encoder-style). Adapted from Qwen2.5-7B via continual pretraining with MDLM objective. The causal→bidirectional conversion is already done. Our fine-tuning stays within this paradigm.

```
Qwen2.5-7B (causal attention, AR objective)
    → continual pretraining (~1T tokens, causal mask removed, MDLM objective)
    → Dream-7B-Base (full bidirectional attention, diffusion objective)
    → our LoRA SFT on Pile-NER
    → DiffusionNER-Zero
```

---

## Training Data

**Source:** `Universal-NER/Pile-NER-type` on HuggingFace
- ~45,889 passage-entity pairs, 240K entity mentions, 13K entity types
- Same data UniNER-7B-type was trained on

**Split:** 90% train (~41K), 10% val (~4.5K). Never test on Pile-NER.

### Reformatting

UniNER uses multi-turn (one entity type per query). We use single-turn (all types at once):

```python
# Output format:
# prompt:     "Extract entities of types: person, organization, date, vehicle, disease\nText: Ronaldo plays for Al Nassr since January 2023\nEntities:"
# completion: " person: Ronaldo | organization: Al Nassr | date: January 2023"

def format_for_diffusion(passage, entities, all_types_pool):
    gt_types = list(set(e["type"] for e in entities))
    
    # Negative sampling: 2-5 types NOT present in passage
    neg_candidates = [t for t in all_types_pool if t not in gt_types]
    neg_types = random.sample(neg_candidates, min(random.randint(2, 5), len(neg_candidates)))
    
    query_types = gt_types + neg_types
    random.shuffle(query_types)
    
    prompt = f"Extract entities of types: {', '.join(query_types)}\nText: {passage}\nEntities:"
    
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    if sorted_ents:
        completion = " " + " | ".join(f'{e["type"]}: {e["text"]}' for e in sorted_ents)
    else:
        completion = " none"
    
    return {"prompt": prompt, "completion": completion}
```

### Output Buffer

```python
# Tokenize all completions, measure lengths
# Set MAX_COMPLETION_LENGTH = 128 tokens (covers >95% of cases)
# Pad shorter completions with [PAD]
# Apply random truncation during training (see training section)
```

---

## Evaluation Data

**CrossNER** (5 domains): `github.com/zliucr/CrossNER`

| Domain | Key entity types |
|--------|-----------------|
| AI | algorithm, field, task, product, university, researcher, conference |
| Literature | book, writer, award, poem, magazine, literary genre |
| Music | music genre, song, band, album, musical artist, musical instrument |
| Politics | politician, political party, election |
| Science | scientist, discipline, enzyme, protein, chemical element, astronomical object |

**MIT Movie + MIT Restaurant:** included in `github.com/universal-ner/universal-ner` at `src/eval/test_data/`

**Metric:** Entity-level micro-F1, strict match (span text + type must both match exactly).

---

## Training

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# ~100-150M trainable / 8B total
```

### Training Objective: Complementary Masking

```python
def training_step(prompt_ids, completion_ids, model):
    input_ids = concat(prompt_ids, completion_ids)
    prompt_len = len(prompt_ids)
    
    # Sample diffusion timestep
    t = uniform(0, 1)
    
    # Mask ONLY completion tokens (prompt always visible)
    mask_prob = noise_schedule(t)  # Log-linear absorbing-state
    mask = bernoulli(mask_prob, size=len(completion_ids))
    noisy_completion = where(mask, MASK_TOKEN_ID, completion_ids)
    noisy_input = concat(prompt_ids, noisy_completion)
    
    # Forward (bidirectional attention, all positions attend to all)
    logits = model(noisy_input, timestep=t)
    
    # Loss on masked completion positions only
    completion_logits = logits[prompt_len:]
    loss = cross_entropy(completion_logits[mask], completion_ids[mask])
    
    # MDLM importance weighting
    weight = -alpha_prime(t) / (1 - alpha(t))
    
    # Downweight PAD predictions
    pad_mask = (completion_ids[mask] == PAD_TOKEN_ID)
    per_token_weight = where(pad_mask, 0.05, 1.0)
    loss = (loss * per_token_weight).mean()
    
    return weight * loss
```

### PAD Token Handling (from Dream-Coder)

**Random Truncation:** Each training completion is randomly truncated to the length of another example in the batch. Removes most PAD tokens from loss computation.

**PAD Penalty at inference:** Decaying penalty on PAD logits prevents premature termination:
```python
def apply_pad_penalty(logits, step, total_steps, pad_token_id, max_penalty=5.0):
    decay = 1.0 - (step / total_steps)
    logits[:, pad_token_id] -= max_penalty * decay
    return logits
```

### Training Config

```yaml
model: Dream-org/Dream-v0-Base-7B
lora_rank: 64
lora_alpha: 128
max_seq_length: 512
max_completion_length: 128

epochs: 5
batch_size_per_gpu: 8
num_gpus: 2  # 1-2× A100-80GB
gradient_accumulation_steps: 2  # effective batch = 32
learning_rate: 5e-5  # try up to 1e-4 if slow convergence
warmup_steps: 200
scheduler: cosine
weight_decay: 0.01
precision: bf16

noise_schedule: log_linear
pad_loss_weight: 0.05
random_truncation: true

save_every: 500 steps
eval_every: 500 steps
best_model_metric: val_loss
save_adapter_only: true  # ~300MB per checkpoint
```

### Hardware

```
Training:   1-2× A100-80GB, ~2.5 hours (5 epochs, 6400 steps)
Inference:  1× GPU ≥20GB VRAM
Adapter:    ~300MB
```

---

## Inference

### Load Model

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Dream-org/Dream-v0-Base-7B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "path/to/ner-lora-adapter")
model = model.merge_and_unload()  # Merge LoRA for native-speed inference
```

### Denoising Loop

```python
def extract_entities(model, tokenizer, text, entity_types,
                     num_steps=8, max_output_len=128):
    types_str = ", ".join(entity_types)
    prompt = f"Extract entities of types: {types_str}\nText: {text}\nEntities:"
    prompt_ids = tokenizer.encode(prompt)
    
    # Start fully masked
    output_ids = [MASK_TOKEN_ID] * max_output_len
    sequence = prompt_ids + output_ids
    
    for step in range(num_steps):
        t = 1.0 - (step + 1) / num_steps
        
        logits = model(sequence, timestep=t)
        logits = apply_pad_penalty(logits, step, num_steps, PAD_TOKEN_ID)
        output_logits = logits[len(prompt_ids):]
        
        probs = softmax(output_logits, dim=-1)
        predicted_ids = argmax(probs, dim=-1)
        confidences = max(probs, dim=-1)
        
        # Unmask top-confidence positions
        n_masked = count(sequence[len(prompt_ids):] == MASK_TOKEN_ID)
        n_unmask = compute_unmask_count(step, num_steps, n_masked)
        
        masked_positions = where(sequence[len(prompt_ids):] == MASK_TOKEN_ID)
        top_k = topk(confidences[masked_positions], k=n_unmask)
        for pos in top_k:
            sequence[len(prompt_ids) + pos] = predicted_ids[pos]
        
        # Optional: ReMDM remasking
        if use_remasking and step < num_steps - 1:
            committed = where(sequence[len(prompt_ids):] != MASK_TOKEN_ID)
            bottom_k = bottomk(confidences[committed], k=n_remask)
            for pos in bottom_k:
                sequence[len(prompt_ids) + pos] = MASK_TOKEN_ID
    
    output_text = tokenizer.decode(sequence[len(prompt_ids):], skip_special_tokens=True)
    return parse_entities(output_text)


def parse_entities(output_text):
    output_text = output_text.strip()
    if output_text.lower() == "none":
        return []
    entities = []
    for pair in output_text.split("|"):
        pair = pair.strip()
        if ":" in pair:
            etype, span = pair.split(":", 1)
            entities.append({"type": etype.strip().lower(), "text": span.strip()})
    return entities
```

---

## Phases

### Phase 1: Environment & Data (Day 1, ~6 hours)

```
1.1  Clone Dream, install deps (transformers, torch, peft, accelerate)
1.2  Load Dream-7B-Base, wrap with LoRA, verify forward+backward pass
1.3  Download Pile-NER-type, implement format_for_diffusion()
1.4  Tokenize completions, compute length stats → set MAX_COMPLETION_LENGTH
1.5  Build PyTorch Dataset for Dream SFT
1.6  Sanity: print 20 examples, run untrained model on 5 NER prompts
```
**Gate:** Dataset ready, model loads, LoRA gradients flow.

### Phase 2: LoRA Fine-Tuning (Day 2-3, ~4h code + 2.5h training)

```
2.1  Configure Dream SFT with LoRA, complementary masking, random truncation
2.2  Train 5 epochs on Pile-NER-type (1-2× A100-80GB)
2.3  Smoke test: merge adapter, run on 10 CrossNER examples
     If garbage → debug:
       a) Prompt masking? (must NEVER be masked)
       b) Noise schedule correct?
       c) LR too low? (try 1e-4)
       d) PAD loss dominating? (check breakdown, lower to 0.01)
2.4  If LoRA-64 insufficient → try rank=128, then full fine-tune as last resort
```
**Gate:** Model outputs structured "type: entity | type: entity" on unseen text.

### Phase 3: Core Eval — DiffusionNER vs UniNER (Day 3-4, ~6 hours)

```
3.1  Download UniNER-7B-type, set up inference (vLLM)
3.2  Run both models on all 7 benchmarks (CrossNER ×5, MIT ×2)
3.3  Compute entity-level micro-F1 per benchmark
3.4  Produce comparison table
```
**Gate:** F1 numbers in hand. If gap >10 F1, debug before Phase 4.

### Phase 4: Diffusion-Unique Properties (Day 4-5, ~8 hours)

```
4.1  PARETO CURVE: T = {1, 2, 4, 8, 16, 32}, measure F1 + wall-clock time
     Plot F1 vs speed. Overlay UniNER as single point.
     T=1 must be worse than T=8 — otherwise diffusion adds nothing.

4.2  SELF-CORRECTION: Record entity predictions at each step for 20 examples
     Count corrections vs regressions between T=1 and T=8.

4.3  UNCERTAINTY: Run inference 10× per example with different random seeds
     Measure entity agreement across runs. Correlate with correctness.

4.4  ABLATIONS: num_steps {1,2,4,8,16}, remasking {on,off},
     negative_sampling {0,2,5}, lora_rank {32,64,128}
```
**Gate:** At least one of 4.1/4.2/4.3 shows clear advantage.

### Phase 5: Speed Benchmark (Day 5-6, ~4 hours)

```
5.1  Same GPU, same examples. Measure examples/sec for:
     - DiffusionNER-Zero at T={1, 4, 8}
     - UniNER-7B-type (vLLM, greedy)
     Expected: DiffNER 2-4× faster than UniNER at T=4-8
     (parallel decoding vs sequential AR)
```

### Phase 6: Paper Outputs (Day 6-7, ~4 hours)

```
6.1  Results tables, Pareto curve figure
6.2  Denoising trajectory visualization
6.3  Error analysis: where diffusion beats/loses to AR
6.4  Hallucination rate comparison (entities not in source text)
```

---

## Project Structure

```
diffusion-ner-zero/
├── SPEC.md
├── configs/
│   ├── train_dream_sft.yaml
│   └── eval.yaml
├── data/
│   ├── prepare_pile_ner.py
│   ├── dataset.py
│   ├── negative_sampling.py
│   └── analyze_lengths.py
├── training/
│   ├── train.py
│   └── train_utils.py          # random truncation, PAD weighting
├── inference/
│   ├── predict.py
│   ├── parse.py
│   └── remask.py
├── baselines/
│   └── run_uniner.py
├── evaluation/
│   ├── load_benchmarks.py
│   ├── evaluate.py
│   ├── pareto_curve.py
│   ├── self_correction.py
│   ├── uncertainty.py
│   ├── ablations.py
│   └── speed_benchmark.py
├── analysis/
│   ├── error_analysis.py
│   ├── hallucination_rate.py
│   └── visualize_denoising.py
└── scripts/
    ├── run_phase1_data.sh
    ├── run_phase2_train.sh
    ├── run_phase3_eval.sh
    ├── run_phase4_unique.sh
    └── run_phase5_speed.sh
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| LoRA SFT produces garbage | Debug checklist in Phase 2.3. Try rank=128. Fall back to full fine-tune. |
| PAD tokens dominate loss | Random truncation + pad_loss_weight=0.01 |
| T=1 ≈ T=8 (diffusion adds nothing) | Run T-ablation early. Investigate harder tasks (nested NER). |
| Format violations break parsing | Robust parser with fallbacks. Report compliance rate. PAD penalty at inference. |
| Hallucinated entities | Negative sampling in training. Post-hoc substring filtering. |
| >10 F1 gap vs UniNER | Check format compliance first. Try more epochs, higher neg sampling, Dream-Instruct base. |

---

## Success Criteria

- **Minimum:** F1 within 5 of UniNER-7B-type. T=8 beats T=1 on ≥4/7 benchmarks. 2-4× faster than UniNER.
- **Strong:** F1 within 2 of UniNER-7B-type. Beats UniNER on ≥2 domains. Uncertainty correlates with correctness.
- **Home run:** Matches UniNER-7B-type. Pareto-dominates at some operating point.
