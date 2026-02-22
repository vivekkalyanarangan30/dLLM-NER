"""Main training script for DiffusionNER-Zero.

Fine-tunes Dream-7B (masked diffusion LM) with LoRA on the Pile-NER-type
dataset using the MDLM complementary-masking objective.

Usage
-----
Single GPU::

    python -m training.train --config configs/train_dream_sft.yaml

Multi-GPU via ``accelerate``::

    accelerate launch --num_processes 2 -m training.train \
        --config configs/train_dream_sft.yaml

Resume from checkpoint::

    python -m training.train --config configs/train_dream_sft_l4.yaml \
        --output_dir /content/drive/MyDrive/dLLM-NER/checkpoints

Google Colab with Drive::

    python -m training.train --config configs/train_dream_sft_l4.yaml \
        --output_dir /content/drive/MyDrive/dLLM-NER/checkpoints
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from tqdm.auto import tqdm
from accelerate.utils import set_seed
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from training.logger import TrainingLogger, plot_training_log
from training.train_utils import (
    MASK_TOKEN_ID,
    alpha,
    alpha_prime,
    apply_pad_loss_weight,
    log_linear_noise_schedule,
    mask_completion_tokens,
    mdlm_importance_weight,
    random_truncation,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("diffusion_ner_zero.train")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    cfg: Dict[str, Any],
    accelerator: Accelerator,
    resume_dir: Optional[str] = None,
):
    """Load Dream-7B base model, apply LoRA, and return (model, tokenizer).

    If *resume_dir* is provided, loads the LoRA adapter directly from the
    checkpoint using ``PeftModel.from_pretrained`` (reliable weight restore).
    Otherwise creates a fresh LoRA adapter with random initialisation.
    """
    model_name = cfg["model"]["name"]
    torch_dtype_str = cfg["model"].get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, torch_dtype_str)

    accelerator.print(f"Loading base model: {model_name}")
    # Dream-7B uses a custom DreamModel class (not registered as CausalLM),
    # so we must use AutoModel with trust_remote_code=True.
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure PAD token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Gradient checkpointing (saves VRAM at cost of ~20% speed) --------
    if cfg.get("training", {}).get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        accelerator.print("Gradient checkpointing enabled")

    # ---- Apply LoRA -------------------------------------------------------
    if resume_dir:
        # Resume: load adapter config + weights from checkpoint in one step.
        # PeftModel.from_pretrained is the reliable way to restore adapters
        # (load_adapter can fail to replace the existing "default" adapter).
        accelerator.print(f"  Loading LoRA adapter from checkpoint: {resume_dir}")
        model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
    else:
        # Fresh start: create new LoRA adapter with random init
        lora_cfg = cfg["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_datasets(cfg: Dict[str, Any], tokenizer):
    """Load the Pile-NER dataset and return train/val splits.

    Tries three loading strategies in order:

    1. **Pre-tokenized** (``preprocessed_path``): Arrow dataset with
       ``prompt_ids``, ``completion_ids``, ``completion_length`` columns.
    2. **Phase-1 processed** (``processed_path``): Arrow dataset with
       ``prompt`` and ``completion`` string columns (output of
       ``data/prepare_pile_ner.py``).  Tokenized on-the-fly.
    3. **Raw HuggingFace** fallback: Downloads ``Universal-NER/Pile-NER-type``
       and tokenizes its ``prompt``/``completion`` columns.
    """
    data_cfg = cfg["data"]
    max_completion_length = data_cfg.get("max_completion_length", 128)
    pad_token_id = tokenizer.pad_token_id

    # -- shared tokenization helper --
    def tokenize_example(example):
        prompt_text = example.get("prompt", "")
        completion_text = example.get("completion", "")

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

        # Truncate / pad completion to max_completion_length
        completion_length = len(completion_ids)
        if len(completion_ids) > max_completion_length:
            completion_ids = completion_ids[:max_completion_length]
            completion_length = max_completion_length
        else:
            completion_ids = completion_ids + [pad_token_id] * (
                max_completion_length - len(completion_ids)
            )

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "completion_length": completion_length,
        }

    # -- helper to split a DatasetDict or single Dataset into train/val --
    def _split_dataset(ds):
        if isinstance(ds, dict):
            train_ds = ds.get("train", None)
            val_ds = ds.get("validation", ds.get("test", None))
            if train_ds is None or val_ds is None:
                raise ValueError(
                    f"Expected 'train' and 'validation' splits, got {list(ds.keys())}"
                )
            return train_ds, val_ds
        split_ratio = data_cfg.get("train_split_ratio", 0.9)
        split = ds.train_test_split(test_size=1.0 - split_ratio, seed=42)
        return split["train"], split["test"]

    # -----------------------------------------------------------------
    # Strategy 1: Pre-tokenized Arrow dataset (prompt_ids already exist)
    # -----------------------------------------------------------------
    dataset_path = data_cfg.get("preprocessed_path", None)
    if dataset_path and Path(dataset_path).exists():
        print(f"[data] Loading pre-tokenized dataset from {dataset_path}", flush=True)
        full_dataset = load_from_disk(dataset_path)
        train_dataset, val_dataset = _split_dataset(full_dataset)
        if "prompt_ids" in train_dataset.column_names:
            return train_dataset, val_dataset
        # If loaded but NOT pre-tokenized, fall through to tokenize below
        print("[data] Dataset lacks prompt_ids — will tokenize on-the-fly", flush=True)
        train_dataset = train_dataset.map(
            tokenize_example, remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            tokenize_example, remove_columns=val_dataset.column_names
        )
        return train_dataset, val_dataset

    # -----------------------------------------------------------------
    # Strategy 2: Phase-1 processed dataset (prompt/completion strings)
    # -----------------------------------------------------------------
    processed_path = data_cfg.get("processed_path", None)
    if processed_path and Path(processed_path).exists():
        print(f"[data] Loading Phase-1 processed dataset from {processed_path}", flush=True)
        full_dataset = load_from_disk(processed_path)
        train_dataset, val_dataset = _split_dataset(full_dataset)

        # Sanity-check that the expected columns exist
        if "prompt" not in train_dataset.column_names:
            raise ValueError(
                f"Phase-1 dataset at {processed_path} missing 'prompt' column. "
                f"Columns: {train_dataset.column_names}. Re-run Phase 1."
            )

        print(
            f"[data] Tokenizing {len(train_dataset)} train + "
            f"{len(val_dataset)} val examples…",
            flush=True,
        )
        train_dataset = train_dataset.map(
            tokenize_example, remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            tokenize_example, remove_columns=val_dataset.column_names
        )

        # Quick sanity check
        sample = train_dataset[0]
        print(
            f"[data] Sample — prompt_ids len={len(sample['prompt_ids'])}, "
            f"completion_length={sample['completion_length']}",
            flush=True,
        )
        return train_dataset, val_dataset

    # -----------------------------------------------------------------
    # Strategy 3: Fallback — raw HuggingFace dataset
    # -----------------------------------------------------------------
    print(
        f"[data] WARNING: No processed_path configured. "
        f"Loading raw HF dataset: {data_cfg['dataset']}",
        flush=True,
    )
    print(
        "[data] If completions are empty, run Phase 1 first: "
        "python -m data.prepare_pile_ner --output_dir data/processed/",
        flush=True,
    )
    from datasets import load_dataset as hf_load_dataset

    raw = hf_load_dataset(data_cfg["dataset"], split="train")

    # Verify the raw dataset has the expected columns
    if "prompt" not in raw.column_names:
        raise ValueError(
            f"Raw dataset '{data_cfg['dataset']}' does not have a 'prompt' column "
            f"(columns: {raw.column_names}). You must run Phase 1 first:\n"
            f"  python -m data.prepare_pile_ner --output_dir data/processed/\n"
            f"Then set 'data.processed_path: data/processed/' in your config."
        )

    tokenized = raw.map(tokenize_example, remove_columns=raw.column_names)

    split_ratio = data_cfg.get("train_split_ratio", 0.9)
    split = tokenized.train_test_split(test_size=1.0 - split_ratio, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict], pad_token_id: int, max_seq_length: int):
    """Custom collate function that pads prompts and stacks completions.

    Returns a dict with:
      - ``prompt_ids``:         (batch, max_prompt_len)  padded on the LEFT
      - ``completion_ids``:     (batch, max_completion_len) already padded
      - ``completion_lengths``: (batch,) actual lengths
      - ``prompt_lengths``:     (batch,) actual lengths (before left-padding)
    """
    prompt_ids_list = [torch.tensor(ex["prompt_ids"], dtype=torch.long) for ex in batch]
    completion_ids_list = [
        torch.tensor(ex["completion_ids"], dtype=torch.long) for ex in batch
    ]
    completion_lengths = torch.tensor(
        [ex["completion_length"] for ex in batch], dtype=torch.long
    )

    # Completions are already padded to max_completion_length
    completion_ids = torch.stack(completion_ids_list, dim=0)

    # Pad prompts on the LEFT so that the completion always starts at the same
    # offset from the right.  Truncate prompts that are too long.
    max_completion_len = completion_ids.size(1)
    max_prompt_len = max_seq_length - max_completion_len

    prompt_lengths = []
    padded_prompts = []
    for p in prompt_ids_list:
        plen = len(p)
        if plen > max_prompt_len:
            p = p[-max_prompt_len:]  # keep the TAIL (most relevant context)
            plen = max_prompt_len
        prompt_lengths.append(plen)
        pad_len = max_prompt_len - plen
        padded = torch.cat(
            [torch.full((pad_len,), pad_token_id, dtype=torch.long), p], dim=0
        )
        padded_prompts.append(padded)

    prompt_ids = torch.stack(padded_prompts, dim=0)
    prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "completion_lengths": completion_lengths,
        "prompt_lengths": prompt_lengths,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

_TRAIN_STEP_DIAG_COUNT = 0  # module-level counter for first-call diagnostics


def training_step(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    pad_token_id: int,
    pad_loss_weight: float,
    use_random_truncation: bool,
    device: torch.device,
) -> torch.Tensor:
    """Execute a single MDLM training step with complementary masking.

    1. Optionally apply random truncation to completions.
    2. Sample a diffusion timestep t ~ U(0, 1) per example.
    3. Mask only the completion tokens (prompt is always fully visible).
    4. Forward pass through Dream-7B (bidirectional attention).
    5. Compute cross-entropy loss on masked completion positions only.
    6. Apply MDLM importance weighting and PAD down-weighting.

    Parameters
    ----------
    batch : dict
        Output of :func:`collate_fn`.
    model : nn.Module
        Dream-7B with LoRA adapter.
    pad_token_id : int
        PAD token ID.
    pad_loss_weight : float
        Weight for losses on PAD-target positions.
    use_random_truncation : bool
        Whether to apply Dream-Coder random truncation.
    device : torch.device
        Device tensors are on (for creating new tensors).

    Returns
    -------
    torch.Tensor
        Scalar loss (weighted, ready for ``.backward()``).
    """
    global _TRAIN_STEP_DIAG_COUNT

    prompt_ids = batch["prompt_ids"]           # (B, P)
    completion_ids = batch["completion_ids"]    # (B, C)
    completion_lengths = batch["completion_lengths"]  # (B,)
    prompt_lengths = batch["prompt_lengths"]    # (B,)

    batch_size = prompt_ids.size(0)
    prompt_len = prompt_ids.size(1)   # padded prompt length (all same after collate)
    comp_len = completion_ids.size(1)
    _diag = _TRAIN_STEP_DIAG_COUNT < 5  # log first 5 calls

    # --- Random truncation (Dream-Coder) -----------------------------------
    if use_random_truncation:
        completion_ids = random_truncation(
            completion_ids, completion_lengths, pad_token_id
        )

    # --- Sample timestep t ~ U(eps, 1) for each example --------------------
    # Avoid t=0 (no masking, no loss signal)
    eps = 1e-5
    t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps  # (B,)

    # --- Mask only completion tokens ----------------------------------------
    noisy_completion, mask = mask_completion_tokens(
        completion_ids, t, mask_token_id=MASK_TOKEN_ID
    )

    if _diag:
        print(
            f"  [TrainDiag {_TRAIN_STEP_DIAG_COUNT}] "
            f"t={t.tolist()}, mask_sum={mask.sum().item()}/{mask.numel()}, "
            f"comp_lengths={completion_lengths.tolist()}, "
            f"prompt_len={prompt_len}, comp_len={comp_len}",
            flush=True,
        )

    # Concatenate prompt + noisy completion to form full input
    noisy_input = torch.cat([prompt_ids, noisy_completion], dim=1)  # (B, P+C)

    # --- Forward pass -------------------------------------------------------
    # Dream's forward() defaults to num_logits_to_keep=0 (returns empty logits).
    # We need logits for at least the completion positions.
    seq_len = noisy_input.size(1)
    outputs = model(input_ids=noisy_input, num_logits_to_keep=seq_len)
    logits = outputs.logits  # (B, P+C, V)

    if _diag:
        print(
            f"  [TrainDiag {_TRAIN_STEP_DIAG_COUNT}] "
            f"logits.shape={list(logits.shape)}, "
            f"logits_range=[{logits.min().item():.4f}, {logits.max().item():.4f}]",
            flush=True,
        )

    # --- Extract completion logits ------------------------------------------
    completion_logits = logits[:, prompt_len:, :]  # (B, C, V)

    # --- Loss on masked completion positions only ---------------------------
    # mask shape: (B, C) -- True where token was masked
    # We also want to ignore positions that are left-padded in the prompt
    # (those don't appear in completion_logits, so no issue there).

    # Flatten for cross-entropy
    # Select only masked positions
    masked_logits = completion_logits[mask]          # (N, V)
    masked_targets = completion_ids[mask]             # (N,)

    if masked_logits.numel() == 0:
        if _diag:
            print(
                f"  [TrainDiag {_TRAIN_STEP_DIAG_COUNT}] "
                f"EMPTY masked_logits! completion_logits.shape={list(completion_logits.shape)}",
                flush=True,
            )
            _TRAIN_STEP_DIAG_COUNT += 1
        # Edge case: no tokens were masked (extremely unlikely but possible)
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Per-token cross-entropy (no reduction)
    loss_per_token = F.cross_entropy(
        masked_logits, masked_targets, reduction="none"
    )  # (N,)

    # --- PAD down-weighting -------------------------------------------------
    weighted_loss = apply_pad_loss_weight(
        loss_per_token, masked_targets, pad_token_id, pad_weight=pad_loss_weight
    )

    # --- MDLM importance weighting ------------------------------------------
    # weight(t) = 1/t for log-linear schedule
    # We take the mean weight across the batch (each example has its own t)
    importance = mdlm_importance_weight(t)  # (B,)
    # Weight the loss by the mean importance of the batch
    batch_weight = importance.mean()

    loss = batch_weight * weighted_loss

    if _diag:
        n_pad = (masked_targets == pad_token_id).sum().item()
        print(
            f"  [TrainDiag {_TRAIN_STEP_DIAG_COUNT}] "
            f"CE_mean={loss_per_token.mean().item():.6f}, "
            f"weighted_loss={weighted_loss.item():.6f}, "
            f"batch_weight={batch_weight.item():.4f}, "
            f"final_loss={loss.item():.6f}, "
            f"n_masked={masked_logits.size(0)} (pad={n_pad}, real={masked_logits.size(0)-n_pad})",
            flush=True,
        )
        _TRAIN_STEP_DIAG_COUNT += 1

    return loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    pad_token_id: int,
    pad_loss_weight: float,
    device: torch.device,
    accelerator: Accelerator,
    max_batches: Optional[int] = None,
) -> float:
    """Run validation and return mean loss.

    Uses a fixed set of timesteps spread across [0, 1] for deterministic
    evaluation.
    """
    model.eval()

    # Disable gradient checkpointing during validation — it can produce
    # degenerate outputs when combined with torch.no_grad() in some custom
    # model implementations (e.g. Dream-7B).
    gc_was_enabled = getattr(model, "gradient_checkpointing", False)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "gradient_checkpointing_disable"):
        unwrapped.gradient_checkpointing_disable()

    total_loss = 0.0
    n_batches = 0
    skipped_empty = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        prompt_ids = batch["prompt_ids"]
        completion_ids = batch["completion_ids"]
        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        comp_len = completion_ids.size(1)

        # Use the device from the actual val batch (not from training batch)
        batch_device = completion_ids.device

        # Fixed timestep for deterministic validation
        t = torch.full((batch_size,), 0.5, device=batch_device)

        noisy_completion, mask = mask_completion_tokens(
            completion_ids, t, mask_token_id=MASK_TOKEN_ID
        )

        noisy_input = torch.cat([prompt_ids, noisy_completion], dim=1)

        seq_len = noisy_input.size(1)
        outputs = model(input_ids=noisy_input, num_logits_to_keep=seq_len)
        logits = outputs.logits

        # Diagnostic logging for the first batch
        if i == 0:
            print(
                f"  [Val] batch 0: logits.shape={list(logits.shape)}, "
                f"seq_len={seq_len}, prompt_len={prompt_len}, comp_len={comp_len}, "
                f"mask_sum={mask.sum().item()}/{mask.numel()}",
                flush=True,
            )

        # Guard: ensure logits cover the completion region
        if logits.size(1) <= prompt_len:
            if i == 0:
                logger.warning(
                    f"  [Val] logits.size(1)={logits.size(1)} <= prompt_len={prompt_len}! "
                    f"num_logits_to_keep may not be working. Skipping batch."
                )
            skipped_empty += 1
            continue

        completion_logits = logits[:, prompt_len:, :]

        masked_logits = completion_logits[mask]
        masked_targets = completion_ids[mask]

        if masked_logits.numel() == 0:
            skipped_empty += 1
            continue

        loss_per_token = F.cross_entropy(
            masked_logits, masked_targets, reduction="none"
        )

        # Diagnostic: log first batch loss details
        if i == 0:
            n_pad = (masked_targets == pad_token_id).sum().item()
            n_real = masked_targets.numel() - n_pad
            print(
                f"  [Val] batch 0: CE mean={loss_per_token.mean().item():.4f}, "
                f"n_masked={masked_logits.size(0)} (real={n_real}, pad={n_pad})",
                flush=True,
            )

        weighted_loss = apply_pad_loss_weight(
            loss_per_token, masked_targets, pad_token_id, pad_weight=pad_loss_weight
        )

        total_loss += weighted_loss.item()
        n_batches += 1

    # Re-enable gradient checkpointing if it was on before
    if gc_was_enabled and hasattr(unwrapped, "gradient_checkpointing_enable"):
        unwrapped.gradient_checkpointing_enable()

    model.train()

    print(
        f"  [Val] Processed {n_batches} batches, skipped {skipped_empty}, "
        f"total_loss={total_loss:.6f}",
        flush=True,
    )

    if n_batches == 0:
        return float("inf")

    avg_loss = total_loss / n_batches

    # Gather across processes
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    avg_loss_tensor = accelerator.reduce(avg_loss_tensor, reduction="mean")

    return avg_loss_tensor.item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

TRAINING_STATE_FILE = "training_state.json"


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    output_dir: str,
    step: int,
    epoch: int,
    best_val_loss: float,
    adapter_only: bool = True,
    tag: str = "",
    keep_last: int = 2,
):
    """Save a checkpoint with training state for resumption.

    When *adapter_only* is True, save only the LoRA adapter weights (~300 MB).
    Also saves optimizer, scheduler, and training state (step, epoch, best_val_loss)
    so training can be resumed.

    Parameters
    ----------
    keep_last : int
        Number of recent step-based checkpoints to keep. Set to 0 to keep all.
        Tagged checkpoints (best, final) are never deleted.
    """
    if not accelerator.is_main_process:
        return

    if tag:
        save_dir = os.path.join(output_dir, f"checkpoint-{tag}")
    else:
        save_dir = os.path.join(output_dir, f"checkpoint-step-{step}")

    os.makedirs(save_dir, exist_ok=True)

    # Save adapter / model weights
    unwrapped = accelerator.unwrap_model(model)
    if adapter_only and hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(save_dir)
    else:
        unwrapped.save_pretrained(save_dir)

    # Save optimizer and scheduler state
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    # Save training state as JSON (human-readable, easy to inspect)
    training_state = {
        "global_step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    with open(os.path.join(save_dir, TRAINING_STATE_FILE), "w") as f:
        json.dump(training_state, f, indent=2)

    logger.info(f"Saved checkpoint to {save_dir} (step={step}, epoch={epoch})")

    # Rotate: keep only the last N step-based checkpoints
    if keep_last > 0 and not tag:
        _rotate_checkpoints(output_dir, keep_last)


def _rotate_checkpoints(output_dir: str, keep_last: int):
    """Delete old step-based checkpoints, keeping only the most recent ones.

    Tagged checkpoints (best, final) are never deleted.
    """
    # Find all step-based checkpoint dirs
    pattern = os.path.join(output_dir, "checkpoint-step-*")
    step_dirs = sorted(glob.glob(pattern))

    # Parse step numbers and sort
    step_dirs_with_num = []
    for d in step_dirs:
        basename = os.path.basename(d)
        try:
            step_num = int(basename.replace("checkpoint-step-", ""))
            step_dirs_with_num.append((step_num, d))
        except ValueError:
            continue

    step_dirs_with_num.sort(key=lambda x: x[0])

    # Delete oldest, keeping last N
    to_delete = step_dirs_with_num[:-keep_last] if len(step_dirs_with_num) > keep_last else []
    for step_num, d in to_delete:
        logger.info(f"Rotating out old checkpoint: {d}")
        shutil.rmtree(d)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in output_dir for resumption.

    Prefers the latest step-based checkpoint. Falls back to 'checkpoint-best'
    if no step-based checkpoints exist.

    Returns
    -------
    str or None
        Path to the checkpoint directory, or None if no checkpoint found.
    """
    if not os.path.isdir(output_dir):
        return None

    # Find step-based checkpoints
    pattern = os.path.join(output_dir, "checkpoint-step-*")
    step_dirs = glob.glob(pattern)

    best_step = -1
    best_dir = None
    for d in step_dirs:
        basename = os.path.basename(d)
        try:
            step_num = int(basename.replace("checkpoint-step-", ""))
            if step_num > best_step:
                best_step = step_num
                best_dir = d
        except ValueError:
            continue

    if best_dir and os.path.exists(os.path.join(best_dir, TRAINING_STATE_FILE)):
        return best_dir

    # Fall back to checkpoint-best
    best_ckpt = os.path.join(output_dir, "checkpoint-best")
    if os.path.isdir(best_ckpt) and os.path.exists(
        os.path.join(best_ckpt, TRAINING_STATE_FILE)
    ):
        return best_ckpt

    return None


def load_training_state(checkpoint_dir: str) -> Dict[str, Any]:
    """Load the training state JSON from a checkpoint directory."""
    state_path = os.path.join(checkpoint_dir, TRAINING_STATE_FILE)
    with open(state_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train DiffusionNER-Zero (Dream-7B + LoRA) on Pile-NER"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override checkpoint output directory (e.g. /content/drive/MyDrive/checkpoints). "
             "If not set, uses the value from the config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Force start from scratch even if checkpoints exist.",
    )
    args = parser.parse_args()

    # ---- Load config -------------------------------------------------------
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    diff_cfg = cfg["diffusion"]
    ckpt_cfg = cfg["checkpointing"]
    data_cfg = cfg["data"]

    # CLI --output_dir overrides config
    output_dir = args.output_dir or ckpt_cfg.get("output_dir", "checkpoints/")
    keep_last = ckpt_cfg.get("keep_last", 2)

    # ---- Check for existing checkpoint to resume --------------------------
    resume_dir = None
    if not args.no_resume:
        resume_dir = find_latest_checkpoint(output_dir)
        if resume_dir:
            logger.info(f"Found checkpoint to resume from: {resume_dir}")
        else:
            logger.info("No existing checkpoint found. Starting from scratch.")

    # ---- Accelerator -------------------------------------------------------
    precision_map = {
        "bf16": "bf16",
        "fp16": "fp16",
        "fp32": "no",
    }
    mixed_precision = precision_map.get(train_cfg.get("precision", "bf16"), "bf16")

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        log_with="tensorboard",
        project_dir=output_dir,
    )
    set_seed(args.seed)

    accelerator.print("=" * 60)
    accelerator.print("  DiffusionNER-Zero Training")
    accelerator.print("=" * 60)
    accelerator.print(f"  Output dir: {output_dir}")

    # ---- Model & tokenizer -------------------------------------------------
    # Pass resume_dir so adapter weights are loaded via PeftModel.from_pretrained
    # (more reliable than get_peft_model + load_adapter which can fail silently).
    model, tokenizer = load_model_and_tokenizer(cfg, accelerator, resume_dir=resume_dir)
    pad_token_id = tokenizer.pad_token_id

    # ---- Dataset -----------------------------------------------------------
    train_dataset, val_dataset = load_datasets(cfg, tokenizer)
    accelerator.print(f"Train examples: {len(train_dataset)}")
    accelerator.print(f"Val   examples: {len(val_dataset)}")

    max_seq_length = data_cfg.get("max_seq_length", 512)

    # ---- DataLoaders -------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size_per_gpu"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id, max_seq_length),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size_per_gpu"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id, max_seq_length),
        drop_last=False,
    )

    # ---- Optimizer & scheduler ---------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    num_epochs = train_cfg["epochs"]
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_training_steps = steps_per_epoch * num_epochs

    warmup_steps = train_cfg.get("warmup_steps", 200)
    scheduler_type = train_cfg.get("scheduler", "cosine")

    if scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

    # ---- Resume optimizer/scheduler state ----------------------------------
    resume_step = 0
    resume_epoch = 0
    best_val_loss = float("inf")

    if resume_dir:
        training_state = load_training_state(resume_dir)
        resume_step = training_state["global_step"]
        resume_epoch = training_state["epoch"]
        best_val_loss = training_state["best_val_loss"]

        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sched_path = os.path.join(resume_dir, "scheduler.pt")

        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            accelerator.print(f"  Restored optimizer state")
        if os.path.exists(sched_path):
            lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
            accelerator.print(f"  Restored scheduler state")

        accelerator.print(
            f"  Resuming from step={resume_step}, epoch={resume_epoch}, "
            f"best_val_loss={best_val_loss:.4f}"
        )

    # ---- Prepare with accelerator ------------------------------------------
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # ---- Training config ---------------------------------------------------
    pad_loss_weight = diff_cfg.get("pad_loss_weight", 0.05)
    use_random_truncation = diff_cfg.get("random_truncation", True)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    save_every = ckpt_cfg.get("save_every", 500)
    eval_every = ckpt_cfg.get("eval_every", 500)
    adapter_only = ckpt_cfg.get("save_adapter_only", True)

    # ---- Tracking ----------------------------------------------------------
    global_step = resume_step

    # CSV logger (appends to output_dir/training_log.csv, survives restarts)
    csv_logger = TrainingLogger(output_dir) if accelerator.is_main_process else None

    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "diffusion_ner_zero",
            config={
                "learning_rate": train_cfg["learning_rate"],
                "epochs": num_epochs,
                "batch_size_per_gpu": train_cfg["batch_size_per_gpu"],
                "gradient_accumulation_steps": grad_accum_steps,
                "lora_r": cfg["lora"]["r"],
                "noise_schedule": diff_cfg.get("noise_schedule", "log_linear"),
                "resumed_from_step": resume_step,
            },
        )

    # ---- Sanity check: verify model works before training loop --------------
    if resume_dir and accelerator.is_main_process:
        print("=" * 50, flush=True)
        print("RESUME SANITY CHECK", flush=True)
        # Check adapter weights are non-zero
        unwrapped_check = accelerator.unwrap_model(model)
        lora_norms = {}
        for name, param in unwrapped_check.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_norms[name] = param.data.abs().mean().item()
                if len(lora_norms) >= 4:
                    break
        print(f"  LoRA weight norms (first 4): {lora_norms}", flush=True)

        # Test forward pass on the first val batch
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(val_loader))
            test_prompt = test_batch["prompt_ids"]
            test_comp = test_batch["completion_ids"]
            test_t = torch.full((test_prompt.size(0),), 0.5, device=test_prompt.device)
            test_noisy, test_mask = mask_completion_tokens(test_comp, test_t, MASK_TOKEN_ID)
            test_input = torch.cat([test_prompt, test_noisy], dim=1)
            test_out = unwrapped_check(input_ids=test_input, num_logits_to_keep=test_input.size(1))
            test_logits = test_out.logits
            test_comp_logits = test_logits[:, test_prompt.size(1):, :]
            test_masked_logits = test_comp_logits[test_mask]
            test_masked_targets = test_comp[test_mask]
            if test_masked_logits.numel() > 0:
                test_ce = F.cross_entropy(test_masked_logits, test_masked_targets)
                print(f"  Sanity check CE loss: {test_ce.item():.4f}", flush=True)
                print(f"  logits.shape: {list(test_logits.shape)}", flush=True)
                print(f"  mask_sum: {test_mask.sum().item()}/{test_mask.numel()}", flush=True)
            else:
                print(f"  WARNING: sanity check got empty masked_logits!", flush=True)
        model.train()
        print("=" * 50, flush=True)

    # ---- Training loop -----------------------------------------------------
    accelerator.print(f"\nStarting training for {num_epochs} epochs")
    accelerator.print(f"  Steps per epoch:      {steps_per_epoch}")
    accelerator.print(f"  Total training steps: {total_training_steps}")
    accelerator.print(f"  Gradient accum steps: {grad_accum_steps}")
    accelerator.print(f"  Effective batch size: "
                       f"{train_cfg['batch_size_per_gpu'] * grad_accum_steps * accelerator.num_processes}")
    accelerator.print(f"  Mixed precision:      {mixed_precision}")
    accelerator.print(f"  Random truncation:    {use_random_truncation}")
    accelerator.print(f"  PAD loss weight:      {pad_loss_weight}")
    accelerator.print(f"  Keep last checkpoints: {keep_last}")
    if resume_step > 0:
        accelerator.print(f"  Resuming from step:   {resume_step}")
    accelerator.print()

    model.train()
    t_start = time.time()
    _diag_batches_logged = 0  # counter for first-batch diagnostics

    # Progress bar (shows in Colab/terminal)
    pbar = tqdm(
        total=total_training_steps,
        initial=resume_step,
        desc="Training",
        disable=not accelerator.is_main_process,
        unit="step",
    )

    for epoch in range(num_epochs):
        # Skip completed epochs when resuming
        if epoch < resume_epoch:
            continue

        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            # Skip batches already processed in resumed epoch
            if epoch == resume_epoch and resume_step > 0:
                batches_in_step = grad_accum_steps
                completed_batches = (resume_step - (resume_epoch * steps_per_epoch)) * batches_in_step
                if batch_idx < completed_batches:
                    continue

            with accelerator.accumulate(model):
                device = batch["prompt_ids"].device

                loss = training_step(
                    batch=batch,
                    model=model,
                    pad_token_id=pad_token_id,
                    pad_loss_weight=pad_loss_weight,
                    use_random_truncation=use_random_truncation,
                    device=device,
                )

                # Diagnostic: log details for the first 3 batches after resume
                if _diag_batches_logged < 3 and accelerator.is_main_process:
                    _diag_batches_logged += 1
                    print(
                        f"  [Diag] batch {_diag_batches_logged}: loss={loss.item():.6f}, "
                        f"prompt.shape={list(batch['prompt_ids'].shape)}, "
                        f"comp.shape={list(batch['completion_ids'].shape)}, "
                        f"comp_lengths={batch['completion_lengths'].tolist()}",
                        flush=True,
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Track loss (use the un-accumulated loss for logging)
            epoch_loss += loss.detach().item()
            epoch_steps += 1

            # Count global steps (after gradient accumulation)
            if accelerator.sync_gradients:
                global_step += 1

                # ---- Logging ---------------------------------------------------
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start

                    # Progress bar update
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{current_lr:.1e}",
                        epoch=f"{epoch+1}/{num_epochs}",
                    )

                    if accelerator.is_main_process:
                        # CSV log
                        csv_logger.log_train(
                            step=global_step, epoch=epoch,
                            train_loss=avg_loss,
                            learning_rate=current_lr,
                            elapsed_sec=elapsed,
                        )
                        # TensorBoard log
                        accelerator.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/epoch": epoch + 1,
                                "train/step_loss": loss.detach().item(),
                            },
                            step=global_step,
                        )

                pbar.update(1)

                # ---- Validation ------------------------------------------------
                if global_step % eval_every == 0:
                    accelerator.print(f"\n  Running validation at step {global_step}...")
                    val_loss = validate(
                        model=model,
                        val_loader=val_loader,
                        pad_token_id=pad_token_id,
                        pad_loss_weight=pad_loss_weight,
                        device=device,
                        accelerator=accelerator,
                        max_batches=50,  # Cap for speed
                    )
                    accelerator.print(f"  Val loss: {val_loss:.4f}")

                    if accelerator.is_main_process:
                        elapsed = time.time() - t_start
                        csv_logger.log_val(
                            step=global_step, epoch=epoch,
                            val_loss=val_loss, elapsed_sec=elapsed,
                        )
                        accelerator.log(
                            {"val/loss": val_loss},
                            step=global_step,
                        )

                    # Best model tracking
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        accelerator.print(
                            f"  New best val loss: {best_val_loss:.4f} -- saving best adapter"
                        )
                        save_checkpoint(
                            model, optimizer, lr_scheduler,
                            accelerator, output_dir,
                            step=global_step, epoch=epoch,
                            best_val_loss=best_val_loss,
                            adapter_only=adapter_only,
                            tag="best", keep_last=keep_last,
                        )

                    model.train()

                # ---- Periodic checkpoint ---------------------------------------
                if global_step % save_every == 0:
                    save_checkpoint(
                        model, optimizer, lr_scheduler,
                        accelerator, output_dir,
                        step=global_step, epoch=epoch,
                        best_val_loss=best_val_loss,
                        adapter_only=adapter_only,
                        keep_last=keep_last,
                    )

        # End-of-epoch logging
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        accelerator.print(
            f"\n  Epoch {epoch+1}/{num_epochs} complete. "
            f"Avg loss: {avg_epoch_loss:.4f}\n"
        )

    # ---- Final save --------------------------------------------------------
    pbar.close()
    accelerator.print("Training complete!")
    accelerator.print(f"  Best val loss: {best_val_loss:.4f}")
    save_checkpoint(
        model, optimizer, lr_scheduler,
        accelerator, output_dir,
        step=global_step, epoch=num_epochs - 1,
        best_val_loss=best_val_loss,
        adapter_only=adapter_only,
        tag="final", keep_last=keep_last,
    )

    # End tracking
    if accelerator.is_main_process:
        accelerator.end_training()
        # Save final loss curve plot alongside the CSV
        log_path = os.path.join(output_dir, "training_log.csv")
        if os.path.exists(log_path):
            plot_path = os.path.join(output_dir, "training_curves.png")
            plot_training_log(log_path, save_path=plot_path, show=False)
            accelerator.print(f"  Loss curves saved to {plot_path}")

    total_time = time.time() - t_start
    accelerator.print(f"  Total training time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
