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
from accelerate.utils import set_seed
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

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

def load_model_and_tokenizer(cfg: Dict[str, Any], accelerator: Accelerator):
    """Load Dream-7B base model, apply LoRA, and return (model, tokenizer).

    The base model weights are frozen; only LoRA adapter parameters are
    trainable.
    """
    model_name = cfg["model"]["name"]
    torch_dtype_str = cfg["model"].get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, torch_dtype_str)

    accelerator.print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
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
    """Load the preprocessed Pile-NER dataset and return train/val splits.

    Expects a pre-tokenized Arrow dataset on disk with columns:
      - ``prompt_ids``: List[int]
      - ``completion_ids``: List[int]
      - ``completion_length``: int  (unpadded length of the completion)

    If the dataset is not yet pre-tokenized, it falls back to loading the
    raw HuggingFace dataset and tokenizing on-the-fly.
    """
    data_cfg = cfg["data"]
    max_completion_length = data_cfg.get("max_completion_length", 128)

    # Try loading pre-tokenized dataset from disk first
    dataset_path = data_cfg.get("preprocessed_path", None)
    if dataset_path and Path(dataset_path).exists():
        logger.info(f"Loading pre-tokenized dataset from {dataset_path}")
        full_dataset = load_from_disk(dataset_path)
        if isinstance(full_dataset, dict):
            train_dataset = full_dataset["train"]
            val_dataset = full_dataset["validation"]
        else:
            split_ratio = data_cfg.get("train_split_ratio", 0.9)
            split = full_dataset.train_test_split(
                test_size=1.0 - split_ratio, seed=42
            )
            train_dataset = split["train"]
            val_dataset = split["test"]
        return train_dataset, val_dataset

    # Fallback: load raw dataset from HuggingFace and tokenize
    logger.info(
        f"Loading raw dataset: {data_cfg['dataset']} (will tokenize on-the-fly)"
    )
    from datasets import load_dataset as hf_load_dataset

    raw = hf_load_dataset(data_cfg["dataset"], split="train")

    # Tokenize
    pad_token_id = tokenizer.pad_token_id

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
    prompt_ids = batch["prompt_ids"]           # (B, P)
    completion_ids = batch["completion_ids"]    # (B, C)
    completion_lengths = batch["completion_lengths"]  # (B,)
    prompt_lengths = batch["prompt_lengths"]    # (B,)

    batch_size = prompt_ids.size(0)
    prompt_len = prompt_ids.size(1)   # padded prompt length (all same after collate)
    comp_len = completion_ids.size(1)

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

    # Concatenate prompt + noisy completion to form full input
    noisy_input = torch.cat([prompt_ids, noisy_completion], dim=1)  # (B, P+C)

    # --- Forward pass -------------------------------------------------------
    outputs = model(input_ids=noisy_input)
    logits = outputs.logits  # (B, P+C, V)

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
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        prompt_ids = batch["prompt_ids"]
        completion_ids = batch["completion_ids"]
        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        comp_len = completion_ids.size(1)

        # Fixed timestep for deterministic validation
        t = torch.full((batch_size,), 0.5, device=device)

        noisy_completion, mask = mask_completion_tokens(
            completion_ids, t, mask_token_id=MASK_TOKEN_ID
        )

        noisy_input = torch.cat([prompt_ids, noisy_completion], dim=1)

        outputs = model(input_ids=noisy_input)
        logits = outputs.logits
        completion_logits = logits[:, prompt_len:, :]

        masked_logits = completion_logits[mask]
        masked_targets = completion_ids[mask]

        if masked_logits.numel() == 0:
            continue

        loss_per_token = F.cross_entropy(
            masked_logits, masked_targets, reduction="none"
        )

        weighted_loss = apply_pad_loss_weight(
            loss_per_token, masked_targets, pad_token_id, pad_weight=pad_loss_weight
        )

        total_loss += weighted_loss.item()
        n_batches += 1

    model.train()

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
    model, tokenizer = load_model_and_tokenizer(cfg, accelerator)
    pad_token_id = tokenizer.pad_token_id

    # ---- Load adapter weights if resuming ----------------------------------
    if resume_dir:
        accelerator.print(f"  Resuming adapter weights from {resume_dir}")
        unwrapped = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else model
        unwrapped.load_adapter(resume_dir, adapter_name="default")

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
                    accelerator.print(
                        f"  [Epoch {epoch+1}/{num_epochs}] "
                        f"Step {global_step}/{total_training_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/epoch": epoch + 1,
                                "train/step_loss": loss.detach().item(),
                            },
                            step=global_step,
                        )

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

    total_time = time.time() - t_start
    accelerator.print(f"  Total training time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
