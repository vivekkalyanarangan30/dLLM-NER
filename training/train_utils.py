"""Utility functions for MDLM (Masked Diffusion Language Model) training.

Implements the log-linear absorbing-state noise schedule, MDLM importance
weighting, masking helpers, PAD loss weighting, and random truncation
(Dream-Coder style).

Dream-7B uses MASK_TOKEN_ID = 126336 as its absorbing-state mask token.
"""

import torch
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASK_TOKEN_ID = 126336


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def log_linear_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """Log-linear absorbing-state noise schedule.

    alpha(t) = 1 - t  (linear schedule in log space for MDLM).
    Returns mask probability at timestep *t*.

    Parameters
    ----------
    t : torch.Tensor
        Timestep values in [0, 1].

    Returns
    -------
    torch.Tensor
        Mask probability at each timestep.  For log-linear: mask_prob = t.
    """
    return t  # mask_prob = t, where t in [0, 1]


def alpha(t: torch.Tensor) -> torch.Tensor:
    """Alpha function: probability of a token being UNMASKED at time *t*.

    For the log-linear schedule: alpha(t) = 1 - t.
    """
    return 1.0 - t


def alpha_prime(t: torch.Tensor) -> torch.Tensor:
    """Derivative of alpha(t) with respect to t.

    For log-linear: d/dt (1 - t) = -1.
    """
    return torch.ones_like(t) * (-1.0)


def mdlm_importance_weight(t: torch.Tensor) -> torch.Tensor:
    """MDLM importance weighting factor.

    weight(t) = -alpha'(t) / (1 - alpha(t))

    For the log-linear schedule this simplifies to 1/t.  We clamp *t* away
    from zero to avoid division-by-zero.

    Parameters
    ----------
    t : torch.Tensor
        Timestep values in (0, 1].

    Returns
    -------
    torch.Tensor
        Per-sample importance weights.
    """
    return 1.0 / t.clamp(min=1e-5)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask_completion_tokens(
    completion_ids: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
) -> tuple:
    """Apply absorbing-state masking to completion tokens at timestep *t*.

    Each token is independently masked with probability equal to the noise
    schedule value at *t*.

    Parameters
    ----------
    completion_ids : torch.Tensor
        Shape ``(batch, seq_len)`` -- clean completion token IDs.
    t : torch.Tensor
        Shape ``(batch,)`` or scalar -- timestep in [0, 1].
    mask_token_id : int
        Token ID used for the absorbing mask state.

    Returns
    -------
    noisy_completion : torch.Tensor
        Completion with some tokens replaced by *mask_token_id*.
    mask : torch.BoolTensor
        Boolean tensor (same shape as *completion_ids*) indicating which
        positions were masked (``True`` = masked).
    """
    mask_prob = log_linear_noise_schedule(t)
    if mask_prob.dim() == 0:
        mask_prob = mask_prob.unsqueeze(0)
    # (batch, 1) for broadcasting across the sequence dimension
    mask_prob = mask_prob.view(-1, 1)

    # Bernoulli mask -- each token independently masked
    mask = torch.bernoulli(mask_prob.expand_as(completion_ids)).bool()

    noisy_completion = completion_ids.clone()
    noisy_completion[mask] = mask_token_id

    return noisy_completion, mask


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def apply_pad_loss_weight(
    loss_per_token: torch.Tensor,
    target_ids: torch.Tensor,
    pad_token_id: int,
    pad_weight: float = 0.05,
) -> torch.Tensor:
    """Down-weight loss on PAD token predictions.

    Tokens whose ground-truth target is the PAD token receive a reduced
    weight in the loss computation.  All other tokens receive weight 1.0.

    Parameters
    ----------
    loss_per_token : torch.Tensor
        Shape ``(N,)`` -- per-token cross-entropy losses for masked positions.
    target_ids : torch.Tensor
        Shape ``(N,)`` -- target (clean) token IDs for those same positions.
    pad_token_id : int
        The token ID that represents padding.
    pad_weight : float
        Multiplicative weight applied to PAD-target losses (default 0.05).

    Returns
    -------
    torch.Tensor
        Scalar -- weighted mean loss.
    """
    weights = torch.where(target_ids == pad_token_id, pad_weight, 1.0)
    return (loss_per_token * weights).sum() / weights.sum()


# ---------------------------------------------------------------------------
# Random truncation (Dream-Coder)
# ---------------------------------------------------------------------------

def random_truncation(
    completion_ids_batch: torch.Tensor,
    completion_lengths: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Random truncation from Dream-Coder.

    Each completion in the batch is truncated to the length of another
    randomly chosen example.  This removes the majority of PAD tokens from
    the loss computation, preventing the model from over-learning to predict
    PAD.

    Parameters
    ----------
    completion_ids_batch : torch.Tensor
        Shape ``(batch, max_completion_len)`` -- padded completions.
    completion_lengths : torch.Tensor
        Shape ``(batch,)`` -- actual (unpadded) length of each completion.
    pad_token_id : int
        Token ID used for padding.

    Returns
    -------
    torch.Tensor
        Truncated completions with the same shape as input, where excess
        positions are filled with *pad_token_id*.
    """
    batch_size = completion_ids_batch.size(0)
    max_len = completion_ids_batch.size(1)

    # Randomly assign each example another example's length
    perm = torch.randperm(batch_size)
    target_lengths = completion_lengths[perm]

    # Truncate: set positions beyond the target length to PAD
    positions = torch.arange(max_len, device=completion_ids_batch.device).unsqueeze(0)
    truncation_mask = positions >= target_lengths.unsqueeze(1)

    result = completion_ids_batch.clone()
    result[truncation_mask] = pad_token_id

    return result


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def compute_unmask_count(step: int, total_steps: int, n_masked: int) -> int:
    """Compute how many tokens to unmask at a given denoising step.

    Uses a linear schedule: at each step we unmask a proportional fraction
    so that by the final step all tokens are unmasked.

    Parameters
    ----------
    step : int
        Current step index (0-based).
    total_steps : int
        Total number of denoising steps.
    n_masked : int
        Number of currently masked tokens.

    Returns
    -------
    int
        Number of tokens to unmask (at least 1 if any remain masked).
    """
    if n_masked == 0:
        return 0
    # Target: remaining masked = n_masked * (1 - (step+1)/total_steps)
    target_remaining = int(n_masked * (1.0 - (step + 1) / total_steps))
    target_remaining = max(0, target_remaining)
    n_unmask = n_masked - target_remaining
    return max(1, n_unmask)  # always unmask at least 1
