"""ReMDM (Re-Masking Diffusion Model) remasking logic for improved inference.

During the iterative denoising loop, the standard approach only unmasks tokens
and never re-masks committed decisions.  ReMDM improves upon this by allowing
the model to "change its mind": at each step, a fraction of the
lowest-confidence committed tokens are re-masked so the model can reconsider
them on subsequent steps.

This is particularly useful for NER because entity boundaries and types are
inter-dependent -- correcting one token may require revising earlier choices.

Reference:
    Shi et al., "Diffusion Language Models Can Perform Many Tasks with
    Scaling and Instruction-Following", 2024.
"""

import torch


def remask_low_confidence(
    sequence: torch.Tensor,
    confidences: torch.Tensor,
    prompt_len: int,
    n_remask: int,
    mask_token_id: int,
) -> torch.Tensor:
    """Re-mask the lowest-confidence committed tokens in the output region.

    Only tokens in the output region (positions ``prompt_len`` onward) that are
    currently *not* masked are candidates for re-masking.  Among those, the
    ``n_remask`` tokens with the lowest confidence scores are set back to
    ``mask_token_id``.

    Parameters
    ----------
    sequence : torch.Tensor
        Shape ``(seq_len,)`` -- the full sequence (prompt + output), 1-D.
    confidences : torch.Tensor
        Shape ``(output_len,)`` -- confidence scores (e.g. max softmax
        probability) for each position in the output region.
    prompt_len : int
        Length of the prompt prefix.  Tokens before this index are never
        touched.
    n_remask : int
        Number of committed tokens to re-mask.  If this exceeds the number
        of committed tokens, all committed tokens are re-masked.
    mask_token_id : int
        Token ID used for the ``[MASK]`` token (126336 for Dream-7B).

    Returns
    -------
    torch.Tensor
        Updated sequence (same shape) with ``n_remask`` committed output
        tokens set back to ``mask_token_id``.
    """
    if n_remask <= 0:
        return sequence

    output_region = sequence[prompt_len:]

    # Identify committed (non-masked) positions in the output
    committed_mask = output_region != mask_token_id
    committed_positions = committed_mask.nonzero(as_tuple=True)[0]

    n_committed = len(committed_positions)
    if n_committed == 0:
        return sequence

    # Clamp n_remask to the number of committed tokens
    n_remask = min(n_remask, n_committed)

    # Get confidence scores for committed positions only
    committed_confidences = confidences[committed_positions]

    # Select the n_remask positions with the LOWEST confidence
    _, bottom_k_local = committed_confidences.topk(n_remask, largest=False)
    positions_to_remask = committed_positions[bottom_k_local]

    # Apply re-masking
    sequence = sequence.clone()
    sequence[prompt_len + positions_to_remask] = mask_token_id

    return sequence


def compute_remask_count(
    step: int,
    total_steps: int,
    n_committed: int,
    remask_ratio: float = 0.3,
) -> int:
    """Compute the number of tokens to re-mask at a given denoising step.

    The re-mask count decays linearly with the step number: early steps
    re-mask more aggressively, and later steps re-mask less to allow the
    sequence to converge.  At the last step, no tokens are re-masked.

    The formula is::

        decay = 1.0 - (step + 1) / total_steps
        n_remask = round(remask_ratio * decay * n_committed)

    Parameters
    ----------
    step : int
        Current denoising step (0-indexed).
    total_steps : int
        Total number of denoising steps.
    n_committed : int
        Number of currently committed (non-masked) tokens in the output.
    remask_ratio : float, optional
        Base fraction of committed tokens eligible for re-masking
        (default 0.3).

    Returns
    -------
    int
        Number of tokens to re-mask.  Always non-negative and at most
        ``n_committed``.  Returns 0 if ``n_committed == 0`` or if this
        is the last step.
    """
    if n_committed == 0:
        return 0

    if total_steps <= 0:
        return 0

    # Linear decay: full ratio at step 0, zero at the final step
    decay = 1.0 - (step + 1) / total_steps

    if decay <= 0.0:
        return 0

    n_remask = round(remask_ratio * decay * n_committed)
    return min(max(n_remask, 0), n_committed)
