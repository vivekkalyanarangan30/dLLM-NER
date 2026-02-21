"""Main inference module for DiffusionNER-Zero.

Implements the iterative denoising loop for entity extraction using
Dream-7B (masked diffusion language model) fine-tuned with LoRA.

The denoising process:
1. Encode the prompt and create a fully masked output region.
2. At each step, run a forward pass to get logits for all positions.
3. Apply a decaying PAD penalty to discourage premature termination.
4. Unmask the top-confidence masked positions.
5. (Optional) Re-mask the lowest-confidence committed positions (ReMDM).
6. After all steps, decode the output and parse structured entities.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .parse import parse_entities
from .remask import compute_remask_count, remask_low_confidence

# Dream-7B uses token ID 126336 for the [MASK] token.
MASK_TOKEN_ID = 126336


def apply_pad_penalty(
    logits: torch.Tensor,
    step: int,
    total_steps: int,
    pad_token_id: int,
    max_penalty: float = 5.0,
) -> torch.Tensor:
    """Apply a decaying penalty on PAD logits to prevent premature termination.

    At step 0 the full penalty is applied; it decays linearly to zero at the
    final step, allowing the model to legitimately produce PAD tokens towards
    the end of generation when the entity list is shorter than the output buffer.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(seq_len, vocab_size)`` -- logits for the output region.
    step : int
        Current denoising step (0-indexed).
    total_steps : int
        Total number of denoising steps.
    pad_token_id : int
        Token ID of the PAD token.
    max_penalty : float, optional
        Maximum penalty applied at step 0 (default 5.0).

    Returns
    -------
    torch.Tensor
        Modified logits with PAD penalty applied (same shape, in-place).
    """
    decay = 1.0 - (step / total_steps)
    logits[:, pad_token_id] -= max_penalty * decay
    return logits


def compute_unmask_count(step: int, total_steps: int, n_masked: int) -> int:
    """Compute how many tokens to unmask at this denoising step.

    Uses a linear schedule: at each step we target a fraction
    ``(step + 1) / total_steps`` of the original masked tokens to be
    unmasked.  The function returns the difference between the current
    number of masked tokens and the target remaining count, ensuring at
    least 1 token is unmasked per step (when tokens remain).

    Parameters
    ----------
    step : int
        Current denoising step (0-indexed).
    total_steps : int
        Total number of denoising steps.
    n_masked : int
        Number of currently masked tokens.

    Returns
    -------
    int
        Number of tokens to unmask at this step.  Returns 0 only when
        ``n_masked == 0``.
    """
    if n_masked == 0:
        return 0

    target_remaining = int(n_masked * (1.0 - (step + 1) / total_steps))
    target_remaining = max(0, target_remaining)
    n_unmask = n_masked - target_remaining
    return max(1, n_unmask)


@torch.no_grad()
def extract_entities(
    model,
    tokenizer,
    text: str,
    entity_types: List[str],
    num_steps: int = 8,
    max_output_len: int = 128,
    use_remasking: bool = False,
    remask_ratio: float = 0.3,
    pad_penalty_max: float = 5.0,
) -> List[Dict[str, str]]:
    """Extract named entities from text using iterative denoising.

    Constructs a prompt from the input text and entity types, initialises
    the output region with MASK tokens, and iteratively unmasks tokens in
    order of descending confidence.

    Parameters
    ----------
    model : PreTrainedModel
        Dream-7B model (with merged LoRA adapter).
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    text : str
        Input passage to extract entities from.
    entity_types : List[str]
        Entity types to look for (e.g. ``["person", "organization", "date"]``).
    num_steps : int, optional
        Number of denoising steps (default 8).
    max_output_len : int, optional
        Maximum number of output tokens (default 128).
    use_remasking : bool, optional
        Whether to apply ReMDM re-masking of low-confidence committed
        tokens (default False).
    remask_ratio : float, optional
        Fraction of committed tokens eligible for re-masking (default 0.3).
    pad_penalty_max : float, optional
        Maximum PAD penalty at step 0 (default 5.0).

    Returns
    -------
    List[Dict[str, str]]
        List of ``{"type": str, "text": str}`` entity dicts parsed from
        the model output.
    """
    # Build prompt
    types_str = ", ".join(entity_types)
    prompt = f"Extract entities of types: {types_str}\nText: {text}\nEntities:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Initialise fully masked output region
    output_ids = [MASK_TOKEN_ID] * max_output_len
    sequence = torch.tensor(
        prompt_ids + output_ids, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    prompt_len = len(prompt_ids)

    for step in range(num_steps):
        # Forward pass
        logits = model(input_ids=sequence).logits[0]  # (seq_len, vocab_size)
        output_logits = logits[prompt_len:]

        # Penalise PAD tokens
        output_logits = apply_pad_penalty(
            output_logits, step, num_steps, pad_token_id, pad_penalty_max
        )

        # Compute predictions and confidences
        probs = F.softmax(output_logits, dim=-1)
        predicted_ids = probs.argmax(dim=-1)
        confidences = probs.max(dim=-1).values

        # Identify still-masked positions in the output region
        current_output = sequence[0, prompt_len:]
        masked_positions = (current_output == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        n_masked = len(masked_positions)

        if n_masked == 0:
            break

        # Determine how many tokens to unmask
        n_unmask = compute_unmask_count(step, num_steps, n_masked)

        # Pick the top-confidence masked positions to unmask
        masked_confidences = confidences[masked_positions]
        k = min(n_unmask, len(masked_positions))
        top_k_indices = masked_confidences.topk(k).indices
        positions_to_unmask = masked_positions[top_k_indices]

        # Unmask selected positions
        for pos in positions_to_unmask:
            sequence[0, prompt_len + pos] = predicted_ids[pos]

        # Optional ReMDM re-masking (skip on the final step)
        if use_remasking and step < num_steps - 1:
            committed = (sequence[0, prompt_len:] != MASK_TOKEN_ID).nonzero(
                as_tuple=True
            )[0]
            n_committed = len(committed)
            n_remask = compute_remask_count(
                step, num_steps, n_committed, remask_ratio
            )
            if n_remask > 0:
                sequence[0] = remask_low_confidence(
                    sequence[0], confidences, prompt_len, n_remask, MASK_TOKEN_ID
                )

    # Decode output, filtering out MASK and PAD tokens
    output_token_ids = sequence[0, prompt_len:].tolist()
    clean_ids = [
        tid
        for tid in output_token_ids
        if tid != MASK_TOKEN_ID and tid != pad_token_id
    ]
    output_text = tokenizer.decode(clean_ids, skip_special_tokens=True)

    return parse_entities(output_text)


@torch.no_grad()
def extract_entities_with_trajectory(
    model,
    tokenizer,
    text: str,
    entity_types: List[str],
    num_steps: int = 8,
    max_output_len: int = 128,
    use_remasking: bool = False,
    remask_ratio: float = 0.3,
    pad_penalty_max: float = 5.0,
) -> Dict[str, object]:
    """Extract entities and record the full denoising trajectory.

    Identical to :func:`extract_entities` but also captures the intermediate
    state at every denoising step.  This is used for:

    - **Self-correction analysis**: tracking when the model fixes mistakes
      across steps.
    - **Denoising visualisation**: showing the progressive unmasking process.
    - **Debugging**: inspecting confidence distributions and token choices.

    Parameters
    ----------
    model : PreTrainedModel
        Dream-7B model (with merged LoRA adapter).
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    text : str
        Input passage to extract entities from.
    entity_types : List[str]
        Entity types to look for.
    num_steps : int, optional
        Number of denoising steps (default 8).
    max_output_len : int, optional
        Maximum number of output tokens (default 128).
    use_remasking : bool, optional
        Whether to apply ReMDM re-masking (default False).
    remask_ratio : float, optional
        Fraction of committed tokens eligible for re-masking (default 0.3).
    pad_penalty_max : float, optional
        Maximum PAD penalty at step 0 (default 5.0).

    Returns
    -------
    Dict[str, object]
        Dictionary with keys:

        - ``"entities"`` (List[Dict[str, str]]): Final parsed entities.
        - ``"trajectory"`` (List[Dict]): One entry per step, each containing:

          - ``"step"`` (int): Step index (0-based).
          - ``"text"`` (str): Decoded output at this step (MASK/PAD filtered).
          - ``"entities"`` (List[Dict[str, str]]): Entities parsed at this step.
          - ``"n_masked"`` (int): Number of remaining masked positions.
          - ``"n_committed"`` (int): Number of committed (unmasked) positions.
          - ``"mean_confidence"`` (float): Mean confidence over committed positions.
    """
    # Build prompt
    types_str = ", ".join(entity_types)
    prompt = f"Extract entities of types: {types_str}\nText: {text}\nEntities:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Initialise fully masked output region
    output_ids = [MASK_TOKEN_ID] * max_output_len
    sequence = torch.tensor(
        prompt_ids + output_ids, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    prompt_len = len(prompt_ids)

    trajectory: List[Dict] = []

    for step in range(num_steps):
        # Forward pass
        logits = model(input_ids=sequence).logits[0]  # (seq_len, vocab_size)
        output_logits = logits[prompt_len:]

        # Penalise PAD tokens
        output_logits = apply_pad_penalty(
            output_logits, step, num_steps, pad_token_id, pad_penalty_max
        )

        # Compute predictions and confidences
        probs = F.softmax(output_logits, dim=-1)
        predicted_ids = probs.argmax(dim=-1)
        confidences = probs.max(dim=-1).values

        # Identify still-masked positions
        current_output = sequence[0, prompt_len:]
        masked_positions = (current_output == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        n_masked = len(masked_positions)

        if n_masked == 0:
            # Record final converged state and stop
            step_ids = sequence[0, prompt_len:].tolist()
            step_clean = [
                tid
                for tid in step_ids
                if tid != MASK_TOKEN_ID and tid != pad_token_id
            ]
            step_text = tokenizer.decode(step_clean, skip_special_tokens=True)
            committed_positions = (current_output != MASK_TOKEN_ID).nonzero(
                as_tuple=True
            )[0]
            mean_conf = (
                confidences[committed_positions].mean().item()
                if len(committed_positions) > 0
                else 0.0
            )
            trajectory.append(
                {
                    "step": step,
                    "text": step_text,
                    "entities": parse_entities(step_text),
                    "n_masked": 0,
                    "n_committed": len(committed_positions),
                    "mean_confidence": mean_conf,
                }
            )
            break

        # Unmask top-confidence positions
        n_unmask = compute_unmask_count(step, num_steps, n_masked)
        masked_confidences = confidences[masked_positions]
        k = min(n_unmask, len(masked_positions))
        top_k_indices = masked_confidences.topk(k).indices
        positions_to_unmask = masked_positions[top_k_indices]

        for pos in positions_to_unmask:
            sequence[0, prompt_len + pos] = predicted_ids[pos]

        # Optional ReMDM re-masking
        if use_remasking and step < num_steps - 1:
            committed = (sequence[0, prompt_len:] != MASK_TOKEN_ID).nonzero(
                as_tuple=True
            )[0]
            n_committed_for_remask = len(committed)
            n_remask = compute_remask_count(
                step, num_steps, n_committed_for_remask, remask_ratio
            )
            if n_remask > 0:
                sequence[0] = remask_low_confidence(
                    sequence[0], confidences, prompt_len, n_remask, MASK_TOKEN_ID
                )

        # Record trajectory snapshot AFTER unmasking (and possible re-masking)
        current_output_after = sequence[0, prompt_len:]
        n_masked_after = (current_output_after == MASK_TOKEN_ID).sum().item()
        committed_after = (current_output_after != MASK_TOKEN_ID).nonzero(
            as_tuple=True
        )[0]
        n_committed_after = len(committed_after)

        step_ids = current_output_after.tolist()
        step_clean = [
            tid
            for tid in step_ids
            if tid != MASK_TOKEN_ID and tid != pad_token_id
        ]
        step_text = tokenizer.decode(step_clean, skip_special_tokens=True)

        mean_conf = (
            confidences[committed_after].mean().item()
            if n_committed_after > 0
            else 0.0
        )

        trajectory.append(
            {
                "step": step,
                "text": step_text,
                "entities": parse_entities(step_text),
                "n_masked": n_masked_after,
                "n_committed": n_committed_after,
                "mean_confidence": mean_conf,
            }
        )

    # Final decoded output
    output_token_ids = sequence[0, prompt_len:].tolist()
    clean_ids = [
        tid
        for tid in output_token_ids
        if tid != MASK_TOKEN_ID and tid != pad_token_id
    ]
    output_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
    final_entities = parse_entities(output_text)

    return {
        "entities": final_entities,
        "trajectory": trajectory,
    }
