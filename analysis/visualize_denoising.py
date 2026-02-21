"""Visualize the denoising trajectory of DiffusionNER-Zero.

Shows how tokens get progressively unmasked during the iterative denoising
process.  Produces:

1. A heatmap/grid figure (PNG) where the x-axis is output token position, the
   y-axis is denoising step, and cell colour indicates confidence.
2. A plain-text step-by-step rendering of the denoising process.

Trajectory format::

    [
        {"step": int, "token_ids": list[int], "confidences": list[float]},
        ...
    ]

Dream-7B uses MASK_TOKEN_ID = 126336.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MASK_TOKEN_ID = 126336
MASK_DISPLAY = "[M]"
MASK_COLOR = "#d3d3d3"          # light gray for masked tokens
HIGH_CONF_CMAP = "YlGn"        # yellow-green for confidence


# ---------------------------------------------------------------------------
# Grid / heatmap visualization
# ---------------------------------------------------------------------------

def visualize_trajectory(
    trajectory: List[Dict[str, Any]],
    prompt: str,
    tokenizer: Any,
    save_path: str = "figures/denoising_trajectory.png",
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    max_tokens: int = 40,
    figsize_per_token: float = 0.55,
    figsize_per_step: float = 0.50,
) -> None:
    """Visualize a single denoising trajectory as a colour-coded grid.

    The grid has:

    - **x-axis**: output token positions (left to right).
    - **y-axis**: denoising steps (top = step 0 / fully masked, bottom = final).
    - **Colour**: masked positions are gray; unmasked positions are coloured by
      confidence (darker green = higher confidence).
    - **Text**: decoded token text is shown inside each cell.

    Parameters
    ----------
    trajectory : list of dict
        Each dict has ``"step"`` (int), ``"token_ids"`` (list of int), and
        ``"confidences"`` (list of float).
    prompt : str
        The prompt text (shown as the figure title / subtitle).
    tokenizer
        A HuggingFace-compatible tokenizer with a ``decode`` method.
    save_path : str
        File path for saving the figure (PNG).
    mask_token_id : int
        The token ID used for [MASK] (default 126336).
    max_tokens : int
        Maximum number of output tokens to display.
    figsize_per_token : float
        Figure width per token position.
    figsize_per_step : float
        Figure height per denoising step.
    """
    n_steps = len(trajectory)
    n_tokens = min(len(trajectory[0]["token_ids"]), max_tokens)

    fig_width = max(6, n_tokens * figsize_per_token)
    fig_height = max(3, n_steps * figsize_per_step + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap(HIGH_CONF_CMAP)

    for row_idx, step_data in enumerate(trajectory):
        step = step_data["step"]
        token_ids = step_data["token_ids"][:n_tokens]
        confidences = step_data["confidences"][:n_tokens]

        for col_idx, (tid, conf) in enumerate(zip(token_ids, confidences)):
            is_mask = (tid == mask_token_id)

            if is_mask:
                color = MASK_COLOR
                text = MASK_DISPLAY
                text_color = "#888888"
            else:
                color = cmap(conf)
                text = tokenizer.decode([tid]).strip()
                if not text:
                    text = f"<{tid}>"
                # Truncate long tokens for display
                if len(text) > 6:
                    text = text[:5] + "..."
                text_color = "black" if conf > 0.3 else "white"

            rect = plt.Rectangle(
                (col_idx, n_steps - 1 - row_idx), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.5,
            )
            ax.add_patch(rect)

            ax.text(
                col_idx + 0.5, n_steps - 1 - row_idx + 0.5,
                text, ha="center", va="center",
                fontsize=7, color=text_color, fontweight="bold",
            )

    ax.set_xlim(0, n_tokens)
    ax.set_ylim(0, n_steps)
    ax.set_xticks([i + 0.5 for i in range(n_tokens)])
    ax.set_xticklabels([str(i) for i in range(n_tokens)], fontsize=6)
    ax.set_xlabel("Output token position", fontsize=9)

    ax.set_yticks([i + 0.5 for i in range(n_steps)])
    ax.set_yticklabels(
        [f"Step {trajectory[n_steps - 1 - i]['step']}" for i in range(n_steps)],
        fontsize=7,
    )
    ax.set_ylabel("Denoising step", fontsize=9)

    # Title with prompt (truncated if too long)
    prompt_display = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    ax.set_title(f"Denoising Trajectory\n{prompt_display}", fontsize=10, pad=10)

    # Legend
    mask_patch = mpatches.Patch(color=MASK_COLOR, label="[MASK]")
    conf_patch = mpatches.Patch(color=cmap(0.8), label="High confidence")
    low_patch = mpatches.Patch(color=cmap(0.2), label="Low confidence")
    ax.legend(
        handles=[mask_patch, conf_patch, low_patch],
        loc="upper right", fontsize=7, framealpha=0.9,
    )

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved denoising trajectory figure to %s", save_path)


# ---------------------------------------------------------------------------
# Multiple trajectories
# ---------------------------------------------------------------------------

def visualize_multiple_trajectories(
    trajectories: List[List[Dict[str, Any]]],
    prompts: List[str],
    tokenizer: Any,
    save_dir: str = "figures/",
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    max_tokens: int = 40,
) -> List[str]:
    """Visualize multiple denoising trajectories, one figure per example.

    Parameters
    ----------
    trajectories : list of list of dict
        Each outer element is a single trajectory (list of step dicts).
    prompts : list of str
        Prompt text for each trajectory.
    tokenizer
        HuggingFace-compatible tokenizer.
    save_dir : str
        Directory to save all figures.
    mask_token_id : int
        The mask token ID.
    max_tokens : int
        Maximum output tokens to display per figure.

    Returns
    -------
    list of str
        Paths to all saved figure files.
    """
    assert len(trajectories) == len(prompts), (
        f"Length mismatch: trajectories={len(trajectories)}, prompts={len(prompts)}"
    )

    os.makedirs(save_dir, exist_ok=True)
    saved_paths: List[str] = []

    for idx, (traj, prompt) in enumerate(zip(trajectories, prompts)):
        save_path = os.path.join(save_dir, f"denoising_trajectory_{idx:03d}.png")
        visualize_trajectory(
            traj, prompt, tokenizer,
            save_path=save_path,
            mask_token_id=mask_token_id,
            max_tokens=max_tokens,
        )
        saved_paths.append(save_path)

    logger.info("Saved %d trajectory figures to %s", len(saved_paths), save_dir)
    return saved_paths


# ---------------------------------------------------------------------------
# Text-based trajectory rendering
# ---------------------------------------------------------------------------

def create_step_by_step_text(
    trajectory: List[Dict[str, Any]],
    tokenizer: Any,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
) -> str:
    """Create a plain-text representation of the denoising process.

    Example output::

        Step 0: [MASK] [MASK] [MASK] [MASK] [MASK]
        Step 1: person [MASK] Ronaldo [MASK] [MASK]
        Step 2: person : Ronaldo | [MASK]
        Step 3: person : Ronaldo | organization : Al Nassr

    Parameters
    ----------
    trajectory : list of dict
        Each dict has ``"step"`` (int), ``"token_ids"`` (list of int), and
        ``"confidences"`` (list of float).
    tokenizer
        HuggingFace-compatible tokenizer.
    mask_token_id : int
        The mask token ID (default 126336).

    Returns
    -------
    str
        Multi-line string showing the denoising process step by step.
    """
    lines: List[str] = []

    for step_data in trajectory:
        step = step_data["step"]
        token_ids = step_data["token_ids"]

        tokens: List[str] = []
        for tid in token_ids:
            if tid == mask_token_id:
                tokens.append("[MASK]")
            else:
                decoded = tokenizer.decode([tid]).strip()
                tokens.append(decoded if decoded else f"<{tid}>")

        # Remove trailing [MASK] tokens for cleaner display
        while tokens and tokens[-1] == "[MASK]":
            tokens.pop()

        line = f"Step {step}: {' '.join(tokens)}" if tokens else f"Step {step}: (empty)"
        lines.append(line)

    return "\n".join(lines)


def create_multiple_step_by_step_texts(
    trajectories: List[List[Dict[str, Any]]],
    prompts: List[str],
    tokenizer: Any,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    save_path: Optional[str] = None,
) -> str:
    """Create text representations for multiple trajectories.

    Parameters
    ----------
    trajectories : list of list of dict
        Each outer element is a single trajectory.
    prompts : list of str
        Prompt for each trajectory.
    tokenizer
        HuggingFace-compatible tokenizer.
    mask_token_id : int
        The mask token ID.
    save_path : str, optional
        If provided, the combined text is written to this file.

    Returns
    -------
    str
        Combined text for all trajectories.
    """
    assert len(trajectories) == len(prompts), (
        f"Length mismatch: trajectories={len(trajectories)}, prompts={len(prompts)}"
    )

    sections: List[str] = []

    for idx, (traj, prompt) in enumerate(zip(trajectories, prompts)):
        header = f"{'='*60}\nExample {idx}\nPrompt: {prompt}\n{'='*60}"
        text = create_step_by_step_text(traj, tokenizer, mask_token_id)
        sections.append(f"{header}\n{text}")

    combined = "\n\n".join(sections)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(combined)
        logger.info("Saved step-by-step text to %s", save_path)

    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Visualize denoising trajectories from DiffusionNER-Zero."
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        required=True,
        help=(
            "Path to JSON file containing trajectories. Expected format: "
            "list of trajectories, where each trajectory is a list of "
            "step dicts with 'step', 'token_ids', and 'confidences'."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to JSON file containing prompt strings (one per trajectory).",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="Dream-org/Dream-v0-Base-7B",
        help="HuggingFace tokenizer name or path.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="figures/",
        help="Directory to save trajectory figures.",
    )
    parser.add_argument(
        "--text-output",
        type=str,
        default=None,
        help="Optional path to save step-by-step text output.",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=DEFAULT_MASK_TOKEN_ID,
        help="Mask token ID (default: 126336 for Dream-7B).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Maximum output tokens to display per figure.",
    )
    args = parser.parse_args()

    # Lazy import to avoid requiring transformers at module level
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    with open(args.trajectories, "r", encoding="utf-8") as f:
        trajectories = json.load(f)
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Generate figures
    saved_paths = visualize_multiple_trajectories(
        trajectories, prompts, tokenizer,
        save_dir=args.save_dir,
        mask_token_id=args.mask_token_id,
        max_tokens=args.max_tokens,
    )
    print(f"Saved {len(saved_paths)} trajectory figures to {args.save_dir}")

    # Generate text output
    if args.text_output:
        text = create_multiple_step_by_step_texts(
            trajectories, prompts, tokenizer,
            mask_token_id=args.mask_token_id,
            save_path=args.text_output,
        )
        print(f"Saved step-by-step text to {args.text_output}")
    else:
        # Print to stdout
        text = create_multiple_step_by_step_texts(
            trajectories, prompts, tokenizer,
            mask_token_id=args.mask_token_id,
        )
        print("\n" + text)
