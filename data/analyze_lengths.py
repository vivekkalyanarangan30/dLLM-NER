"""Analyze tokenized completion lengths to validate MAX_COMPLETION_LENGTH.

Computes percentile statistics and optionally renders a histogram so we can
confirm that ``MAX_COMPLETION_LENGTH = 128`` covers >95% of all completions
in the reformatted Pile-NER-type dataset.

Usage::

    python -m data.analyze_lengths --data-dir data/processed

Or programmatically::

    from data.analyze_lengths import analyze_completion_lengths
    stats = analyze_completion_lengths(data, tokenizer)
    print(stats)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Target completion length from SPEC
DEFAULT_MAX_COMPLETION_LENGTH = 128


def analyze_completion_lengths(
    data: List[Dict[str, str]],
    tokenizer: Any,
    max_completion_length: int = DEFAULT_MAX_COMPLETION_LENGTH,
) -> Dict[str, Any]:
    """Compute length statistics for tokenized completions.

    Parameters
    ----------
    data : list[dict]
        Each dict must have a ``"completion"`` string field.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for Dream-7B.
    max_completion_length : int, optional
        The proposed maximum completion length (default 128).

    Returns
    -------
    dict
        Statistics dictionary with keys:

        - ``count`` : int -- total number of examples
        - ``mean`` : float -- mean token length
        - ``std`` : float -- standard deviation
        - ``min`` : int -- minimum length
        - ``max`` : int -- maximum length
        - ``median`` : float -- median length
        - ``percentiles`` : dict mapping percentile labels to values
          (50, 75, 90, 95, 99, 99.5)
        - ``coverage_at_max`` : float -- fraction of examples that fit
          within *max_completion_length*
        - ``num_exceeding`` : int -- count of examples exceeding the limit
        - ``histogram_bins`` : list[int] -- bin edges for the histogram
        - ``histogram_counts`` : list[int] -- counts per bin
    """
    lengths: List[int] = []

    for example in data:
        completion_text = example["completion"]
        token_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        lengths.append(len(token_ids))

    lengths_arr = np.array(lengths)

    # Core statistics
    percentile_keys = [50, 75, 90, 95, 99, 99.5]
    percentile_values = np.percentile(lengths_arr, percentile_keys).tolist()
    percentiles = {str(p): round(v, 1) for p, v in zip(percentile_keys, percentile_values)}

    coverage = float(np.mean(lengths_arr <= max_completion_length))
    num_exceeding = int(np.sum(lengths_arr > max_completion_length))

    # Histogram (bins of width 8, from 0 to max observed length + margin)
    bin_width = 8
    max_bin = int(np.ceil((lengths_arr.max() + bin_width) / bin_width) * bin_width)
    bins = list(range(0, max_bin + bin_width, bin_width))
    counts, _ = np.histogram(lengths_arr, bins=bins)

    stats = {
        "count": len(lengths),
        "mean": round(float(lengths_arr.mean()), 2),
        "std": round(float(lengths_arr.std()), 2),
        "min": int(lengths_arr.min()),
        "max": int(lengths_arr.max()),
        "median": round(float(np.median(lengths_arr)), 1),
        "percentiles": percentiles,
        "coverage_at_max": round(coverage, 4),
        "max_completion_length": max_completion_length,
        "num_exceeding": num_exceeding,
        "histogram_bins": bins,
        "histogram_counts": counts.tolist(),
    }

    return stats


def print_report(stats: Dict[str, Any]) -> None:
    """Pretty-print the analysis report to stdout.

    Parameters
    ----------
    stats : dict
        Output of :func:`analyze_completion_lengths`.
    """
    mcl = stats["max_completion_length"]
    print("=" * 60)
    print("  Completion Token Length Analysis")
    print("=" * 60)
    print(f"  Total examples:        {stats['count']:,}")
    print(f"  Mean length:           {stats['mean']:.1f}")
    print(f"  Std deviation:         {stats['std']:.1f}")
    print(f"  Min length:            {stats['min']}")
    print(f"  Max length:            {stats['max']}")
    print(f"  Median length:         {stats['median']:.1f}")
    print()
    print("  Percentiles:")
    for pct, val in stats["percentiles"].items():
        print(f"    P{pct:>5s}: {val:.1f} tokens")
    print()
    print(f"  MAX_COMPLETION_LENGTH: {mcl}")
    print(f"  Coverage:              {stats['coverage_at_max'] * 100:.2f}%")
    print(f"  Examples exceeding:    {stats['num_exceeding']:,}")
    if stats["coverage_at_max"] >= 0.95:
        print(f"  --> OK: {mcl} covers >= 95% of completions.")
    else:
        print(f"  --> WARNING: {mcl} covers < 95% of completions.")
        p95 = stats["percentiles"].get("95", stats["percentiles"].get("95.0", "N/A"))
        print(f"      Consider increasing to at least {p95} tokens (P95).")
    print("=" * 60)


def plot_histogram(
    stats: Dict[str, Any],
    output_path: Optional[str] = None,
) -> None:
    """Render a histogram of completion lengths.

    Parameters
    ----------
    stats : dict
        Output of :func:`analyze_completion_lengths`.
    output_path : str, optional
        If provided, save the figure to this path.  Otherwise, display
        interactively with ``plt.show()``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping histogram plot.")
        return

    bins = stats["histogram_bins"]
    counts = stats["histogram_counts"]
    mcl = stats["max_completion_length"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot bars
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
    bin_width = bins[1] - bins[0]
    colors = [
        "#4CAF50" if center <= mcl else "#F44336" for center in bin_centers
    ]
    ax.bar(bin_centers, counts, width=bin_width * 0.9, color=colors, edgecolor="white")

    # Vertical line at MAX_COMPLETION_LENGTH
    ax.axvline(
        x=mcl, color="red", linestyle="--", linewidth=2,
        label=f"MAX_COMPLETION_LENGTH={mcl}",
    )

    # Annotations
    coverage_pct = stats["coverage_at_max"] * 100
    ax.set_title(
        f"Completion Token Lengths (coverage at {mcl}: {coverage_pct:.1f}%)",
        fontsize=13,
    )
    ax.set_xlabel("Token Length", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved histogram to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Analyze completion token lengths for the processed Pile-NER dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to the processed dataset (saved by prepare_pile_ner.py).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Dream-org/Dream-v0-Base-7B",
        help="HuggingFace tokenizer name or path.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_MAX_COMPLETION_LENGTH,
        help="Proposed max completion length to check coverage for.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to analyze (default: train).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="If set, save histogram to this file path (e.g. 'lengths.png').",
    )
    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="If set, save statistics JSON to this file path.",
    )
    args = parser.parse_args()

    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Loading dataset from: %s", args.data_dir)
    ds = load_from_disk(args.data_dir)
    split_data = ds[args.split]

    # Convert to list of dicts
    data = [{"completion": row["completion"]} for row in split_data]

    logger.info("Analyzing %d completions...", len(data))
    stats = analyze_completion_lengths(
        data, tokenizer, max_completion_length=args.max_completion_length
    )

    print_report(stats)

    if args.plot:
        plot_histogram(stats, output_path=args.plot)

    if args.save_stats:
        with open(args.save_stats, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved stats to %s", args.save_stats)
