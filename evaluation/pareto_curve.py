"""Pareto curve analysis: F1 vs inference speed at different step counts.

Runs the DiffusionNER model at multiple denoising step counts (e.g. 1, 2, 4,
8, 16, 32), measures both entity-level micro-F1 and wall-clock time per
example, then plots the F1 vs speed Pareto frontier.  Optionally overlays a
UniNER baseline point for comparison.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from evaluation.evaluate import compute_micro_f1

logger = logging.getLogger(__name__)

# Use non-interactive backend when no display is available
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Pareto analysis
# ---------------------------------------------------------------------------

def run_pareto_analysis(
    model: Any,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    steps_list: Optional[List[int]] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Run the model at different step counts and measure F1 and wall-clock time.

    Parameters
    ----------
    model :
        The diffusion NER model.
    tokenizer :
        Associated tokenizer.
    benchmark : List[Dict[str, Any]]
        Evaluation examples with ``"text"``, ``"entities"``, ``"entity_types"``.
    entity_types : List[str]
        Entity types to query.
    extract_fn : Callable
        Extraction function with signature::

            extract_fn(model, tokenizer, text, entity_types, num_steps=N)
            -> List[Dict[str, str]]
    steps_list : List[int], optional
        Step counts to evaluate.  Defaults to ``[1, 2, 4, 8, 16, 32]``.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per step count with keys:
        ``"steps"``, ``"f1"``, ``"precision"``, ``"recall"``,
        ``"time_per_example"``, ``"total_time"``, ``"num_examples"``.
    """
    if steps_list is None:
        steps_list = [1, 2, 4, 8, 16, 32]

    results: List[Dict[str, Any]] = []

    for num_steps in steps_list:
        logger.info("Running Pareto analysis with num_steps=%d ...", num_steps)

        all_predictions: List[List[Dict[str, str]]] = []
        all_gold: List[List[Dict[str, str]]] = []
        times: List[float] = []

        iterator = tqdm(
            benchmark,
            desc=f"Steps={num_steps}",
            disable=not show_progress,
        )

        for example in iterator:
            text = example["text"]
            gold_entities = example["entities"]
            types = entity_types if entity_types else example.get("entity_types", [])

            t0 = time.perf_counter()
            try:
                pred_entities = extract_fn(
                    model, tokenizer, text, types, num_steps=num_steps
                )
            except Exception:
                logger.exception("Extraction failed at steps=%d", num_steps)
                pred_entities = []
            elapsed = time.perf_counter() - t0

            all_predictions.append(pred_entities)
            all_gold.append(gold_entities)
            times.append(elapsed)

        metrics = compute_micro_f1(all_predictions, all_gold)
        total_time = sum(times)
        time_per_example = total_time / len(benchmark) if benchmark else 0.0

        step_result = {
            "steps": num_steps,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "time_per_example": time_per_example,
            "total_time": total_time,
            "num_examples": len(benchmark),
        }
        results.append(step_result)

        logger.info(
            "  steps=%d  F1=%.4f  time/ex=%.4fs",
            num_steps, metrics["f1"], time_per_example,
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto_curve(
    results: List[Dict[str, Any]],
    uniner_point: Optional[Dict[str, Any]] = None,
    save_path: str = "figures/pareto_curve.png",
    title: str = "F1 vs Inference Speed (Pareto Curve)",
    figsize: tuple = (8, 6),
) -> None:
    """Plot F1 vs speed Pareto curve and optionally overlay a UniNER baseline.

    The x-axis shows throughput (examples/sec, higher is better).  The y-axis
    shows entity-level micro-F1 (higher is better).

    Parameters
    ----------
    results : List[Dict[str, Any]]
        Output of :func:`run_pareto_analysis`.  Each dict must have
        ``"steps"``, ``"f1"``, and ``"time_per_example"`` keys.
    uniner_point : Dict[str, Any], optional
        Baseline point with ``"f1"`` and ``"time_per_example"`` keys.  If
        provided, it is plotted as a separate marker.
    save_path : str
        File path to save the figure.  Parent directories are created as
        needed.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Compute throughput (examples/sec)
    steps_arr = [r["steps"] for r in results]
    f1_arr = [r["f1"] * 100 for r in results]  # percentage
    throughput_arr = [
        1.0 / r["time_per_example"] if r["time_per_example"] > 0 else 0.0
        for r in results
    ]

    # Plot DiffusionNER points
    ax.plot(
        throughput_arr, f1_arr,
        marker="o", linewidth=2, markersize=8,
        color="#2196F3", label="DiffusionNER-Zero",
        zorder=3,
    )

    # Annotate each point with the step count
    for i, steps in enumerate(steps_arr):
        ax.annotate(
            f"T={steps}",
            (throughput_arr[i], f1_arr[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color="#1565C0",
        )

    # Overlay UniNER baseline if provided
    if uniner_point is not None:
        uni_f1 = uniner_point["f1"] * 100
        uni_throughput = (
            1.0 / uniner_point["time_per_example"]
            if uniner_point.get("time_per_example", 0) > 0
            else 0.0
        )
        ax.scatter(
            [uni_throughput], [uni_f1],
            marker="*", s=200, color="#F44336",
            label="UniNER-7B-type", zorder=4,
        )
        ax.annotate(
            "UniNER",
            (uni_throughput, uni_f1),
            textcoords="offset points",
            xytext=(10, -10),
            fontsize=10,
            fontweight="bold",
            color="#C62828",
        )

    ax.set_xlabel("Throughput (examples/sec)", fontsize=12)
    ax.set_ylabel("Entity-level Micro-F1 (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    if throughput_arr:
        x_margin = max(throughput_arr) * 0.1
        ax.set_xlim(left=max(0, min(throughput_arr) - x_margin))
    if f1_arr:
        y_lo = min(f1_arr) - 3
        y_hi = min(100, max(f1_arr) + 3)
        ax.set_ylim(bottom=max(0, y_lo), top=y_hi)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Pareto curve to %s", save_path)


def plot_pareto_curve_time(
    results: List[Dict[str, Any]],
    uniner_point: Optional[Dict[str, Any]] = None,
    save_path: str = "figures/pareto_curve_time.png",
    title: str = "F1 vs Latency (Pareto Curve)",
    figsize: tuple = (8, 6),
) -> None:
    """Alternative Pareto plot with latency (ms/example) on the x-axis.

    Lower latency is better (left is better), so this gives the classic
    Pareto trade-off view.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        Output of :func:`run_pareto_analysis`.
    uniner_point : Dict[str, Any], optional
        Baseline point.
    save_path : str
        File path to save the figure.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    steps_arr = [r["steps"] for r in results]
    f1_arr = [r["f1"] * 100 for r in results]
    latency_arr = [r["time_per_example"] * 1000 for r in results]  # ms

    ax.plot(
        latency_arr, f1_arr,
        marker="o", linewidth=2, markersize=8,
        color="#2196F3", label="DiffusionNER-Zero",
        zorder=3,
    )

    for i, steps in enumerate(steps_arr):
        ax.annotate(
            f"T={steps}",
            (latency_arr[i], f1_arr[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color="#1565C0",
        )

    if uniner_point is not None:
        uni_f1 = uniner_point["f1"] * 100
        uni_latency = uniner_point.get("time_per_example", 0) * 1000
        ax.scatter(
            [uni_latency], [uni_f1],
            marker="*", s=200, color="#F44336",
            label="UniNER-7B-type", zorder=4,
        )
        ax.annotate(
            "UniNER",
            (uni_latency, uni_f1),
            textcoords="offset points",
            xytext=(10, -10),
            fontsize=10,
            fontweight="bold",
            color="#C62828",
        )

    ax.set_xlabel("Latency (ms/example)", fontsize=12)
    ax.set_ylabel("Entity-level Micro-F1 (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Pareto latency curve to %s", save_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quick demo with synthetic data
    demo_results = [
        {"steps": 1,  "f1": 0.32, "time_per_example": 0.05},
        {"steps": 2,  "f1": 0.41, "time_per_example": 0.09},
        {"steps": 4,  "f1": 0.48, "time_per_example": 0.17},
        {"steps": 8,  "f1": 0.52, "time_per_example": 0.33},
        {"steps": 16, "f1": 0.53, "time_per_example": 0.65},
        {"steps": 32, "f1": 0.54, "time_per_example": 1.28},
    ]
    demo_uniner = {"f1": 0.50, "time_per_example": 1.10}

    plot_pareto_curve(
        demo_results,
        uniner_point=demo_uniner,
        save_path="figures/pareto_curve_demo.png",
    )
    plot_pareto_curve_time(
        demo_results,
        uniner_point=demo_uniner,
        save_path="figures/pareto_curve_time_demo.png",
    )
    print("Demo Pareto plots saved to figures/")
