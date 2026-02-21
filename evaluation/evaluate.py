"""Core evaluation for DiffusionNER-Zero: entity-level micro-F1 with strict match.

Strict matching means both the entity span text AND the entity type must match
exactly for a prediction to count as a true positive.  Evaluation is computed
at the entity level (not token level), and aggregated across all examples to
produce micro-averaged precision, recall, and F1.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Micro-F1 computation
# ---------------------------------------------------------------------------

def _normalize_entity(entity: Dict[str, str]) -> tuple:
    """Produce a hashable, normalised representation of a predicted entity.

    Normalisation:
        - Entity type is lowercased and stripped, spaces replaced with ``_``.
        - Entity text is lowercased and stripped.

    Parameters
    ----------
    entity : Dict[str, str]
        Must contain ``"type"`` and ``"text"`` keys.

    Returns
    -------
    tuple
        ``(normalised_type, normalised_text)``
    """
    etype = entity.get("type", "").strip().lower().replace(" ", "_")
    etext = entity.get("text", "").strip().lower()
    return (etype, etext)


def compute_micro_f1(
    predictions: List[List[Dict[str, str]]],
    gold: List[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """Compute entity-level micro-averaged precision, recall, and F1.

    Both *predictions* and *gold* are lists of examples.  Each example is
    itself a list of entity dicts with at least ``"type"`` and ``"text"``
    keys.  Matching is **strict**: the normalised (type, text) pair must be
    identical.

    For each example, we convert gold and predicted entities to multisets
    (to handle duplicates) and count true positives as the overlap.

    Parameters
    ----------
    predictions : List[List[Dict[str, str]]]
        Predicted entities per example.
    gold : List[List[Dict[str, str]]]
        Gold entities per example.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``"precision"``, ``"recall"``, ``"f1"``,
        ``"tp"``, ``"fp"``, ``"fn"``.
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold."
        )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_entities, gold_entities in zip(predictions, gold):
        pred_norm = [_normalize_entity(e) for e in pred_entities]
        gold_norm = [_normalize_entity(e) for e in gold_entities]

        # Build multisets (counts) to handle duplicate entities
        pred_counts: Dict[tuple, int] = {}
        for e in pred_norm:
            pred_counts[e] = pred_counts.get(e, 0) + 1

        gold_counts: Dict[tuple, int] = {}
        for e in gold_norm:
            gold_counts[e] = gold_counts.get(e, 0) + 1

        # True positives = overlap between pred and gold multisets
        tp = 0
        all_keys = set(pred_counts.keys()) | set(gold_counts.keys())
        for key in all_keys:
            tp += min(pred_counts.get(key, 0), gold_counts.get(key, 0))

        fp = sum(pred_counts.values()) - tp
        fn = sum(gold_counts.values()) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


# ---------------------------------------------------------------------------
# Per-example evaluation (for diagnostics)
# ---------------------------------------------------------------------------

def compute_per_example_f1(
    pred_entities: List[Dict[str, str]],
    gold_entities: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compute precision, recall, and F1 for a single example.

    Parameters
    ----------
    pred_entities : List[Dict[str, str]]
        Predicted entities for one example.
    gold_entities : List[Dict[str, str]]
        Gold entities for one example.

    Returns
    -------
    Dict[str, Any]
        ``{"precision", "recall", "f1", "tp", "fp", "fn"}``
    """
    result = compute_micro_f1([pred_entities], [gold_entities])
    return result


# ---------------------------------------------------------------------------
# Model evaluation driver
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    num_steps: int = 8,
    batch_size: int = 1,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate a model on a single benchmark.

    Parameters
    ----------
    model :
        The language model (DiffusionNER or baseline).
    tokenizer :
        The associated tokenizer.
    benchmark : List[Dict[str, Any]]
        List of examples, each with ``"text"``, ``"entities"``, and
        ``"entity_types"`` keys.
    entity_types : List[str]
        Entity type strings to query for.  If ``None``, uses the per-example
        ``"entity_types"`` field.
    extract_fn : Callable
        Extraction function with signature::

            extract_fn(model, tokenizer, text, entity_types, num_steps=N)
            -> List[Dict[str, str]]

        Each returned dict must have ``"type"`` and ``"text"`` keys.
    num_steps : int
        Number of denoising steps (passed to *extract_fn*).
    batch_size : int
        Not currently used (reserved for future batched inference).
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    Dict[str, Any]
        Dictionary with micro-F1 metrics (``"precision"``, ``"recall"``,
        ``"f1"``, ``"tp"``, ``"fp"``, ``"fn"``), plus:
        - ``"per_example"``: list of per-example results
        - ``"predictions"``: raw predictions per example
        - ``"num_examples"``: total number of examples
        - ``"wall_time"``: total inference wall-clock time in seconds
    """
    all_predictions: List[List[Dict[str, str]]] = []
    all_gold: List[List[Dict[str, str]]] = []
    per_example_results: List[Dict[str, Any]] = []

    iterator = tqdm(benchmark, desc="Evaluating", disable=not show_progress)
    start_time = time.time()

    for example in iterator:
        text = example["text"]
        gold_entities = example["entities"]
        types = entity_types if entity_types else example.get("entity_types", [])

        try:
            pred_entities = extract_fn(
                model, tokenizer, text, types, num_steps=num_steps
            )
        except Exception:
            logger.exception("Extraction failed for text: %s...", text[:80])
            pred_entities = []

        all_predictions.append(pred_entities)
        all_gold.append(gold_entities)

        # Per-example metrics
        ex_metrics = compute_per_example_f1(pred_entities, gold_entities)
        ex_metrics["text"] = text
        ex_metrics["pred_entities"] = pred_entities
        ex_metrics["gold_entities"] = gold_entities
        per_example_results.append(ex_metrics)

    wall_time = time.time() - start_time

    # Aggregate micro-F1
    metrics = compute_micro_f1(all_predictions, all_gold)
    metrics["per_example"] = per_example_results
    metrics["predictions"] = all_predictions
    metrics["num_examples"] = len(benchmark)
    metrics["wall_time"] = wall_time

    logger.info(
        "Evaluation complete: P=%.4f R=%.4f F1=%.4f (TP=%d FP=%d FN=%d) in %.1fs",
        metrics["precision"], metrics["recall"], metrics["f1"],
        metrics["tp"], metrics["fp"], metrics["fn"],
        wall_time,
    )

    return metrics


# ---------------------------------------------------------------------------
# Multi-benchmark evaluation
# ---------------------------------------------------------------------------

def evaluate_all_benchmarks(
    model: Any,
    tokenizer: Any,
    benchmarks: Dict[str, List[Dict[str, Any]]],
    extract_fn: Callable,
    num_steps: int = 8,
    show_progress: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run evaluation across all benchmarks.

    Parameters
    ----------
    model :
        The language model.
    tokenizer :
        The tokenizer.
    benchmarks : Dict[str, List[Dict[str, Any]]]
        Mapping from benchmark name to list of examples (as returned by
        :func:`evaluation.load_benchmarks.load_all_benchmarks`).
    extract_fn : Callable
        Entity extraction function (see :func:`evaluate_model`).
    num_steps : int
        Number of denoising steps.
    show_progress : bool
        Whether to show progress bars.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        ``{benchmark_name: metrics_dict}`` for each benchmark.
    """
    all_results: Dict[str, Dict[str, Any]] = {}

    for bench_name, bench_examples in benchmarks.items():
        if not bench_examples:
            logger.warning("Skipping empty benchmark: %s", bench_name)
            continue

        logger.info("Evaluating on %s (%d examples)...", bench_name, len(bench_examples))

        # Use the entity types from the first example (all examples in a
        # benchmark share the same type list)
        entity_types = bench_examples[0].get("entity_types", [])

        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            benchmark=bench_examples,
            entity_types=entity_types,
            extract_fn=extract_fn,
            num_steps=num_steps,
            show_progress=show_progress,
        )

        # Strip per-example data for the summary (keep it available in full metrics)
        summary = {
            k: v for k, v in metrics.items()
            if k not in ("per_example", "predictions")
        }
        all_results[bench_name] = metrics

        logger.info(
            "  %s: F1=%.4f P=%.4f R=%.4f",
            bench_name, summary["f1"], summary["precision"], summary["recall"],
        )

    return all_results


# ---------------------------------------------------------------------------
# Pretty-printed results table
# ---------------------------------------------------------------------------

def print_results_table(
    results: Dict[str, Dict[str, Any]],
    title: str = "Evaluation Results",
) -> None:
    """Print a formatted results table to stdout.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        ``{benchmark_name: metrics_dict}``.  Each metrics dict should have
        at least ``"precision"``, ``"recall"``, ``"f1"``.
    title : str
        Title printed above the table.
    """
    header = f"{'Benchmark':25s} | {'P':>8s} | {'R':>8s} | {'F1':>8s} | {'TP':>6s} | {'FP':>6s} | {'FN':>6s}"
    separator = "-" * len(header)

    print(f"\n{title}")
    print(separator)
    print(header)
    print(separator)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for bench_name, metrics in sorted(results.items()):
        tp = metrics.get("tp", 0)
        fp = metrics.get("fp", 0)
        fn = metrics.get("fn", 0)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(
            f"{bench_name:25s} | {metrics.get('precision', 0.0):8.4f} | "
            f"{metrics.get('recall', 0.0):8.4f} | {metrics.get('f1', 0.0):8.4f} | "
            f"{tp:6d} | {fp:6d} | {fn:6d}"
        )

    # Compute overall micro-F1
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0
        else 0.0
    )

    print(separator)
    print(
        f"{'OVERALL (micro)':25s} | {overall_p:8.4f} | "
        f"{overall_r:8.4f} | {overall_f1:8.4f} | "
        f"{total_tp:6d} | {total_fp:6d} | {total_fn:6d}"
    )
    print(separator)
    print()


def print_comparison_table(
    results_a: Dict[str, Dict[str, Any]],
    results_b: Dict[str, Dict[str, Any]],
    name_a: str = "DiffusionNER",
    name_b: str = "UniNER",
) -> None:
    """Print a side-by-side comparison table of two models.

    Parameters
    ----------
    results_a, results_b : Dict[str, Dict[str, Any]]
        Per-benchmark metrics for each model.
    name_a, name_b : str
        Display names for the two models.
    """
    all_benchmarks = sorted(
        set(list(results_a.keys()) + list(results_b.keys()))
    )

    header = (
        f"{'Benchmark':25s} | {name_a + ' F1':>14s} | {name_b + ' F1':>14s} | "
        f"{'Delta':>8s}"
    )
    separator = "-" * len(header)

    print(f"\nComparison: {name_a} vs {name_b}")
    print(separator)
    print(header)
    print(separator)

    for bench in all_benchmarks:
        f1_a = results_a.get(bench, {}).get("f1", 0.0)
        f1_b = results_b.get(bench, {}).get("f1", 0.0)
        delta = f1_a - f1_b
        sign = "+" if delta >= 0 else ""
        print(
            f"{bench:25s} | {f1_a:14.4f} | {f1_b:14.4f} | {sign}{delta:7.4f}"
        )

    print(separator)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage with dummy data
    logging.basicConfig(level=logging.INFO)

    gold = [
        [{"type": "person", "text": "Alice"}, {"type": "org", "text": "ACME"}],
        [{"type": "person", "text": "Bob"}],
    ]
    pred = [
        [{"type": "person", "text": "Alice"}, {"type": "org", "text": "Acme Inc"}],
        [{"type": "person", "text": "Bob"}, {"type": "loc", "text": "London"}],
    ]

    result = compute_micro_f1(pred, gold)
    print("Demo micro-F1 result:", result)
