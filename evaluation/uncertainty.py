"""Uncertainty quantification for DiffusionNER-Zero.

Diffusion models are inherently stochastic: running inference multiple times
with different random seeds can produce different entity predictions.  We
exploit this to measure model confidence.

For each example, we run inference ``num_runs`` times and measure the agreement
across runs.  High agreement suggests the model is confident; low agreement
suggests uncertainty.  We then correlate agreement with correctness to evaluate
whether the model's implicit confidence signal is calibrated.
"""

import logging
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity normalisation (consistent with evaluate.py)
# ---------------------------------------------------------------------------

def _normalize_entity(entity: Dict[str, str]) -> Tuple[str, str]:
    """Produce a normalised (type, text) pair."""
    etype = entity.get("type", "").strip().lower().replace(" ", "_")
    etext = entity.get("text", "").strip().lower()
    return (etype, etext)


# ---------------------------------------------------------------------------
# Agreement computation
# ---------------------------------------------------------------------------

def compute_entity_agreement(runs: List[List[Dict[str, str]]]) -> float:
    """Compute entity agreement ratio across multiple inference runs.

    For each unique entity (type, text) pair that appears in ANY run, we
    compute the fraction of runs in which it appeared.  The agreement score
    is the mean of these fractions, weighted by frequency.

    An agreement of 1.0 means all runs produced exactly the same entity set.
    Lower values indicate more stochastic variation.

    Parameters
    ----------
    runs : List[List[Dict[str, str]]]
        One list of predicted entities per run.

    Returns
    -------
    float
        Agreement score in [0, 1].  Returns 1.0 if all runs are empty.
    """
    num_runs = len(runs)
    if num_runs == 0:
        return 1.0

    # Count how many runs each entity appears in
    entity_run_counts: Counter = Counter()
    for run_preds in runs:
        # De-duplicate within a single run (use set)
        unique_in_run = set(_normalize_entity(e) for e in run_preds)
        for entity in unique_in_run:
            entity_run_counts[entity] += 1

    if not entity_run_counts:
        # All runs produced empty predictions
        return 1.0

    # Agreement = mean fraction of runs in which each entity appears
    fractions = [count / num_runs for count in entity_run_counts.values()]
    return float(np.mean(fractions))


def compute_pairwise_jaccard(runs: List[List[Dict[str, str]]]) -> float:
    """Compute mean pairwise Jaccard similarity across runs.

    Parameters
    ----------
    runs : List[List[Dict[str, str]]]
        One list of predicted entities per run.

    Returns
    -------
    float
        Mean pairwise Jaccard similarity in [0, 1].
    """
    num_runs = len(runs)
    if num_runs < 2:
        return 1.0

    entity_sets = [
        set(_normalize_entity(e) for e in run_preds)
        for run_preds in runs
    ]

    jaccard_sum = 0.0
    pair_count = 0

    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            a, b = entity_sets[i], entity_sets[j]
            if not a and not b:
                jaccard = 1.0
            else:
                intersection = len(a & b)
                union = len(a | b)
                jaccard = intersection / union if union > 0 else 0.0
            jaccard_sum += jaccard
            pair_count += 1

    return jaccard_sum / pair_count if pair_count > 0 else 1.0


def compute_majority_vote_entities(
    runs: List[List[Dict[str, str]]],
    threshold: float = 0.5,
) -> List[Dict[str, str]]:
    """Produce majority-vote entity predictions from multiple runs.

    An entity is included if it appears in at least ``threshold`` fraction
    of runs.

    Parameters
    ----------
    runs : List[List[Dict[str, str]]]
        Predictions from multiple runs.
    threshold : float
        Minimum fraction of runs in which an entity must appear.

    Returns
    -------
    List[Dict[str, str]]
        Majority-vote entity predictions.
    """
    num_runs = len(runs)
    if num_runs == 0:
        return []

    entity_run_counts: Counter = Counter()
    # Also keep one canonical raw entity dict per normalised key
    entity_examples: Dict[Tuple[str, str], Dict[str, str]] = {}

    for run_preds in runs:
        seen_in_run: Set[Tuple[str, str]] = set()
        for entity in run_preds:
            norm = _normalize_entity(entity)
            if norm not in seen_in_run:
                entity_run_counts[norm] += 1
                seen_in_run.add(norm)
                if norm not in entity_examples:
                    entity_examples[norm] = entity

    min_count = threshold * num_runs
    majority = []
    for norm, count in entity_run_counts.items():
        if count >= min_count:
            majority.append(entity_examples[norm])

    return majority


# ---------------------------------------------------------------------------
# Main uncertainty analysis
# ---------------------------------------------------------------------------

def analyze_uncertainty(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    num_runs: int = 10,
    num_steps: int = 8,
    seeds: Optional[List[int]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run inference multiple times per example and measure entity agreement.

    Parameters
    ----------
    model :
        The diffusion NER model.
    tokenizer :
        Associated tokenizer.
    examples : List[Dict[str, Any]]
        Evaluation examples with ``"text"``, ``"entities"``, ``"entity_types"``.
    entity_types : List[str]
        Entity types to query.
    extract_fn : Callable
        Extraction function with signature::

            extract_fn(model, tokenizer, text, entity_types, num_steps=N)
            -> List[Dict[str, str]]

        The function should honour the current random seed state for
        stochastic behavior.
    num_runs : int
        Number of inference runs per example.
    num_steps : int
        Number of denoising steps per run.
    seeds : List[int], optional
        Explicit list of random seeds to use.  Must have length >= num_runs.
        If ``None``, seeds are generated as ``[42 + i for i in range(num_runs)]``.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    Dict[str, Any]
        - ``"agreement_scores"``: List[float] -- per-example agreement
        - ``"jaccard_scores"``: List[float] -- per-example pairwise Jaccard
        - ``"mean_agreement"``: float -- mean agreement across examples
        - ``"mean_jaccard"``: float -- mean pairwise Jaccard
        - ``"correctness_scores"``: List[float] -- per-example F1 (majority vote)
        - ``"agreement_correctness_correlation"``: float -- Pearson correlation
        - ``"majority_vote_f1"``: float -- F1 using majority-vote predictions
        - ``"single_run_f1"``: float -- F1 from the first run alone
        - ``"per_example"``: list of per-example details
    """
    import torch
    from evaluation.evaluate import compute_micro_f1, compute_per_example_f1

    if seeds is None:
        seeds = [42 + i for i in range(num_runs)]
    if len(seeds) < num_runs:
        raise ValueError(
            f"Need at least {num_runs} seeds, got {len(seeds)}."
        )

    agreement_scores: List[float] = []
    jaccard_scores: List[float] = []
    per_example_data: List[Dict[str, Any]] = []
    all_majority_preds: List[List[Dict[str, str]]] = []
    all_first_run_preds: List[List[Dict[str, str]]] = []
    all_gold: List[List[Dict[str, str]]] = []

    iterator = tqdm(examples, desc="Uncertainty analysis", disable=not show_progress)

    for example in iterator:
        text = example["text"]
        gold_entities = example["entities"]
        types = entity_types if entity_types else example.get("entity_types", [])

        # Run inference multiple times
        runs: List[List[Dict[str, str]]] = []
        for run_idx in range(num_runs):
            seed = seeds[run_idx]
            # Set seeds for reproducibility of each run
            random.seed(seed)
            np.random.seed(seed)
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass  # torch may not be available in all environments

            try:
                pred_entities = extract_fn(
                    model, tokenizer, text, types, num_steps=num_steps
                )
            except Exception:
                logger.exception("Run %d failed for: %s...", run_idx, text[:80])
                pred_entities = []
            runs.append(pred_entities)

        # Compute agreement
        agreement = compute_entity_agreement(runs)
        jaccard = compute_pairwise_jaccard(runs)
        agreement_scores.append(agreement)
        jaccard_scores.append(jaccard)

        # Majority vote predictions
        majority_preds = compute_majority_vote_entities(runs, threshold=0.5)
        all_majority_preds.append(majority_preds)
        all_first_run_preds.append(runs[0] if runs else [])
        all_gold.append(gold_entities)

        per_example_data.append({
            "text": text,
            "gold_entities": gold_entities,
            "num_runs": len(runs),
            "agreement": agreement,
            "jaccard": jaccard,
            "majority_vote_preds": majority_preds,
            "all_run_preds": runs,
        })

    # Compute F1 for majority-vote and single-run approaches
    majority_metrics = compute_micro_f1(all_majority_preds, all_gold)
    single_metrics = compute_micro_f1(all_first_run_preds, all_gold)

    # Per-example correctness (majority vote F1)
    correctness_scores: List[float] = []
    for majority_preds, gold in zip(all_majority_preds, all_gold):
        ex_metrics = compute_per_example_f1(majority_preds, gold)
        correctness_scores.append(ex_metrics["f1"])

    # Correlation between agreement and correctness
    if len(agreement_scores) > 1 and np.std(agreement_scores) > 0 and np.std(correctness_scores) > 0:
        correlation = float(
            np.corrcoef(agreement_scores, correctness_scores)[0, 1]
        )
    else:
        correlation = 0.0

    result = {
        "agreement_scores": agreement_scores,
        "jaccard_scores": jaccard_scores,
        "mean_agreement": float(np.mean(agreement_scores)) if agreement_scores else 0.0,
        "mean_jaccard": float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        "correctness_scores": correctness_scores,
        "agreement_correctness_correlation": correlation,
        "majority_vote_f1": majority_metrics["f1"],
        "single_run_f1": single_metrics["f1"],
        "majority_vote_metrics": majority_metrics,
        "single_run_metrics": single_metrics,
        "per_example": per_example_data,
        "num_examples": len(examples),
        "num_runs": num_runs,
    }

    logger.info(
        "Uncertainty analysis: mean_agreement=%.3f, mean_jaccard=%.3f, "
        "correlation=%.3f, majority_F1=%.4f, single_F1=%.4f",
        result["mean_agreement"], result["mean_jaccard"],
        correlation, result["majority_vote_f1"], result["single_run_f1"],
    )

    return result


def print_uncertainty_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of uncertainty analysis results.

    Parameters
    ----------
    result : Dict[str, Any]
        Output of :func:`analyze_uncertainty`.
    """
    print("\nUncertainty Analysis")
    print("=" * 60)
    print(f"  Examples analysed:           {result['num_examples']}")
    print(f"  Runs per example:            {result['num_runs']}")
    print(f"  Mean entity agreement:       {result['mean_agreement']:.4f}")
    print(f"  Mean pairwise Jaccard:       {result['mean_jaccard']:.4f}")
    print(f"  Agreement-correctness corr:  {result['agreement_correctness_correlation']:.4f}")
    print(f"  Single-run F1:               {result['single_run_f1']:.4f}")
    print(f"  Majority-vote F1:            {result['majority_vote_f1']:.4f}")
    mv_gain = result['majority_vote_f1'] - result['single_run_f1']
    print(f"  Majority-vote gain:          {mv_gain:+.4f}")
    print("=" * 60)

    # Show distribution of agreement scores
    scores = result.get("agreement_scores", [])
    if scores:
        arr = np.array(scores)
        print("\n  Agreement score distribution:")
        print(f"    Min:    {arr.min():.4f}")
        print(f"    25th:   {np.percentile(arr, 25):.4f}")
        print(f"    Median: {np.median(arr):.4f}")
        print(f"    75th:   {np.percentile(arr, 75):.4f}")
        print(f"    Max:    {arr.max():.4f}")
    print()
