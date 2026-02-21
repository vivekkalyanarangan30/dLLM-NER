"""Self-correction analysis for DiffusionNER-Zero.

Track entity predictions at each denoising step to study how the diffusion
process refines its output.  A "correction" occurs when the model fixes an
incorrect prediction between early and late steps.  A "regression" occurs when
a correct early prediction becomes incorrect later.

This analysis supports the claim that multi-step diffusion inference
meaningfully improves entity extraction quality over single-step prediction.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity normalisation (matches evaluate.py)
# ---------------------------------------------------------------------------

def _normalize_entity(entity: Dict[str, str]) -> Tuple[str, str]:
    """Produce a normalised (type, text) pair for comparison."""
    etype = entity.get("type", "").strip().lower().replace(" ", "_")
    etext = entity.get("text", "").strip().lower()
    return (etype, etext)


def _entity_set(entities: List[Dict[str, str]]) -> Set[Tuple[str, str]]:
    """Convert a list of entity dicts to a set of normalised tuples."""
    return set(_normalize_entity(e) for e in entities)


# ---------------------------------------------------------------------------
# Trajectory analysis helpers
# ---------------------------------------------------------------------------

def _classify_step_changes(
    gold_set: Set[Tuple[str, str]],
    prev_set: Set[Tuple[str, str]],
    curr_set: Set[Tuple[str, str]],
) -> Dict[str, int]:
    """Classify entity changes between two consecutive steps.

    Parameters
    ----------
    gold_set : Set
        Normalised gold entity tuples.
    prev_set : Set
        Normalised predicted entities at the previous step.
    curr_set : Set
        Normalised predicted entities at the current step.

    Returns
    -------
    Dict[str, int]
        Counts of ``"corrections"`` (wrong -> right), ``"regressions"``
        (right -> wrong), ``"new_correct"`` (absent -> right),
        ``"new_incorrect"`` (absent -> wrong), ``"dropped_correct"``
        (right -> absent), ``"dropped_incorrect"`` (wrong -> absent).
    """
    added = curr_set - prev_set
    removed = prev_set - curr_set

    corrections = 0
    regressions = 0
    new_correct = 0
    new_incorrect = 0
    dropped_correct = 0
    dropped_incorrect = 0

    for entity in added:
        if entity in gold_set:
            # Was it replacing a wrong entity or entirely new?
            new_correct += 1
        else:
            new_incorrect += 1

    for entity in removed:
        if entity in gold_set:
            dropped_correct += 1
        else:
            dropped_incorrect += 1

    # A "correction" is dropping an incorrect entity or gaining a correct one
    corrections = new_correct + dropped_incorrect
    # A "regression" is dropping a correct entity or gaining an incorrect one
    regressions = dropped_correct + new_incorrect

    return {
        "corrections": corrections,
        "regressions": regressions,
        "new_correct": new_correct,
        "new_incorrect": new_incorrect,
        "dropped_correct": dropped_correct,
        "dropped_incorrect": dropped_incorrect,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_self_correction(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    entity_types: List[str],
    extract_with_trajectory_fn: Callable,
    num_steps: int = 8,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Record entity predictions at each denoising step and count corrections.

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
    extract_with_trajectory_fn : Callable
        Function that returns the full trajectory of predictions across steps.
        Signature::

            extract_with_trajectory_fn(
                model, tokenizer, text, entity_types, num_steps=N
            ) -> List[List[Dict[str, str]]]

        Returns a list of length ``num_steps``, where element ``i`` is the
        list of entity predictions after step ``i``.
    num_steps : int
        Number of denoising steps.
    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    Dict[str, Any]
        - ``"total_corrections"``: int -- total corrections across all steps and examples
        - ``"total_regressions"``: int -- total regressions across all steps and examples
        - ``"correction_ratio"``: float -- corrections / (corrections + regressions)
        - ``"per_step_corrections"``: List[int] -- corrections at each step transition
        - ``"per_step_regressions"``: List[int] -- regressions at each step transition
        - ``"step1_f1"``: float -- micro-F1 at step 1 (earliest)
        - ``"final_f1"``: float -- micro-F1 at the final step
        - ``"f1_per_step"``: List[float] -- F1 at each step
        - ``"per_example"``: list of per-example trajectory data
    """
    from evaluation.evaluate import compute_micro_f1

    # Accumulators across all examples
    per_step_corrections = [0] * (num_steps - 1) if num_steps > 1 else []
    per_step_regressions = [0] * (num_steps - 1) if num_steps > 1 else []

    # For computing per-step F1: predictions at each step, indexed by step
    per_step_predictions: List[List[List[Dict[str, str]]]] = [
        [] for _ in range(num_steps)
    ]
    all_gold: List[List[Dict[str, str]]] = []

    per_example_data: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Self-correction analysis", disable=not show_progress)

    for example in iterator:
        text = example["text"]
        gold_entities = example["entities"]
        types = entity_types if entity_types else example.get("entity_types", [])
        gold_set = _entity_set(gold_entities)

        try:
            trajectory = extract_with_trajectory_fn(
                model, tokenizer, text, types, num_steps=num_steps
            )
        except Exception:
            logger.exception("Trajectory extraction failed for: %s...", text[:80])
            trajectory = [[] for _ in range(num_steps)]

        # Pad or truncate trajectory to exactly num_steps entries
        while len(trajectory) < num_steps:
            trajectory.append(trajectory[-1] if trajectory else [])
        trajectory = trajectory[:num_steps]

        # Store predictions per step
        all_gold.append(gold_entities)
        for step_idx, step_preds in enumerate(trajectory):
            per_step_predictions[step_idx].append(step_preds)

        # Analyze step-by-step transitions
        example_corrections = 0
        example_regressions = 0
        step_changes = []

        for step_idx in range(1, num_steps):
            prev_set = _entity_set(trajectory[step_idx - 1])
            curr_set = _entity_set(trajectory[step_idx])
            changes = _classify_step_changes(gold_set, prev_set, curr_set)

            per_step_corrections[step_idx - 1] += changes["corrections"]
            per_step_regressions[step_idx - 1] += changes["regressions"]
            example_corrections += changes["corrections"]
            example_regressions += changes["regressions"]
            step_changes.append(changes)

        per_example_data.append({
            "text": text,
            "gold_entities": gold_entities,
            "trajectory": trajectory,
            "corrections": example_corrections,
            "regressions": example_regressions,
            "step_changes": step_changes,
        })

    # Compute per-step F1
    f1_per_step: List[float] = []
    for step_idx in range(num_steps):
        step_metrics = compute_micro_f1(per_step_predictions[step_idx], all_gold)
        f1_per_step.append(step_metrics["f1"])

    total_corrections = sum(per_step_corrections)
    total_regressions = sum(per_step_regressions)
    correction_ratio = (
        total_corrections / (total_corrections + total_regressions)
        if (total_corrections + total_regressions) > 0
        else 0.0
    )

    result = {
        "total_corrections": total_corrections,
        "total_regressions": total_regressions,
        "correction_ratio": correction_ratio,
        "per_step_corrections": per_step_corrections,
        "per_step_regressions": per_step_regressions,
        "step1_f1": f1_per_step[0] if f1_per_step else 0.0,
        "final_f1": f1_per_step[-1] if f1_per_step else 0.0,
        "f1_per_step": f1_per_step,
        "per_example": per_example_data,
        "num_examples": len(examples),
        "num_steps": num_steps,
    }

    logger.info(
        "Self-correction analysis: %d corrections, %d regressions "
        "(ratio=%.3f). F1 step1=%.4f -> final=%.4f",
        total_corrections, total_regressions, correction_ratio,
        result["step1_f1"], result["final_f1"],
    )

    return result


def print_self_correction_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of self-correction analysis results.

    Parameters
    ----------
    result : Dict[str, Any]
        Output of :func:`analyze_self_correction`.
    """
    print("\nSelf-Correction Analysis")
    print("=" * 60)
    print(f"  Examples analysed:   {result['num_examples']}")
    print(f"  Denoising steps:     {result['num_steps']}")
    print(f"  Total corrections:   {result['total_corrections']}")
    print(f"  Total regressions:   {result['total_regressions']}")
    print(f"  Correction ratio:    {result['correction_ratio']:.3f}")
    print(f"  F1 at step 1:        {result['step1_f1']:.4f}")
    print(f"  F1 at final step:    {result['final_f1']:.4f}")
    print(f"  F1 improvement:      {result['final_f1'] - result['step1_f1']:+.4f}")
    print()

    if result.get("per_step_corrections"):
        print("  Per-step transitions:")
        print(f"    {'Step':>6s}  {'Corrections':>12s}  {'Regressions':>12s}  {'Net':>8s}")
        for i, (corr, reg) in enumerate(
            zip(result["per_step_corrections"], result["per_step_regressions"])
        ):
            net = corr - reg
            sign = "+" if net >= 0 else ""
            print(f"    {i+1}->{i+2}  {corr:>12d}  {reg:>12d}  {sign}{net:>7d}")
    print()

    if result.get("f1_per_step"):
        print("  F1 per denoising step:")
        for i, f1 in enumerate(result["f1_per_step"]):
            bar = "#" * int(f1 * 50)
            print(f"    Step {i+1:2d}: {f1:.4f}  {bar}")
    print("=" * 60)
