"""Hallucination rate measurement for DiffusionNER-Zero.

A hallucination is defined as a predicted entity whose text does NOT appear as
a substring of the original source text.  This module computes hallucination
rates for a single model and provides a comparison utility for two models
(diffusion vs UniNER).

Expected entity format::

    {"type": str, "text": str}
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core hallucination computation
# ---------------------------------------------------------------------------

def compute_hallucination_rate(
    predictions: List[List[Dict[str, str]]],
    source_texts: List[str],
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """Compute the hallucination rate for model predictions.

    An entity is hallucinated if its text span does not appear as a substring
    of the corresponding source text.

    Parameters
    ----------
    predictions : list of list of dict
        Per-example predictions.  Each example is a list of entity dicts with
        ``"type"`` and ``"text"`` keys.
    source_texts : list of str
        The source passage for each example (same length as *predictions*).
    case_sensitive : bool, optional
        If False (default), the substring check is case-insensitive.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"hallucination_rate"`` (float): fraction of total predicted
          entities that are hallucinated.  0.0 if no entities predicted.
        - ``"total_entities"`` (int): total number of predicted entities
          across all examples.
        - ``"hallucinated_entities"`` (int): number of hallucinated entities.
        - ``"grounded_entities"`` (int): number of entities found in source.
        - ``"per_example_rates"`` (list of float): per-example hallucination
          rates (NaN-free; 0.0 when an example has no predictions).
        - ``"examples"`` (list of dict): hallucinated entity instances with
          ``"example_idx"``, ``"entity_type"``, ``"entity_text"``, and
          ``"source_snippet"`` (first 120 chars of source).
    """
    assert len(predictions) == len(source_texts), (
        f"Length mismatch: predictions={len(predictions)}, "
        f"source_texts={len(source_texts)}"
    )

    total_entities = 0
    hallucinated_entities = 0
    hallucinated_examples: List[Dict[str, Any]] = []
    per_example_rates: List[float] = []

    for idx, (preds, source) in enumerate(zip(predictions, source_texts)):
        source_check = source if case_sensitive else source.lower()
        n_pred = len(preds)
        n_halluc = 0

        for ent in preds:
            ent_text = ent["text"].strip()
            text_check = ent_text if case_sensitive else ent_text.lower()

            if text_check not in source_check:
                n_halluc += 1
                hallucinated_examples.append({
                    "example_idx": idx,
                    "entity_type": ent["type"].strip().lower(),
                    "entity_text": ent_text,
                    "source_snippet": source[:120],
                })

        total_entities += n_pred
        hallucinated_entities += n_halluc
        per_example_rates.append(n_halluc / n_pred if n_pred > 0 else 0.0)

    rate = hallucinated_entities / total_entities if total_entities > 0 else 0.0

    return {
        "hallucination_rate": rate,
        "total_entities": total_entities,
        "hallucinated_entities": hallucinated_entities,
        "grounded_entities": total_entities - hallucinated_entities,
        "per_example_rates": per_example_rates,
        "examples": hallucinated_examples,
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_hallucination_rates(
    diffusion_preds: List[List[Dict[str, str]]],
    uniner_preds: List[List[Dict[str, str]]],
    source_texts: List[str],
    case_sensitive: bool = False,
) -> Dict[str, Any]:
    """Compare hallucination rates between the diffusion model and UniNER.

    Parameters
    ----------
    diffusion_preds : list of list of dict
        Per-example diffusion model predictions.
    uniner_preds : list of list of dict
        Per-example UniNER predictions.
    source_texts : list of str
        Source passages (same length as both prediction lists).
    case_sensitive : bool, optional
        If False (default), the substring check is case-insensitive.

    Returns
    -------
    dict
        Dictionary with keys ``"diffusion"``, ``"uniner"`` (each the output of
        :func:`compute_hallucination_rate`), and ``"summary"`` (a compact
        comparison dict).
    """
    diff_result = compute_hallucination_rate(
        diffusion_preds, source_texts, case_sensitive=case_sensitive,
    )
    uni_result = compute_hallucination_rate(
        uniner_preds, source_texts, case_sensitive=case_sensitive,
    )

    summary = {
        "diffusion_hallucination_rate": diff_result["hallucination_rate"],
        "uniner_hallucination_rate": uni_result["hallucination_rate"],
        "diffusion_total_entities": diff_result["total_entities"],
        "uniner_total_entities": uni_result["total_entities"],
        "diffusion_hallucinated": diff_result["hallucinated_entities"],
        "uniner_hallucinated": uni_result["hallucinated_entities"],
    }

    return {
        "diffusion": diff_result,
        "uniner": uni_result,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_hallucination_report(
    comparison: Dict[str, Any],
    output_path: str = "results/hallucination_report.txt",
) -> str:
    """Generate a human-readable hallucination rate report.

    Parameters
    ----------
    comparison : dict
        Output of :func:`compare_hallucination_rates`.
    output_path : str
        Path to write the report.

    Returns
    -------
    str
        The full report text.
    """
    summary = comparison["summary"]
    diff_result = comparison["diffusion"]
    uni_result = comparison["uniner"]

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Hallucination Rate Comparison")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"{'Metric':<35} {'DiffusionNER':>12} {'UniNER':>12}")
    lines.append("-" * 60)
    lines.append(
        f"{'Total predicted entities':<35} "
        f"{summary['diffusion_total_entities']:>12} "
        f"{summary['uniner_total_entities']:>12}"
    )
    lines.append(
        f"{'Hallucinated entities':<35} "
        f"{summary['diffusion_hallucinated']:>12} "
        f"{summary['uniner_hallucinated']:>12}"
    )
    lines.append(
        f"{'Hallucination rate':<35} "
        f"{summary['diffusion_hallucination_rate']:>11.4f} "
        f"{summary['uniner_hallucination_rate']:>12.4f}"
    )
    lines.append("")

    # Show a few hallucinated examples from each model
    max_examples = 5
    for model_name, result in [
        ("DiffusionNER-Zero", diff_result),
        ("UniNER", uni_result),
    ]:
        examples = result["examples"]
        lines.append(f"--- {model_name}: Hallucinated examples "
                      f"({len(examples)} total, showing up to {max_examples}) ---")
        for ex in examples[:max_examples]:
            lines.append(
                f"  [{ex['entity_type']}] \"{ex['entity_text']}\" "
                f"(example {ex['example_idx']})"
            )
            lines.append(f"    source: \"{ex['source_snippet']}...\"")
        lines.append("")

    report = "\n".join(lines)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Hallucination report written to %s", output_path)
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compute and compare hallucination rates for NER models."
    )
    parser.add_argument(
        "--diffusion-preds",
        type=str,
        required=True,
        help="Path to JSON file with diffusion model predictions.",
    )
    parser.add_argument(
        "--uniner-preds",
        type=str,
        required=True,
        help="Path to JSON file with UniNER predictions.",
    )
    parser.add_argument(
        "--source-texts",
        type=str,
        required=True,
        help="Path to JSON file with source text strings.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/hallucination_report.txt",
        help="Path to write the hallucination report.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        default=False,
        help="Use case-sensitive substring matching.",
    )
    args = parser.parse_args()

    with open(args.diffusion_preds, "r", encoding="utf-8") as f:
        diffusion_preds = json.load(f)
    with open(args.uniner_preds, "r", encoding="utf-8") as f:
        uniner_preds = json.load(f)
    with open(args.source_texts, "r", encoding="utf-8") as f:
        source_texts = json.load(f)

    comparison = compare_hallucination_rates(
        diffusion_preds, uniner_preds, source_texts,
        case_sensitive=args.case_sensitive,
    )

    report = generate_hallucination_report(comparison, args.output_path)

    print(f"\nSummary: {comparison['summary']}")
    print(f"\nReport written to: {args.output_path}")
