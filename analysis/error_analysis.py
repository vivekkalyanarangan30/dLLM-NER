"""Error analysis for DiffusionNER-Zero vs UniNER predictions.

Compares predictions from the diffusion model and the autoregressive baseline
(UniNER-7B-type) against gold-standard annotations.  Categorizes outcomes into
four buckets (both correct, diffusion-only, UniNER-only, both wrong) and
performs fine-grained error typing (type errors, boundary errors, missing
entities, hallucinated entities).

Expected entity format throughout::

    {"type": str, "text": str}

Gold, diffusion, and UniNER predictions are all lists of examples, where each
example is itself a list of entity dicts.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_set(entities: List[Dict[str, str]]) -> Set[Tuple[str, str]]:
    """Convert a list of entity dicts to a set of (type, text) tuples."""
    return {(e["type"].strip().lower(), e["text"].strip()) for e in entities}


def _entity_texts(entities: List[Dict[str, str]]) -> Set[str]:
    """Extract the set of entity text spans (lowered) from a list."""
    return {e["text"].strip().lower() for e in entities}


def _entity_types(entities: List[Dict[str, str]]) -> Set[str]:
    """Extract the set of entity types (lowered) from a list."""
    return {e["type"].strip().lower() for e in entities}


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_predictions(
    diffusion_preds: List[List[Dict[str, str]]],
    uniner_preds: List[List[Dict[str, str]]],
    gold: List[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """Compare predictions from both models against gold annotations.

    For each gold entity, we check whether each model predicted it (strict
    match on both type and text).  Entities are then categorized into:

    - **both_correct**: both models predicted this entity correctly.
    - **diffusion_only**: only the diffusion model got it right.
    - **uniner_only**: only UniNER got it right.
    - **both_wrong**: neither model predicted the entity.

    Parameters
    ----------
    diffusion_preds : list of list of dict
        Per-example diffusion model predictions.
    uniner_preds : list of list of dict
        Per-example UniNER predictions.
    gold : list of list of dict
        Per-example gold entity annotations.

    Returns
    -------
    dict
        Dictionary with keys ``"counts"`` (mapping category name to int) and
        ``"examples"`` (mapping category name to a list of example dicts with
        ``"example_idx"``, ``"entity_type"``, and ``"entity_text"``).
    """
    assert len(diffusion_preds) == len(uniner_preds) == len(gold), (
        f"Length mismatch: diffusion={len(diffusion_preds)}, "
        f"uniner={len(uniner_preds)}, gold={len(gold)}"
    )

    categories = {
        "both_correct": [],
        "diffusion_only": [],
        "uniner_only": [],
        "both_wrong": [],
    }

    for idx, (diff_ents, uni_ents, gold_ents) in enumerate(
        zip(diffusion_preds, uniner_preds, gold)
    ):
        diff_set = _entity_set(diff_ents)
        uni_set = _entity_set(uni_ents)

        for g in gold_ents:
            key = (g["type"].strip().lower(), g["text"].strip())
            in_diff = key in diff_set
            in_uni = key in uni_set

            entry = {
                "example_idx": idx,
                "entity_type": key[0],
                "entity_text": key[1],
            }

            if in_diff and in_uni:
                categories["both_correct"].append(entry)
            elif in_diff and not in_uni:
                categories["diffusion_only"].append(entry)
            elif not in_diff and in_uni:
                categories["uniner_only"].append(entry)
            else:
                categories["both_wrong"].append(entry)

    counts = {k: len(v) for k, v in categories.items()}

    return {"counts": counts, "examples": categories}


# ---------------------------------------------------------------------------
# Error typing
# ---------------------------------------------------------------------------

def analyze_error_types(
    predictions: List[List[Dict[str, str]]],
    gold: List[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """Analyze the types of errors a model makes.

    Error categories:

    - **type_error**: predicted entity text matches a gold span but with the
      wrong entity type.
    - **boundary_error**: predicted entity text partially overlaps a gold span
      (substring in either direction) but is not an exact match, and the type
      matches.
    - **missing_entity**: a gold entity that was not predicted at all.
    - **hallucinated_entity**: a predicted entity that does not match any gold
      entity (even partially).

    Parameters
    ----------
    predictions : list of list of dict
        Per-example model predictions.
    gold : list of list of dict
        Per-example gold entity annotations.

    Returns
    -------
    dict
        Dictionary with ``"counts"`` (error type -> int) and ``"examples"``
        (error type -> list of example dicts).
    """
    assert len(predictions) == len(gold), (
        f"Length mismatch: predictions={len(predictions)}, gold={len(gold)}"
    )

    errors: Dict[str, List[Dict[str, Any]]] = {
        "type_error": [],
        "boundary_error": [],
        "missing_entity": [],
        "hallucinated_entity": [],
    }

    for idx, (pred_ents, gold_ents) in enumerate(zip(predictions, gold)):
        pred_set = _entity_set(pred_ents)
        gold_set = _entity_set(gold_ents)

        gold_texts = _entity_texts(gold_ents)
        pred_texts = _entity_texts(pred_ents)

        # --- Missing entities ---
        for g in gold_ents:
            gkey = (g["type"].strip().lower(), g["text"].strip())
            if gkey in pred_set:
                continue  # correctly predicted

            g_text_lower = g["text"].strip().lower()
            g_type_lower = g["type"].strip().lower()

            # Check for type error (same text, different type)
            matching_text_preds = [
                p for p in pred_ents
                if p["text"].strip().lower() == g_text_lower
                and p["type"].strip().lower() != g_type_lower
            ]
            if matching_text_preds:
                errors["type_error"].append({
                    "example_idx": idx,
                    "gold_type": g_type_lower,
                    "gold_text": g["text"].strip(),
                    "predicted_type": matching_text_preds[0]["type"].strip().lower(),
                })
                continue

            # Check for boundary error (partial overlap with matching type)
            found_boundary = False
            for p in pred_ents:
                p_text_lower = p["text"].strip().lower()
                p_type_lower = p["type"].strip().lower()
                if p_type_lower != g_type_lower:
                    continue
                # Partial overlap: one is a substring of the other
                if (
                    (p_text_lower in g_text_lower or g_text_lower in p_text_lower)
                    and p_text_lower != g_text_lower
                ):
                    errors["boundary_error"].append({
                        "example_idx": idx,
                        "gold_type": g_type_lower,
                        "gold_text": g["text"].strip(),
                        "predicted_text": p["text"].strip(),
                    })
                    found_boundary = True
                    break

            if not found_boundary:
                errors["missing_entity"].append({
                    "example_idx": idx,
                    "entity_type": g_type_lower,
                    "entity_text": g["text"].strip(),
                })

        # --- Hallucinated entities ---
        for p in pred_ents:
            pkey = (p["type"].strip().lower(), p["text"].strip())
            if pkey in gold_set:
                continue  # correctly predicted

            p_text_lower = p["text"].strip().lower()
            p_type_lower = p["type"].strip().lower()

            # Already accounted for as type error or boundary error?
            # We check if this predicted entity's text matches any gold text
            is_type_error = p_text_lower in gold_texts
            is_boundary = any(
                (p_text_lower in gt.lower() or gt.lower() in p_text_lower)
                and p_text_lower != gt.lower()
                for gt in [g["text"].strip() for g in gold_ents
                           if g["type"].strip().lower() == p_type_lower]
            )

            if not is_type_error and not is_boundary:
                errors["hallucinated_entity"].append({
                    "example_idx": idx,
                    "entity_type": p_type_lower,
                    "entity_text": p["text"].strip(),
                })

    counts = {k: len(v) for k, v in errors.items()}
    return {"counts": counts, "examples": errors}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_error_report(
    diffusion_results: Dict[str, Any],
    uniner_results: Dict[str, Any],
    gold: List[List[Dict[str, str]]],
    output_path: str = "results/error_analysis.txt",
) -> str:
    """Generate a detailed error analysis report and write it to disk.

    Parameters
    ----------
    diffusion_results : dict
        Output of :func:`analyze_error_types` for the diffusion model.
    uniner_results : dict
        Output of :func:`analyze_error_types` for UniNER.
    gold : list of list of dict
        Per-example gold entity annotations.
    output_path : str
        Path to write the report file.

    Returns
    -------
    str
        The full report text.
    """
    total_gold = sum(len(g) for g in gold)

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DiffusionNER-Zero vs UniNER: Error Analysis Report")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Total gold entities: {total_gold}")
    lines.append(f"Number of examples:  {len(gold)}")
    lines.append("")

    # --- Error type comparison ---
    lines.append("-" * 72)
    lines.append("Error Type Breakdown")
    lines.append("-" * 72)
    lines.append(
        f"{'Error Type':<25} {'DiffusionNER':>15} {'UniNER':>15}"
    )
    lines.append("-" * 55)

    error_types = ["type_error", "boundary_error", "missing_entity", "hallucinated_entity"]
    for etype in error_types:
        diff_count = diffusion_results["counts"].get(etype, 0)
        uni_count = uniner_results["counts"].get(etype, 0)
        lines.append(f"{etype:<25} {diff_count:>15} {uni_count:>15}")

    lines.append("")

    # --- Example errors for each model ---
    max_examples_per_category = 5

    for model_name, results in [
        ("DiffusionNER-Zero", diffusion_results),
        ("UniNER", uniner_results),
    ]:
        lines.append("-" * 72)
        lines.append(f"Example Errors: {model_name}")
        lines.append("-" * 72)

        for etype in error_types:
            examples = results["examples"].get(etype, [])
            lines.append(f"\n  {etype} ({len(examples)} total):")
            for ex in examples[:max_examples_per_category]:
                lines.append(f"    - {ex}")

        lines.append("")

    report = "\n".join(lines)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Error analysis report written to %s", output_path)
    return report


# ---------------------------------------------------------------------------
# End-to-end analysis
# ---------------------------------------------------------------------------

def run_full_analysis(
    diffusion_preds: List[List[Dict[str, str]]],
    uniner_preds: List[List[Dict[str, str]]],
    gold: List[List[Dict[str, str]]],
    output_path: str = "results/error_analysis.txt",
) -> Dict[str, Any]:
    """Run the full error analysis pipeline.

    Compares both models, analyzes error types for each, and generates a
    report.

    Parameters
    ----------
    diffusion_preds : list of list of dict
        Per-example diffusion model predictions.
    uniner_preds : list of list of dict
        Per-example UniNER predictions.
    gold : list of list of dict
        Per-example gold entity annotations.
    output_path : str
        Path for the output report.

    Returns
    -------
    dict
        Combined results with keys ``"comparison"``, ``"diffusion_errors"``,
        ``"uniner_errors"``, and ``"report_path"``.
    """
    comparison = compare_predictions(diffusion_preds, uniner_preds, gold)

    diffusion_errors = analyze_error_types(diffusion_preds, gold)
    uniner_errors = analyze_error_types(uniner_preds, gold)

    report = generate_error_report(diffusion_errors, uniner_errors, gold, output_path)

    logger.info("Comparison counts: %s", comparison["counts"])
    logger.info("Diffusion error counts: %s", diffusion_errors["counts"])
    logger.info("UniNER error counts: %s", uniner_errors["counts"])

    return {
        "comparison": comparison,
        "diffusion_errors": diffusion_errors,
        "uniner_errors": uniner_errors,
        "report_path": output_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run error analysis comparing DiffusionNER-Zero and UniNER."
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
        "--gold",
        type=str,
        required=True,
        help="Path to JSON file with gold annotations.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/error_analysis.txt",
        help="Path to write the error analysis report.",
    )
    args = parser.parse_args()

    with open(args.diffusion_preds, "r", encoding="utf-8") as f:
        diffusion_preds = json.load(f)
    with open(args.uniner_preds, "r", encoding="utf-8") as f:
        uniner_preds = json.load(f)
    with open(args.gold, "r", encoding="utf-8") as f:
        gold = json.load(f)

    results = run_full_analysis(diffusion_preds, uniner_preds, gold, args.output_path)

    print(f"\nComparison counts: {results['comparison']['counts']}")
    print(f"Diffusion error counts: {results['diffusion_errors']['counts']}")
    print(f"UniNER error counts: {results['uniner_errors']['counts']}")
    print(f"\nReport written to: {results['report_path']}")
