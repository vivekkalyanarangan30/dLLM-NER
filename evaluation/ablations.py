"""Ablation experiments for DiffusionNER-Zero.

Systematically evaluates the impact of key hyperparameters on NER performance:
    - **num_steps**: number of denoising steps (1, 2, 4, 8, 16)
    - **remasking**: ReMDM-style remasking during inference (on/off)
    - **negative_sampling**: number of negative entity types during training (0, 2, 5)

Each ablation sweeps one dimension while holding others at their default values.
Results are stored as nested dicts for easy comparison and table generation.
"""

import copy
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from evaluation.evaluate import compute_micro_f1

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default ablation settings (from eval.yaml / SPEC)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "num_steps": [1, 2, 4, 8, 16],
    "remasking": [True, False],
    "negative_sampling": [0, 2, 5],
}

DEFAULT_BASELINE: Dict[str, Any] = {
    "num_steps": 8,
    "remasking": False,
    "negative_sampling": 2,
}


# ---------------------------------------------------------------------------
# Single-setting evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_single_setting(
    model: Any,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    num_steps: int = 8,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run evaluation for a single configuration setting.

    Parameters
    ----------
    model :
        The model (or a wrapper that respects configuration).
    tokenizer :
        Tokenizer.
    benchmark : List[Dict[str, Any]]
        Evaluation examples.
    entity_types : List[str]
        Entity types to query.
    extract_fn : Callable
        Extraction function with signature::

            extract_fn(model, tokenizer, text, entity_types, num_steps=N)
            -> List[Dict[str, str]]
    num_steps : int
        Number of denoising steps.
    show_progress : bool
        Whether to display a progress bar.

    Returns
    -------
    Dict[str, Any]
        Metrics dict with ``"f1"``, ``"precision"``, ``"recall"``, etc.
    """
    all_predictions: List[List[Dict[str, str]]] = []
    all_gold: List[List[Dict[str, str]]] = []

    iterator = tqdm(benchmark, desc=f"Steps={num_steps}", disable=not show_progress)
    start = time.time()

    for example in iterator:
        text = example["text"]
        gold_entities = example["entities"]
        types = entity_types if entity_types else example.get("entity_types", [])

        try:
            pred_entities = extract_fn(
                model, tokenizer, text, types, num_steps=num_steps
            )
        except Exception:
            logger.exception("Extraction failed for: %s...", text[:80])
            pred_entities = []

        all_predictions.append(pred_entities)
        all_gold.append(gold_entities)

    wall_time = time.time() - start
    metrics = compute_micro_f1(all_predictions, all_gold)
    metrics["wall_time"] = wall_time
    metrics["num_examples"] = len(benchmark)

    return metrics


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------

def ablate_num_steps(
    model: Any,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    steps_list: Optional[List[int]] = None,
    show_progress: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Ablation over the number of denoising steps.

    Parameters
    ----------
    model, tokenizer, benchmark, entity_types, extract_fn :
        Standard evaluation arguments.
    steps_list : List[int], optional
        Step counts to test.  Defaults to ``[1, 2, 4, 8, 16]``.
    show_progress : bool
        Whether to display progress bars.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        ``{num_steps: metrics}``
    """
    if steps_list is None:
        steps_list = DEFAULT_CONFIG["num_steps"]

    results: Dict[int, Dict[str, Any]] = {}
    for num_steps in steps_list:
        logger.info("Ablation: num_steps=%d", num_steps)
        metrics = _evaluate_single_setting(
            model, tokenizer, benchmark, entity_types, extract_fn,
            num_steps=num_steps, show_progress=show_progress,
        )
        results[num_steps] = metrics
        logger.info("  num_steps=%d -> F1=%.4f", num_steps, metrics["f1"])

    return results


def ablate_remasking(
    model: Any,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn_no_remask: Callable,
    extract_fn_with_remask: Callable,
    num_steps: int = 8,
    show_progress: bool = True,
) -> Dict[bool, Dict[str, Any]]:
    """Ablation over ReMDM-style remasking.

    Requires two extraction functions: one with remasking enabled and one
    without.

    Parameters
    ----------
    extract_fn_no_remask : Callable
        Extraction function with remasking disabled.
    extract_fn_with_remask : Callable
        Extraction function with remasking enabled.
    num_steps : int
        Number of denoising steps.

    Returns
    -------
    Dict[bool, Dict[str, Any]]
        ``{True: metrics_with_remask, False: metrics_without_remask}``
    """
    results: Dict[bool, Dict[str, Any]] = {}

    for use_remask, fn in [(False, extract_fn_no_remask), (True, extract_fn_with_remask)]:
        logger.info("Ablation: remasking=%s", use_remask)
        metrics = _evaluate_single_setting(
            model, tokenizer, benchmark, entity_types, fn,
            num_steps=num_steps, show_progress=show_progress,
        )
        results[use_remask] = metrics
        logger.info("  remasking=%s -> F1=%.4f", use_remask, metrics["f1"])

    return results


def ablate_negative_sampling(
    model_loader_fn: Callable,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    neg_counts: Optional[List[int]] = None,
    num_steps: int = 8,
    show_progress: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Ablation over negative sampling counts.

    Since negative sampling affects training, this requires models trained with
    different negative sampling rates.  The ``model_loader_fn`` is called with
    the negative count to load the appropriate checkpoint.

    Parameters
    ----------
    model_loader_fn : Callable
        Function with signature ``model_loader_fn(neg_count) -> model`` that
        loads the model checkpoint trained with ``neg_count`` negative types.
    neg_counts : List[int], optional
        Negative sampling counts to test.  Defaults to ``[0, 2, 5]``.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        ``{neg_count: metrics}``
    """
    if neg_counts is None:
        neg_counts = DEFAULT_CONFIG["negative_sampling"]

    results: Dict[int, Dict[str, Any]] = {}

    for neg_count in neg_counts:
        logger.info("Ablation: negative_sampling=%d", neg_count)
        model = model_loader_fn(neg_count)
        metrics = _evaluate_single_setting(
            model, tokenizer, benchmark, entity_types, extract_fn,
            num_steps=num_steps, show_progress=show_progress,
        )
        results[neg_count] = metrics
        logger.info("  negative_sampling=%d -> F1=%.4f", neg_count, metrics["f1"])

    return results


# ---------------------------------------------------------------------------
# Unified ablation driver
# ---------------------------------------------------------------------------

def run_ablations(
    model: Any,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    config: Optional[Dict[str, Any]] = None,
    extract_fn_with_remask: Optional[Callable] = None,
    model_loader_fn: Optional[Callable] = None,
    show_progress: bool = True,
) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    """Run all ablation experiments specified in the config.

    Parameters
    ----------
    model :
        The default model.
    tokenizer :
        Tokenizer.
    benchmark : List[Dict[str, Any]]
        Evaluation examples.
    entity_types : List[str]
        Entity types to query.
    extract_fn : Callable
        Default extraction function.
    config : Dict[str, Any], optional
        Ablation configuration.  Defaults to :data:`DEFAULT_CONFIG`.  Keys:

        - ``"num_steps"`` (List[int]): step counts to ablate.
        - ``"remasking"`` (List[bool]): whether to test remasking.
        - ``"negative_sampling"`` (List[int]): negative type counts.
    extract_fn_with_remask : Callable, optional
        Extraction function with remasking enabled.  Required if the config
        includes a remasking ablation.
    model_loader_fn : Callable, optional
        Function to load models with different negative sampling.  Required
        if the config includes a negative sampling ablation.
    show_progress : bool
        Whether to display progress bars.

    Returns
    -------
    Dict[str, Dict[Any, Dict[str, Any]]]
        ``{ablation_name: {setting_value: metrics}}``
    """
    if config is None:
        config = DEFAULT_CONFIG

    all_results: Dict[str, Dict[Any, Dict[str, Any]]] = {}

    # 1. Number of denoising steps
    if "num_steps" in config:
        logger.info("--- Ablation: num_steps ---")
        all_results["num_steps"] = ablate_num_steps(
            model, tokenizer, benchmark, entity_types, extract_fn,
            steps_list=config["num_steps"],
            show_progress=show_progress,
        )

    # 2. Remasking
    if "remasking" in config and config["remasking"]:
        if extract_fn_with_remask is None:
            logger.warning(
                "Skipping remasking ablation: extract_fn_with_remask not provided."
            )
        else:
            logger.info("--- Ablation: remasking ---")
            all_results["remasking"] = ablate_remasking(
                model, tokenizer, benchmark, entity_types,
                extract_fn_no_remask=extract_fn,
                extract_fn_with_remask=extract_fn_with_remask,
                show_progress=show_progress,
            )

    # 3. Negative sampling
    if "negative_sampling" in config:
        if model_loader_fn is None:
            logger.warning(
                "Skipping negative_sampling ablation: model_loader_fn not provided."
            )
        else:
            logger.info("--- Ablation: negative_sampling ---")
            all_results["negative_sampling"] = ablate_negative_sampling(
                model_loader_fn, tokenizer, benchmark, entity_types, extract_fn,
                neg_counts=config["negative_sampling"],
                show_progress=show_progress,
            )

    return all_results


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def print_ablation_results(results: Dict[str, Dict[Any, Dict[str, Any]]]) -> None:
    """Print formatted ablation results.

    Parameters
    ----------
    results : Dict[str, Dict[Any, Dict[str, Any]]]
        Output of :func:`run_ablations`.
    """
    for ablation_name, settings in results.items():
        print(f"\nAblation: {ablation_name}")
        print("-" * 60)
        header = f"  {'Setting':>20s} | {'P':>8s} | {'R':>8s} | {'F1':>8s} | {'Time (s)':>10s}"
        print(header)
        print("  " + "-" * 56)

        for setting_value, metrics in sorted(settings.items(), key=lambda x: str(x[0])):
            wall_time = metrics.get("wall_time", 0.0)
            print(
                f"  {str(setting_value):>20s} | "
                f"{metrics['precision']:8.4f} | "
                f"{metrics['recall']:8.4f} | "
                f"{metrics['f1']:8.4f} | "
                f"{wall_time:10.2f}"
            )
        print()


def ablation_results_to_dict(
    results: Dict[str, Dict[Any, Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert ablation results to a flat list format for serialisation.

    Parameters
    ----------
    results : Dict[str, Dict[Any, Dict[str, Any]]]
        Nested ablation results.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        ``{ablation_name: [{"setting": value, "f1": ..., "precision": ..., ...}]}``
    """
    flat: Dict[str, List[Dict[str, Any]]] = {}
    for ablation_name, settings in results.items():
        rows = []
        for setting_value, metrics in sorted(settings.items(), key=lambda x: str(x[0])):
            row = {"setting": setting_value}
            for key in ("f1", "precision", "recall", "tp", "fp", "fn", "wall_time"):
                if key in metrics:
                    row[key] = metrics[key]
            rows.append(row)
        flat[ablation_name] = rows
    return flat
