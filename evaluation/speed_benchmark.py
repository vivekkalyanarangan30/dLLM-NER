"""Inference speed benchmarking for DiffusionNER-Zero vs UniNER.

Measures throughput (examples/sec) and average latency (ms/example) for:
    - DiffusionNER-Zero at various step counts (e.g. 1, 4, 8)
    - UniNER-7B-type (autoregressive baseline)

Both models are benchmarked on the same examples with warmup runs excluded
from timing.  GPU synchronisation is performed before each timing measurement
to ensure accurate wall-clock measurement.
"""

import gc
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU synchronisation helper
# ---------------------------------------------------------------------------

def _sync_gpu() -> None:
    """Synchronize CUDA if available, to ensure accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# DiffusionNER speed benchmark
# ---------------------------------------------------------------------------

def benchmark_speed(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Callable,
    num_steps_list: Optional[List[int]] = None,
    num_warmup: int = 5,
    num_runs: int = 50,
    show_progress: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Benchmark DiffusionNER inference speed at multiple step counts.

    Parameters
    ----------
    model :
        The diffusion NER model.
    tokenizer :
        Associated tokenizer.
    examples : List[Dict[str, Any]]
        Evaluation examples with ``"text"`` and ``"entity_types"`` keys.
        If fewer than ``num_warmup + num_runs``, examples are cycled.
    entity_types : List[str]
        Entity types to query.
    extract_fn : Callable
        Extraction function with signature::

            extract_fn(model, tokenizer, text, entity_types, num_steps=N)
            -> List[Dict[str, str]]
    num_steps_list : List[int], optional
        Step counts to benchmark.  Defaults to ``[1, 4, 8]``.
    num_warmup : int
        Number of warmup iterations (excluded from timing).
    num_runs : int
        Number of timed iterations.
    show_progress : bool
        Whether to display progress bars.

    Returns
    -------
    Dict[int, Dict[str, float]]
        ``{num_steps: {"examples_per_sec": float, "avg_time_ms": float,
        "std_time_ms": float, "total_time_s": float, "num_runs": int}}``
    """
    if num_steps_list is None:
        num_steps_list = [1, 4, 8]

    # Cycle through examples if we don't have enough
    def _get_example(idx: int) -> Dict[str, Any]:
        return examples[idx % len(examples)]

    results: Dict[int, Dict[str, float]] = {}

    for num_steps in num_steps_list:
        logger.info("Benchmarking speed: num_steps=%d ...", num_steps)

        # Warmup
        for i in range(num_warmup):
            ex = _get_example(i)
            types = entity_types if entity_types else ex.get("entity_types", [])
            try:
                extract_fn(model, tokenizer, ex["text"], types, num_steps=num_steps)
            except Exception:
                pass
        _sync_gpu()

        # Timed runs
        times: List[float] = []
        iterator = range(num_runs)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Speed T={num_steps}")

        for i in iterator:
            ex = _get_example(num_warmup + i)
            types = entity_types if entity_types else ex.get("entity_types", [])

            _sync_gpu()
            t0 = time.perf_counter()
            try:
                extract_fn(model, tokenizer, ex["text"], types, num_steps=num_steps)
            except Exception:
                pass
            _sync_gpu()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg_time = sum(times) / len(times) if times else 0.0
        std_time = (
            (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            if len(times) > 1
            else 0.0
        )

        step_result = {
            "examples_per_sec": 1.0 / avg_time if avg_time > 0 else 0.0,
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "total_time_s": sum(times),
            "num_runs": len(times),
        }
        results[num_steps] = step_result

        logger.info(
            "  num_steps=%d: %.2f examples/sec (%.1f +/- %.1f ms/example)",
            num_steps,
            step_result["examples_per_sec"],
            step_result["avg_time_ms"],
            step_result["std_time_ms"],
        )

    return results


# ---------------------------------------------------------------------------
# UniNER speed benchmark
# ---------------------------------------------------------------------------

def benchmark_uniner_speed(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    entity_types: List[str],
    extract_fn: Optional[Callable] = None,
    num_warmup: int = 5,
    num_runs: int = 50,
    show_progress: bool = True,
) -> Dict[str, float]:
    """Benchmark UniNER-7B-type inference speed.

    UniNER uses one query per entity type, so total inference time scales
    with the number of entity types.

    Parameters
    ----------
    model :
        The UniNER model (or vLLM engine).
    tokenizer :
        Associated tokenizer.
    examples : List[Dict[str, Any]]
        Evaluation examples.
    entity_types : List[str]
        Entity types to query.
    extract_fn : Callable, optional
        UniNER extraction function.  If ``None``, attempts to import from
        ``baselines.run_uniner.extract_entities_uniner``.
    num_warmup : int
        Number of warmup iterations.
    num_runs : int
        Number of timed iterations.
    show_progress : bool
        Whether to display progress bars.

    Returns
    -------
    Dict[str, float]
        ``{"examples_per_sec": float, "avg_time_ms": float,
        "std_time_ms": float, "total_time_s": float, "num_runs": int,
        "num_entity_types": int}``
    """
    if extract_fn is None:
        from baselines.run_uniner import extract_entities_uniner
        extract_fn = extract_entities_uniner

    def _get_example(idx: int) -> Dict[str, Any]:
        return examples[idx % len(examples)]

    # Warmup
    for i in range(num_warmup):
        ex = _get_example(i)
        types = entity_types if entity_types else ex.get("entity_types", [])
        try:
            extract_fn(model, tokenizer, ex["text"], types)
        except Exception:
            pass
    _sync_gpu()

    # Timed runs
    times: List[float] = []
    iterator = range(num_runs)
    if show_progress:
        iterator = tqdm(iterator, desc="UniNER speed")

    for i in iterator:
        ex = _get_example(num_warmup + i)
        types = entity_types if entity_types else ex.get("entity_types", [])

        _sync_gpu()
        t0 = time.perf_counter()
        try:
            extract_fn(model, tokenizer, ex["text"], types)
        except Exception:
            pass
        _sync_gpu()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg_time = sum(times) / len(times) if times else 0.0
    std_time = (
        (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        if len(times) > 1
        else 0.0
    )

    result = {
        "examples_per_sec": 1.0 / avg_time if avg_time > 0 else 0.0,
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "total_time_s": sum(times),
        "num_runs": len(times),
        "num_entity_types": len(entity_types),
    }

    logger.info(
        "UniNER speed: %.2f examples/sec (%.1f +/- %.1f ms/example) "
        "with %d entity types",
        result["examples_per_sec"],
        result["avg_time_ms"],
        result["std_time_ms"],
        result["num_entity_types"],
    )

    return result


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def run_speed_comparison(
    diffusion_model: Any,
    diffusion_tokenizer: Any,
    uniner_model: Any,
    uniner_tokenizer: Any,
    examples: List[Dict[str, Any]],
    entity_types: List[str],
    diffusion_extract_fn: Callable,
    uniner_extract_fn: Optional[Callable] = None,
    num_steps_list: Optional[List[int]] = None,
    num_warmup: int = 5,
    num_runs: int = 50,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run a full speed comparison between DiffusionNER and UniNER.

    Parameters
    ----------
    diffusion_model, diffusion_tokenizer :
        DiffusionNER model and tokenizer.
    uniner_model, uniner_tokenizer :
        UniNER model and tokenizer.
    examples, entity_types :
        Shared evaluation examples and entity types.
    diffusion_extract_fn :
        DiffusionNER extraction function.
    uniner_extract_fn : Callable, optional
        UniNER extraction function.
    num_steps_list : List[int], optional
        Step counts for DiffusionNER.
    num_warmup, num_runs : int
        Warmup and timed run counts.
    show_progress : bool
        Whether to show progress bars.

    Returns
    -------
    Dict[str, Any]
        ``{"diffusion": {steps: metrics}, "uniner": metrics, "speedup": {steps: float}}``
    """
    if num_steps_list is None:
        num_steps_list = [1, 4, 8]

    logger.info("=== Speed benchmark: DiffusionNER-Zero ===")
    diffusion_results = benchmark_speed(
        diffusion_model, diffusion_tokenizer, examples, entity_types,
        diffusion_extract_fn,
        num_steps_list=num_steps_list,
        num_warmup=num_warmup,
        num_runs=num_runs,
        show_progress=show_progress,
    )

    # Clear GPU memory between models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=== Speed benchmark: UniNER-7B-type ===")
    uniner_results = benchmark_uniner_speed(
        uniner_model, uniner_tokenizer, examples, entity_types,
        extract_fn=uniner_extract_fn,
        num_warmup=num_warmup,
        num_runs=num_runs,
        show_progress=show_progress,
    )

    # Compute speedup ratios
    uniner_time = uniner_results["avg_time_ms"]
    speedup: Dict[int, float] = {}
    for steps, diff_metrics in diffusion_results.items():
        diff_time = diff_metrics["avg_time_ms"]
        if diff_time > 0:
            speedup[steps] = uniner_time / diff_time
        else:
            speedup[steps] = float("inf")

    comparison = {
        "diffusion": diffusion_results,
        "uniner": uniner_results,
        "speedup": speedup,
    }

    return comparison


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def print_speed_results(
    diffusion_results: Dict[int, Dict[str, float]],
    uniner_results: Optional[Dict[str, float]] = None,
) -> None:
    """Print a formatted speed benchmark comparison table.

    Parameters
    ----------
    diffusion_results : Dict[int, Dict[str, float]]
        DiffusionNER results from :func:`benchmark_speed`.
    uniner_results : Dict[str, float], optional
        UniNER results from :func:`benchmark_uniner_speed`.
    """
    print("\nInference Speed Benchmark")
    print("=" * 72)
    header = (
        f"  {'Model':25s} | {'Ex/sec':>10s} | {'Avg (ms)':>10s} | "
        f"{'Std (ms)':>10s} | {'Speedup':>8s}"
    )
    print(header)
    print("  " + "-" * 68)

    uniner_time = uniner_results["avg_time_ms"] if uniner_results else 0.0

    for steps in sorted(diffusion_results.keys()):
        metrics = diffusion_results[steps]
        name = f"DiffusionNER (T={steps})"
        speedup_str = ""
        if uniner_time > 0 and metrics["avg_time_ms"] > 0:
            speedup = uniner_time / metrics["avg_time_ms"]
            speedup_str = f"{speedup:.2f}x"
        print(
            f"  {name:25s} | {metrics['examples_per_sec']:10.2f} | "
            f"{metrics['avg_time_ms']:10.1f} | {metrics['std_time_ms']:10.1f} | "
            f"{speedup_str:>8s}"
        )

    if uniner_results:
        print(
            f"  {'UniNER-7B-type':25s} | {uniner_results['examples_per_sec']:10.2f} | "
            f"{uniner_results['avg_time_ms']:10.1f} | {uniner_results['std_time_ms']:10.1f} | "
            f"{'1.00x':>8s}"
        )

    print("=" * 72)
    print()
