"""Negative type sampling for DiffusionNER-Zero training.

During training, we present the model with a mix of ground-truth entity types
(those actually present in the passage) and negative entity types (plausible
types that are NOT present). This teaches the model to output "none" when a
queried type has no matching entities, reducing hallucination at inference time.

For each training example we sample 2-5 negative types from a global pool of
all entity types seen in the dataset.
"""

import random
from typing import Dict, List, Set, Sequence

from datasets import Dataset


def build_type_pool(dataset: Dataset) -> List[str]:
    """Extract all unique entity types from the reformatted dataset.

    Parameters
    ----------
    dataset : datasets.Dataset or list[dict]
        Each element must have an ``"entities"`` field, which is a list of
        dicts each containing a ``"type"`` key.

    Returns
    -------
    List[str]
        Sorted list of unique entity type strings found across the entire
        dataset.  Sorting ensures deterministic ordering for reproducibility.
    """
    type_set: Set[str] = set()
    for example in dataset:
        for entity in example.get("entities", []):
            etype = entity.get("type")
            if etype:
                type_set.add(etype)
    return sorted(type_set)


def sample_negative_types(
    gt_types: Sequence[str],
    type_pool: List[str],
    min_neg: int = 2,
    max_neg: int = 5,
) -> List[str]:
    """Sample negative entity types that are NOT present in the ground truth.

    Parameters
    ----------
    gt_types : Sequence[str]
        Ground-truth entity types for the current passage.
    type_pool : List[str]
        Global pool of all entity types in the dataset.
    min_neg : int, optional
        Minimum number of negative types to sample (default 2).
    max_neg : int, optional
        Maximum number of negative types to sample (default 5).

    Returns
    -------
    List[str]
        A list of ``n`` entity types not in *gt_types*, where
        ``n = min(randint(min_neg, max_neg), len(neg_candidates))``.
        Returns an empty list when no negative candidates are available.
    """
    gt_set = set(gt_types)
    neg_candidates = [t for t in type_pool if t not in gt_set]
    n = min(random.randint(min_neg, max_neg), len(neg_candidates))
    return random.sample(neg_candidates, n) if n > 0 else []


def sample_negative_types_deterministic(
    gt_types: Sequence[str],
    type_pool: List[str],
    n_neg: int = 3,
) -> List[str]:
    """Deterministic variant that always samples exactly *n_neg* negatives.

    Useful for evaluation / reproducible preprocessing.  Falls back to fewer
    types if the candidate pool is too small.

    Parameters
    ----------
    gt_types : Sequence[str]
        Ground-truth entity types for the current passage.
    type_pool : List[str]
        Global pool of all entity types in the dataset.
    n_neg : int, optional
        Exact number of negatives to sample (default 3).

    Returns
    -------
    List[str]
        Up to *n_neg* entity types not in *gt_types*.
    """
    gt_set = set(gt_types)
    neg_candidates = [t for t in type_pool if t not in gt_set]
    n = min(n_neg, len(neg_candidates))
    return random.sample(neg_candidates, n) if n > 0 else []
