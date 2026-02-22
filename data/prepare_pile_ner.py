"""Download and reformat Pile-NER-type data for diffusion-based NER training.

The source dataset ``Universal-NER/Pile-NER-type`` on HuggingFace stores data
in a multi-turn conversational format: each row contains a single
(passage, entity_type) query and a GPT response listing entity mentions of that
type as a JSON list.

We reformat into a **single-turn** layout where each passage is paired with
ALL of its entity types at once, plus a handful of randomly-sampled negative
types (types not present in the passage) to teach the model to output "none".

Output format per example::

    {
      "prompt":     "Extract entities of types: person, org, ...\nText: ...\nEntities:",
      "completion": " person: Ronaldo | org: Al Nassr | date: January 2023",
      "entities":   [{"type": "person", "text": "Ronaldo", "start": 0}, ...]
    }
"""

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from data.negative_sampling import build_type_pool, sample_negative_types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PILE_NER_HF_ID = "Universal-NER/Pile-NER-type"
DEFAULT_OUTPUT_DIR = Path("data/processed")
TRAIN_SPLIT_RATIO = 0.9


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_passage_and_type(human_message: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse the passage text and entity type from a human-turn message.

    The human message in Pile-NER-type has several format variants::

        # Variant 1 (most common):
        Text: <passage>

        What describes <type> in the text?

        # Variant 2:
        Text: <passage>
        Use the provided text ... entities that belong to the following category: <type>

        # Variant 3:
        Text: <passage>
        Please identify entities of category: <type>

    We extract the passage and the queried entity type.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        ``(passage, entity_type)`` or ``(None, None)`` on parse failure.
    """
    # Extract passage: everything after "Text: " up to the instruction line
    # Try all known instruction markers
    passage_match = re.search(
        r"Text:\s*(.+?)(?:\n+What describes\b|\nUse the provided text|\nPlease)",
        human_message,
        re.DOTALL,
    )
    if not passage_match:
        # Fallback: split on "Text: " and find the instruction boundary
        parts = human_message.split("Text: ", 1)
        if len(parts) < 2:
            return None, None
        rest = parts[1]
        for marker in ["What describes ", "Use the provided text", "Please "]:
            idx = rest.find(marker)
            if idx != -1:
                passage = rest[:idx].strip()
                break
        else:
            passage = rest.strip()
    else:
        passage = passage_match.group(1).strip()

    # Extract entity type — try multiple patterns
    # Pattern 1: "What describes <type> in the text?"
    type_match = re.search(r"What describes\s+(.+?)\s+in the text", human_message)
    if not type_match:
        # Pattern 2: "category: <type>"
        type_match = re.search(r"category:\s*(.+?)(?:\n|$)", human_message)
    if not type_match:
        # Pattern 3: "type: <type>"
        type_match = re.search(r"type:\s*(.+?)(?:\n|$)", human_message)
    if not type_match:
        return passage, None

    entity_type = type_match.group(1).strip().rstrip(".?")
    return passage, entity_type


def _parse_entity_list(gpt_response: str) -> List[str]:
    """Parse the GPT response which is a JSON list of entity strings.

    Parameters
    ----------
    gpt_response : str
        Raw GPT response, expected to be a JSON array like
        ``["entity1", "entity2"]``.

    Returns
    -------
    List[str]
        Parsed entity mention strings.  Returns empty list on failure.
    """
    gpt_response = gpt_response.strip()
    try:
        entities = json.loads(gpt_response)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if str(e).strip()]
    except json.JSONDecodeError:
        # Try to salvage by finding the first JSON array in the response
        match = re.search(r"\[.*?\]", gpt_response, re.DOTALL)
        if match:
            try:
                entities = json.loads(match.group(0))
                if isinstance(entities, list):
                    return [str(e).strip() for e in entities if str(e).strip()]
            except json.JSONDecodeError:
                pass
    return []


def _find_entity_start(passage: str, entity_text: str, used_offsets: set) -> int:
    """Find the character-level start offset of an entity mention in the passage.

    If the entity appears multiple times, we pick the first occurrence whose
    offset has not been used yet (to handle duplicate mentions).

    Parameters
    ----------
    passage : str
        The source passage text.
    entity_text : str
        The entity mention to locate.
    used_offsets : set
        Set of already-assigned start offsets (mutated in place).

    Returns
    -------
    int
        Character offset, or -1 if not found.
    """
    start = 0
    while True:
        idx = passage.find(entity_text, start)
        if idx == -1:
            # Case-insensitive fallback
            idx_lower = passage.lower().find(entity_text.lower(), start)
            if idx_lower == -1:
                return -1
            if idx_lower not in used_offsets:
                used_offsets.add(idx_lower)
                return idx_lower
            start = idx_lower + 1
        else:
            if idx not in used_offsets:
                used_offsets.add(idx)
                return idx
            start = idx + 1
        if start >= len(passage):
            return -1


# ---------------------------------------------------------------------------
# Grouping: multi-turn -> single passage
# ---------------------------------------------------------------------------

def group_by_passage(dataset: Dataset) -> Dict[str, List[Dict[str, Any]]]:
    """Group the multi-turn Pile-NER-type data by passage.

    Each row in the source dataset has a ``"conversations"`` field (list of
    dicts with ``"from"`` and ``"value"`` keys).  The human turn contains the
    passage and a single entity type; the GPT turn has the entity mentions.

    We group all (entity_type, mentions) pairs that share the same passage.

    Parameters
    ----------
    dataset : datasets.Dataset
        The raw Pile-NER-type dataset (single split).

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Mapping from passage text to a list of dicts, each with keys
        ``"type"`` (str) and ``"mentions"`` (List[str]).
    """
    passage_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for row in dataset:
        conversations = row.get("conversations", [])
        if len(conversations) < 2:
            skipped += 1
            continue

        # Each row can have MULTIPLE human/gpt pairs (one per entity type).
        # Process them in consecutive pairs.
        pairs_found = 0
        i = 0
        while i < len(conversations) - 1:
            if conversations[i].get("from") == "human" and \
               conversations[i + 1].get("from") == "gpt":
                human_msg = conversations[i].get("value", "")
                gpt_msg = conversations[i + 1].get("value", "")
                i += 2

                if not human_msg or not gpt_msg:
                    continue

                passage, entity_type = _extract_passage_and_type(human_msg)
                if not passage or not entity_type:
                    continue

                mentions = _parse_entity_list(gpt_msg)
                passage_groups[passage].append({
                    "type": entity_type,
                    "mentions": mentions,
                })
                pairs_found += 1
            else:
                i += 1

        if pairs_found == 0:
            skipped += 1

    if skipped:
        logger.warning("Skipped %d rows during grouping (parse failures).", skipped)
    logger.info(
        "Grouped %d unique passages from %d rows.", len(passage_groups), len(dataset)
    )
    return passage_groups


# ---------------------------------------------------------------------------
# Passage -> entities list
# ---------------------------------------------------------------------------

def _build_entity_list(
    passage: str,
    type_mention_pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert grouped (type, mentions) pairs into a flat entity list.

    Each entity dict has keys ``"type"``, ``"text"``, and ``"start"``
    (character offset in the passage).

    Parameters
    ----------
    passage : str
        The source passage text.
    type_mention_pairs : List[Dict[str, Any]]
        Each dict has ``"type"`` (str) and ``"mentions"`` (List[str]).

    Returns
    -------
    List[Dict[str, Any]]
        Flat list of entity dicts with ``type``, ``text``, ``start`` keys.
    """
    entities: List[Dict[str, Any]] = []
    used_offsets: set = set()

    for pair in type_mention_pairs:
        etype = pair["type"]
        for mention in pair["mentions"]:
            start = _find_entity_start(passage, mention, used_offsets)
            entities.append({
                "type": etype,
                "text": mention,
                "start": start,
            })

    return entities


# ---------------------------------------------------------------------------
# Formatting for diffusion training
# ---------------------------------------------------------------------------

def format_for_diffusion(
    passage: str,
    entities: List[Dict[str, Any]],
    all_types_pool: List[str],
) -> Dict[str, Any]:
    """Format a single passage + entities into a diffusion training example.

    The prompt lists a shuffled mix of ground-truth types and 2-5 negative
    types.  The completion lists entities sorted by their start offset in the
    passage, or ``" none"`` if no entities are present.

    Parameters
    ----------
    passage : str
        Source passage text.
    entities : List[Dict[str, Any]]
        Entity dicts with ``"type"``, ``"text"``, ``"start"`` keys.
    all_types_pool : List[str]
        Global pool of all entity types for negative sampling.

    Returns
    -------
    Dict[str, Any]
        Dict with ``"prompt"``, ``"completion"``, and ``"entities"`` keys.
    """
    gt_types = list(set(e["type"] for e in entities))

    # Negative sampling: 2-5 types NOT present in passage
    neg_types = sample_negative_types(gt_types, all_types_pool, min_neg=2, max_neg=5)

    query_types = gt_types + neg_types
    random.shuffle(query_types)

    prompt = (
        f"Extract entities of types: {', '.join(query_types)}\n"
        f"Text: {passage}\n"
        f"Entities:"
    )

    # Sort entities by start offset for consistent ordering
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    if sorted_ents:
        completion = " " + " | ".join(
            f'{e["type"]}: {e["text"]}' for e in sorted_ents
        )
    else:
        completion = " none"

    return {
        "prompt": prompt,
        "completion": completion,
        "entities": entities,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_reformat(
    output_dir: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = TRAIN_SPLIT_RATIO,
) -> DatasetDict:
    """Load Pile-NER-type from HuggingFace, reformat, split, and save.

    Steps:
        1. Load the dataset from HuggingFace.
        2. Group conversations by passage.
        3. Extract passage text, entity types, and entity mentions.
        4. Build a global type pool and call :func:`format_for_diffusion`
           for each passage.
        5. Split into train / val (default 90/10).
        6. Save to disk.

    Parameters
    ----------
    output_dir : str, optional
        Directory to write the processed dataset.  Defaults to
        ``data/processed``.
    seed : int, optional
        Random seed for reproducibility (default 42).
    train_ratio : float, optional
        Fraction of data for training (default 0.9).

    Returns
    -------
    datasets.DatasetDict
        A ``DatasetDict`` with ``"train"`` and ``"validation"`` splits.
    """
    random.seed(seed)
    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load raw dataset
    logger.info("Loading %s from HuggingFace...", PILE_NER_HF_ID)
    raw_dataset = load_dataset(PILE_NER_HF_ID, split="train")
    logger.info("Loaded %d rows.", len(raw_dataset))

    # 2. Group by passage
    passage_groups = group_by_passage(raw_dataset)

    # 3-4. Build entity lists and the global type pool
    # First pass: build all entity lists so we can extract the type pool
    passage_entity_pairs: List[Tuple[str, List[Dict[str, Any]]]] = []
    for passage, type_mention_pairs in passage_groups.items():
        entities = _build_entity_list(passage, type_mention_pairs)
        passage_entity_pairs.append((passage, entities))

    # Build the global type pool from all entities
    all_entities_dataset = [{"entities": ents} for _, ents in passage_entity_pairs]
    all_types_pool = build_type_pool(all_entities_dataset)
    logger.info("Built type pool with %d unique types.", len(all_types_pool))

    # Format each passage for diffusion training
    formatted_examples: List[Dict[str, Any]] = []
    for passage, entities in passage_entity_pairs:
        example = format_for_diffusion(passage, entities, all_types_pool)
        formatted_examples.append(example)

    logger.info("Formatted %d examples.", len(formatted_examples))

    # 5. Shuffle and split
    random.shuffle(formatted_examples)
    split_idx = int(len(formatted_examples) * train_ratio)
    train_data = formatted_examples[:split_idx]
    val_data = formatted_examples[split_idx:]
    logger.info("Split: %d train, %d val.", len(train_data), len(val_data))

    # Convert to HuggingFace datasets
    # Serialize entities as JSON strings for HF Dataset compatibility
    def serialize_entities(examples: List[Dict[str, Any]]) -> Dict[str, list]:
        return {
            "prompt": [ex["prompt"] for ex in examples],
            "completion": [ex["completion"] for ex in examples],
            "entities": [json.dumps(ex["entities"]) for ex in examples],
        }

    train_dict = serialize_entities(train_data)
    val_dict = serialize_entities(val_data)

    ds = DatasetDict({
        "train": Dataset.from_dict(train_dict),
        "validation": Dataset.from_dict(val_dict),
    })

    # 6. Save to disk
    ds.save_to_disk(str(output_path))
    logger.info("Saved processed dataset to %s.", output_path)

    # Also save the type pool for later use
    type_pool_path = output_path / "type_pool.json"
    with open(type_pool_path, "w") as f:
        json.dump(all_types_pool, f, indent=2)
    logger.info("Saved type pool (%d types) to %s.", len(all_types_pool), type_pool_path)

    return ds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download and reformat Pile-NER-type for diffusion NER training."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_SPLIT_RATIO,
        help="Fraction of data for training split.",
    )
    args = parser.parse_args()

    load_and_reformat(
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
