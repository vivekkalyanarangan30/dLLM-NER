"""Robust parser for DiffusionNER-Zero model output.

The model produces structured output in the format:
    " person: Ronaldo | organization: Al Nassr | date: January 2023"
or simply:
    " none"

This module provides parsing utilities that handle edge cases such as
malformed separators, extra whitespace, empty segments, and missing colons.
It also provides a compliance checker to measure how well the model adheres
to the expected output format.
"""

import re
from typing import Dict, List, Optional


def parse_entities(output_text: str) -> List[Dict[str, str]]:
    """Parse model output into a list of entity dicts.

    Handles common edge cases:
    - Leading/trailing whitespace
    - "none" (case-insensitive) returns empty list
    - Empty or whitespace-only input returns empty list
    - Segments without a colon are silently skipped
    - Multiple colons in a segment: first colon is the delimiter
    - Empty type or text after splitting are skipped

    Parameters
    ----------
    output_text : str
        Raw model output, e.g. " person: Ronaldo | organization: Al Nassr".

    Returns
    -------
    List[Dict[str, str]]
        List of ``{"type": str, "text": str}`` dicts. Types are lower-cased
        and stripped. Text spans are stripped but otherwise preserved.
    """
    output_text = output_text.strip()

    if not output_text:
        return []

    if output_text.lower() == "none":
        return []

    entities: List[Dict[str, str]] = []

    for pair in output_text.split("|"):
        pair = pair.strip()
        if not pair:
            continue

        if ":" not in pair:
            # Malformed segment -- skip silently
            continue

        etype, span = pair.split(":", 1)
        etype = etype.strip().lower()
        span = span.strip()

        # Skip entries where either the type or text is empty
        if not etype or not span:
            continue

        entities.append({"type": etype, "text": span})

    return entities


def deduplicate_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate entities (same type and text).

    Preserves first-occurrence order.

    Parameters
    ----------
    entities : List[Dict[str, str]]
        Parsed entity list from :func:`parse_entities`.

    Returns
    -------
    List[Dict[str, str]]
        De-duplicated entity list.
    """
    seen: set = set()
    unique: List[Dict[str, str]] = []

    for ent in entities:
        key = (ent["type"], ent["text"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)

    return unique


def filter_entities_by_source(
    entities: List[Dict[str, str]],
    source_text: str,
    case_sensitive: bool = False,
) -> List[Dict[str, str]]:
    """Filter entities to only those whose text appears in the source.

    This is a post-hoc hallucination filter: any entity whose span text does
    not appear as a substring of the original passage is removed.

    Parameters
    ----------
    entities : List[Dict[str, str]]
        Parsed entity list.
    source_text : str
        Original input passage.
    case_sensitive : bool, optional
        Whether the substring check is case-sensitive (default False).

    Returns
    -------
    List[Dict[str, str]]
        Filtered entity list.
    """
    if case_sensitive:
        return [e for e in entities if e["text"] in source_text]

    source_lower = source_text.lower()
    return [e for e in entities if e["text"].lower() in source_lower]


def format_compliance_check(output_text: str) -> Dict[str, object]:
    """Check whether the model output follows the expected format.

    Expected format is one of:
    - ``"none"`` (case-insensitive, possibly with leading/trailing whitespace)
    - One or more ``"type: text"`` segments separated by ``" | "``

    Parameters
    ----------
    output_text : str
        Raw model output string.

    Returns
    -------
    Dict[str, object]
        Dictionary with the following keys:

        - ``"is_compliant"`` (bool): True if the output fully matches the
          expected format.
        - ``"is_none_output"`` (bool): True if the output is the "none" token.
        - ``"num_segments"`` (int): Number of pipe-delimited segments found.
        - ``"num_valid_segments"`` (int): Segments that have a valid
          ``type: text`` structure.
        - ``"num_malformed_segments"`` (int): Segments missing a colon or
          with empty type/text.
        - ``"malformed_segments"`` (List[str]): The raw text of any malformed
          segments for debugging.
        - ``"raw_output"`` (str): The original output text.
    """
    stripped = output_text.strip()

    result: Dict[str, object] = {
        "is_compliant": False,
        "is_none_output": False,
        "num_segments": 0,
        "num_valid_segments": 0,
        "num_malformed_segments": 0,
        "malformed_segments": [],
        "raw_output": output_text,
    }

    # Empty output is non-compliant
    if not stripped:
        return result

    # "none" output
    if stripped.lower() == "none":
        result["is_compliant"] = True
        result["is_none_output"] = True
        return result

    segments = stripped.split("|")
    result["num_segments"] = len(segments)

    malformed: List[str] = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            malformed.append(seg)
            continue

        if ":" not in seg:
            malformed.append(seg)
            continue

        etype, span = seg.split(":", 1)
        etype = etype.strip()
        span = span.strip()

        if not etype or not span:
            malformed.append(seg)
            continue

        # Valid segment -- check that the type looks reasonable (no special
        # characters other than spaces, hyphens, underscores)
        if not re.match(r"^[a-zA-Z0-9\s\-_]+$", etype):
            malformed.append(seg)
            continue

        result["num_valid_segments"] += 1  # type: ignore[operator]

    result["num_malformed_segments"] = len(malformed)
    result["malformed_segments"] = malformed
    result["is_compliant"] = len(malformed) == 0 and result["num_valid_segments"] > 0  # type: ignore[operator]

    return result
