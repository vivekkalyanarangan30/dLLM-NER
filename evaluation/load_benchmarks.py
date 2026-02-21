"""Load and parse NER evaluation benchmarks for DiffusionNER-Zero.

Supports:
    - **CrossNER** (5 domains): AI, Literature, Music, Politics, Science.
      Source: ``github.com/zliucr/CrossNER``
    - **MIT Movie**: Movie-related NER.
    - **MIT Restaurant**: Restaurant-related NER.
      Both MIT datasets sourced from ``github.com/universal-ner/universal-ner``
      under ``src/eval/test_data/``.

All benchmarks use CoNLL-style BIO tagging (one token + tag per line, blank
lines separating sentences).  We convert them to a unified representation::

    {
        "text": str,           # reconstructed sentence
        "entities": [          # entity spans
            {"type": str, "text": str, "start": int},  # start = char offset
            ...
        ],
        "entity_types": [str, ...]  # unique entity types in the domain
    }
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity type catalogues per domain
# ---------------------------------------------------------------------------

CROSSNER_ENTITY_TYPES: Dict[str, List[str]] = {
    "ai": [
        "algorithm", "field", "task", "product", "university",
        "researcher", "conference", "country", "person",
        "organisation", "location", "programming_language",
        "metrics", "misc",
    ],
    "literature": [
        "book", "writer", "award", "poem", "magazine",
        "literary_genre", "person", "location", "organisation",
        "country", "event", "misc",
    ],
    "music": [
        "music_genre", "song", "band", "album", "musical_artist",
        "musical_instrument", "award", "event", "country",
        "location", "organisation", "person", "misc",
    ],
    "politics": [
        "politician", "political_party", "election", "person",
        "organisation", "location", "country", "event", "misc",
    ],
    "science": [
        "scientist", "discipline", "enzyme", "protein",
        "chemical_element", "chemical_compound", "astronomical_object",
        "academic_journal", "university", "country", "person",
        "organisation", "location", "event", "theory", "misc",
    ],
}

MIT_MOVIE_ENTITY_TYPES: List[str] = [
    "actor", "character", "director", "genre", "song",
    "title", "year", "rating", "ratings_average",
    "review", "plot", "trailer",
]

MIT_RESTAURANT_ENTITY_TYPES: List[str] = [
    "rating", "amenity", "location", "restaurant_name",
    "price", "hours", "dish", "cuisine",
]

# ---------------------------------------------------------------------------
# Repository URLs
# ---------------------------------------------------------------------------

CROSSNER_REPO_URL = "https://github.com/zliucr/CrossNER.git"
UNIVERSAL_NER_REPO_URL = "https://github.com/universal-ner/universal-ner.git"

# CrossNER domain names as they appear in the repo directory structure
CROSSNER_DOMAIN_DIRS: Dict[str, str] = {
    "ai": "ai",
    "literature": "literature",
    "music": "music",
    "politics": "politics",
    "science": "science",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _clone_repo_if_needed(repo_url: str, target_dir: str) -> None:
    """Clone a git repository if the target directory does not exist.

    Parameters
    ----------
    repo_url : str
        HTTPS URL of the git repository.
    target_dir : str
        Local directory path to clone into.
    """
    target_path = Path(target_dir)
    if target_path.exists() and any(target_path.iterdir()):
        logger.info("Repository already present at %s, skipping clone.", target_dir)
        return

    target_path.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s ...", repo_url, target_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Clone complete.")
    except subprocess.CalledProcessError as exc:
        logger.error("git clone failed: %s", exc.stderr)
        raise


def download_crossner(data_dir: str = "data/benchmarks/crossner") -> str:
    """Ensure the CrossNER repository is available locally.

    Parameters
    ----------
    data_dir : str
        Directory where the CrossNER repo will be cloned.

    Returns
    -------
    str
        Path to the cloned repository root.
    """
    _clone_repo_if_needed(CROSSNER_REPO_URL, data_dir)
    return data_dir


def download_universal_ner(data_dir: str = "data/benchmarks/universal-ner") -> str:
    """Ensure the Universal-NER repository is available locally.

    Parameters
    ----------
    data_dir : str
        Directory where the Universal-NER repo will be cloned.

    Returns
    -------
    str
        Path to the cloned repository root.
    """
    _clone_repo_if_needed(UNIVERSAL_NER_REPO_URL, data_dir)
    return data_dir


# ---------------------------------------------------------------------------
# BIO tag parsing
# ---------------------------------------------------------------------------

def bio_to_entities(tokens: List[str], tags: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert parallel lists of tokens and BIO tags into entity spans.

    The reconstructed text joins tokens with single spaces.  Entity character
    offsets (``start``) are computed relative to this reconstructed text.

    Parameters
    ----------
    tokens : List[str]
        Word-level tokens from the CoNLL file.
    tags : List[str]
        Corresponding BIO tags (e.g. ``O``, ``B-person``, ``I-person``).

    Returns
    -------
    Tuple[str, List[Dict[str, Any]]]
        ``(text, entities)`` where *text* is the reconstructed sentence and
        *entities* is a list of dicts with ``"type"``, ``"text"``, ``"start"``
        keys.
    """
    if len(tokens) != len(tags):
        raise ValueError(
            f"Token/tag length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
        )

    text = " ".join(tokens)
    entities: List[Dict[str, Any]] = []

    # Track character offsets for each token
    char_offsets: List[int] = []
    offset = 0
    for token in tokens:
        char_offsets.append(offset)
        offset += len(token) + 1  # +1 for the space separator

    current_entity_tokens: List[str] = []
    current_entity_type: Optional[str] = None
    current_entity_start: int = 0

    def _flush_entity() -> None:
        """Write the current entity (if any) to the entities list."""
        if current_entity_tokens and current_entity_type:
            entity_text = " ".join(current_entity_tokens)
            entities.append({
                "type": current_entity_type,
                "text": entity_text,
                "start": current_entity_start,
            })

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            # Flush any previous entity
            _flush_entity()
            current_entity_type = tag[2:].lower().replace(" ", "_")
            current_entity_tokens = [token]
            current_entity_start = char_offsets[i]
        elif tag.startswith("I-"):
            tag_type = tag[2:].lower().replace(" ", "_")
            if current_entity_type == tag_type and current_entity_tokens:
                # Continue the current entity
                current_entity_tokens.append(token)
            else:
                # Orphaned I- tag (no matching B-): treat as B-
                _flush_entity()
                current_entity_type = tag_type
                current_entity_tokens = [token]
                current_entity_start = char_offsets[i]
        else:
            # O tag or anything else
            _flush_entity()
            current_entity_tokens = []
            current_entity_type = None

    # Flush final entity if sentence ends mid-entity
    _flush_entity()

    return text, entities


def parse_bio_file(filepath: str) -> List[Dict[str, Any]]:
    """Parse a CoNLL-style BIO-tagged file into a list of examples.

    Each sentence (separated by blank lines) becomes one example dict with
    ``"text"`` (str) and ``"entities"`` (list) keys.

    Parameters
    ----------
    filepath : str
        Path to the BIO-tagged file.

    Returns
    -------
    List[Dict[str, Any]]
        Parsed examples, one per sentence.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"BIO file not found: {filepath}")

    examples: List[Dict[str, Any]] = []
    current_tokens: List[str] = []
    current_tags: List[str] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip() == "":
                # End of sentence
                if current_tokens:
                    text, entities = bio_to_entities(current_tokens, current_tags)
                    examples.append({"text": text, "entities": entities})
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[-1]  # Tag is typically the last column
                    current_tokens.append(token)
                    current_tags.append(tag)
                elif len(parts) == 1:
                    # Some formats have only the token when tag is O
                    current_tokens.append(parts[0])
                    current_tags.append("O")

    # Handle file that does not end with a blank line
    if current_tokens:
        text, entities = bio_to_entities(current_tokens, current_tags)
        examples.append({"text": text, "entities": entities})

    logger.info("Parsed %d examples from %s.", len(examples), filepath)
    return examples


# ---------------------------------------------------------------------------
# CrossNER loading
# ---------------------------------------------------------------------------

def _find_crossner_test_file(crossner_dir: str, domain: str) -> str:
    """Locate the test BIO file for a CrossNER domain.

    The CrossNER repo has the structure::

        CrossNER/ner_data/<domain>/test.txt

    Parameters
    ----------
    crossner_dir : str
        Root of the cloned CrossNER repository.
    domain : str
        One of: ai, literature, music, politics, science.

    Returns
    -------
    str
        Absolute path to the test file.

    Raises
    ------
    FileNotFoundError
        If the test file cannot be located.
    """
    domain_key = CROSSNER_DOMAIN_DIRS.get(domain.lower())
    if domain_key is None:
        raise ValueError(
            f"Unknown CrossNER domain: {domain!r}. "
            f"Choose from {list(CROSSNER_DOMAIN_DIRS.keys())}."
        )

    # Try standard paths
    candidates = [
        Path(crossner_dir) / "ner_data" / domain_key / "test.txt",
        Path(crossner_dir) / domain_key / "test.txt",
        Path(crossner_dir) / "data" / domain_key / "test.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fallback: search recursively
    for path in Path(crossner_dir).rglob("test.txt"):
        if domain_key in str(path).lower():
            return str(path)

    raise FileNotFoundError(
        f"Could not find CrossNER test file for domain '{domain}' "
        f"under {crossner_dir}. Searched: {[str(c) for c in candidates]}"
    )


def load_crossner(
    domain: str,
    data_dir: str = "data/benchmarks/crossner",
) -> List[Dict[str, Any]]:
    """Load the CrossNER test set for a specific domain.

    Downloads the CrossNER repository if not already present, parses the
    BIO-tagged test file, and attaches domain entity types to each example.

    Parameters
    ----------
    domain : str
        One of: ``ai``, ``literature``, ``music``, ``politics``, ``science``.
    data_dir : str
        Directory where the CrossNER repo is (or will be) cloned.

    Returns
    -------
    List[Dict[str, Any]]
        Examples with ``"text"``, ``"entities"``, and ``"entity_types"`` keys.
    """
    domain = domain.lower()
    download_crossner(data_dir)

    test_file = _find_crossner_test_file(data_dir, domain)
    examples = parse_bio_file(test_file)

    entity_types = CROSSNER_ENTITY_TYPES.get(domain, [])
    for example in examples:
        example["entity_types"] = entity_types

    logger.info(
        "Loaded CrossNER/%s: %d examples, %d entity types.",
        domain, len(examples), len(entity_types),
    )
    return examples


# ---------------------------------------------------------------------------
# MIT Movie / MIT Restaurant loading
# ---------------------------------------------------------------------------

def _find_mit_test_file(universal_ner_dir: str, dataset_name: str) -> str:
    """Locate a MIT test file inside the Universal-NER repo.

    Expected path::

        universal-ner/src/eval/test_data/<dataset_name>/test.txt

    Also tries common variations.

    Parameters
    ----------
    universal_ner_dir : str
        Root of the cloned Universal-NER repository.
    dataset_name : str
        Either ``"mit-movie"`` or ``"mit-restaurant"``.

    Returns
    -------
    str
        Absolute path to the test file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be located.
    """
    # Normalize variations in naming
    name_variants = [
        dataset_name,
        dataset_name.replace("-", "_"),
        dataset_name.replace("-", ""),
    ]

    for name in name_variants:
        candidates = [
            Path(universal_ner_dir) / "src" / "eval" / "test_data" / name / "test.txt",
            Path(universal_ner_dir) / "src" / "eval" / "test_data" / name / "test",
            Path(universal_ner_dir) / "eval" / "test_data" / name / "test.txt",
            Path(universal_ner_dir) / "test_data" / name / "test.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    # Fallback: search recursively
    search_root = Path(universal_ner_dir)
    for path in search_root.rglob("test.txt"):
        path_str = str(path).lower()
        for name in name_variants:
            if name in path_str:
                return str(path)

    # Also try without extension
    for path in search_root.rglob("test"):
        if path.is_file():
            path_str = str(path).lower()
            for name in name_variants:
                if name in path_str:
                    return str(path)

    raise FileNotFoundError(
        f"Could not find {dataset_name} test file under {universal_ner_dir}."
    )


def load_mit_movie(
    data_dir: str = "data/benchmarks/universal-ner",
) -> List[Dict[str, Any]]:
    """Load the MIT Movie test set.

    Downloads the Universal-NER repository if not already present and parses
    the BIO-tagged test file.

    Parameters
    ----------
    data_dir : str
        Directory where the Universal-NER repo is (or will be) cloned.

    Returns
    -------
    List[Dict[str, Any]]
        Examples with ``"text"``, ``"entities"``, and ``"entity_types"`` keys.
    """
    download_universal_ner(data_dir)

    test_file = _find_mit_test_file(data_dir, "mit-movie")
    examples = parse_bio_file(test_file)

    for example in examples:
        example["entity_types"] = MIT_MOVIE_ENTITY_TYPES

    logger.info("Loaded MIT Movie: %d examples.", len(examples))
    return examples


def load_mit_restaurant(
    data_dir: str = "data/benchmarks/universal-ner",
) -> List[Dict[str, Any]]:
    """Load the MIT Restaurant test set.

    Downloads the Universal-NER repository if not already present and parses
    the BIO-tagged test file.

    Parameters
    ----------
    data_dir : str
        Directory where the Universal-NER repo is (or will be) cloned.

    Returns
    -------
    List[Dict[str, Any]]
        Examples with ``"text"``, ``"entities"``, and ``"entity_types"`` keys.
    """
    download_universal_ner(data_dir)

    test_file = _find_mit_test_file(data_dir, "mit-restaurant")
    examples = parse_bio_file(test_file)

    for example in examples:
        example["entity_types"] = MIT_RESTAURANT_ENTITY_TYPES

    logger.info("Loaded MIT Restaurant: %d examples.", len(examples))
    return examples


# ---------------------------------------------------------------------------
# Load all benchmarks
# ---------------------------------------------------------------------------

def load_all_benchmarks(
    data_dir: str = "data/benchmarks",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load all 7 evaluation benchmarks.

    Downloads repositories as needed.

    Parameters
    ----------
    data_dir : str
        Root directory under which benchmark repos are cloned.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Mapping from benchmark name to a list of examples.  Keys are:
        ``crossner_ai``, ``crossner_literature``, ``crossner_music``,
        ``crossner_politics``, ``crossner_science``, ``mit_movie``,
        ``mit_restaurant``.
    """
    crossner_dir = os.path.join(data_dir, "crossner")
    uniner_dir = os.path.join(data_dir, "universal-ner")

    benchmarks: Dict[str, List[Dict[str, Any]]] = {}

    # CrossNER domains
    for domain in CROSSNER_DOMAIN_DIRS:
        key = f"crossner_{domain}"
        try:
            benchmarks[key] = load_crossner(domain, data_dir=crossner_dir)
            logger.info("Loaded %s: %d examples.", key, len(benchmarks[key]))
        except Exception:
            logger.exception("Failed to load %s.", key)
            benchmarks[key] = []

    # MIT Movie
    try:
        benchmarks["mit_movie"] = load_mit_movie(data_dir=uniner_dir)
    except Exception:
        logger.exception("Failed to load MIT Movie.")
        benchmarks["mit_movie"] = []

    # MIT Restaurant
    try:
        benchmarks["mit_restaurant"] = load_mit_restaurant(data_dir=uniner_dir)
    except Exception:
        logger.exception("Failed to load MIT Restaurant.")
        benchmarks["mit_restaurant"] = []

    total = sum(len(v) for v in benchmarks.values())
    logger.info(
        "Loaded %d benchmarks with %d total examples.",
        len(benchmarks), total,
    )
    return benchmarks


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
        description="Download and inspect NER evaluation benchmarks."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/benchmarks",
        help="Root directory for benchmark data.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Load a specific benchmark (e.g. crossner_ai, mit_movie). "
             "If omitted, loads all.",
    )
    args = parser.parse_args()

    if args.benchmark:
        if args.benchmark.startswith("crossner_"):
            domain = args.benchmark.replace("crossner_", "")
            examples = load_crossner(
                domain,
                data_dir=os.path.join(args.data_dir, "crossner"),
            )
        elif args.benchmark == "mit_movie":
            examples = load_mit_movie(
                data_dir=os.path.join(args.data_dir, "universal-ner"),
            )
        elif args.benchmark == "mit_restaurant":
            examples = load_mit_restaurant(
                data_dir=os.path.join(args.data_dir, "universal-ner"),
            )
        else:
            raise ValueError(f"Unknown benchmark: {args.benchmark}")

        print(f"\n{args.benchmark}: {len(examples)} examples")
        if examples:
            ex = examples[0]
            print(f"  Text: {ex['text'][:120]}...")
            print(f"  Entities ({len(ex['entities'])}): {ex['entities'][:3]}")
            print(f"  Types: {ex['entity_types']}")
    else:
        all_benchmarks = load_all_benchmarks(args.data_dir)
        print("\nBenchmark summary:")
        for name, examples in all_benchmarks.items():
            n_ents = sum(len(ex["entities"]) for ex in examples)
            print(f"  {name:25s}  {len(examples):5d} examples, {n_ents:6d} entities")
