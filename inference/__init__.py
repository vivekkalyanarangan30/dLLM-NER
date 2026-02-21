"""DiffusionNER-Zero inference modules.

Provides entity extraction via iterative denoising with Dream-7B,
output parsing, and ReMDM re-masking logic.
"""

from .parse import (
    deduplicate_entities,
    filter_entities_by_source,
    format_compliance_check,
    parse_entities,
)
from .predict import (
    MASK_TOKEN_ID,
    apply_pad_penalty,
    compute_unmask_count,
    extract_entities,
    extract_entities_with_trajectory,
)
from .remask import compute_remask_count, remask_low_confidence

__all__ = [
    # parse
    "parse_entities",
    "deduplicate_entities",
    "filter_entities_by_source",
    "format_compliance_check",
    # predict
    "MASK_TOKEN_ID",
    "apply_pad_penalty",
    "compute_unmask_count",
    "extract_entities",
    "extract_entities_with_trajectory",
    # remask
    "remask_low_confidence",
    "compute_remask_count",
]
