"""UniNER-7B-type baseline inference for comparison with DiffusionNER-Zero.

UniNER uses a multi-turn autoregressive format: one query per entity type.
Each query follows the template::

    A virtual assistant that extracts named entities from text.
    [INST] Text: {text}
    Use the provided text to identify entities that belong to the
    following category: {entity_type} [/INST]

The model responds with a JSON array of entity strings.  We aggregate
predictions across all entity types and deduplicate.

This module supports both HuggingFace Transformers inference and optional
vLLM acceleration for higher throughput.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UniNER prompt template
# ---------------------------------------------------------------------------

UNINER_SYSTEM_MSG = "A virtual assistant that extracts named entities from text."

UNINER_PROMPT_TEMPLATE = (
    "{system_msg}\n"
    "[INST] Text: {text}\n"
    "Use the provided text to identify entities that belong to the "
    "following category: {entity_type} [/INST]"
)


def _format_uniner_prompt(text: str, entity_type: str) -> str:
    """Format a single UniNER query prompt.

    Parameters
    ----------
    text : str
        Source passage text.
    entity_type : str
        Entity type to query for.

    Returns
    -------
    str
        Formatted prompt string.
    """
    return UNINER_PROMPT_TEMPLATE.format(
        system_msg=UNINER_SYSTEM_MSG,
        text=text,
        entity_type=entity_type,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_uniner_response(response: str) -> List[str]:
    """Parse a UniNER response string into a list of entity mentions.

    The model typically outputs a JSON array like ``["entity1", "entity2"]``.
    We handle common edge cases: extra whitespace, non-JSON responses, and
    markdown code blocks.

    Parameters
    ----------
    response : str
        Raw model output text.

    Returns
    -------
    List[str]
        Parsed entity mention strings.  Returns empty list on parse failure.
    """
    response = response.strip()

    # Remove markdown code block if present
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove first and last lines (``` markers)
        inner_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                break
            elif in_block:
                inner_lines.append(line)
        if inner_lines:
            response = "\n".join(inner_lines).strip()

    # Try direct JSON parse
    try:
        entities = json.loads(response)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if str(e).strip()]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the response
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            entities = json.loads(match.group(0))
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()]
        except json.JSONDecodeError:
            pass

    # Handle "none" or empty responses
    if response.lower() in ("none", "[]", "n/a", "null", ""):
        return []

    # Last resort: split on commas (some models produce comma-separated lists)
    if "," in response:
        candidates = [s.strip().strip('"').strip("'") for s in response.split(",")]
        return [c for c in candidates if c]

    # Single entity?
    if response and response.lower() not in ("none", "n/a", "null"):
        return [response]

    return []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_uniner(
    model_name: str = "Universal-NER/UniNER-7B-type",
    use_vllm: bool = True,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.85,
) -> Tuple[Any, Any]:
    """Load a UniNER model for inference.

    Attempts to use vLLM for faster inference.  Falls back to HuggingFace
    Transformers if vLLM is unavailable or ``use_vllm=False``.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    use_vllm : bool
        Whether to try loading with vLLM.
    dtype : str
        Data type for model weights (e.g. ``"bfloat16"``, ``"float16"``).
    gpu_memory_utilization : float
        GPU memory fraction for vLLM (only used when ``use_vllm=True``).

    Returns
    -------
    Tuple[Any, Any]
        ``(model, tokenizer)`` tuple.  When using vLLM, the "tokenizer"
        is the vLLM tokenizer embedded in the engine.
    """
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams

            logger.info("Loading UniNER via vLLM: %s", model_name)
            model = LLM(
                model=model_name,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
            tokenizer = model.get_tokenizer()
            logger.info("UniNER loaded via vLLM successfully.")
            return model, tokenizer

        except ImportError:
            logger.warning("vLLM not available. Falling back to HuggingFace Transformers.")
        except Exception:
            logger.exception("vLLM loading failed. Falling back to HuggingFace Transformers.")

    # HuggingFace Transformers fallback
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading UniNER via HuggingFace Transformers: %s", model_name)

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info("UniNER loaded via Transformers successfully.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _generate_vllm(
    model: Any,
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> List[str]:
    """Generate responses using a vLLM engine.

    Parameters
    ----------
    model :
        vLLM ``LLM`` instance.
    prompts : List[str]
        Input prompts.
    max_tokens : int
        Maximum output tokens.
    temperature : float
        Sampling temperature (0.0 = greedy).

    Returns
    -------
    List[str]
        Generated text outputs.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )

    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def _generate_hf(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Generate a response using HuggingFace Transformers.

    Parameters
    ----------
    model :
        HuggingFace model.
    tokenizer :
        HuggingFace tokenizer.
    prompt : str
        Input prompt.
    max_new_tokens : int
        Maximum output tokens.
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        Generated text output.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # Decode only the new tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _is_vllm_model(model: Any) -> bool:
    """Check if the model is a vLLM LLM instance."""
    try:
        from vllm import LLM
        return isinstance(model, LLM)
    except ImportError:
        return False


def extract_entities_uniner(
    model: Any,
    tokenizer: Any,
    text: str,
    entity_types: List[str],
    max_tokens: int = 256,
    batch_vllm: bool = True,
) -> List[Dict[str, str]]:
    """Extract entities using UniNER (one query per entity type).

    Parameters
    ----------
    model :
        UniNER model (vLLM or HuggingFace).
    tokenizer :
        Associated tokenizer.
    text : str
        Source passage text.
    entity_types : List[str]
        Entity types to query for.
    max_tokens : int
        Maximum response tokens per query.
    batch_vllm : bool
        If True and model is vLLM, batch all type queries in a single
        ``generate`` call for efficiency.

    Returns
    -------
    List[Dict[str, str]]
        Entity predictions with ``"type"`` and ``"text"`` keys.
    """
    if not entity_types:
        return []

    is_vllm = _is_vllm_model(model)

    if is_vllm and batch_vllm:
        # Batch all type queries for vLLM
        prompts = [_format_uniner_prompt(text, etype) for etype in entity_types]
        responses = _generate_vllm(model, prompts, max_tokens=max_tokens)
    else:
        responses = []
        for etype in entity_types:
            prompt = _format_uniner_prompt(text, etype)
            if is_vllm:
                resp = _generate_vllm(model, [prompt], max_tokens=max_tokens)[0]
            else:
                resp = _generate_hf(model, tokenizer, prompt, max_new_tokens=max_tokens)
            responses.append(resp)

    # Parse and aggregate entities
    all_entities: List[Dict[str, str]] = []
    seen: set = set()

    for etype, response in zip(entity_types, responses):
        mentions = _parse_uniner_response(response)
        for mention in mentions:
            key = (etype.lower(), mention.lower())
            if key not in seen:
                seen.add(key)
                all_entities.append({
                    "type": etype.lower(),
                    "text": mention,
                })

    return all_entities


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def run_uniner_on_benchmark(
    model: Any,
    tokenizer: Any,
    benchmark: List[Dict[str, Any]],
    entity_types: Optional[List[str]] = None,
    show_progress: bool = True,
) -> List[List[Dict[str, str]]]:
    """Run UniNER on a full benchmark dataset.

    Parameters
    ----------
    model :
        UniNER model.
    tokenizer :
        UniNER tokenizer.
    benchmark : List[Dict[str, Any]]
        Evaluation examples with ``"text"`` and optionally ``"entity_types"``.
    entity_types : List[str], optional
        Entity types to query.  If ``None``, uses per-example types.
    show_progress : bool
        Whether to display a progress bar.

    Returns
    -------
    List[List[Dict[str, str]]]
        Predictions for each example.
    """
    all_predictions: List[List[Dict[str, str]]] = []

    iterator = tqdm(benchmark, desc="UniNER inference", disable=not show_progress)

    for example in iterator:
        text = example["text"]
        types = entity_types if entity_types else example.get("entity_types", [])

        try:
            preds = extract_entities_uniner(model, tokenizer, text, types)
        except Exception:
            logger.exception("UniNER extraction failed for: %s...", text[:80])
            preds = []

        all_predictions.append(preds)

    logger.info(
        "UniNER inference complete: %d examples, %d total entities predicted.",
        len(all_predictions),
        sum(len(p) for p in all_predictions),
    )

    return all_predictions


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
        description="Run UniNER-7B-type baseline inference."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Universal-NER/UniNER-7B-type",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM, use HuggingFace Transformers.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Ronaldo plays for Al Nassr since January 2023.",
        help="Text to extract entities from.",
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        nargs="+",
        default=["person", "organization", "date"],
        help="Entity types to query.",
    )
    args = parser.parse_args()

    print(f"Loading UniNER: {args.model_name} (vLLM={not args.no_vllm})")
    model, tokenizer = load_uniner(
        model_name=args.model_name,
        use_vllm=not args.no_vllm,
    )

    print(f"\nText: {args.text}")
    print(f"Types: {args.entity_types}")
    entities = extract_entities_uniner(model, tokenizer, args.text, args.entity_types)
    print(f"Extracted entities: {entities}")
