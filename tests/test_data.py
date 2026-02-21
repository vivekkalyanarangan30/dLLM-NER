"""Comprehensive unit tests for DiffusionNER-Zero data pipeline modules.

Tests cover:
  - data.negative_sampling (build_type_pool, sample_negative_types)
  - data.prepare_pile_ner (format_for_diffusion, parsing helpers, group_by_passage)
  - data.dataset (DreamNERDataset, random_truncation_collate_fn)
  - data.analyze_lengths (analyze_completion_lengths, percentiles, coverage)

All HuggingFace model/tokenizer/dataset loading is mocked so tests run on
CPU-only machines without downloading any large artefacts.
"""

import json
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Mock tokenizer used across dataset and analyze_lengths tests
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal mock tokenizer that splits on whitespace and maps words to ints.

    Every unique word gets a deterministic integer id (via a growing vocab dict).
    ``encode`` returns a list[int]; ``decode`` joins ids back to words.
    """

    def __init__(
        self,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._vocab: Dict[str, int] = {}
        self._next_id: int = 10  # reserve low ids for special tokens

    def _get_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._next_id += 1
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = text.split()
        return [self._get_id(w) for w in tokens]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, f"<unk:{i}>") for i in ids)


# =========================================================================
# Tests for data.negative_sampling
# =========================================================================


class TestBuildTypePool:
    """Tests for negative_sampling.build_type_pool."""

    def test_basic_list_of_dicts(self):
        """build_type_pool should extract unique sorted types from a list of dicts."""
        from data.negative_sampling import build_type_pool

        dataset = [
            {"entities": [{"type": "person"}, {"type": "org"}]},
            {"entities": [{"type": "org"}, {"type": "location"}]},
        ]
        pool = build_type_pool(dataset)
        assert pool == ["location", "org", "person"]

    def test_empty_dataset(self):
        """build_type_pool with empty dataset returns empty list."""
        from data.negative_sampling import build_type_pool

        assert build_type_pool([]) == []

    def test_no_entities_field(self):
        """Rows without 'entities' key should be silently skipped."""
        from data.negative_sampling import build_type_pool

        dataset = [{"text": "hello"}, {"entities": [{"type": "date"}]}]
        assert build_type_pool(dataset) == ["date"]

    def test_empty_type_string_skipped(self):
        """Entities with empty-string type should be skipped."""
        from data.negative_sampling import build_type_pool

        dataset = [{"entities": [{"type": ""}, {"type": "person"}]}]
        assert build_type_pool(dataset) == ["person"]

    def test_none_type_skipped(self):
        """Entities with None type (missing key) should be skipped."""
        from data.negative_sampling import build_type_pool

        dataset = [{"entities": [{"name": "foo"}, {"type": "org"}]}]
        assert build_type_pool(dataset) == ["org"]

    def test_duplicates_collapsed(self):
        """Duplicate types across rows should appear only once."""
        from data.negative_sampling import build_type_pool

        dataset = [
            {"entities": [{"type": "person"}]},
            {"entities": [{"type": "person"}]},
            {"entities": [{"type": "person"}]},
        ]
        assert build_type_pool(dataset) == ["person"]


class TestSampleNegativeTypes:
    """Tests for negative_sampling.sample_negative_types."""

    def test_returns_correct_count_range(self):
        """Returned negatives count should be within [min_neg, max_neg]."""
        from data.negative_sampling import sample_negative_types

        type_pool = ["a", "b", "c", "d", "e", "f", "g", "h"]
        gt_types = ["a"]

        random.seed(123)
        for _ in range(50):
            negs = sample_negative_types(gt_types, type_pool, min_neg=2, max_neg=5)
            assert 2 <= len(negs) <= 5

    def test_never_returns_ground_truth_types(self):
        """Sampled negatives must never include any ground-truth type."""
        from data.negative_sampling import sample_negative_types

        type_pool = ["a", "b", "c", "d", "e"]
        gt_types = ["a", "b"]

        random.seed(42)
        for _ in range(100):
            negs = sample_negative_types(gt_types, type_pool, min_neg=1, max_neg=3)
            for neg in negs:
                assert neg not in gt_types

    def test_fewer_candidates_than_requested(self):
        """When pool has fewer candidates than min_neg, return all candidates."""
        from data.negative_sampling import sample_negative_types

        type_pool = ["a", "b", "c"]
        gt_types = ["a", "b"]  # only "c" is a candidate

        negs = sample_negative_types(gt_types, type_pool, min_neg=5, max_neg=10)
        assert negs == ["c"]

    def test_empty_gt_types(self):
        """With empty gt_types, all pool items are candidates."""
        from data.negative_sampling import sample_negative_types

        type_pool = ["a", "b", "c"]
        random.seed(0)
        negs = sample_negative_types([], type_pool, min_neg=2, max_neg=3)
        assert 2 <= len(negs) <= 3
        for neg in negs:
            assert neg in type_pool

    def test_all_types_are_gt(self):
        """When every pool type is in gt_types, return empty list."""
        from data.negative_sampling import sample_negative_types

        type_pool = ["a", "b"]
        gt_types = ["a", "b"]

        negs = sample_negative_types(gt_types, type_pool, min_neg=2, max_neg=5)
        assert negs == []

    def test_empty_pool(self):
        """Empty type pool should return empty list."""
        from data.negative_sampling import sample_negative_types

        negs = sample_negative_types(["a"], [], min_neg=2, max_neg=5)
        assert negs == []


class TestSampleNegativeTypesDeterministic:
    """Tests for negative_sampling.sample_negative_types_deterministic."""

    def test_returns_exact_count(self):
        """Deterministic variant should return exactly n_neg items."""
        from data.negative_sampling import sample_negative_types_deterministic

        type_pool = ["a", "b", "c", "d", "e", "f"]
        gt_types = ["a"]

        negs = sample_negative_types_deterministic(gt_types, type_pool, n_neg=3)
        assert len(negs) == 3

    def test_fallback_fewer_candidates(self):
        """Returns fewer if candidates < n_neg."""
        from data.negative_sampling import sample_negative_types_deterministic

        type_pool = ["a", "b", "c"]
        gt_types = ["a", "b"]

        negs = sample_negative_types_deterministic(gt_types, type_pool, n_neg=5)
        assert len(negs) == 1
        assert negs == ["c"]


# =========================================================================
# Tests for data.prepare_pile_ner
# =========================================================================


class TestExtractPassageAndType:
    """Tests for prepare_pile_ner._extract_passage_and_type."""

    def test_standard_format(self):
        """Parse a well-formed human message with Text: and category:."""
        from data.prepare_pile_ner import _extract_passage_and_type

        human_msg = (
            "Text: The quick brown fox jumped over the lazy dog.\n"
            "Use the provided text to identify entities that belong to the following category: animal"
        )
        passage, etype = _extract_passage_and_type(human_msg)
        assert passage == "The quick brown fox jumped over the lazy dog."
        assert etype == "animal"

    def test_please_variant(self):
        """Parse human message using 'Please' instead of 'Use the provided text'."""
        from data.prepare_pile_ner import _extract_passage_and_type

        human_msg = (
            "Text: Paris is the capital of France.\n"
            "Please identify entities of category: location"
        )
        passage, etype = _extract_passage_and_type(human_msg)
        assert passage == "Paris is the capital of France."
        assert etype == "location"

    def test_parse_failure_no_text(self):
        """Return (None, None) when 'Text:' is missing."""
        from data.prepare_pile_ner import _extract_passage_and_type

        passage, etype = _extract_passage_and_type("Some random message without markers")
        assert passage is None
        assert etype is None

    def test_type_with_trailing_period_stripped(self):
        """Trailing period on the entity type should be stripped."""
        from data.prepare_pile_ner import _extract_passage_and_type

        human_msg = (
            "Text: Test passage.\n"
            "Use the provided text to identify entities that belong to the following category: person."
        )
        passage, etype = _extract_passage_and_type(human_msg)
        assert etype == "person"

    def test_no_category_keyword(self):
        """If neither 'category:' nor 'type:' is found, entity_type is None."""
        from data.prepare_pile_ner import _extract_passage_and_type

        human_msg = (
            "Text: Some passage.\n"
            "Use the provided text to identify entities."
        )
        passage, etype = _extract_passage_and_type(human_msg)
        assert passage == "Some passage."
        assert etype is None

    def test_fallback_type_keyword(self):
        """Fall back to 'type:' when 'category:' is not present."""
        from data.prepare_pile_ner import _extract_passage_and_type

        human_msg = (
            "Text: A sentence.\n"
            "Use the provided text to find type: organization"
        )
        passage, etype = _extract_passage_and_type(human_msg)
        assert etype == "organization"


class TestParseEntityList:
    """Tests for prepare_pile_ner._parse_entity_list."""

    def test_valid_json_array(self):
        from data.prepare_pile_ner import _parse_entity_list

        result = _parse_entity_list('["Alice", "Bob"]')
        assert result == ["Alice", "Bob"]

    def test_empty_json_array(self):
        from data.prepare_pile_ner import _parse_entity_list

        result = _parse_entity_list("[]")
        assert result == []

    def test_whitespace_stripped(self):
        from data.prepare_pile_ner import _parse_entity_list

        result = _parse_entity_list('  ["  Alice  ", "Bob"]  ')
        assert result == ["Alice", "Bob"]

    def test_invalid_json_with_embedded_array(self):
        """Salvage an array embedded inside non-JSON text."""
        from data.prepare_pile_ner import _parse_entity_list

        result = _parse_entity_list('Here are the entities: ["foo", "bar"] end')
        assert result == ["foo", "bar"]

    def test_completely_invalid(self):
        """Return empty list for completely unparseable input."""
        from data.prepare_pile_ner import _parse_entity_list

        result = _parse_entity_list("not json at all")
        assert result == []

    def test_empty_strings_filtered(self):
        """Empty strings in the array should be filtered out."""
        from data.prepare_pile_ner import _parse_entity_list

        result = _parse_entity_list('["Alice", "", "  ", "Bob"]')
        assert result == ["Alice", "Bob"]

    def test_non_list_json(self):
        """If JSON decodes to a non-list (e.g. dict), return empty list."""
        from data.prepare_pile_ner import _parse_entity_list

        result = _parse_entity_list('{"key": "value"}')
        assert result == []


class TestFindEntityStart:
    """Tests for prepare_pile_ner._find_entity_start."""

    def test_basic_offset(self):
        from data.prepare_pile_ner import _find_entity_start

        used = set()
        idx = _find_entity_start("Hello World", "World", used)
        assert idx == 6
        assert 6 in used

    def test_duplicate_mentions(self):
        """Second call for same entity text should find next occurrence."""
        from data.prepare_pile_ner import _find_entity_start

        passage = "cat and cat"
        used = set()
        idx1 = _find_entity_start(passage, "cat", used)
        idx2 = _find_entity_start(passage, "cat", used)
        assert idx1 == 0
        assert idx2 == 8
        assert used == {0, 8}

    def test_case_insensitive_fallback(self):
        """If exact match fails, try case-insensitive."""
        from data.prepare_pile_ner import _find_entity_start

        used = set()
        idx = _find_entity_start("Hello WORLD", "world", used)
        assert idx == 6

    def test_not_found(self):
        """Return -1 when entity text is not in passage at all."""
        from data.prepare_pile_ner import _find_entity_start

        used = set()
        idx = _find_entity_start("Hello World", "foobar", used)
        assert idx == -1

    def test_all_occurrences_used(self):
        """Return -1 when all matching offsets are already used."""
        from data.prepare_pile_ner import _find_entity_start

        passage = "ab ab"
        used = {0, 3}
        idx = _find_entity_start(passage, "ab", used)
        assert idx == -1


class TestGroupByPassage:
    """Tests for prepare_pile_ner.group_by_passage."""

    def test_basic_grouping(self):
        from data.prepare_pile_ner import group_by_passage

        passage_text = "Ronaldo plays for Al Nassr."
        mock_dataset = [
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Text: {passage_text}\n"
                            "Use the provided text to identify entities that belong to the following category: person"
                        ),
                    },
                    {"from": "gpt", "value": '["Ronaldo"]'},
                ]
            },
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Text: {passage_text}\n"
                            "Use the provided text to identify entities that belong to the following category: organization"
                        ),
                    },
                    {"from": "gpt", "value": '["Al Nassr"]'},
                ]
            },
        ]

        groups = group_by_passage(mock_dataset)
        assert passage_text in groups
        assert len(groups[passage_text]) == 2
        types = {g["type"] for g in groups[passage_text]}
        assert types == {"person", "organization"}

    def test_skips_malformed_rows(self):
        """Rows with insufficient conversation turns are skipped."""
        from data.prepare_pile_ner import group_by_passage

        mock_dataset = [
            {"conversations": [{"from": "human", "value": "incomplete"}]},
            {"conversations": []},
        ]
        groups = group_by_passage(mock_dataset)
        assert len(groups) == 0

    def test_skips_missing_human_or_gpt(self):
        """Rows without a human or gpt turn are skipped."""
        from data.prepare_pile_ner import group_by_passage

        mock_dataset = [
            {
                "conversations": [
                    {"from": "system", "value": "hi"},
                    {"from": "system", "value": "there"},
                ]
            },
        ]
        groups = group_by_passage(mock_dataset)
        assert len(groups) == 0


class TestFormatForDiffusion:
    """Tests for prepare_pile_ner.format_for_diffusion."""

    def test_normal_entities(self):
        from data.prepare_pile_ner import format_for_diffusion

        entities = [
            {"type": "person", "text": "Alice", "start": 0},
            {"type": "org", "text": "Acme", "start": 20},
        ]
        pool = ["person", "org", "location", "date", "event", "product", "misc"]
        random.seed(42)
        result = format_for_diffusion("Alice works at Acme Corp.", entities, pool)

        assert "prompt" in result
        assert "completion" in result
        assert "entities" in result
        assert "person" in result["prompt"]
        assert "org" in result["prompt"]
        assert "Alice" in result["completion"]
        assert "Acme" in result["completion"]

    def test_no_entities_returns_none(self):
        """When there are no entities, completion should be ' none'."""
        from data.prepare_pile_ner import format_for_diffusion

        result = format_for_diffusion(
            "Some text.", [], ["person", "org", "location"]
        )
        assert result["completion"] == " none"

    def test_includes_negative_types_in_prompt(self):
        """Prompt should contain more types than just the ground-truth ones."""
        from data.prepare_pile_ner import format_for_diffusion

        entities = [{"type": "person", "text": "Bob", "start": 0}]
        pool = ["person", "org", "location", "date", "event", "product", "misc"]
        random.seed(42)
        result = format_for_diffusion("Bob is here.", entities, pool)

        # The prompt should list multiple types (gt + negatives)
        prompt_types_line = result["prompt"].split("\n")[0]
        # Extract the types from "Extract entities of types: ..."
        listed_types = prompt_types_line.replace("Extract entities of types: ", "").split(", ")
        assert len(listed_types) >= 3  # at least 1 gt + 2 neg (min_neg=2)

    def test_entities_sorted_by_start_offset(self):
        """Entities in the completion must be sorted by start offset."""
        from data.prepare_pile_ner import format_for_diffusion

        entities = [
            {"type": "org", "text": "Corp", "start": 30},
            {"type": "person", "text": "Alice", "start": 0},
            {"type": "date", "text": "2023", "start": 15},
        ]
        pool = ["person", "org", "date"]
        random.seed(99)
        result = format_for_diffusion("Alice said on 2023 that Corp is great.", entities, pool)

        completion = result["completion"]
        # The completion should have Alice before 2023 before Corp
        alice_pos = completion.index("Alice")
        date_pos = completion.index("2023")
        corp_pos = completion.index("Corp")
        assert alice_pos < date_pos < corp_pos

    def test_prompt_structure(self):
        """Validate the structural format of the prompt string."""
        from data.prepare_pile_ner import format_for_diffusion

        entities = [{"type": "person", "text": "Eve", "start": 0}]
        pool = ["person", "org", "location"]
        random.seed(1)
        result = format_for_diffusion("Eve is here.", entities, pool)
        prompt = result["prompt"]

        assert prompt.startswith("Extract entities of types:")
        assert "\nText:" in prompt
        assert prompt.endswith("Entities:")


# =========================================================================
# Tests for data.dataset
# =========================================================================


class TestDreamNERDataset:
    """Tests for dataset.DreamNERDataset."""

    def _make_dataset(
        self,
        data=None,
        max_seq_length=64,
        max_completion_length=16,
        pad_token_id=0,
        eos_token_id=1,
    ):
        from data.dataset import DreamNERDataset

        if data is None:
            data = [
                {"prompt": "Extract person", "completion": " person: Alice"},
                {"prompt": "Extract org date", "completion": " org: Acme | date: Jan"},
            ]
        tok = MockTokenizer(pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        ds = DreamNERDataset(
            data, tok,
            max_seq_length=max_seq_length,
            max_completion_length=max_completion_length,
        )
        return ds, tok

    def test_length(self):
        ds, _ = self._make_dataset()
        assert len(ds) == 2

    def test_completion_padded_to_max_length(self):
        """All completion_ids tensors should have length == max_completion_length."""
        ds, _ = self._make_dataset()
        for i in range(len(ds)):
            item = ds[i]
            assert item["completion_ids"].shape[0] == 16

    def test_completion_truncation(self):
        """Completions longer than max_completion_length should be truncated."""
        long_completion = " " + " ".join([f"word{i}" for i in range(50)])
        data = [{"prompt": "p", "completion": long_completion}]
        ds, tok = self._make_dataset(data=data, max_completion_length=8)

        item = ds[0]
        assert item["completion_ids"].shape[0] == 8
        # Unpadded length should be capped at max_completion_length
        assert item["completion_length"].item() == 8

    def test_prompt_truncation_from_left(self):
        """Prompts exceeding max_prompt_length should be truncated from the left."""
        long_prompt = " ".join([f"token{i}" for i in range(100)])
        data = [{"prompt": long_prompt, "completion": " ok"}]
        ds, tok = self._make_dataset(
            data=data, max_seq_length=32, max_completion_length=8,
        )
        # max_prompt_length = 32 - 8 = 24
        item = ds[0]
        prompt_len = item["prompt_ids"].shape[0]
        assert prompt_len <= 24

        # Should keep the RIGHT side of the prompt (truncate from left)
        full_prompt_ids = tok.encode(long_prompt, add_special_tokens=False)
        expected_kept = full_prompt_ids[-24:]
        assert item["prompt_ids"].tolist() == expected_kept

    def test_pad_token_id_used(self):
        """PAD tokens in completion should use the tokenizer's pad_token_id."""
        data = [{"prompt": "p", "completion": " short"}]
        ds, tok = self._make_dataset(
            data=data, max_completion_length=16, pad_token_id=99,
        )
        item = ds[0]
        comp_ids = item["completion_ids"].tolist()
        # "short" encodes to 1 token (whitespace-split), so 15 pads
        num_short_tokens = len(tok.encode(" short", add_special_tokens=False))
        pad_region = comp_ids[num_short_tokens:]
        assert all(pid == 99 for pid in pad_region)

    def test_fallback_to_eos_token_id(self):
        """When pad_token_id is None, eos_token_id should be used."""
        data = [{"prompt": "p", "completion": " x"}]
        ds, tok = self._make_dataset(
            data=data, pad_token_id=None, eos_token_id=77,
        )
        item = ds[0]
        comp_ids = item["completion_ids"].tolist()
        num_real = len(tok.encode(" x", add_special_tokens=False))
        pad_region = comp_ids[num_real:]
        assert all(pid == 77 for pid in pad_region)

    def test_no_pad_no_eos_raises(self):
        """If both pad_token_id and eos_token_id are None, raise ValueError."""
        from data.dataset import DreamNERDataset

        tok = MockTokenizer(pad_token_id=None, eos_token_id=None)
        with pytest.raises(ValueError, match="neither pad_token_id nor eos_token_id"):
            DreamNERDataset(
                [{"prompt": "a", "completion": "b"}], tok,
                max_seq_length=32, max_completion_length=8,
            )

    def test_attention_mask_shape(self):
        """attention_mask length should equal prompt_len + max_completion_length."""
        ds, _ = self._make_dataset()
        item = ds[0]
        expected_len = item["prompt_ids"].shape[0] + 16
        assert item["attention_mask"].shape[0] == expected_len

    def test_attention_mask_values(self):
        """Attention mask should be True for real tokens, False for pads."""
        data = [{"prompt": "hello world", "completion": " one"}]
        ds, tok = self._make_dataset(data=data, max_completion_length=8)
        item = ds[0]

        prompt_len = item["prompt_ids"].shape[0]
        comp_real_len = item["completion_length"].item()

        mask = item["attention_mask"]
        # Prompt portion: all True
        assert mask[:prompt_len].all()
        # Completion real tokens: True
        assert mask[prompt_len:prompt_len + comp_real_len].all()
        # Completion pad tokens: False
        if comp_real_len < 8:
            assert not mask[prompt_len + comp_real_len:].any()

    def test_completion_lengths_property(self):
        """The completion_lengths property should return unpadded lengths."""
        ds, tok = self._make_dataset()
        lengths = ds.completion_lengths
        assert len(lengths) == 2
        for length in lengths:
            assert isinstance(length, int)
            assert length > 0

    def test_getitem_returns_tensors(self):
        """All values in __getitem__ output should be torch tensors."""
        ds, _ = self._make_dataset()
        item = ds[0]
        assert isinstance(item["prompt_ids"], torch.Tensor)
        assert isinstance(item["completion_ids"], torch.Tensor)
        assert isinstance(item["completion_length"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)

    def test_getitem_dtypes(self):
        """Verify tensor dtypes."""
        ds, _ = self._make_dataset()
        item = ds[0]
        assert item["prompt_ids"].dtype == torch.long
        assert item["completion_ids"].dtype == torch.long
        assert item["completion_length"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.bool


class TestRandomTruncationCollateFn:
    """Tests for dataset.random_truncation_collate_fn."""

    def _make_batch(self, comp_lengths, max_comp_len=16, pad_id=0):
        """Helper: build a batch of fake items for collation."""
        batch = []
        for comp_len in comp_lengths:
            prompt_len = 4  # arbitrary fixed prompt length
            prompt_ids = torch.arange(100, 100 + prompt_len, dtype=torch.long)
            comp_ids = torch.cat([
                torch.arange(200, 200 + comp_len, dtype=torch.long),
                torch.full((max_comp_len - comp_len,), pad_id, dtype=torch.long),
            ])
            prompt_mask = torch.ones(prompt_len, dtype=torch.bool)
            comp_mask = torch.cat([
                torch.ones(comp_len, dtype=torch.bool),
                torch.zeros(max_comp_len - comp_len, dtype=torch.bool),
            ])
            attn_mask = torch.cat([prompt_mask, comp_mask])
            batch.append({
                "prompt_ids": prompt_ids,
                "completion_ids": comp_ids,
                "completion_length": torch.tensor(comp_len, dtype=torch.long),
                "attention_mask": attn_mask,
            })
        return batch

    def test_output_keys(self):
        from data.dataset import random_truncation_collate_fn

        batch = self._make_batch([5, 8, 3])
        result = random_truncation_collate_fn(batch)
        expected_keys = {
            "prompt_ids", "completion_ids", "completion_length",
            "original_completion_length", "attention_mask",
        }
        assert set(result.keys()) == expected_keys

    def test_completion_ids_shape(self):
        from data.dataset import random_truncation_collate_fn

        batch = self._make_batch([5, 8, 3], max_comp_len=16)
        result = random_truncation_collate_fn(batch)
        assert result["completion_ids"].shape == (3, 16)

    def test_truncated_lengths_bounded(self):
        """Truncated lengths should not exceed original lengths."""
        from data.dataset import random_truncation_collate_fn

        random.seed(0)
        batch = self._make_batch([5, 8, 3])
        result = random_truncation_collate_fn(batch)

        orig = result["original_completion_length"]
        trunc = result["completion_length"]
        for i in range(3):
            assert trunc[i].item() <= orig[i].item()

    def test_pads_after_truncation(self):
        """Tokens beyond truncated length should be PAD."""
        from data.dataset import random_truncation_collate_fn

        # Use a batch where truncation will definitely happen
        # One item has length 10, will be truncated to length of another item
        random.seed(1)
        batch = self._make_batch([10, 2], max_comp_len=16, pad_id=0)
        result = random_truncation_collate_fn(batch)

        for i in range(2):
            trunc_len = result["completion_length"][i].item()
            comp = result["completion_ids"][i]
            # Everything from trunc_len onward should be pad (0)
            if trunc_len < 16:
                pad_region = comp[trunc_len:]
                assert (pad_region == 0).all(), (
                    f"Item {i}: expected pad after position {trunc_len}, "
                    f"got {pad_region.tolist()}"
                )

    def test_attention_mask_updated(self):
        """Attention mask should reflect truncated lengths, not original."""
        from data.dataset import random_truncation_collate_fn

        random.seed(2)
        batch = self._make_batch([10, 3], max_comp_len=16)
        result = random_truncation_collate_fn(batch)

        for i in range(2):
            trunc_len = result["completion_length"][i].item()
            mask = result["attention_mask"][i]
            prompt_len = result["prompt_ids"][i].shape[0]
            # Prompt portion: True
            assert mask[:prompt_len].all()
            # Completion: True up to trunc_len, False after
            comp_mask = mask[prompt_len:]
            assert comp_mask[:trunc_len].all()
            if trunc_len < 16:
                assert not comp_mask[trunc_len:].any()

    def test_original_lengths_preserved(self):
        """original_completion_length should match the input lengths."""
        from data.dataset import random_truncation_collate_fn

        random.seed(0)
        batch = self._make_batch([5, 8, 3])
        result = random_truncation_collate_fn(batch)
        assert result["original_completion_length"].tolist() == [5, 8, 3]

    def test_single_item_batch(self):
        """Collate should work with a single-item batch."""
        from data.dataset import random_truncation_collate_fn

        random.seed(0)
        batch = self._make_batch([5])
        result = random_truncation_collate_fn(batch)
        assert result["completion_ids"].shape == (1, 16)
        # With single item, j must be 0, so trunc_len = min(5, 5) = 5
        assert result["completion_length"][0].item() == 5


# =========================================================================
# Tests for data.analyze_lengths
# =========================================================================


class TestAnalyzeCompletionLengths:
    """Tests for analyze_lengths.analyze_completion_lengths."""

    def _make_data_and_tokenizer(self, completions):
        """Build mock data list and tokenizer from raw completion strings."""
        data = [{"completion": c} for c in completions]
        tok = MockTokenizer(pad_token_id=0, eos_token_id=1)
        return data, tok

    def test_basic_stats(self):
        from data.analyze_lengths import analyze_completion_lengths

        completions = [
            "a b c",        # 3 tokens
            "d e f g h",    # 5 tokens
            "i j",          # 2 tokens
            "k l m n",      # 4 tokens
        ]
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok, max_completion_length=128)

        assert stats["count"] == 4
        assert stats["min"] == 2
        assert stats["max"] == 5
        assert 2.0 <= stats["mean"] <= 5.0
        assert stats["median"] == 3.5  # median of [2, 3, 4, 5]

    def test_percentiles_present(self):
        from data.analyze_lengths import analyze_completion_lengths

        completions = [" ".join(["w"] * i) for i in range(1, 101)]  # 1..100 tokens
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok)

        assert "percentiles" in stats
        expected_pct_keys = {"50", "75", "90", "95", "99", "99.5"}
        assert set(stats["percentiles"].keys()) == expected_pct_keys

        # P50 should be around 50 for uniform 1..100
        assert 40 <= stats["percentiles"]["50"] <= 60

    def test_coverage_all_fit(self):
        """When all completions fit within max, coverage should be 1.0."""
        from data.analyze_lengths import analyze_completion_lengths

        completions = ["a b c", "d e", "f"]
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok, max_completion_length=128)

        assert stats["coverage_at_max"] == 1.0
        assert stats["num_exceeding"] == 0

    def test_coverage_some_exceed(self):
        """Verify coverage and num_exceeding when some completions are too long."""
        from data.analyze_lengths import analyze_completion_lengths

        completions = [
            "a",                                # 1 token
            "a b",                              # 2 tokens
            " ".join(["x"] * 10),               # 10 tokens
        ]
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok, max_completion_length=5)

        assert stats["num_exceeding"] == 1
        # 2 out of 3 fit => coverage ~ 0.6667
        assert abs(stats["coverage_at_max"] - (2 / 3)) < 0.01

    def test_coverage_threshold_check_ok(self):
        """When coverage >= 0.95, the report says OK."""
        from data.analyze_lengths import analyze_completion_lengths, print_report

        completions = [" ".join(["w"] * i) for i in range(1, 21)]
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok, max_completion_length=200)

        assert stats["coverage_at_max"] >= 0.95

    def test_coverage_threshold_check_warning(self):
        """When coverage < 0.95, print_report should warn."""
        from data.analyze_lengths import analyze_completion_lengths, print_report
        import io
        import sys

        # 100 completions: 50 of them exceed max
        completions = [" ".join(["w"] * (i + 1)) for i in range(100)]
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok, max_completion_length=10)

        assert stats["coverage_at_max"] < 0.95

        # Capture print_report output
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            print_report(stats)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "WARNING" in output

    def test_histogram_bins_and_counts(self):
        """histogram_bins and histogram_counts should be present and consistent."""
        from data.analyze_lengths import analyze_completion_lengths

        completions = ["a b c", "d e f g h i j k"]
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok)

        assert "histogram_bins" in stats
        assert "histogram_counts" in stats
        # Number of count entries should be len(bins) - 1
        assert len(stats["histogram_counts"]) == len(stats["histogram_bins"]) - 1
        # Total counts should equal number of examples
        assert sum(stats["histogram_counts"]) == 2

    def test_std_deviation(self):
        """Standard deviation for identical lengths should be 0."""
        from data.analyze_lengths import analyze_completion_lengths

        completions = ["a b c", "d e f", "g h i"]  # all 3 tokens
        data, tok = self._make_data_and_tokenizer(completions)
        stats = analyze_completion_lengths(data, tok)

        assert stats["std"] == 0.0

    def test_single_example(self):
        """Should handle a single-example dataset without errors."""
        from data.analyze_lengths import analyze_completion_lengths

        data, tok = self._make_data_and_tokenizer(["hello world"])
        stats = analyze_completion_lengths(data, tok)
        assert stats["count"] == 1
        assert stats["min"] == stats["max"] == 2
        assert stats["mean"] == 2.0


# =========================================================================
# Integration-style test: DreamNERDataset + collate
# =========================================================================


class TestDatasetCollateIntegration:
    """End-to-end test combining DreamNERDataset with the collate function."""

    def test_dataset_items_collate_correctly(self):
        from data.dataset import DreamNERDataset, random_truncation_collate_fn

        data = [
            {"prompt": "Extract person org", "completion": " person: Alice | org: Acme"},
            {"prompt": "Extract date", "completion": " date: January"},
            {"prompt": "Extract location event", "completion": " location: Paris | event: Olympics"},
        ]
        tok = MockTokenizer(pad_token_id=0, eos_token_id=1)
        ds = DreamNERDataset(data, tok, max_seq_length=64, max_completion_length=16)

        batch = [ds[i] for i in range(len(ds))]
        random.seed(42)
        result = random_truncation_collate_fn(batch)

        assert result["completion_ids"].shape[0] == 3
        assert result["completion_ids"].shape[1] == 16
        assert len(result["prompt_ids"]) == 3
        assert len(result["attention_mask"]) == 3
        assert result["completion_length"].shape == (3,)
        assert result["original_completion_length"].shape == (3,)


# =========================================================================
# Tests for data.dataset.pad_prompt_batch
# =========================================================================


class TestPadPromptBatch:
    """Tests for dataset.pad_prompt_batch utility."""

    def test_left_padding(self):
        from data.dataset import pad_prompt_batch

        prompts = [
            torch.tensor([10, 20, 30], dtype=torch.long),
            torch.tensor([40, 50], dtype=torch.long),
        ]
        padded, mask = pad_prompt_batch(prompts, pad_token_id=0, padding_side="left")

        assert padded.shape == (2, 3)
        assert padded[0].tolist() == [10, 20, 30]
        assert padded[1].tolist() == [0, 40, 50]
        assert mask[0].tolist() == [True, True, True]
        assert mask[1].tolist() == [False, True, True]

    def test_right_padding(self):
        from data.dataset import pad_prompt_batch

        prompts = [
            torch.tensor([10, 20], dtype=torch.long),
            torch.tensor([30, 40, 50], dtype=torch.long),
        ]
        padded, mask = pad_prompt_batch(prompts, pad_token_id=0, padding_side="right")

        assert padded.shape == (2, 3)
        assert padded[0].tolist() == [10, 20, 0]
        assert padded[1].tolist() == [30, 40, 50]
        assert mask[0].tolist() == [True, True, False]
        assert mask[1].tolist() == [True, True, True]

    def test_all_same_length(self):
        """When all prompts have the same length, no padding is needed."""
        from data.dataset import pad_prompt_batch

        prompts = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5, 6], dtype=torch.long),
        ]
        padded, mask = pad_prompt_batch(prompts, pad_token_id=0)

        assert padded.shape == (2, 3)
        assert mask.all()
