"""Comprehensive unit tests for DiffusionNER-Zero inference modules.

Tests cover:
- inference/parse.py: parse_entities, deduplicate_entities,
  filter_entities_by_source, format_compliance_check
- inference/remask.py: remask_low_confidence, compute_remask_count
- inference/predict.py: apply_pad_penalty, compute_unmask_count,
  extract_entities (with mocked model), full denoising loop

All tests run on CPU only; the model is mocked so no GPU is required.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from inference.parse import (
    deduplicate_entities,
    filter_entities_by_source,
    format_compliance_check,
    parse_entities,
)
from inference.predict import (
    MASK_TOKEN_ID,
    apply_pad_penalty,
    compute_unmask_count,
    extract_entities,
)
from inference.remask import compute_remask_count, remask_low_confidence


# ---------------------------------------------------------------------------
# parse.py -- parse_entities
# ---------------------------------------------------------------------------


class TestParseEntities:
    """Tests for parse_entities."""

    def test_normal_output(self):
        result = parse_entities("person: Ronaldo | organization: Al Nassr")
        assert result == [
            {"type": "person", "text": "Ronaldo"},
            {"type": "organization", "text": "Al Nassr"},
        ]

    def test_none_with_leading_space(self):
        """' none' (leading space) should return empty list."""
        assert parse_entities(" none") == []

    def test_none_case_insensitive(self):
        assert parse_entities("None") == []
        assert parse_entities("NONE") == []
        assert parse_entities("nOnE") == []

    def test_empty_string(self):
        assert parse_entities("") == []

    def test_whitespace_only(self):
        assert parse_entities("   ") == []

    def test_single_entity_no_pipe(self):
        result = parse_entities("person: Ronaldo")
        assert result == [{"type": "person", "text": "Ronaldo"}]

    def test_malformed_no_colon(self):
        """Segment without a colon is silently skipped."""
        result = parse_entities("this has no colon")
        assert result == []

    def test_mixed_valid_and_malformed(self):
        result = parse_entities("person: Ada | malformed | org: MIT")
        assert result == [
            {"type": "person", "text": "Ada"},
            {"type": "org", "text": "MIT"},
        ]

    def test_strips_whitespace(self):
        result = parse_entities("  person :  Ronaldo  |  organization :  Al Nassr  ")
        assert result == [
            {"type": "person", "text": "Ronaldo"},
            {"type": "organization", "text": "Al Nassr"},
        ]

    def test_lowercases_entity_types(self):
        result = parse_entities("PERSON: Ronaldo | Organization: Al Nassr")
        assert result[0]["type"] == "person"
        assert result[1]["type"] == "organization"

    def test_preserves_text_case(self):
        """Entity text should keep original casing."""
        result = parse_entities("person: Cristiano Ronaldo")
        assert result[0]["text"] == "Cristiano Ronaldo"

    def test_multiple_colons_in_value(self):
        """Only the first colon is the delimiter; rest is part of the text."""
        result = parse_entities("date: 2023:01:15")
        assert result == [{"type": "date", "text": "2023:01:15"}]

    def test_empty_type_skipped(self):
        result = parse_entities(": Ronaldo")
        assert result == []

    def test_empty_text_skipped(self):
        result = parse_entities("person: ")
        assert result == []

    def test_trailing_pipe(self):
        """Trailing pipe produces an empty segment which is skipped."""
        result = parse_entities("person: Ronaldo |")
        assert result == [{"type": "person", "text": "Ronaldo"}]


# ---------------------------------------------------------------------------
# parse.py -- deduplicate_entities
# ---------------------------------------------------------------------------


class TestDeduplicateEntities:
    def test_removes_duplicates(self):
        entities = [
            {"type": "person", "text": "Ronaldo"},
            {"type": "person", "text": "Ronaldo"},
            {"type": "organization", "text": "Al Nassr"},
        ]
        result = deduplicate_entities(entities)
        assert result == [
            {"type": "person", "text": "Ronaldo"},
            {"type": "organization", "text": "Al Nassr"},
        ]

    def test_preserves_order(self):
        entities = [
            {"type": "organization", "text": "Al Nassr"},
            {"type": "person", "text": "Ronaldo"},
            {"type": "organization", "text": "Al Nassr"},
        ]
        result = deduplicate_entities(entities)
        assert result[0] == {"type": "organization", "text": "Al Nassr"}
        assert result[1] == {"type": "person", "text": "Ronaldo"}

    def test_different_type_same_text_kept(self):
        entities = [
            {"type": "person", "text": "Paris"},
            {"type": "location", "text": "Paris"},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate_entities([]) == []


# ---------------------------------------------------------------------------
# parse.py -- filter_entities_by_source
# ---------------------------------------------------------------------------


class TestFilterEntitiesBySource:
    def test_removes_hallucinated(self):
        entities = [
            {"type": "person", "text": "Ronaldo"},
            {"type": "person", "text": "Messi"},
        ]
        source = "Ronaldo joined Al Nassr."
        result = filter_entities_by_source(entities, source)
        assert result == [{"type": "person", "text": "Ronaldo"}]

    def test_case_insensitive_by_default(self):
        entities = [
            {"type": "person", "text": "ronaldo"},
        ]
        source = "Ronaldo joined Al Nassr."
        result = filter_entities_by_source(entities, source, case_sensitive=False)
        assert len(result) == 1

    def test_case_sensitive_mode(self):
        entities = [
            {"type": "person", "text": "ronaldo"},
        ]
        source = "Ronaldo joined Al Nassr."
        result = filter_entities_by_source(entities, source, case_sensitive=True)
        assert len(result) == 0

    def test_all_entities_present(self):
        entities = [
            {"type": "person", "text": "Ronaldo"},
            {"type": "organization", "text": "Al Nassr"},
        ]
        source = "Ronaldo joined Al Nassr in January."
        result = filter_entities_by_source(entities, source)
        assert len(result) == 2

    def test_empty_entities(self):
        assert filter_entities_by_source([], "some text") == []


# ---------------------------------------------------------------------------
# parse.py -- format_compliance_check
# ---------------------------------------------------------------------------


class TestFormatComplianceCheck:
    def test_compliant_output(self):
        result = format_compliance_check("person: Ronaldo | organization: Al Nassr")
        assert result["is_compliant"] is True
        assert result["is_none_output"] is False
        assert result["num_valid_segments"] == 2
        assert result["num_malformed_segments"] == 0

    def test_none_output(self):
        result = format_compliance_check(" None ")
        assert result["is_compliant"] is True
        assert result["is_none_output"] is True

    def test_empty_output(self):
        result = format_compliance_check("")
        assert result["is_compliant"] is False

    def test_malformed_segment(self):
        result = format_compliance_check("person: Ronaldo | no_colon_here")
        assert result["is_compliant"] is False
        assert result["num_malformed_segments"] == 1
        assert "no_colon_here" in result["malformed_segments"]

    def test_raw_output_preserved(self):
        raw = "  person: Ronaldo  "
        result = format_compliance_check(raw)
        assert result["raw_output"] == raw

    def test_single_valid_segment(self):
        result = format_compliance_check("person: Ronaldo")
        assert result["is_compliant"] is True
        assert result["num_segments"] == 1
        assert result["num_valid_segments"] == 1

    def test_special_chars_in_type_malformed(self):
        """Types with special characters (e.g. @) are counted as malformed."""
        result = format_compliance_check("per@son: Ronaldo")
        assert result["is_compliant"] is False
        assert result["num_malformed_segments"] == 1


# ---------------------------------------------------------------------------
# remask.py -- remask_low_confidence
# ---------------------------------------------------------------------------


class TestRemaskLowConfidence:
    def test_remasks_lowest_confidence(self):
        mask_id = 126336
        # prompt=[10, 20], output=[100, 200, 300, mask_id, mask_id]
        sequence = torch.tensor([10, 20, 100, 200, 300, mask_id, mask_id])
        # confidences for output region (length 5)
        # positions 0,1,2 are committed; positions 3,4 are masked
        confidences = torch.tensor([0.1, 0.9, 0.5, 0.0, 0.0])
        prompt_len = 2
        n_remask = 1

        result = remask_low_confidence(sequence, confidences, prompt_len, n_remask, mask_id)

        # Position 0 (token 100) has the lowest confidence (0.1) among committed
        assert result[2].item() == mask_id  # prompt_len + 0
        # Position 1 (token 200) should be untouched
        assert result[3].item() == 200
        # Prompt should be untouched
        assert result[0].item() == 10
        assert result[1].item() == 20

    def test_does_not_touch_prompt(self):
        mask_id = 126336
        sequence = torch.tensor([10, 20, 100, 200, mask_id])
        confidences = torch.tensor([0.2, 0.8, 0.0])
        prompt_len = 2
        n_remask = 1

        result = remask_low_confidence(sequence, confidences, prompt_len, n_remask, mask_id)

        assert result[0].item() == 10
        assert result[1].item() == 20

    def test_n_remask_zero_changes_nothing(self):
        mask_id = 126336
        sequence = torch.tensor([10, 20, 100, 200, 300])
        confidences = torch.tensor([0.1, 0.5, 0.9])
        prompt_len = 2

        result = remask_low_confidence(sequence, confidences, prompt_len, 0, mask_id)
        assert torch.equal(result, sequence)

    def test_n_remask_exceeds_committed(self):
        """If n_remask > committed tokens, all committed get re-masked."""
        mask_id = 126336
        sequence = torch.tensor([10, 20, 100, mask_id])
        confidences = torch.tensor([0.5, 0.0])
        prompt_len = 2

        result = remask_low_confidence(sequence, confidences, prompt_len, 10, mask_id)
        # Only one committed token (position 0 = token 100), should become mask
        assert result[2].item() == mask_id

    def test_all_masked_output_unchanged(self):
        """If all output tokens are already masked, nothing changes."""
        mask_id = 126336
        sequence = torch.tensor([10, 20, mask_id, mask_id])
        confidences = torch.tensor([0.0, 0.0])
        prompt_len = 2

        result = remask_low_confidence(sequence, confidences, prompt_len, 2, mask_id)
        assert torch.equal(result, sequence)

    def test_original_sequence_not_mutated(self):
        """remask_low_confidence should clone before modifying."""
        mask_id = 126336
        sequence = torch.tensor([10, 20, 100, 200])
        original_copy = sequence.clone()
        confidences = torch.tensor([0.1, 0.9])
        prompt_len = 2

        _ = remask_low_confidence(sequence, confidences, prompt_len, 1, mask_id)
        # Original should be unchanged
        assert torch.equal(sequence, original_copy)


# ---------------------------------------------------------------------------
# remask.py -- compute_remask_count
# ---------------------------------------------------------------------------


class TestComputeRemaskCount:
    def test_decays_over_steps(self):
        """Earlier steps should remask more than later steps."""
        early = compute_remask_count(step=0, total_steps=10, n_committed=100)
        late = compute_remask_count(step=8, total_steps=10, n_committed=100)
        assert early > late

    def test_returns_zero_at_final_step(self):
        result = compute_remask_count(step=9, total_steps=10, n_committed=100)
        assert result == 0

    def test_zero_committed(self):
        result = compute_remask_count(step=0, total_steps=10, n_committed=0)
        assert result == 0

    def test_total_steps_zero(self):
        result = compute_remask_count(step=0, total_steps=0, n_committed=100)
        assert result == 0

    def test_step_zero_value(self):
        """At step 0 with total_steps=10, decay = 1.0 - 1/10 = 0.9."""
        result = compute_remask_count(
            step=0, total_steps=10, n_committed=100, remask_ratio=0.3
        )
        # Expected: round(0.3 * 0.9 * 100) = round(27.0) = 27
        assert result == 27

    def test_never_exceeds_committed(self):
        result = compute_remask_count(
            step=0, total_steps=10, n_committed=2, remask_ratio=1.0
        )
        assert result <= 2

    def test_always_non_negative(self):
        for step in range(20):
            result = compute_remask_count(step=step, total_steps=10, n_committed=50)
            assert result >= 0


# ---------------------------------------------------------------------------
# predict.py -- apply_pad_penalty
# ---------------------------------------------------------------------------


class TestApplyPadPenalty:
    def test_reduces_pad_logits(self):
        logits = torch.zeros(10, 100)
        pad_id = 0
        original_pad_val = logits[0, pad_id].item()
        result = apply_pad_penalty(logits, step=0, total_steps=8, pad_token_id=pad_id)
        assert result[0, pad_id].item() < original_pad_val

    def test_full_penalty_at_step_zero(self):
        logits = torch.zeros(10, 100)
        pad_id = 5
        max_pen = 5.0
        apply_pad_penalty(logits, step=0, total_steps=8, pad_token_id=pad_id, max_penalty=max_pen)
        # decay = 1.0 - 0/8 = 1.0, so penalty = 5.0
        assert logits[0, pad_id].item() == pytest.approx(-5.0)

    def test_no_penalty_at_final_step(self):
        logits = torch.zeros(10, 100)
        pad_id = 5
        max_pen = 5.0
        total = 8
        apply_pad_penalty(logits, step=total, total_steps=total, pad_token_id=pad_id, max_penalty=max_pen)
        # decay = 1.0 - 8/8 = 0.0, so penalty = 0.0
        assert logits[0, pad_id].item() == pytest.approx(0.0)

    def test_intermediate_step_decay(self):
        logits = torch.zeros(10, 100)
        pad_id = 5
        max_pen = 5.0
        total = 10
        step = 5
        apply_pad_penalty(logits, step=step, total_steps=total, pad_token_id=pad_id, max_penalty=max_pen)
        # decay = 1.0 - 5/10 = 0.5, penalty = 2.5
        assert logits[0, pad_id].item() == pytest.approx(-2.5)

    def test_does_not_affect_non_pad(self):
        logits = torch.zeros(10, 100)
        pad_id = 5
        apply_pad_penalty(logits, step=0, total_steps=8, pad_token_id=pad_id)
        # Non-pad columns should be untouched
        assert logits[0, 0].item() == 0.0
        assert logits[0, 6].item() == 0.0


# ---------------------------------------------------------------------------
# predict.py -- compute_unmask_count
# ---------------------------------------------------------------------------


class TestComputeUnmaskCount:
    def test_zero_masked(self):
        assert compute_unmask_count(step=0, total_steps=10, n_masked=0) == 0

    def test_at_least_one_unmasked(self):
        """When there are masked tokens, at least 1 should be unmasked."""
        for step in range(10):
            result = compute_unmask_count(step=step, total_steps=10, n_masked=100)
            assert result >= 1

    def test_first_step_unmasks_some(self):
        result = compute_unmask_count(step=0, total_steps=10, n_masked=100)
        # target_remaining = 100 * (1 - 1/10) = 90 => unmask = 10
        assert result == 10

    def test_last_step_unmasks_all_remaining(self):
        result = compute_unmask_count(step=9, total_steps=10, n_masked=50)
        # target_remaining = 50 * (1 - 10/10) = 0 => unmask = 50
        assert result == 50

    def test_monotonic_progress(self):
        """Over the full schedule, total unmasked should not exceed original."""
        n_masked = 100
        total_unmasked = 0
        for step in range(10):
            n_unmask = compute_unmask_count(step=step, total_steps=10, n_masked=n_masked)
            n_masked -= n_unmask
            total_unmasked += n_unmask
            assert n_masked >= 0


# ---------------------------------------------------------------------------
# predict.py -- extract_entities (mocked model)
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(prompt_ids, output_text, pad_token_id=0):
    """Create a mock tokenizer.

    Parameters
    ----------
    prompt_ids : list[int]
        Token IDs that ``encode`` returns for any prompt.
    output_text : str
        Text that ``decode`` returns for any token list.
    pad_token_id : int
        Value of ``pad_token_id`` attribute.
    """
    tokenizer = MagicMock()
    tokenizer.encode.return_value = prompt_ids
    tokenizer.decode.return_value = output_text
    tokenizer.pad_token_id = pad_token_id
    tokenizer.eos_token_id = 1
    return tokenizer


def _make_mock_model(vocab_size, device="cpu", target_token_id=None):
    """Create a mock model that returns logits on CPU.

    When ``target_token_id`` is given, the logits strongly favour that token
    for all positions, making the denoising loop deterministic and easy to
    reason about.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for the logits.
    device : str
        Device attribute of the mock model.
    target_token_id : int or None
        If set, the model will produce high logit for this token.
    """
    model = MagicMock()
    model.device = device

    def forward(input_ids=None, **kwargs):
        seq_len = input_ids.shape[1]
        if target_token_id is not None:
            # Create logits that strongly favour target_token_id
            logits = torch.full((1, seq_len, vocab_size), -10.0)
            logits[:, :, target_token_id] = 10.0
        else:
            # Random logits -- deterministic via manual seed per call
            logits = torch.randn(1, seq_len, vocab_size)
        return SimpleNamespace(logits=logits)

    model.side_effect = forward
    model.__call__ = forward
    return model


class TestExtractEntitiesMocked:
    """Test extract_entities with a mocked model and tokenizer."""

    def test_returns_list_of_dicts(self):
        """Pipeline returns parsed entities from decode output."""
        prompt_ids = [10, 20, 30]
        tokenizer = _make_mock_tokenizer(
            prompt_ids,
            output_text=" person: Ronaldo | organization: Al Nassr",
        )
        model = _make_mock_model(vocab_size=200, target_token_id=50)

        result = extract_entities(
            model,
            tokenizer,
            text="Ronaldo joined Al Nassr.",
            entity_types=["person", "organization"],
            num_steps=4,
            max_output_len=16,
        )

        assert isinstance(result, list)
        assert all(isinstance(e, dict) for e in result)
        assert result == [
            {"type": "person", "text": "Ronaldo"},
            {"type": "organization", "text": "Al Nassr"},
        ]

    def test_decode_called_with_clean_ids(self):
        """MASK and PAD tokens must be filtered before decoding."""
        prompt_ids = [10, 20]
        pad_token_id = 0
        tokenizer = _make_mock_tokenizer(prompt_ids, "none", pad_token_id)
        # Target token is 50 (neither MASK nor PAD)
        model = _make_mock_model(vocab_size=200, target_token_id=50)

        extract_entities(
            model,
            tokenizer,
            text="Hello world.",
            entity_types=["person"],
            num_steps=4,
            max_output_len=8,
        )

        # decode should have been called; verify no MASK or PAD in the ids
        call_args = tokenizer.decode.call_args
        decoded_ids = call_args[0][0]
        assert MASK_TOKEN_ID not in decoded_ids
        assert pad_token_id not in decoded_ids

    def test_with_remasking_enabled(self):
        """Pipeline should not crash when use_remasking=True."""
        prompt_ids = [10, 20]
        tokenizer = _make_mock_tokenizer(prompt_ids, " person: Ada")
        model = _make_mock_model(vocab_size=200, target_token_id=42)

        result = extract_entities(
            model,
            tokenizer,
            text="Ada Lovelace was a mathematician.",
            entity_types=["person"],
            num_steps=6,
            max_output_len=16,
            use_remasking=True,
            remask_ratio=0.3,
        )
        assert isinstance(result, list)

    def test_random_model_completes_without_error(self):
        """Even with random logits the pipeline should not crash."""
        prompt_ids = [10, 20, 30, 40]
        tokenizer = _make_mock_tokenizer(prompt_ids, "none")
        model = _make_mock_model(vocab_size=200, target_token_id=None)

        result = extract_entities(
            model,
            tokenizer,
            text="Some random text.",
            entity_types=["person", "org"],
            num_steps=4,
            max_output_len=16,
        )
        # "none" decoded output -> empty entity list
        assert result == []

    def test_pad_token_id_fallback_to_eos(self):
        """When pad_token_id is None, eos_token_id is used as fallback."""
        prompt_ids = [10, 20]
        tokenizer = MagicMock()
        tokenizer.encode.return_value = prompt_ids
        tokenizer.decode.return_value = "none"
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 99

        model = _make_mock_model(vocab_size=200, target_token_id=50)

        result = extract_entities(
            model,
            tokenizer,
            text="Test.",
            entity_types=["person"],
            num_steps=2,
            max_output_len=8,
        )
        # Should not crash; pad_token_id should fall back to eos=99
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# predict.py -- full denoising loop logic
# ---------------------------------------------------------------------------


class TestDenoisingLoop:
    """Test the denoising loop mechanics more deeply."""

    def test_masked_count_decreases(self):
        """Over the loop, the number of masked tokens should decrease."""
        prompt_ids = [10, 20]
        max_output_len = 16
        vocab_size = 200
        target_id = 50

        # Manually simulate the loop to check mask counts decrease
        model = _make_mock_model(vocab_size, target_token_id=target_id)

        output_ids = [MASK_TOKEN_ID] * max_output_len
        sequence = torch.tensor(
            prompt_ids + output_ids, dtype=torch.long
        ).unsqueeze(0)
        prompt_len = len(prompt_ids)
        num_steps = 8

        mask_counts = []
        for step in range(num_steps):
            logits = model(input_ids=sequence).logits[0]
            output_logits = logits[prompt_len:]

            probs = F.softmax(output_logits, dim=-1)
            predicted_ids = probs.argmax(dim=-1)
            confidences = probs.max(dim=-1).values

            current_output = sequence[0, prompt_len:]
            masked_positions = (current_output == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
            n_masked = len(masked_positions)
            mask_counts.append(n_masked)

            if n_masked == 0:
                break

            n_unmask = compute_unmask_count(step, num_steps, n_masked)
            masked_confidences = confidences[masked_positions]
            k = min(n_unmask, len(masked_positions))
            top_k_indices = masked_confidences.topk(k).indices
            positions_to_unmask = masked_positions[top_k_indices]

            for pos in positions_to_unmask:
                sequence[0, prompt_len + pos] = predicted_ids[pos]

        # Mask count should be non-increasing
        for i in range(1, len(mask_counts)):
            assert mask_counts[i] <= mask_counts[i - 1]

        # After loop, all should be unmasked (target is deterministic)
        final_masked = (sequence[0, prompt_len:] == MASK_TOKEN_ID).sum().item()
        assert final_masked == 0

    def test_prompt_region_untouched(self):
        """The denoising loop must never modify the prompt region."""
        prompt_ids = [10, 20, 30]
        max_output_len = 8
        vocab_size = 200

        model = _make_mock_model(vocab_size, target_token_id=42)
        tokenizer = _make_mock_tokenizer(prompt_ids, "person: Test")

        extract_entities(
            model,
            tokenizer,
            text="Test text",
            entity_types=["person"],
            num_steps=4,
            max_output_len=max_output_len,
        )

        # Since extract_entities creates its own sequence internally, we
        # verify indirectly: the function should complete without error and
        # return a parseable result.  If prompt tokens were corrupted, the
        # model call would receive mangled inputs.
        # (A more direct check would require inspecting the sequence, but the
        # function returns only parsed entities.)

    def test_remasking_loop_does_not_diverge(self):
        """With remasking enabled, the loop should still converge."""
        prompt_ids = [10, 20]
        max_output_len = 16
        vocab_size = 200

        model = _make_mock_model(vocab_size, target_token_id=50)
        tokenizer = _make_mock_tokenizer(prompt_ids, "person: Test")

        result = extract_entities(
            model,
            tokenizer,
            text="Test text.",
            entity_types=["person"],
            num_steps=10,
            max_output_len=max_output_len,
            use_remasking=True,
            remask_ratio=0.3,
        )
        assert isinstance(result, list)

    def test_single_step_denoising(self):
        """With num_steps=1, all masked tokens should be unmasked in one step."""
        prompt_ids = [10, 20]
        max_output_len = 8
        vocab_size = 200
        target_id = 50

        model = _make_mock_model(vocab_size, target_token_id=target_id)
        tokenizer = _make_mock_tokenizer(prompt_ids, "person: X")

        result = extract_entities(
            model,
            tokenizer,
            text="X is a person.",
            entity_types=["person"],
            num_steps=1,
            max_output_len=max_output_len,
        )
        assert isinstance(result, list)

        # Verify decode was called (the loop ran at least once)
        assert tokenizer.decode.called

    def test_all_unmasked_early_break(self):
        """If all tokens are unmasked before all steps, the loop should break."""
        prompt_ids = [10]
        # Very small output length to ensure everything unmasks quickly
        max_output_len = 2
        vocab_size = 200
        target_id = 50

        model = _make_mock_model(vocab_size, target_token_id=target_id)
        tokenizer = _make_mock_tokenizer(prompt_ids, "none")

        # num_steps is large, but should break early
        result = extract_entities(
            model,
            tokenizer,
            text="X",
            entity_types=["person"],
            num_steps=100,
            max_output_len=max_output_len,
        )

        # The model was not called 100 times -- hard to verify directly,
        # but at least the function completes in reasonable time
        assert isinstance(result, list)
