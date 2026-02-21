"""Comprehensive unit tests for evaluation and analysis modules.

Tests cover:
- evaluation.evaluate: compute_micro_f1, _normalize_entity, compute_per_example_f1
- evaluation.load_benchmarks: bio_to_entities
- analysis.error_analysis: compare_predictions, analyze_error_types
- analysis.hallucination_rate: compute_hallucination_rate
"""

import math
import pytest

from evaluation.evaluate import compute_micro_f1, _normalize_entity, compute_per_example_f1
from evaluation.load_benchmarks import bio_to_entities
from analysis.error_analysis import compare_predictions, _entity_set, analyze_error_types
from analysis.hallucination_rate import compute_hallucination_rate


# =====================================================================
# Helper to build entity dicts concisely
# =====================================================================

def _ent(etype: str, text: str) -> dict:
    """Shorthand to build an entity dict."""
    return {"type": etype, "text": text}


# =====================================================================
# Tests for evaluate.py -- _normalize_entity
# =====================================================================

class TestNormalizeEntity:
    """Tests for the _normalize_entity helper."""

    def test_basic_normalization(self):
        result = _normalize_entity({"type": "PER", "text": "Alice"})
        assert result == ("per", "alice")

    def test_strips_whitespace(self):
        result = _normalize_entity({"type": "  PER  ", "text": "  Alice  "})
        assert result == ("per", "alice")

    def test_replaces_spaces_in_type(self):
        result = _normalize_entity({"type": "chemical compound", "text": "H2O"})
        assert result == ("chemical_compound", "h2o")

    def test_empty_fields(self):
        result = _normalize_entity({"type": "", "text": ""})
        assert result == ("", "")

    def test_missing_fields_default_to_empty(self):
        result = _normalize_entity({})
        assert result == ("", "")

    def test_mixed_case_type_and_text(self):
        result = _normalize_entity({"type": "OrGaNiZaTiOn", "text": "ACME Corp"})
        assert result == ("organization", "acme corp")


# =====================================================================
# Tests for evaluate.py -- compute_micro_f1
# =====================================================================

class TestComputeMicroF1:
    """Tests for compute_micro_f1."""

    def test_perfect_predictions(self):
        """All predictions match gold exactly -> F1=1.0."""
        gold = [
            [_ent("person", "Alice"), _ent("org", "ACME")],
            [_ent("person", "Bob")],
        ]
        preds = [
            [_ent("person", "Alice"), _ent("org", "ACME")],
            [_ent("person", "Bob")],
        ]
        result = compute_micro_f1(preds, gold)
        assert result["f1"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["tp"] == 3
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_no_predictions(self):
        """No predictions at all -> F1=0.0, recall=0.0."""
        gold = [[_ent("person", "Alice")]]
        preds = [[]]
        result = compute_micro_f1(preds, gold)
        assert result["f1"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 1

    def test_partial_overlap(self):
        """One correct, one wrong prediction, one missed gold entity."""
        gold = [[_ent("person", "Alice"), _ent("org", "ACME")]]
        preds = [[_ent("person", "Alice"), _ent("loc", "London")]]
        result = compute_micro_f1(preds, gold)
        # TP=1 (Alice), FP=1 (London), FN=1 (ACME)
        assert result["tp"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["precision"] == pytest.approx(1 / 2)
        assert result["recall"] == pytest.approx(1 / 2)
        expected_f1 = 2 * 0.5 * 0.5 / (0.5 + 0.5)
        assert result["f1"] == pytest.approx(expected_f1)

    def test_multiple_examples_aggregated(self):
        """Micro-F1 aggregates across examples, not averages per-example."""
        gold = [
            [_ent("person", "Alice")],
            [_ent("person", "Bob"), _ent("org", "ACME")],
        ]
        preds = [
            [_ent("person", "Alice")],  # 1 TP
            [_ent("person", "Bob"), _ent("loc", "London")],  # 1 TP, 1 FP, 1 FN (ACME)
        ]
        result = compute_micro_f1(preds, gold)
        # total: TP=2, FP=1, FN=1
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)

    def test_case_sensitivity_types_lowercased(self):
        """Entity types should be lowercased for comparison."""
        gold = [[_ent("PERSON", "Alice")]]
        preds = [[_ent("person", "Alice")]]
        result = compute_micro_f1(preds, gold)
        assert result["f1"] == pytest.approx(1.0)
        assert result["tp"] == 1

    def test_case_sensitivity_text_lowercased(self):
        """Entity text should be lowercased for comparison."""
        gold = [[_ent("person", "ALICE")]]
        preds = [[_ent("person", "alice")]]
        result = compute_micro_f1(preds, gold)
        assert result["f1"] == pytest.approx(1.0)
        assert result["tp"] == 1

    def test_duplicate_entities_handled(self):
        """Duplicate entities in gold/pred use multiset matching."""
        gold = [[_ent("person", "Alice"), _ent("person", "Alice")]]
        preds = [[_ent("person", "Alice")]]
        result = compute_micro_f1(preds, gold)
        # Gold has 2 "Alice", pred has 1 -> TP=1, FN=1, FP=0
        assert result["tp"] == 1
        assert result["fn"] == 1
        assert result["fp"] == 0

    def test_duplicate_entities_both_sides(self):
        """Both gold and pred have duplicates."""
        gold = [[_ent("person", "Alice"), _ent("person", "Alice")]]
        preds = [[_ent("person", "Alice"), _ent("person", "Alice")]]
        result = compute_micro_f1(preds, gold)
        assert result["tp"] == 2
        assert result["fn"] == 0
        assert result["fp"] == 0
        assert result["f1"] == pytest.approx(1.0)

    def test_duplicate_entities_more_preds_than_gold(self):
        """More predicted duplicates than gold -> extra count as FP."""
        gold = [[_ent("person", "Alice")]]
        preds = [[_ent("person", "Alice"), _ent("person", "Alice"), _ent("person", "Alice")]]
        result = compute_micro_f1(preds, gold)
        # TP=1, FP=2, FN=0
        assert result["tp"] == 1
        assert result["fp"] == 2
        assert result["fn"] == 0

    def test_empty_gold_with_predictions(self):
        """Empty gold but model makes predictions -> precision=0 (FP only)."""
        gold = [[]]
        preds = [[_ent("person", "Alice"), _ent("org", "ACME")]]
        result = compute_micro_f1(preds, gold)
        assert result["tp"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 0
        # precision = 0/(0+2) = 0, recall = 0/(0+0) = 0
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_empty_predictions_with_gold(self):
        """Empty predictions with gold entities -> recall=0 (FN only)."""
        gold = [[_ent("person", "Alice"), _ent("org", "ACME")]]
        preds = [[]]
        result = compute_micro_f1(preds, gold)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 2
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_both_empty(self):
        """Both gold and predictions empty -> all zeros, F1=0."""
        gold = [[]]
        preds = [[]]
        result = compute_micro_f1(preds, gold)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["f1"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_micro_f1([[]], [[], []])

    def test_whitespace_stripping_in_match(self):
        """Leading/trailing whitespace in entity fields is stripped."""
        gold = [[_ent("  person  ", "  Alice  ")]]
        preds = [[_ent("person", "Alice")]]
        result = compute_micro_f1(preds, gold)
        assert result["f1"] == pytest.approx(1.0)

    def test_type_with_spaces_normalized(self):
        """Spaces in entity types become underscores during normalization."""
        gold = [[_ent("chemical compound", "H2O")]]
        preds = [[_ent("chemical_compound", "H2O")]]
        result = compute_micro_f1(preds, gold)
        assert result["f1"] == pytest.approx(1.0)


# =====================================================================
# Tests for evaluate.py -- compute_per_example_f1
# =====================================================================

class TestComputePerExampleF1:
    """Tests for compute_per_example_f1."""

    def test_single_example_perfect(self):
        pred = [_ent("person", "Alice")]
        gold = [_ent("person", "Alice")]
        result = compute_per_example_f1(pred, gold)
        assert result["f1"] == pytest.approx(1.0)

    def test_single_example_no_overlap(self):
        pred = [_ent("loc", "London")]
        gold = [_ent("person", "Alice")]
        result = compute_per_example_f1(pred, gold)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1


# =====================================================================
# Tests for load_benchmarks.py -- bio_to_entities
# =====================================================================

class TestBioToEntities:
    """Tests for the bio_to_entities function."""

    def test_simple_bio_one_entity(self):
        """B-PER I-PER O -> one entity."""
        tokens = ["John", "Smith", "runs"]
        tags = ["B-PER", "I-PER", "O"]
        text, entities = bio_to_entities(tokens, tags)
        assert text == "John Smith runs"
        assert len(entities) == 1
        assert entities[0]["type"] == "per"
        assert entities[0]["text"] == "John Smith"
        assert entities[0]["start"] == 0

    def test_multiple_entities(self):
        """B-PER O B-LOC -> two distinct entities."""
        tokens = ["Alice", "visited", "Paris"]
        tags = ["B-PER", "O", "B-LOC"]
        text, entities = bio_to_entities(tokens, tags)
        assert text == "Alice visited Paris"
        assert len(entities) == 2
        assert entities[0]["type"] == "per"
        assert entities[0]["text"] == "Alice"
        assert entities[0]["start"] == 0
        assert entities[1]["type"] == "loc"
        assert entities[1]["text"] == "Paris"
        assert entities[1]["start"] == 14  # "Alice visited " = 14 chars

    def test_consecutive_same_type(self):
        """B-PER B-PER -> two separate entities of same type."""
        tokens = ["Alice", "Bob"]
        tags = ["B-PER", "B-PER"]
        text, entities = bio_to_entities(tokens, tags)
        assert text == "Alice Bob"
        assert len(entities) == 2
        assert entities[0]["text"] == "Alice"
        assert entities[1]["text"] == "Bob"
        assert entities[0]["type"] == "per"
        assert entities[1]["type"] == "per"

    def test_o_only_tags(self):
        """All O tags -> no entities."""
        tokens = ["The", "quick", "fox"]
        tags = ["O", "O", "O"]
        text, entities = bio_to_entities(tokens, tags)
        assert text == "The quick fox"
        assert len(entities) == 0

    def test_orphan_i_tag_treated_as_b(self):
        """An I- tag without a preceding B- of the same type starts a new entity."""
        tokens = ["Alice", "runs"]
        tags = ["I-PER", "O"]
        text, entities = bio_to_entities(tokens, tags)
        assert len(entities) == 1
        assert entities[0]["type"] == "per"
        assert entities[0]["text"] == "Alice"

    def test_orphan_i_tag_different_type(self):
        """An I- tag whose type differs from the current entity starts a new entity."""
        tokens = ["Alice", "London", "runs"]
        tags = ["B-PER", "I-LOC", "O"]
        text, entities = bio_to_entities(tokens, tags)
        # "Alice" is B-PER, "London" is I-LOC which doesn't match PER -> flush Alice, start LOC
        assert len(entities) == 2
        assert entities[0]["type"] == "per"
        assert entities[0]["text"] == "Alice"
        assert entities[1]["type"] == "loc"
        assert entities[1]["text"] == "London"

    def test_text_reconstruction(self):
        """Reconstructed text joins tokens with single spaces."""
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
        tags = ["O", "O", "O", "O", "O", "O"]
        text, entities = bio_to_entities(tokens, tags)
        assert text == "The cat sat on the mat"

    def test_entity_at_end_of_sentence(self):
        """Entity at the end (no trailing O) is correctly flushed."""
        tokens = ["lives", "in", "New", "York"]
        tags = ["O", "O", "B-LOC", "I-LOC"]
        text, entities = bio_to_entities(tokens, tags)
        assert len(entities) == 1
        assert entities[0]["type"] == "loc"
        assert entities[0]["text"] == "New York"

    def test_character_offsets_correct(self):
        """Character offsets match the positions in the reconstructed text."""
        tokens = ["Alice", "visited", "New", "York"]
        tags = ["B-PER", "O", "B-LOC", "I-LOC"]
        text, entities = bio_to_entities(tokens, tags)
        assert text == "Alice visited New York"
        # Verify offsets by slicing the text
        for ent in entities:
            start = ent["start"]
            extracted = text[start:start + len(ent["text"])]
            assert extracted == ent["text"]

    def test_token_tag_length_mismatch_raises(self):
        """Mismatched token/tag lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Token/tag length mismatch"):
            bio_to_entities(["Alice", "Bob"], ["O"])

    def test_type_lowercased(self):
        """Entity types are lowercased."""
        tokens = ["Alice"]
        tags = ["B-PERSON"]
        text, entities = bio_to_entities(tokens, tags)
        assert entities[0]["type"] == "person"

    def test_type_spaces_become_underscores(self):
        """Spaces in BIO tag types become underscores."""
        # BIO tags normally don't have spaces, but the code replaces them
        tokens = ["H2O"]
        tags = ["B-chemical compound"]
        text, entities = bio_to_entities(tokens, tags)
        assert entities[0]["type"] == "chemical_compound"

    def test_empty_tokens_and_tags(self):
        """Empty input produces empty output."""
        text, entities = bio_to_entities([], [])
        assert text == ""
        assert entities == []

    def test_single_b_tag_only(self):
        """A single B- tag with no continuation or O tag."""
        tokens = ["Alice"]
        tags = ["B-PER"]
        text, entities = bio_to_entities(tokens, tags)
        assert len(entities) == 1
        assert entities[0]["text"] == "Alice"
        assert entities[0]["type"] == "per"

    def test_multiple_consecutive_i_orphans(self):
        """Multiple consecutive orphan I- tags of same type continue as one entity."""
        tokens = ["New", "York", "City"]
        tags = ["I-LOC", "I-LOC", "I-LOC"]
        text, entities = bio_to_entities(tokens, tags)
        # First I-LOC is orphan -> treated as B-LOC, subsequent I-LOC continue
        assert len(entities) == 1
        assert entities[0]["text"] == "New York City"
        assert entities[0]["type"] == "loc"


# =====================================================================
# Tests for error_analysis.py -- compare_predictions
# =====================================================================

class TestComparePredictions:
    """Tests for the compare_predictions function."""

    def test_both_correct(self):
        """Both models predict a gold entity -> both_correct."""
        gold = [[_ent("person", "Alice")]]
        diff = [[_ent("person", "Alice")]]
        uni = [[_ent("person", "Alice")]]
        result = compare_predictions(diff, uni, gold)
        assert result["counts"]["both_correct"] == 1
        assert result["counts"]["diffusion_only"] == 0
        assert result["counts"]["uniner_only"] == 0
        assert result["counts"]["both_wrong"] == 0

    def test_diffusion_only(self):
        """Only diffusion model predicts a gold entity -> diffusion_only."""
        gold = [[_ent("person", "Alice")]]
        diff = [[_ent("person", "Alice")]]
        uni = [[]]
        result = compare_predictions(diff, uni, gold)
        assert result["counts"]["both_correct"] == 0
        assert result["counts"]["diffusion_only"] == 1
        assert result["counts"]["uniner_only"] == 0
        assert result["counts"]["both_wrong"] == 0

    def test_uniner_only(self):
        """Only UniNER predicts a gold entity -> uniner_only."""
        gold = [[_ent("person", "Alice")]]
        diff = [[]]
        uni = [[_ent("person", "Alice")]]
        result = compare_predictions(diff, uni, gold)
        assert result["counts"]["both_correct"] == 0
        assert result["counts"]["diffusion_only"] == 0
        assert result["counts"]["uniner_only"] == 1
        assert result["counts"]["both_wrong"] == 0

    def test_both_wrong(self):
        """Neither model predicts a gold entity -> both_wrong."""
        gold = [[_ent("person", "Alice")]]
        diff = [[]]
        uni = [[]]
        result = compare_predictions(diff, uni, gold)
        assert result["counts"]["both_correct"] == 0
        assert result["counts"]["diffusion_only"] == 0
        assert result["counts"]["uniner_only"] == 0
        assert result["counts"]["both_wrong"] == 1

    def test_mixed_categories(self):
        """Multiple gold entities across examples fall into different categories."""
        gold = [
            [_ent("person", "Alice"), _ent("org", "ACME")],
            [_ent("loc", "Paris"), _ent("person", "Bob")],
        ]
        diff = [
            [_ent("person", "Alice"), _ent("org", "ACME")],  # both correct
            [_ent("person", "Bob")],  # Bob: diffusion_only=no, Paris: both_wrong
        ]
        uni = [
            [_ent("person", "Alice")],  # Alice: both_correct, ACME: diffusion_only
            [_ent("loc", "Paris"), _ent("person", "Bob")],  # Paris: uniner_only, Bob: both_correct
        ]
        result = compare_predictions(diff, uni, gold)
        # Alice: both_correct, ACME: diffusion_only
        # Paris: uniner_only, Bob: both_correct
        assert result["counts"]["both_correct"] == 2  # Alice, Bob
        assert result["counts"]["diffusion_only"] == 1  # ACME
        assert result["counts"]["uniner_only"] == 1  # Paris
        assert result["counts"]["both_wrong"] == 0

    def test_case_insensitive_matching(self):
        """Entity type is lowercased for comparison in _entity_set."""
        gold = [[_ent("PERSON", "Alice")]]
        diff = [[_ent("person", "Alice")]]
        uni = [[_ent("Person", "Alice")]]
        result = compare_predictions(diff, uni, gold)
        assert result["counts"]["both_correct"] == 1

    def test_examples_field_populated(self):
        """The examples field contains correct details."""
        gold = [[_ent("person", "Alice")]]
        diff = [[_ent("person", "Alice")]]
        uni = [[]]
        result = compare_predictions(diff, uni, gold)
        examples = result["examples"]["diffusion_only"]
        assert len(examples) == 1
        assert examples[0]["example_idx"] == 0
        assert examples[0]["entity_type"] == "person"
        assert examples[0]["entity_text"] == "Alice"

    def test_length_mismatch_raises(self):
        """Mismatched input lengths should raise AssertionError."""
        with pytest.raises(AssertionError):
            compare_predictions([[]], [[]], [[], []])

    def test_empty_gold_no_categories(self):
        """No gold entities means no entries in any category."""
        gold = [[]]
        diff = [[_ent("person", "Alice")]]
        uni = [[_ent("person", "Bob")]]
        result = compare_predictions(diff, uni, gold)
        total = sum(result["counts"].values())
        assert total == 0

    def test_text_not_lowered_in_comparison(self):
        """Entity text is NOT lowered in _entity_set (only stripped)."""
        # _entity_set lowercases type but only strips text
        gold = [[_ent("person", "Alice")]]
        diff = [[_ent("person", "alice")]]  # different case text
        uni = [[_ent("person", "Alice")]]
        result = compare_predictions(diff, uni, gold)
        # diff has "alice" vs gold "Alice" -> not matching since _entity_set strips but doesn't lower text
        assert result["counts"]["uniner_only"] == 1
        assert result["counts"]["diffusion_only"] == 0


# =====================================================================
# Tests for error_analysis.py -- analyze_error_types
# =====================================================================

class TestAnalyzeErrorTypes:
    """Tests for the analyze_error_types function."""

    def test_type_error_detected(self):
        """Predicted text matches gold but with wrong type -> type_error."""
        gold = [[_ent("person", "Alice")]]
        preds = [[_ent("org", "Alice")]]
        result = analyze_error_types(preds, gold)
        assert result["counts"]["type_error"] == 1
        assert result["counts"]["missing_entity"] == 0

    def test_missing_entity_detected(self):
        """Gold entity not predicted at all -> missing_entity."""
        gold = [[_ent("person", "Alice")]]
        preds = [[]]
        result = analyze_error_types(preds, gold)
        assert result["counts"]["missing_entity"] == 1

    def test_hallucinated_entity_detected(self):
        """Predicted entity not in gold at all -> hallucinated_entity."""
        gold = [[]]
        preds = [[_ent("person", "Nobody")]]
        result = analyze_error_types(preds, gold)
        assert result["counts"]["hallucinated_entity"] == 1

    def test_boundary_error_detected(self):
        """Predicted text partially overlaps gold with same type -> boundary_error."""
        gold = [[_ent("loc", "New York City")]]
        preds = [[_ent("loc", "New York")]]
        result = analyze_error_types(preds, gold)
        assert result["counts"]["boundary_error"] == 1

    def test_correct_prediction_no_errors(self):
        """Correctly predicted entity does not generate errors."""
        gold = [[_ent("person", "Alice")]]
        preds = [[_ent("person", "Alice")]]
        result = analyze_error_types(preds, gold)
        assert all(v == 0 for v in result["counts"].values())


# =====================================================================
# Tests for hallucination_rate.py -- compute_hallucination_rate
# =====================================================================

class TestComputeHallucinationRate:
    """Tests for compute_hallucination_rate."""

    def test_no_hallucinations(self):
        """All predicted entities appear in source -> rate=0.0."""
        preds = [[_ent("person", "Alice"), _ent("org", "ACME")]]
        sources = ["Alice works at ACME corporation"]
        result = compute_hallucination_rate(preds, sources)
        assert result["hallucination_rate"] == pytest.approx(0.0)
        assert result["total_entities"] == 2
        assert result["hallucinated_entities"] == 0
        assert result["grounded_entities"] == 2

    def test_all_hallucinations(self):
        """No predicted entity text appears in source -> rate=1.0."""
        preds = [[_ent("person", "Zephyr"), _ent("org", "Nonexistent")]]
        sources = ["Alice works at ACME corporation"]
        result = compute_hallucination_rate(preds, sources)
        assert result["hallucination_rate"] == pytest.approx(1.0)
        assert result["hallucinated_entities"] == 2
        assert result["grounded_entities"] == 0

    def test_mixed_hallucinations(self):
        """Some entities grounded, some hallucinated -> correct rate."""
        preds = [[_ent("person", "Alice"), _ent("person", "Zephyr")]]
        sources = ["Alice works at ACME"]
        result = compute_hallucination_rate(preds, sources)
        assert result["hallucination_rate"] == pytest.approx(0.5)
        assert result["total_entities"] == 2
        assert result["hallucinated_entities"] == 1
        assert result["grounded_entities"] == 1

    def test_case_insensitive_matching_default(self):
        """Default case_sensitive=False means 'alice' matches 'ALICE' in source."""
        preds = [[_ent("person", "alice")]]
        sources = ["ALICE works at ACME"]
        result = compute_hallucination_rate(preds, sources)
        assert result["hallucination_rate"] == pytest.approx(0.0)
        assert result["hallucinated_entities"] == 0

    def test_case_sensitive_matching(self):
        """With case_sensitive=True, 'alice' does NOT match 'ALICE' in source."""
        preds = [[_ent("person", "alice")]]
        sources = ["ALICE works at ACME"]
        result = compute_hallucination_rate(preds, sources, case_sensitive=True)
        assert result["hallucination_rate"] == pytest.approx(1.0)
        assert result["hallucinated_entities"] == 1

    def test_no_predictions(self):
        """No predictions at all -> rate=0.0 (vacuously no hallucinations)."""
        preds = [[]]
        sources = ["Alice works at ACME"]
        result = compute_hallucination_rate(preds, sources)
        assert result["hallucination_rate"] == pytest.approx(0.0)
        assert result["total_entities"] == 0
        assert result["hallucinated_entities"] == 0

    def test_multiple_examples(self):
        """Hallucination rate aggregated across multiple examples."""
        preds = [
            [_ent("person", "Alice")],  # grounded
            [_ent("person", "Bob"), _ent("person", "Zephyr")],  # Bob grounded, Zephyr halluc
        ]
        sources = [
            "Alice works here",
            "Bob and Carol are friends",
        ]
        result = compute_hallucination_rate(preds, sources)
        # 3 total entities, 1 hallucinated
        assert result["total_entities"] == 3
        assert result["hallucinated_entities"] == 1
        assert result["hallucination_rate"] == pytest.approx(1 / 3)

    def test_per_example_rates(self):
        """per_example_rates correctly computed for each example."""
        preds = [
            [_ent("person", "Alice")],  # 0/1 hallucinated
            [_ent("person", "Zephyr"), _ent("person", "Nobody")],  # 2/2 hallucinated
        ]
        sources = [
            "Alice works here",
            "Bob and Carol are friends",
        ]
        result = compute_hallucination_rate(preds, sources)
        assert result["per_example_rates"][0] == pytest.approx(0.0)
        assert result["per_example_rates"][1] == pytest.approx(1.0)

    def test_per_example_rate_zero_for_no_predictions(self):
        """An example with 0 predictions gets per-example rate 0.0."""
        preds = [[]]
        sources = ["Some text"]
        result = compute_hallucination_rate(preds, sources)
        assert result["per_example_rates"][0] == pytest.approx(0.0)

    def test_hallucination_examples_populated(self):
        """The examples field contains details of hallucinated entities."""
        preds = [[_ent("person", "Zephyr")]]
        sources = ["Alice works at ACME"]
        result = compute_hallucination_rate(preds, sources)
        assert len(result["examples"]) == 1
        ex = result["examples"][0]
        assert ex["example_idx"] == 0
        assert ex["entity_type"] == "person"
        assert ex["entity_text"] == "Zephyr"
        assert ex["source_snippet"] == "Alice works at ACME"

    def test_substring_matching(self):
        """Entity text must be a substring of source text (not just word match)."""
        preds = [[_ent("org", "AC")]]
        sources = ["ACME corporation"]
        result = compute_hallucination_rate(preds, sources)
        # "ac" is substring of "acme corporation" (case-insensitive)
        assert result["hallucination_rate"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        """Mismatched predictions/sources lengths should raise AssertionError."""
        with pytest.raises(AssertionError):
            compute_hallucination_rate([[]], [])

    def test_whitespace_stripped_from_entity_text(self):
        """Entity text is stripped before checking against source."""
        preds = [[_ent("person", "  Alice  ")]]
        sources = ["Alice works here"]
        result = compute_hallucination_rate(preds, sources)
        assert result["hallucination_rate"] == pytest.approx(0.0)


# =====================================================================
# Tests for _entity_set helper in error_analysis.py
# =====================================================================

class TestEntitySet:
    """Tests for the _entity_set helper."""

    def test_basic_set(self):
        entities = [_ent("person", "Alice"), _ent("org", "ACME")]
        result = _entity_set(entities)
        assert result == {("person", "Alice"), ("org", "ACME")}

    def test_type_lowercased(self):
        entities = [_ent("PERSON", "Alice")]
        result = _entity_set(entities)
        assert ("person", "Alice") in result

    def test_text_stripped(self):
        entities = [_ent("person", "  Alice  ")]
        result = _entity_set(entities)
        assert ("person", "Alice") in result

    def test_empty_list(self):
        assert _entity_set([]) == set()
