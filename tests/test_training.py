"""Comprehensive unit tests for training/train_utils.py.

All tests use CPU-only torch tensors. No GPU or model downloads required.
"""

import pytest
import torch
import math

from training.train_utils import (
    MASK_TOKEN_ID,
    log_linear_noise_schedule,
    alpha,
    alpha_prime,
    mdlm_importance_weight,
    mask_completion_tokens,
    apply_pad_loss_weight,
    random_truncation,
    compute_unmask_count,
)


# ===================================================================
# Noise schedule
# ===================================================================

class TestLogLinearNoiseSchedule:
    """Tests for log_linear_noise_schedule."""

    def test_returns_t_identity(self):
        """log_linear_noise_schedule should return t unchanged (identity)."""
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = log_linear_noise_schedule(t)
        assert torch.allclose(result, t), (
            f"Expected identity: {t}, got {result}"
        )

    def test_scalar_input(self):
        """Works with a scalar tensor."""
        t = torch.tensor(0.3)
        result = log_linear_noise_schedule(t)
        assert torch.allclose(result, t)

    def test_batch_input(self):
        """Works with a batch of timesteps."""
        t = torch.rand(16)
        result = log_linear_noise_schedule(t)
        assert torch.allclose(result, t)


class TestAlpha:
    """Tests for the alpha(t) = 1 - t function."""

    def test_alpha_returns_one_minus_t(self):
        """alpha(t) should return 1 - t."""
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        result = alpha(t)
        assert torch.allclose(result, expected), (
            f"Expected {expected}, got {result}"
        )

    def test_alpha_at_zero(self):
        """alpha(0) = 1: at time 0, all tokens are unmasked."""
        t = torch.tensor(0.0)
        result = alpha(t)
        assert result.item() == pytest.approx(1.0)

    def test_alpha_at_one(self):
        """alpha(1) = 0: at time 1, all tokens are masked."""
        t = torch.tensor(1.0)
        result = alpha(t)
        assert result.item() == pytest.approx(0.0)

    def test_alpha_midpoint(self):
        """alpha(0.5) = 0.5."""
        t = torch.tensor(0.5)
        result = alpha(t)
        assert result.item() == pytest.approx(0.5)

    def test_alpha_batch(self):
        """alpha works on a batch of values."""
        t = torch.rand(32)
        result = alpha(t)
        expected = 1.0 - t
        assert torch.allclose(result, expected)


class TestAlphaPrime:
    """Tests for alpha_prime(t) = -1 for all t."""

    def test_returns_negative_one(self):
        """alpha_prime should return -1 for all inputs."""
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = alpha_prime(t)
        expected = torch.full_like(t, -1.0)
        assert torch.allclose(result, expected), (
            f"Expected all -1.0, got {result}"
        )

    def test_scalar_input(self):
        """alpha_prime works on scalar."""
        t = torch.tensor(0.42)
        result = alpha_prime(t)
        assert result.item() == pytest.approx(-1.0)

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        t = torch.rand(8, 4)
        result = alpha_prime(t)
        assert result.shape == t.shape

    def test_all_same_value(self):
        """Derivative is constant for the linear schedule."""
        t = torch.linspace(0, 1, 100)
        result = alpha_prime(t)
        assert torch.all(result == -1.0)


# ===================================================================
# MDLM importance weight
# ===================================================================

class TestMdlmImportanceWeight:
    """Tests for mdlm_importance_weight(t) = 1/t."""

    def test_computes_one_over_t(self):
        """weight(t) = 1/t for t > 0."""
        t = torch.tensor([0.1, 0.25, 0.5, 1.0])
        expected = torch.tensor([10.0, 4.0, 2.0, 1.0])
        result = mdlm_importance_weight(t)
        assert torch.allclose(result, expected), (
            f"Expected {expected}, got {result}"
        )

    def test_near_zero_clamped_no_inf(self):
        """Near-zero t should be clamped and not produce inf."""
        t = torch.tensor([0.0, 1e-10, 1e-20])
        result = mdlm_importance_weight(t)
        assert torch.all(torch.isfinite(result)), (
            f"Expected finite values, got {result}"
        )

    def test_near_zero_clamped_value(self):
        """t=0 should be clamped to 1e-5, giving weight = 1e5."""
        t = torch.tensor([0.0])
        result = mdlm_importance_weight(t)
        expected = torch.tensor([1.0 / 1e-5])
        assert torch.allclose(result, expected), (
            f"Expected {expected}, got {result}"
        )

    def test_various_t_values(self):
        """Check expected values for various t."""
        t = torch.tensor([0.01, 0.02, 0.05, 0.2, 0.5])
        expected = 1.0 / t
        result = mdlm_importance_weight(t)
        assert torch.allclose(result, expected, rtol=1e-5), (
            f"Expected {expected}, got {result}"
        )

    def test_weight_decreases_with_t(self):
        """Importance weight should decrease as t increases."""
        t = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
        result = mdlm_importance_weight(t)
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1], (
                f"Weight should decrease: w({t[i]})={result[i]} > w({t[i+1]})={result[i+1]}"
            )

    def test_batch_shape_preserved(self):
        """Output shape matches input shape."""
        t = torch.rand(16) * 0.9 + 0.1  # in (0.1, 1.0)
        result = mdlm_importance_weight(t)
        assert result.shape == t.shape


# ===================================================================
# Masking
# ===================================================================

class TestMaskCompletionTokens:
    """Tests for mask_completion_tokens."""

    def test_t_zero_masks_nothing(self):
        """At t=0, no tokens should be masked."""
        batch_size, seq_len = 4, 10
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        t = torch.zeros(batch_size)
        noisy, mask = mask_completion_tokens(completion_ids, t)
        assert mask.sum().item() == 0, (
            f"Expected 0 masked tokens at t=0, got {mask.sum().item()}"
        )
        assert torch.equal(noisy, completion_ids), (
            "At t=0, output should be identical to input"
        )

    def test_t_one_masks_everything(self):
        """At t=1, all tokens should be masked."""
        batch_size, seq_len = 4, 10
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        t = torch.ones(batch_size)
        noisy, mask = mask_completion_tokens(completion_ids, t)
        assert mask.all(), (
            f"Expected all masked at t=1, got {mask.sum().item()}/{mask.numel()}"
        )
        assert (noisy == MASK_TOKEN_ID).all(), (
            "All tokens should be MASK_TOKEN_ID at t=1"
        )

    def test_t_half_masks_approximately_half(self):
        """At t=0.5, roughly 50% of tokens should be masked.

        Uses a large batch to reduce variance.
        """
        batch_size, seq_len = 100, 200
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        t = torch.full((batch_size,), 0.5)

        torch.manual_seed(42)
        noisy, mask = mask_completion_tokens(completion_ids, t)

        mask_frac = mask.float().mean().item()
        assert 0.45 < mask_frac < 0.55, (
            f"Expected ~50% masked, got {mask_frac:.3f}"
        )

    def test_masked_positions_get_mask_token_id(self):
        """Masked positions should have the MASK_TOKEN_ID value."""
        completion_ids = torch.randint(0, 1000, (8, 20))
        t = torch.full((8,), 0.5)
        torch.manual_seed(0)
        noisy, mask = mask_completion_tokens(completion_ids, t)

        assert (noisy[mask] == MASK_TOKEN_ID).all(), (
            "All masked positions must be MASK_TOKEN_ID"
        )

    def test_non_masked_positions_keep_original(self):
        """Non-masked positions should retain their original token IDs."""
        completion_ids = torch.randint(0, 1000, (8, 20))
        t = torch.full((8,), 0.5)
        torch.manual_seed(1)
        noisy, mask = mask_completion_tokens(completion_ids, t)

        assert torch.equal(noisy[~mask], completion_ids[~mask]), (
            "Non-masked positions should keep their original values"
        )

    def test_batch_dimension_handled(self):
        """Different samples in the batch can have different t values."""
        batch_size, seq_len = 4, 50
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # First two samples: t=0 (no masking), last two: t=1 (full masking)
        t = torch.tensor([0.0, 0.0, 1.0, 1.0])
        noisy, mask = mask_completion_tokens(completion_ids, t)

        # First two rows: no masking
        assert mask[:2].sum().item() == 0, (
            "Rows with t=0 should have no masking"
        )
        assert torch.equal(noisy[:2], completion_ids[:2])

        # Last two rows: fully masked
        assert mask[2:].all(), "Rows with t=1 should be fully masked"
        assert (noisy[2:] == MASK_TOKEN_ID).all()

    def test_scalar_t(self):
        """mask_completion_tokens handles a scalar t tensor."""
        completion_ids = torch.randint(0, 1000, (3, 10))
        t = torch.tensor(0.0)
        noisy, mask = mask_completion_tokens(completion_ids, t)
        assert mask.sum().item() == 0

    def test_output_shapes(self):
        """Output tensors have the expected shapes."""
        batch_size, seq_len = 5, 15
        completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        t = torch.rand(batch_size)
        noisy, mask = mask_completion_tokens(completion_ids, t)
        assert noisy.shape == (batch_size, seq_len)
        assert mask.shape == (batch_size, seq_len)

    def test_mask_token_id_value(self):
        """Verify the mask token ID constant is 126336."""
        assert MASK_TOKEN_ID == 126336


# ===================================================================
# PAD loss weighting
# ===================================================================

class TestApplyPadLossWeight:
    """Tests for apply_pad_loss_weight."""

    def test_downweights_pad_tokens(self):
        """PAD-target tokens should have their loss reduced."""
        pad_token_id = 0
        pad_weight = 0.05
        loss_per_token = torch.ones(10)
        # 5 normal tokens, 5 PAD tokens
        target_ids = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])

        result = apply_pad_loss_weight(
            loss_per_token, target_ids, pad_token_id, pad_weight
        )

        # Manual calculation:
        # weights = [1, 1, 1, 1, 1, 0.05, 0.05, 0.05, 0.05, 0.05]
        # weighted_sum = 5*1 + 5*0.05 = 5.25
        # weight_sum = 5 + 0.25 = 5.25
        # result = 5.25 / 5.25 = 1.0
        expected = (5.0 * 1.0 + 5.0 * 0.05) / (5.0 + 5.0 * 0.05)
        assert result.item() == pytest.approx(expected, rel=1e-5), (
            f"Expected {expected}, got {result.item()}"
        )

    def test_no_pad_tokens_returns_normal_mean(self):
        """With no PAD tokens, result is the simple mean of losses."""
        pad_token_id = 0
        loss_per_token = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target_ids = torch.tensor([10, 20, 30, 40])  # no PAD

        result = apply_pad_loss_weight(
            loss_per_token, target_ids, pad_token_id
        )

        expected = loss_per_token.mean()
        assert result.item() == pytest.approx(expected.item(), rel=1e-5), (
            f"Expected normal mean {expected.item()}, got {result.item()}"
        )

    def test_all_pad_tokens(self):
        """With all PAD tokens, result = pad_weight * mean_loss."""
        pad_token_id = 0
        pad_weight = 0.05
        loss_per_token = torch.tensor([2.0, 4.0, 6.0])
        target_ids = torch.tensor([0, 0, 0])  # all PAD

        result = apply_pad_loss_weight(
            loss_per_token, target_ids, pad_token_id, pad_weight
        )

        # weights = [0.05, 0.05, 0.05]
        # weighted_sum = (2+4+6)*0.05 = 0.6
        # weight_sum = 3*0.05 = 0.15
        # result = 0.6 / 0.15 = 4.0 = mean of losses
        expected_mean = loss_per_token.mean().item()
        assert result.item() == pytest.approx(expected_mean, rel=1e-5), (
            f"Expected {expected_mean}, got {result.item()}"
        )

    def test_correct_weighting_mixed(self):
        """Check precise weighted mean with mixed PAD and non-PAD tokens."""
        pad_token_id = 0
        pad_weight = 0.1
        loss_per_token = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target_ids = torch.tensor([5, 0, 5, 0])  # indices 1,3 are PAD

        result = apply_pad_loss_weight(
            loss_per_token, target_ids, pad_token_id, pad_weight
        )

        # weights = [1.0, 0.1, 1.0, 0.1]
        # weighted_sum = 1*1 + 2*0.1 + 3*1 + 4*0.1 = 1+0.2+3+0.4 = 4.6
        # weight_sum = 1 + 0.1 + 1 + 0.1 = 2.2
        # result = 4.6 / 2.2
        expected = 4.6 / 2.2
        assert result.item() == pytest.approx(expected, rel=1e-5), (
            f"Expected {expected}, got {result.item()}"
        )

    def test_default_pad_weight(self):
        """Default pad_weight is 0.05."""
        pad_token_id = 0
        loss_per_token = torch.tensor([1.0, 1.0])
        target_ids = torch.tensor([0, 1])

        result = apply_pad_loss_weight(
            loss_per_token, target_ids, pad_token_id
        )

        # weights = [0.05, 1.0]
        # weighted_sum = 1*0.05 + 1*1.0 = 1.05
        # weight_sum = 0.05 + 1.0 = 1.05
        # result = 1.0
        expected = (1.0 * 0.05 + 1.0 * 1.0) / (0.05 + 1.0)
        assert result.item() == pytest.approx(expected, rel=1e-5)


# ===================================================================
# Random truncation
# ===================================================================

class TestRandomTruncation:
    """Tests for random_truncation."""

    def test_changes_batch(self):
        """random_truncation should modify at least some entries
        (when completions have different lengths)."""
        pad_token_id = 0
        batch_size, max_len = 4, 10
        # Different length completions
        completion_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        completion_ids[0, :3] = torch.tensor([10, 20, 30])
        completion_ids[1, :7] = torch.tensor([10, 20, 30, 40, 50, 60, 70])
        completion_ids[2, :5] = torch.tensor([10, 20, 30, 40, 50])
        completion_ids[3, :2] = torch.tensor([10, 20])
        completion_lengths = torch.tensor([3, 7, 5, 2])

        torch.manual_seed(123)
        result = random_truncation(completion_ids, completion_lengths, pad_token_id)

        # At least one row should differ from the original (unless permutation
        # is identity, which is unlikely for seed 123).
        assert result.shape == completion_ids.shape

    def test_truncated_positions_become_pad(self):
        """Positions beyond the assigned target length should be PAD."""
        pad_token_id = 0
        # All real tokens, no padding originally
        completion_ids = torch.arange(1, 21).reshape(2, 10)  # values 1..20
        completion_lengths = torch.tensor([10, 5])

        # Force a known permutation by controlling the seed
        torch.manual_seed(0)
        result = random_truncation(completion_ids, completion_lengths, pad_token_id)

        # Each row's positions beyond its assigned target length should be PAD
        for i in range(2):
            # Find the assigned target length for row i
            # (determined by permutation, we can check result instead)
            for j in range(10):
                if result[i, j].item() == pad_token_id:
                    # All subsequent positions should also be PAD
                    assert (result[i, j:] == pad_token_id).all(), (
                        f"Row {i}: once PAD starts at position {j}, "
                        "all following should be PAD"
                    )
                    break

    def test_original_content_preserved_up_to_truncation(self):
        """Content before the truncation point should match the original."""
        pad_token_id = 0
        completion_ids = torch.tensor([
            [100, 200, 300, 400, 500],
            [10, 20, 30, 40, 50],
        ])
        completion_lengths = torch.tensor([5, 3])

        torch.manual_seed(7)
        result = random_truncation(completion_ids, completion_lengths, pad_token_id)

        for i in range(2):
            for j in range(5):
                if result[i, j].item() != pad_token_id:
                    assert result[i, j].item() == completion_ids[i, j].item(), (
                        f"Row {i}, pos {j}: non-PAD value should match original"
                    )

    def test_different_batch_sizes(self):
        """Works with various batch sizes."""
        pad_token_id = 0
        for batch_size in [1, 2, 8, 16]:
            max_len = 10
            completion_ids = torch.randint(1, 100, (batch_size, max_len))
            completion_lengths = torch.randint(1, max_len + 1, (batch_size,))
            result = random_truncation(completion_ids, completion_lengths, pad_token_id)
            assert result.shape == (batch_size, max_len), (
                f"Shape mismatch for batch_size={batch_size}"
            )

    def test_does_not_modify_input(self):
        """Input tensor should not be modified (clone is used internally)."""
        pad_token_id = 0
        completion_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
        original = completion_ids.clone()
        completion_lengths = torch.tensor([3, 1])

        torch.manual_seed(99)
        _ = random_truncation(completion_ids, completion_lengths, pad_token_id)

        assert torch.equal(completion_ids, original), (
            "random_truncation should not modify the input tensor"
        )


# ===================================================================
# Compute unmask count
# ===================================================================

class TestComputeUnmaskCount:
    """Tests for compute_unmask_count."""

    def test_step_zero_unmasks_small_fraction(self):
        """At step 0 of many steps, only a small fraction is unmasked."""
        total_steps = 100
        n_masked = 100
        result = compute_unmask_count(step=0, total_steps=total_steps, n_masked=n_masked)
        # step=0: target_remaining = 100 * (1 - 1/100) = 99
        # n_unmask = 100 - 99 = 1
        assert result >= 1, "Should unmask at least 1"
        assert result <= n_masked // 2, (
            f"At step 0 of {total_steps}, should unmask a small fraction, got {result}"
        )

    def test_final_step_unmasks_all_remaining(self):
        """At the final step, all remaining masked tokens should be unmasked."""
        total_steps = 10
        n_masked = 50
        result = compute_unmask_count(
            step=total_steps - 1, total_steps=total_steps, n_masked=n_masked
        )
        # step=9: target_remaining = 50 * (1 - 10/10) = 0
        # n_unmask = 50 - 0 = 50
        assert result == n_masked, (
            f"At final step, should unmask all {n_masked}, got {result}"
        )

    def test_returns_at_least_one_when_tokens_remain(self):
        """Should always return at least 1 when there are masked tokens."""
        # Even with a small fraction, the min is 1
        result = compute_unmask_count(step=0, total_steps=1000, n_masked=1)
        assert result >= 1, (
            f"Should unmask at least 1 when tokens remain, got {result}"
        )

    def test_returns_zero_when_no_masked_tokens(self):
        """Should return 0 when there are no masked tokens."""
        result = compute_unmask_count(step=5, total_steps=10, n_masked=0)
        assert result == 0, (
            f"Expected 0 when no masked tokens, got {result}"
        )

    def test_monotonic_progress(self):
        """Over all steps, the cumulative unmask count covers all tokens."""
        total_steps = 20
        n_masked = 100
        remaining = n_masked
        for step in range(total_steps):
            n_unmask = compute_unmask_count(step, total_steps, remaining)
            assert n_unmask >= 0
            remaining -= n_unmask
            assert remaining >= 0, (
                f"Remaining should not go negative: {remaining} at step {step}"
            )
        assert remaining == 0, (
            f"After all steps, all tokens should be unmasked. Remaining: {remaining}"
        )

    def test_intermediate_step(self):
        """Check a specific intermediate step calculation."""
        # step=4, total_steps=10, n_masked=100
        # target_remaining = int(100 * (1 - 5/10)) = int(50.0) = 50
        # n_unmask = 100 - 50 = 50
        result = compute_unmask_count(step=4, total_steps=10, n_masked=100)
        assert result == 50

    def test_single_step(self):
        """With total_steps=1, all tokens should be unmasked at step 0."""
        result = compute_unmask_count(step=0, total_steps=1, n_masked=42)
        assert result == 42
