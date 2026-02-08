"""Tests for GRPO learning rate scheduling.

Verifies that the get_lr function correctly implements warmup and decay
schedules for stable GRPO training convergence.
"""

import pytest

from tinker_mcp.utils import get_lr


class TestGetLR:
    """Tests for the get_lr learning rate scheduling function."""

    def test_constant_schedule_no_warmup(self):
        """Constant schedule should return base_lr for all steps."""
        base_lr = 1e-5
        for step in range(10):
            lr = get_lr(step=step, total_steps=10, base_lr=base_lr, warmup_ratio=0.0, scheduler="constant")
            assert lr == base_lr, f"Step {step}: expected {base_lr}, got {lr}"

    def test_constant_schedule_with_warmup(self):
        """Constant schedule with warmup should ramp up then stay flat."""
        base_lr = 1e-4
        total_steps = 100
        warmup_ratio = 0.1  # 10 warmup steps

        # During warmup (steps 0-9), LR should increase
        warmup_lrs = [
            get_lr(step=i, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="constant")
            for i in range(10)
        ]

        # Verify warmup is increasing
        for i in range(1, len(warmup_lrs)):
            assert warmup_lrs[i] > warmup_lrs[i - 1], f"Warmup should be increasing at step {i}"

        # After warmup, should be constant at base_lr
        for step in range(10, 100):
            lr = get_lr(
                step=step, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="constant"
            )
            assert lr == base_lr, f"Post-warmup step {step}: expected {base_lr}, got {lr}"

    def test_linear_decay(self):
        """Linear schedule should decay from base_lr to 0."""
        base_lr = 1e-4
        total_steps = 100

        # At start, should be base_lr
        lr_start = get_lr(step=0, total_steps=total_steps, base_lr=base_lr, warmup_ratio=0.0, scheduler="linear")
        assert lr_start == base_lr

        # At end, should be near 0
        lr_end = get_lr(step=99, total_steps=total_steps, base_lr=base_lr, warmup_ratio=0.0, scheduler="linear")
        assert lr_end < base_lr * 0.1, f"End LR should be near 0, got {lr_end}"

        # Should be monotonically decreasing
        lrs = [
            get_lr(step=i, total_steps=total_steps, base_lr=base_lr, warmup_ratio=0.0, scheduler="linear")
            for i in range(100)
        ]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1], f"Linear decay should be decreasing at step {i}"

    def test_cosine_decay(self):
        """Cosine schedule should smoothly decay from base_lr to 0."""
        base_lr = 1e-4
        total_steps = 100

        # At start, should be base_lr
        lr_start = get_lr(step=0, total_steps=total_steps, base_lr=base_lr, warmup_ratio=0.0, scheduler="cosine")
        assert lr_start == base_lr

        # At end, should be near 0
        lr_end = get_lr(step=99, total_steps=total_steps, base_lr=base_lr, warmup_ratio=0.0, scheduler="cosine")
        assert lr_end < base_lr * 0.1, f"End LR should be near 0, got {lr_end}"

        # Should be monotonically decreasing
        lrs = [
            get_lr(step=i, total_steps=total_steps, base_lr=base_lr, warmup_ratio=0.0, scheduler="cosine")
            for i in range(100)
        ]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1], f"Cosine decay should be decreasing at step {i}"

    def test_cosine_with_warmup(self):
        """Cosine schedule with warmup should ramp up then decay."""
        base_lr = 1e-4
        total_steps = 100
        warmup_ratio = 0.1  # 10 warmup steps

        # During warmup, should be increasing
        warmup_lrs = [
            get_lr(step=i, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="cosine")
            for i in range(10)
        ]

        for i in range(1, len(warmup_lrs)):
            assert warmup_lrs[i] > warmup_lrs[i - 1], f"Warmup should increase at step {i}"

        # At warmup end, should be at base_lr
        lr_at_warmup_end = get_lr(
            step=10, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="cosine"
        )
        assert (
            abs(lr_at_warmup_end - base_lr) < 1e-8
        ), f"At warmup end, LR should be base_lr={base_lr}, got {lr_at_warmup_end}"

        # After warmup, should be decreasing
        post_warmup_lrs = [
            get_lr(step=i, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="cosine")
            for i in range(10, 100)
        ]

        for i in range(1, len(post_warmup_lrs)):
            assert post_warmup_lrs[i] <= post_warmup_lrs[i - 1], f"Post-warmup should decay at step {i + 10}"

    def test_decay_schedules_do_not_oscillate_past_horizon(self):
        """Linear/cosine schedules should clamp once progress reaches 100%.

        This guards against step > total_steps oscillation when a fixed horizon
        is used across repeated calls.
        """
        base_lr = 1e-4
        total_steps = 20
        for scheduler in ["linear", "cosine"]:
            lr_at_horizon = get_lr(
                step=total_steps, total_steps=total_steps, base_lr=base_lr, warmup_ratio=0.0, scheduler=scheduler
            )
            lr_far_past = get_lr(
                step=total_steps * 10,
                total_steps=total_steps,
                base_lr=base_lr,
                warmup_ratio=0.0,
                scheduler=scheduler,
            )
            assert lr_far_past <= lr_at_horizon + 1e-12

    def test_warmup_ratio_zero_skips_warmup(self):
        """With warmup_ratio=0, should start at base_lr."""
        base_lr = 1e-4
        lr = get_lr(step=0, total_steps=100, base_lr=base_lr, warmup_ratio=0.0, scheduler="cosine")
        assert lr == base_lr, f"With no warmup, step 0 should be base_lr={base_lr}, got {lr}"

    def test_unknown_scheduler_falls_back_to_constant(self):
        """Unknown scheduler should fall back to constant."""
        base_lr = 1e-4
        lr = get_lr(step=50, total_steps=100, base_lr=base_lr, warmup_ratio=0.0, scheduler="unknown_scheduler")
        assert lr == base_lr, f"Unknown scheduler should return base_lr, got {lr}"

    def test_warmup_formula(self):
        """Verify warmup uses correct linear formula."""
        base_lr = 1e-4
        total_steps = 100
        warmup_ratio = 0.1  # 10 warmup steps

        # Step 0 should be base_lr * 1/10
        lr_step0 = get_lr(
            step=0, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="constant"
        )
        expected_step0 = base_lr * (0 + 1) / 10
        assert abs(lr_step0 - expected_step0) < 1e-10, f"Step 0 warmup: expected {expected_step0}, got {lr_step0}"

        # Step 4 should be base_lr * 5/10
        lr_step4 = get_lr(
            step=4, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="constant"
        )
        expected_step4 = base_lr * (4 + 1) / 10
        assert abs(lr_step4 - expected_step4) < 1e-10, f"Step 4 warmup: expected {expected_step4}, got {lr_step4}"

        # Step 9 should be base_lr * 10/10 = base_lr
        lr_step9 = get_lr(
            step=9, total_steps=total_steps, base_lr=base_lr, warmup_ratio=warmup_ratio, scheduler="constant"
        )
        expected_step9 = base_lr * (9 + 1) / 10
        assert abs(lr_step9 - expected_step9) < 1e-10, f"Step 9 warmup: expected {expected_step9}, got {lr_step9}"

    def test_grpo_typical_usage(self):
        """Test typical GRPO usage pattern with warmup on first call only."""
        base_lr = 1e-5  # GRPO code task

        # First call: with warmup
        # (simulating single iteration with warmup_ratio=0.1)
        lr_first = get_lr(step=0, total_steps=1, base_lr=base_lr, warmup_ratio=0.1, scheduler="cosine")
        # With only 1 step and warmup_ratio=0.1, warmup_steps=0, so no warmup
        # LR should be base_lr
        assert lr_first == base_lr, f"First call LR should be {base_lr}, got {lr_first}"

        # For multi-iteration call (e.g., num_iterations=5)
        # First iteration should have some warmup, last should have decay
        lrs_multi = [
            get_lr(step=i, total_steps=5, base_lr=base_lr, warmup_ratio=0.1, scheduler="cosine") for i in range(5)
        ]

        # Should not have any 0 LRs (which would stop training)
        assert all(lr > 0 for lr in lrs_multi), f"All LRs should be > 0, got {lrs_multi}"

    def test_edge_case_single_step(self):
        """Single step should return base_lr regardless of schedule."""
        base_lr = 1e-4

        for scheduler in ["constant", "linear", "cosine"]:
            lr = get_lr(step=0, total_steps=1, base_lr=base_lr, warmup_ratio=0.0, scheduler=scheduler)
            assert lr == base_lr, f"Single step with {scheduler} should return base_lr={base_lr}, got {lr}"


class TestLRSchedulingIntegration:
    """Integration tests for LR scheduling in GRPO context."""

    def test_lr_never_zero_during_training(self):
        """LR should never hit exactly 0 during training (would halt
        learning)."""
        base_lr = 1e-5
        total_steps = 25  # Typical GRPO training

        for scheduler in ["constant", "linear", "cosine"]:
            for warmup_ratio in [0.0, 0.1, 0.2]:
                for step in range(total_steps):
                    lr = get_lr(
                        step=step,
                        total_steps=total_steps,
                        base_lr=base_lr,
                        warmup_ratio=warmup_ratio,
                        scheduler=scheduler,
                    )
                    assert lr > 0, f"LR should be > 0: scheduler={scheduler}, warmup={warmup_ratio}, step={step}"

    def test_recommended_grpo_settings(self):
        """Test the recommended GRPO settings from the prompt templates."""
        # MBPP recommended settings
        mbpp_lr = 1e-5
        warmup_ratio = 0.1

        # First call with warmup
        lrs_first_call = [
            get_lr(step=i, total_steps=5, base_lr=mbpp_lr, warmup_ratio=warmup_ratio, scheduler="cosine")
            for i in range(5)
        ]

        # Subsequent calls without warmup
        lrs_subsequent = [
            get_lr(step=i, total_steps=5, base_lr=mbpp_lr, warmup_ratio=0.0, scheduler="cosine") for i in range(5)
        ]

        # Both should have reasonable LRs
        assert all(
            0 < lr <= mbpp_lr for lr in lrs_first_call
        ), f"First call LRs should be in (0, {mbpp_lr}], got {lrs_first_call}"
        assert all(
            0 < lr <= mbpp_lr for lr in lrs_subsequent
        ), f"Subsequent LRs should be in (0, {mbpp_lr}], got {lrs_subsequent}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
