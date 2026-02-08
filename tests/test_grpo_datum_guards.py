"""Tests for GRPO datum construction guards."""


def test_is_trainable_completion_rejects_short_sequences():
    """Completions shorter than 2 tokens cannot build shifted GRPO datums."""
    from tinker_mcp.training.grpo import _is_trainable_completion

    assert _is_trainable_completion([]) is False
    assert _is_trainable_completion([123]) is False


def test_is_trainable_completion_accepts_two_plus_tokens():
    """Two or more completion tokens are trainable."""
    from tinker_mcp.training.grpo import _is_trainable_completion

    assert _is_trainable_completion([123, 456]) is True
    assert _is_trainable_completion([1, 2, 3]) is True
