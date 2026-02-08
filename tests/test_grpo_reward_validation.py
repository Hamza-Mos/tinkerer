"""Tests for GRPO reward function syntax + contract validation."""


def _validate(code: str):
    from tinker_mcp.training.grpo import validate_reward_function_syntax

    return validate_reward_function_syntax(code)


def test_reward_validation_accepts_minimal_valid_function():
    is_valid, err = _validate(
        """
def compute_reward(completion, ground_truth):
    return 1.0
""".strip()
    )
    assert is_valid is True
    assert err == ""


def test_reward_validation_rejects_missing_compute_reward():
    is_valid, err = _validate("x = 1")
    assert is_valid is False
    assert "Missing required function" in err


def test_reward_validation_rejects_async_compute_reward():
    is_valid, err = _validate(
        """
async def compute_reward(completion, ground_truth):
    return 1.0
""".strip()
    )
    assert is_valid is False
    assert "regular function" in err


def test_reward_validation_rejects_extra_required_positional_args():
    is_valid, err = _validate(
        """
def compute_reward(completion, ground_truth, weight):
    return 1.0
""".strip()
    )
    assert is_valid is False
    assert "more than two positional args" in err


def test_reward_validation_rejects_required_keyword_only_args():
    is_valid, err = _validate(
        """
def compute_reward(completion, ground_truth, *, weight):
    return 1.0
""".strip()
    )
    assert is_valid is False
    assert "required keyword-only args" in err


def test_reward_validation_allows_optional_extra_positional_arg():
    is_valid, err = _validate(
        """
def compute_reward(completion, ground_truth, weight=1.0):
    return weight
""".strip()
    )
    assert is_valid is True
    assert err == ""


def test_reward_validation_rejects_duplicate_compute_reward_defs():
    is_valid, err = _validate(
        """
def compute_reward(completion, ground_truth):
    return 1.0

def compute_reward(completion, ground_truth):
    return 0.0
""".strip()
    )
    assert is_valid is False
    assert "exactly one" in err
