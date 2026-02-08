"""Unit tests for common.py validation helpers.

Tests the strict type validation pattern used for API response handling.
"""

import pytest

from tinker_mcp.training.common import (
    extract_metrics,
    validate_reward_output,
    validate_lr_scheduler_name,
    resolve_scheduler_total_steps,
)


class TestExtractMetrics:
    """Tests for extract_metrics() - validates ForwardBackwardOutput.metrics."""

    def test_none_returns_empty_dict(self):
        """None metrics should return empty dict (optional field)."""
        result = extract_metrics(None, context="test")
        assert result == {}

    def test_dict_returns_dict(self):
        """Valid dict should be returned as-is."""
        metrics = {"kl_sample_train_v1": 0.005, "kl_sample_train_v2": 0.003}
        result = extract_metrics(metrics, context="test")
        assert result == metrics

    def test_empty_dict_returns_empty_dict(self):
        """Empty dict should be returned as-is."""
        result = extract_metrics({}, context="test")
        assert result == {}

    def test_list_raises_type_error(self):
        """List should raise TypeError with context."""
        with pytest.raises(TypeError) as excinfo:
            extract_metrics([1, 2, 3], context="GRPO forward_backward")
        assert "GRPO forward_backward" in str(excinfo.value)
        assert "list" in str(excinfo.value)
        assert "dict" in str(excinfo.value)

    def test_string_raises_type_error(self):
        """String should raise TypeError."""
        with pytest.raises(TypeError) as excinfo:
            extract_metrics("not a dict", context="test")
        assert "str" in str(excinfo.value)

    def test_int_raises_type_error(self):
        """Integer should raise TypeError."""
        with pytest.raises(TypeError) as excinfo:
            extract_metrics(42, context="test")
        assert "int" in str(excinfo.value)


class TestValidateRewardOutput:
    """Tests for validate_reward_output() - validates reward subprocess JSON."""

    def test_valid_output_single_completion(self):
        """Valid output with one completion."""
        output = {"rewards": [0.5], "errors": [], "error_count": 0}
        result = validate_reward_output(output, num_completions=1, context="test")
        assert result["rewards"] == [0.5]
        assert result["errors"] == []
        assert result["error_count"] == 0

    def test_valid_output_multiple_completions(self):
        """Valid output with multiple completions."""
        output = {"rewards": [0.0, 0.5, 1.0], "errors": [], "error_count": 0}
        result = validate_reward_output(output, num_completions=3, context="test")
        assert result["rewards"] == [0.0, 0.5, 1.0]

    def test_valid_output_with_errors(self):
        """Valid output with error tracking."""
        output = {
            "rewards": [0.5, 0.0],
            "errors": [{"index": 1, "error": "ValueError", "msg": "test error"}],
            "error_count": 1,
        }
        result = validate_reward_output(output, num_completions=2, context="test")
        assert result["rewards"] == [0.5, 0.0]
        assert len(result["errors"]) == 1
        assert result["error_count"] == 1

    def test_missing_optional_fields_defaults(self):
        """Missing errors/error_count should default to empty/0."""
        output = {"rewards": [0.75]}
        result = validate_reward_output(output, num_completions=1, context="test")
        assert result["rewards"] == [0.75]
        assert result["errors"] == []
        assert result["error_count"] == 0

    def test_list_raises_type_error(self):
        """List output should raise TypeError."""
        with pytest.raises(TypeError) as excinfo:
            validate_reward_output([0.5, 0.75], num_completions=2, context="compute_rewards_batch")
        assert "compute_rewards_batch" in str(excinfo.value)
        assert "list" in str(excinfo.value)
        assert "dict" in str(excinfo.value)

    def test_none_raises_type_error(self):
        """None output should raise TypeError."""
        with pytest.raises(TypeError) as excinfo:
            validate_reward_output(None, num_completions=1, context="test")
        assert "NoneType" in str(excinfo.value)

    def test_missing_rewards_raises_key_error(self):
        """Missing 'rewards' key should raise KeyError."""
        with pytest.raises(KeyError) as excinfo:
            validate_reward_output({"error_count": 0}, num_completions=1, context="test")
        assert "rewards" in str(excinfo.value)

    def test_rewards_not_list_raises_type_error(self):
        """Non-list 'rewards' should raise TypeError."""
        with pytest.raises(TypeError) as excinfo:
            validate_reward_output({"rewards": 0.5}, num_completions=1, context="test")
        assert "list" in str(excinfo.value)

    def test_rewards_length_mismatch_raises_value_error(self):
        """Wrong rewards length should raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            validate_reward_output({"rewards": [0.5, 0.75]}, num_completions=3, context="test")
        assert "2" in str(excinfo.value)
        assert "3" in str(excinfo.value)

    def test_invalid_errors_type_defaults_to_empty(self):
        """Non-list 'errors' should default to empty list."""
        output = {"rewards": [0.5], "errors": "not a list", "error_count": 0}
        result = validate_reward_output(output, num_completions=1, context="test")
        assert result["errors"] == []

    def test_invalid_error_count_type_defaults_to_zero(self):
        """Non-numeric 'error_count' should default to 0."""
        output = {"rewards": [0.5], "errors": [], "error_count": "one"}
        result = validate_reward_output(output, num_completions=1, context="test")
        assert result["error_count"] == 0

    def test_float_error_count_converts_to_int(self):
        """Float 'error_count' should be converted to int."""
        output = {"rewards": [0.5, 0.0], "errors": [], "error_count": 1.0}
        result = validate_reward_output(output, num_completions=2, context="test")
        assert result["error_count"] == 1
        assert isinstance(result["error_count"], int)


class TestSchedulerValidation:
    """Tests for scheduler name and horizon resolution helpers."""

    def test_validate_lr_scheduler_name(self):
        assert validate_lr_scheduler_name("constant") is None
        assert validate_lr_scheduler_name("linear") is None
        assert validate_lr_scheduler_name("cosine") is None
        err = validate_lr_scheduler_name("bad_scheduler")
        assert err is not None
        assert "Invalid lr_scheduler" in err

    def test_resolve_scheduler_total_steps_constant(self):
        total, warning = resolve_scheduler_total_steps(
            scheduler="constant",
            explicit_total_steps=None,
            cumulative_steps=5,
            steps_this_call=1,
            persisted_total_steps=0,
            default_floor_steps=20,
            unit_name="iteration",
        )
        assert total == 6
        assert warning is None

    def test_resolve_scheduler_total_steps_single_step_cosine_uses_floor(self):
        total, warning = resolve_scheduler_total_steps(
            scheduler="cosine",
            explicit_total_steps=None,
            cumulative_steps=1,
            steps_this_call=1,
            persisted_total_steps=0,
            default_floor_steps=20,
            unit_name="iteration",
        )
        assert total == 20
        assert warning is not None
        assert "scheduler_total_steps=20" in warning

    def test_resolve_scheduler_total_steps_preserves_persisted_horizon(self):
        total, warning = resolve_scheduler_total_steps(
            scheduler="linear",
            explicit_total_steps=None,
            cumulative_steps=10,
            steps_this_call=1,
            persisted_total_steps=40,
            default_floor_steps=20,
            unit_name="iteration",
        )
        assert total == 40
        assert warning is not None

    def test_resolve_scheduler_total_steps_rejects_too_small_explicit(self):
        with pytest.raises(ValueError):
            resolve_scheduler_total_steps(
                scheduler="cosine",
                explicit_total_steps=5,
                cumulative_steps=5,
                steps_this_call=1,
                persisted_total_steps=0,
                default_floor_steps=20,
                unit_name="iteration",
            )
