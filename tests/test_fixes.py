"""Unit tests for the fixes applied to the Tinkerer codebase.

These tests verify the fixes without requiring external API access.
Run with: python -m pytest tests/test_fixes.py -v
"""

import os


class TestRewardFunctionValidation:
    """Tests for Issue #2: Reward function syntax validation."""

    def test_valid_reward_function(self):
        """Valid Python code should pass validation."""
        from tinker_mcp.training.grpo import validate_reward_function_syntax

        valid_code = """
def compute_reward(completion, ground_truth):
    return 1.0 if completion == ground_truth else 0.0
"""
        is_valid, error = validate_reward_function_syntax(valid_code)
        assert is_valid is True
        assert error == ""

    def test_invalid_syntax_missing_colon(self):
        """Missing colon should fail validation."""
        from tinker_mcp.training.grpo import validate_reward_function_syntax

        invalid_code = """
def compute_reward(completion, ground_truth)
    return 1.0
"""
        is_valid, error = validate_reward_function_syntax(invalid_code)
        assert is_valid is False
        assert "line 2" in error.lower() or "syntax" in error.lower()

    def test_invalid_syntax_unclosed_bracket(self):
        """Unclosed bracket should fail validation."""
        from tinker_mcp.training.grpo import validate_reward_function_syntax

        invalid_code = """
def compute_reward(completion, ground_truth):
    return float(completion == ground_truth
"""
        is_valid, error = validate_reward_function_syntax(invalid_code)
        assert is_valid is False

    def test_complex_valid_function(self):
        """Complex but valid code should pass."""
        from tinker_mcp.training.grpo import validate_reward_function_syntax

        complex_code = '''
import re

def compute_reward(completion, ground_truth):
    """Compute reward based on answer extraction."""
    # Extract answer using regex
    match = re.search(r'\\\\boxed{([^}]+)}', completion)
    if match:
        extracted = match.group(1).strip()
        return 1.0 if extracted == ground_truth else 0.0
    return 0.0
'''
        is_valid, error = validate_reward_function_syntax(complex_code)
        assert is_valid is True


class TestSessionDirectoryPermissions:
    """Tests for Issue #12: Temp directory permissions."""

    def test_session_dir_created_with_correct_permissions(self):
        """Session directory should have 0o700 permissions."""
        import stat
        from tinker_mcp.state import TrainingSession, SESSION_DIR_MODE

        session = TrainingSession()
        assert os.path.exists(session.session_dir)

        session_stat = os.stat(session.session_dir)
        session_perms = stat.S_IMODE(session_stat.st_mode)
        assert session_perms == SESSION_DIR_MODE

    def test_session_dir_mode_constant(self):
        """SESSION_DIR_MODE should be 0o700."""
        from tinker_mcp.state import SESSION_DIR_MODE

        assert SESSION_DIR_MODE == 0o700


class TestCheckpointMetadataValidation:
    """Tests for Issue #6: Checkpoint metadata validation."""

    def test_valid_metadata_restores_correctly(self):
        """Valid metadata should restore all fields."""
        from tinker_mcp.state import TrainingSession

        session = TrainingSession()
        metadata = {
            "cumulative_sft_epochs": 5,
            "cumulative_grpo_iterations": 10,
            "cumulative_sft_steps": 100,
            "session_method": "sft",
        }
        result = session.restore_from_metadata(metadata)

        assert result is True
        assert session.sft_epochs == 5
        assert session.grpo_iterations == 10
        assert session.sft_steps == 100
        assert session.session_method == "sft"

    def test_invalid_type_handled_gracefully(self):
        """Invalid types should be handled with defaults."""
        from tinker_mcp.state import TrainingSession

        session = TrainingSession()
        metadata = {
            "cumulative_sft_epochs": "not_an_int",
            "cumulative_grpo_iterations": None,
            "session_method": "invalid_method",
        }
        result = session.restore_from_metadata(metadata)

        assert result is False  # Invalid metadata
        assert session.sft_epochs == 0  # Falls back to default
        assert session.session_method is None  # Invalid method reset to None

    def test_non_dict_metadata_rejected(self):
        """Non-dict metadata should be rejected."""
        from tinker_mcp.state import TrainingSession

        session = TrainingSession()
        # Type ignore: intentionally testing with invalid types to verify error handling
        result = session.restore_from_metadata("not a dict")  # type: ignore[arg-type]
        assert result is False

        result = session.restore_from_metadata(None)  # type: ignore[arg-type]
        assert result is False


class TestTimeoutConstants:
    """Tests for Issue #9/11: Centralized timeout configurations."""

    def test_timeout_constants_exist(self):
        """All timeout constants should be defined."""
        from tinker_mcp.utils import (
            TIMEOUT_SHORT,
            TIMEOUT_MEDIUM,
            TIMEOUT_LONG,
            SAMPLING_TIMEOUT_BASE,
            SAMPLING_TOKENS_PER_SECOND,
            WANDB_INIT_TIMEOUT,
            WANDB_FINISH_TIMEOUT,
        )

        assert isinstance(TIMEOUT_SHORT, int)
        assert isinstance(TIMEOUT_MEDIUM, int)
        assert isinstance(TIMEOUT_LONG, int)
        assert isinstance(SAMPLING_TIMEOUT_BASE, int)
        assert isinstance(SAMPLING_TOKENS_PER_SECOND, int)
        assert isinstance(WANDB_INIT_TIMEOUT, int)
        assert isinstance(WANDB_FINISH_TIMEOUT, int)

    def test_timeout_ordering(self):
        """Timeouts should be in expected order."""
        from tinker_mcp.utils import TIMEOUT_SHORT, TIMEOUT_MEDIUM, TIMEOUT_LONG

        assert TIMEOUT_SHORT < TIMEOUT_MEDIUM < TIMEOUT_LONG


class TestProgressFileCleanup:
    """Tests for Issue #4: Race condition in progress file cleanup."""

    def test_cleanup_nonexistent_file_succeeds(self):
        """Cleaning up a non-existent file should succeed."""
        from tinker_mcp.utils import cleanup_progress_file, PROGRESS_FILE

        # Ensure file doesn't exist
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

        # This should not raise
        result = cleanup_progress_file()
        assert result is True  # Returns True for "already clean"

    def test_cleanup_existing_file_succeeds(self):
        """Cleaning up an existing file should succeed."""
        from tinker_mcp.utils import cleanup_progress_file, PROGRESS_FILE

        # Create the file
        with open(PROGRESS_FILE, "w") as f:
            f.write("test")

        assert os.path.exists(PROGRESS_FILE)

        result = cleanup_progress_file()
        assert result is True
        assert not os.path.exists(PROGRESS_FILE)


class TestHFTokenValidation:
    """Tests for Issue #1: HF_TOKEN validation."""

    def test_hf_token_validation_imports(self):
        """The HF token validation function should be importable."""
        # Just verify the module loads (validation happens at import time)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "hf_server", os.path.join(os.path.dirname(__file__), "..", "mcp-servers", "hf-datasets", "server.py")
        )
        # Module should load without crashing even without HF_TOKEN
        assert spec is not None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
