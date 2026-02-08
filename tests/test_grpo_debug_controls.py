"""Tests for GRPO sampling debug control helpers."""


def _helpers():
    from tinker_mcp.training.grpo import (
        _resolve_auto_checkpoint_policy,
        _format_exception,
        _resolve_sampling_debug_prompt_limit,
        _should_log_sampling_debug,
        AUTOCHECKPOINT_MIN_ITERATIONS_ENV,
        AUTOCHECKPOINT_REWARD_THRESHOLD_ENV,
        DEFAULT_AUTOCHECKPOINT_MIN_ITERATIONS,
        DEFAULT_AUTOCHECKPOINT_REWARD_THRESHOLD,
        DEFAULT_SAMPLING_DEBUG_PROMPT_LIMIT,
        SAMPLING_DEBUG_PROMPT_LIMIT_ENV,
    )

    return (
        _resolve_auto_checkpoint_policy,
        _format_exception,
        _resolve_sampling_debug_prompt_limit,
        _should_log_sampling_debug,
        AUTOCHECKPOINT_MIN_ITERATIONS_ENV,
        AUTOCHECKPOINT_REWARD_THRESHOLD_ENV,
        DEFAULT_AUTOCHECKPOINT_MIN_ITERATIONS,
        DEFAULT_AUTOCHECKPOINT_REWARD_THRESHOLD,
        DEFAULT_SAMPLING_DEBUG_PROMPT_LIMIT,
        SAMPLING_DEBUG_PROMPT_LIMIT_ENV,
    )


def test_resolve_sampling_debug_limit_default(monkeypatch):
    """Default limit should be used when env/arg are unset."""
    (
        _resolve_auto_checkpoint_policy,
        _format_exception,
        _resolve_sampling_debug_prompt_limit,
        _should_log_sampling_debug,
        auto_min_env,
        auto_threshold_env,
        default_auto_min,
        default_auto_threshold,
        default_limit,
        env_name,
    ) = _helpers()
    assert _resolve_auto_checkpoint_policy
    assert auto_min_env
    assert auto_threshold_env
    assert default_auto_min >= 0
    assert default_auto_threshold >= 0
    assert _format_exception  # Keep import tuple aligned.
    assert _should_log_sampling_debug

    monkeypatch.delenv(env_name, raising=False)
    assert _resolve_sampling_debug_prompt_limit(None) == default_limit


def test_resolve_sampling_debug_limit_from_env(monkeypatch):
    """Env var should control limit when argument is not provided."""
    _, _, _resolve_sampling_debug_prompt_limit, _, _, _, _, _, _, env_name = _helpers()
    monkeypatch.setenv(env_name, "-1")
    assert _resolve_sampling_debug_prompt_limit(None) == -1


def test_resolve_sampling_debug_limit_arg_overrides_env(monkeypatch):
    """Explicit argument should override env var."""
    _, _, _resolve_sampling_debug_prompt_limit, _, _, _, _, _, _, env_name = _helpers()
    monkeypatch.setenv(env_name, "100")
    assert _resolve_sampling_debug_prompt_limit(2) == 2


def test_resolve_sampling_debug_limit_rejects_invalid_env(monkeypatch):
    """Invalid env values should fail fast."""
    import pytest

    _, _, _resolve_sampling_debug_prompt_limit, _, _, _, _, _, _, env_name = _helpers()
    monkeypatch.setenv(env_name, "not-an-int")
    with pytest.raises(ValueError):
        _resolve_sampling_debug_prompt_limit(None)


def test_resolve_sampling_debug_limit_rejects_too_small():
    """Values below -1 should be rejected."""
    import pytest

    _, _, _resolve_sampling_debug_prompt_limit, _, _, _, _, _, _, _ = _helpers()
    with pytest.raises(ValueError):
        _resolve_sampling_debug_prompt_limit(-2)


def test_should_log_sampling_debug_limit_behavior():
    """Helper should enforce debug gating semantics."""
    _, _, _, _should_log_sampling_debug, _, _, _, _, _, _ = _helpers()

    assert _should_log_sampling_debug(do_debug=False, prompt_index=0, sampling_debug_prompt_limit=-1) is False
    assert _should_log_sampling_debug(do_debug=True, prompt_index=50, sampling_debug_prompt_limit=-1) is True
    assert _should_log_sampling_debug(do_debug=True, prompt_index=0, sampling_debug_prompt_limit=3) is True
    assert _should_log_sampling_debug(do_debug=True, prompt_index=2, sampling_debug_prompt_limit=3) is True
    assert _should_log_sampling_debug(do_debug=True, prompt_index=3, sampling_debug_prompt_limit=3) is False
    assert _should_log_sampling_debug(do_debug=True, prompt_index=0, sampling_debug_prompt_limit=0) is False


def test_format_exception_handles_empty_message():
    """Exception formatter should include type even for empty str(exc)."""
    _, _format_exception, _, _, _, _, _, _, _, _ = _helpers()

    class EmptyError(Exception):
        def __str__(self):
            return ""

    assert _format_exception(EmptyError()) == "EmptyError"
    assert _format_exception(ValueError("bad input")) == "ValueError: bad input"


def test_resolve_auto_checkpoint_policy_default(monkeypatch):
    """Auto-checkpoint policy should use defaults when env/args are unset."""
    (
        _resolve_auto_checkpoint_policy,
        _format_exception,
        _resolve_sampling_debug_prompt_limit,
        _should_log_sampling_debug,
        auto_min_env,
        auto_threshold_env,
        default_auto_min,
        default_auto_threshold,
        default_limit,
        env_name,
    ) = _helpers()
    assert _format_exception
    assert _resolve_sampling_debug_prompt_limit
    assert _should_log_sampling_debug
    assert default_limit >= -1
    assert env_name
    monkeypatch.delenv(auto_threshold_env, raising=False)
    monkeypatch.delenv(auto_min_env, raising=False)
    threshold, min_iters = _resolve_auto_checkpoint_policy(None, None)
    assert threshold == default_auto_threshold
    assert min_iters == default_auto_min


def test_resolve_auto_checkpoint_policy_from_env(monkeypatch):
    """Env vars should drive auto-checkpoint policy when args are unset."""
    (
        _resolve_auto_checkpoint_policy,
        _format_exception,
        _resolve_sampling_debug_prompt_limit,
        _should_log_sampling_debug,
        auto_min_env,
        auto_threshold_env,
        default_auto_min,
        default_auto_threshold,
        default_limit,
        env_name,
    ) = _helpers()
    assert _format_exception
    assert _resolve_sampling_debug_prompt_limit
    assert _should_log_sampling_debug
    assert default_auto_min >= 0
    assert default_auto_threshold >= 0
    assert default_limit >= -1
    assert env_name
    monkeypatch.setenv(auto_threshold_env, "0.55")
    monkeypatch.setenv(auto_min_env, "7")
    threshold, min_iters = _resolve_auto_checkpoint_policy(None, None)
    assert threshold == 0.55
    assert min_iters == 7


def test_resolve_auto_checkpoint_policy_arg_overrides_env(monkeypatch):
    """Explicit args should override environment values."""
    (
        _resolve_auto_checkpoint_policy,
        _format_exception,
        _resolve_sampling_debug_prompt_limit,
        _should_log_sampling_debug,
        auto_min_env,
        auto_threshold_env,
        default_auto_min,
        default_auto_threshold,
        default_limit,
        env_name,
    ) = _helpers()
    assert _format_exception
    assert _resolve_sampling_debug_prompt_limit
    assert _should_log_sampling_debug
    assert default_auto_min >= 0
    assert default_auto_threshold >= 0
    assert default_limit >= -1
    assert env_name
    monkeypatch.setenv(auto_threshold_env, "0.9")
    monkeypatch.setenv(auto_min_env, "99")
    threshold, min_iters = _resolve_auto_checkpoint_policy(0.2, 3)
    assert threshold == 0.2
    assert min_iters == 3


def test_resolve_auto_checkpoint_policy_rejects_invalid_env(monkeypatch):
    """Malformed env values should fail fast."""
    import pytest

    (
        _resolve_auto_checkpoint_policy,
        _format_exception,
        _resolve_sampling_debug_prompt_limit,
        _should_log_sampling_debug,
        auto_min_env,
        auto_threshold_env,
        default_auto_min,
        default_auto_threshold,
        default_limit,
        env_name,
    ) = _helpers()
    assert _format_exception
    assert _resolve_sampling_debug_prompt_limit
    assert _should_log_sampling_debug
    assert default_auto_min >= 0
    assert default_auto_threshold >= 0
    assert default_limit >= -1
    assert env_name
    monkeypatch.setenv(auto_threshold_env, "bad")
    with pytest.raises(ValueError):
        _resolve_auto_checkpoint_policy(None, None)

    monkeypatch.setenv(auto_threshold_env, "0.4")
    monkeypatch.setenv(auto_min_env, "-1")
    with pytest.raises(ValueError):
        _resolve_auto_checkpoint_policy(None, None)
