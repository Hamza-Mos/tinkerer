"""Tests for W&B cleanup policy on recoverable vs fatal errors."""

import types


class _Session:
    def __init__(self):
        self.wandb_run = types.SimpleNamespace(finish=lambda exit_code=0: None)


def test_cleanup_wandb_keeps_run_open_on_recoverable_error():
    """Recoverable errors should not close W&B runs."""
    from tinker_mcp.training.common import cleanup_wandb_on_error

    session = _Session()
    cleanup_wandb_on_error(session, do_debug=False, log_paths={}, fatal=False, reason="retryable failure")
    assert session.wandb_run is not None


def test_cleanup_wandb_finishes_run_on_fatal_error():
    """Fatal errors should close W&B runs and clear session handle."""
    from tinker_mcp.training.common import cleanup_wandb_on_error

    calls = {"finish": 0, "exit_code": None}

    session = _Session()
    session.wandb_run = types.SimpleNamespace(
        finish=lambda exit_code=0: calls.update({"finish": calls["finish"] + 1, "exit_code": exit_code})
    )
    cleanup_wandb_on_error(session, do_debug=False, log_paths={}, fatal=True, reason="fatal init failure")

    assert calls["finish"] == 1
    assert calls["exit_code"] == 1
    assert session.wandb_run is None
