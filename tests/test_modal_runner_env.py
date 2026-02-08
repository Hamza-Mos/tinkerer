from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture()
def runner_module():
    """Load the Modal runner module by path without relying on package imports."""
    root = Path(__file__).resolve().parents[1]
    path = root / "modal" / "run_tinkerer.py"
    spec = importlib.util.spec_from_file_location("tinkerer_run_tinkerer", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_claude_env_omits_empty_wandb_vars(monkeypatch: pytest.MonkeyPatch, runner_module):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_PROJECT", "")
    monkeypatch.setenv("WANDB_ENTITY", "")
    monkeypatch.setenv("WANDB_GROUP", "")
    monkeypatch.setenv("WANDB_RUN_ID", "")
    monkeypatch.setenv("WANDB_RESUME", "")
    monkeypatch.setenv("WANDB_MODE", "")

    env = runner_module._build_claude_env()

    # Empty strings should NOT be present; wandb can fail if these exist as "".
    assert "WANDB_MODE" not in env
    assert "WANDB_RUN_ID" not in env
    assert "WANDB_RESUME" not in env


def test_build_codex_env_omits_empty_wandb_vars(monkeypatch: pytest.MonkeyPatch, runner_module):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_PROJECT", "")
    monkeypatch.setenv("WANDB_ENTITY", "")
    monkeypatch.setenv("WANDB_GROUP", "")
    monkeypatch.setenv("WANDB_RUN_ID", "")
    monkeypatch.setenv("WANDB_RESUME", "")
    monkeypatch.setenv("WANDB_MODE", "")

    env = runner_module._build_codex_env()

    assert "WANDB_MODE" not in env
    assert "WANDB_RUN_ID" not in env
    assert "WANDB_RESUME" not in env
