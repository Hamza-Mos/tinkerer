from __future__ import annotations

import asyncio
import sys
import types

import pytest


class _DummyRun:
    def __init__(self):
        self.url = "https://wandb.example/runs/abc123"
        self.summary = types.SimpleNamespace(update=lambda: None)

    def finish(self, exit_code: int = 0) -> None:
        return None

    def log(self, *_args, **_kwargs) -> None:
        return None


class _DummyWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.init_calls: list[dict] = []
        self.define_metric_calls: list[tuple[tuple, dict]] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return _DummyRun()

    def define_metric(self, *args, **kwargs):
        self.define_metric_calls.append((args, kwargs))


def test_maybe_init_wandb_run_skips_when_no_key(monkeypatch: pytest.MonkeyPatch):
    from tinker_mcp.training.common import maybe_init_wandb_run

    session = types.SimpleNamespace(wandb_run=None)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    status = asyncio.run(
        maybe_init_wandb_run(
            session=session,
            method="grpo",
            base_model="Qwen/Qwen3-8B-Base",
            wandb_project="tinkerer",
            wandb_run_name=None,
            group_default="group",
            name_components=["g4", "r32"],
            step_metric="iteration",
            config={"lora_rank": 32, "group_size": 4},
            extra_tags=None,
            do_debug=False,
            log_paths={},
            timeout_s=1.0,
        )
    )
    assert status == ""
    assert session.wandb_run is None


def test_maybe_init_wandb_run_omits_empty_advanced_env_vars(monkeypatch: pytest.MonkeyPatch):
    from tinker_mcp.training.common import maybe_init_wandb_run

    dummy = _DummyWandb()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    # Minimal required env.
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("TINKERER_CHECKPOINT_PREFIX", "claude")

    # Problematic empties (historically caused pydantic validation failures).
    monkeypatch.setenv("WANDB_RUN_ID", "")
    monkeypatch.setenv("WANDB_RESUME", "")
    monkeypatch.setenv("WANDB_MODE", "")

    session = types.SimpleNamespace(wandb_run=None)
    status = asyncio.run(
        maybe_init_wandb_run(
            session=session,
            method="grpo",
            base_model="Qwen/Qwen3-8B-Base",
            wandb_project="tinkerer",
            wandb_run_name=None,
            group_default="group",
            name_components=["g4", "r32"],
            step_metric="iteration",
            config={"lora_rank": 32, "group_size": 4},
            extra_tags=None,
            do_debug=False,
            log_paths={},
            timeout_s=1.0,
        )
    )

    assert session.wandb_run is not None
    assert "W&B:" in status
    assert len(dummy.init_calls) == 1

    call = dummy.init_calls[0]
    assert call["project"] == "tinkerer"
    assert call["group"] == "group"
    assert call["job_type"] == "grpo"
    assert call["config"]["method"] == "grpo"
    assert call["config"]["agent_prefix"] == "claude"
    assert "id" not in call
    assert "resume" not in call
    assert "mode" not in call


def test_maybe_init_wandb_run_ignores_resume_without_id(monkeypatch: pytest.MonkeyPatch):
    from tinker_mcp.training.common import maybe_init_wandb_run

    dummy = _DummyWandb()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("TINKERER_CHECKPOINT_PREFIX", "claude")

    # Misconfiguration: resume set, but id missing/empty.
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("WANDB_RUN_ID", "")

    session = types.SimpleNamespace(wandb_run=None)
    _ = asyncio.run(
        maybe_init_wandb_run(
            session=session,
            method="grpo",
            base_model="Qwen/Qwen3-8B-Base",
            wandb_project="tinkerer",
            wandb_run_name=None,
            group_default="group",
            name_components=["g4", "r32"],
            step_metric="iteration",
            config={"lora_rank": 32, "group_size": 4},
            extra_tags=None,
            do_debug=False,
            log_paths={},
            timeout_s=1.0,
        )
    )

    call = dummy.init_calls[0]
    assert "id" not in call
    assert "resume" not in call


def test_maybe_init_wandb_run_ignores_unresolved_env_placeholders(monkeypatch: pytest.MonkeyPatch):
    """MCP configs sometimes pass literal '${VAR}' strings when VAR is unset."""
    from tinker_mcp.training.common import maybe_init_wandb_run

    dummy = _DummyWandb()
    monkeypatch.setitem(sys.modules, "wandb", dummy)

    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("TINKERER_CHECKPOINT_PREFIX", "claude")
    monkeypatch.setenv("WANDB_ENTITY", "${WANDB_ENTITY}")

    session = types.SimpleNamespace(wandb_run=None)
    _ = asyncio.run(
        maybe_init_wandb_run(
            session=session,
            method="grpo",
            base_model="Qwen/Qwen3-8B-Base",
            wandb_project="tinkerer",
            wandb_run_name=None,
            group_default="group",
            name_components=["g4", "r32"],
            step_metric="iteration",
            config={"lora_rank": 32, "group_size": 4},
            extra_tags=None,
            do_debug=False,
            log_paths={},
            timeout_s=1.0,
        )
    )

    call = dummy.init_calls[0]
    assert call.get("entity") is None
