from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class _AwaitableResult:
    """Minimal awaitable used to stand in for Tinker APIFuture in unit tests."""

    def __init__(self, result):
        self._result = result

    def __await__(self):
        async def _coro():
            return self._result

        return _coro().__await__()


class _DummyTrainingClientForSave:
    def __init__(self) -> None:
        self.saved_names: list[str] = []

    def save_state(self, name: str):
        self.saved_names.append(name)
        return _AwaitableResult(SimpleNamespace(path=f"tinker://{name}"))

    async def save_state_async(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("save_state_async() should not be called by the harness")


def test_save_tool_awaits_save_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TINKERER_CHECKPOINT_PREFIX", raising=False)

    from tinker_mcp.state import get_session
    from tinker_mcp import server

    session = get_session()
    session.reset()
    session.training_client = _DummyTrainingClientForSave()

    out = asyncio.run(server.save("ckpt"))

    assert "ADAPTER SAVED" in out
    assert "Name: ckpt" in out
    assert "tinker://ckpt" in out
    assert session.saved_adapters["ckpt"] == "tinker://ckpt"


class _DummyTrainingClientForLoad:
    def __init__(self) -> None:
        self.refresh_names: list[str] = []

    async def save_weights_and_get_sampling_client_async(self, name: str):
        self.refresh_names.append(name)
        return object()


class _DummyServiceClient:
    def __init__(self, training_client: _DummyTrainingClientForLoad) -> None:
        self._training_client = training_client
        self.state_paths: list[str] = []

    async def create_training_client_from_state_with_optimizer_async(self, adapter_path: str):
        self.state_paths.append(adapter_path)
        return self._training_client


def test_load_tool_sanitizes_refresh_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TINKERER_CHECKPOINT_PREFIX", raising=False)

    from tinker_mcp.state import get_session
    from tinker_mcp import server

    dummy_training = _DummyTrainingClientForLoad()
    dummy_service = _DummyServiceClient(dummy_training)

    async def _get_client():
        return dummy_service

    monkeypatch.setattr(server, "get_service_client_async", _get_client)

    session = get_session()
    session.reset()
    # load() requires a session to be initialized; any non-None placeholder is fine.
    session.training_client = object()

    adapter_path = "tinker://run-id/state/foo:bar/baz"
    out = asyncio.run(server.load(adapter_path))

    assert "ADAPTER LOADED" in out
    assert dummy_service.state_paths == [adapter_path]
    assert dummy_training.refresh_names, "Expected a refresh save call to create a sampling client"
    refresh_name = dummy_training.refresh_names[0]
    assert ":" not in refresh_name
    assert "/" not in refresh_name
