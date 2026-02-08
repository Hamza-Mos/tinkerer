"""Tests for singleton-session tool-call locking."""

import asyncio


def test_tool_call_lock_serializes_access():
    from tinker_mcp.state import release_tool_call_lock, try_acquire_tool_call_lock

    first_acquired = asyncio.run(try_acquire_tool_call_lock(0.1))
    assert first_acquired is True

    second_acquired = False
    try:
        second_acquired = asyncio.run(try_acquire_tool_call_lock(0.01))
        assert second_acquired is False
    finally:
        # Release once for first holder.
        release_tool_call_lock()
        # Defensive release if test unexpectedly acquired twice.
        if second_acquired:
            release_tool_call_lock()

    third_acquired = asyncio.run(try_acquire_tool_call_lock(0.1))
    assert third_acquired is True
    release_tool_call_lock()
