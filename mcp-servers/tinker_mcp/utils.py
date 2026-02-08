"""Utility functions for the Tinker MCP Server.

This module provides:
- Debug logging with file output
- Progress tracking for external monitoring (Modal watchdog)
- Learning rate scheduling with warmup and decay
- Timeout constants for Tinker API calls
"""

import asyncio
import json
import math
import os
import sys
import threading
import time
from importlib import metadata
from tinker_mcp.state import get_session

# Re-export resolve_api_future for convenience
# Note: write_progress_sync is intentionally not in __all__ (internal use only)
__all__ = [
    "DEBUG_MODE",
    "resolve_api_future",
    "get_sample_sequences",
    "get_checkpoint_prefix",
    "apply_checkpoint_prefix",
    "get_log_paths",
    "debug_log",
    "clear_debug_logs",
    "write_progress",
    "get_lr",
    "TIMEOUT_SHORT",
    "TIMEOUT_MEDIUM",
    "TIMEOUT_LONG",
    "SAMPLING_TIMEOUT_BASE",
    "SAMPLING_TOKENS_PER_SECOND",
    "WANDB_INIT_TIMEOUT",
    "WANDB_FINISH_TIMEOUT",
    "validate_environment",
    "get_tinker_client",
    "get_service_client_async",
    "cleanup_progress_file",
    "cleanup_stale_progress_files",
    "log_warning",
    "log_error",
]


# =============================================================================
# Debug Logging Configuration
# =============================================================================

# Debug mode from environment
DEBUG_MODE = os.environ.get("TINKERER_DEBUG", "0") == "1"
CHECKPOINT_PREFIX_ENV = "TINKERER_CHECKPOINT_PREFIX"

# Thread-safe lock for progress file writes
_progress_lock = threading.Lock()


def get_log_paths() -> dict:
    """Get paths for debug log files.

    Returns dict with log file paths in the session directory.
    """
    session = get_session()
    session_dir = session.session_dir

    return {
        "init": os.path.join(session_dir, "init_debug.log"),
        "training": os.path.join(session_dir, "training_debug.log"),
        "sampling": os.path.join(session_dir, "sampling_debug.log"),
        "reward": os.path.join(session_dir, "reward_debug.log"),
        "grading_function": os.path.join(session_dir, "grading_function.py"),
        "grading_validation": os.path.join(session_dir, "grading_validation.log"),
        "wandb": os.path.join(session_dir, "wandb_debug.log"),
        "save_load": os.path.join(session_dir, "save_load_debug.log"),
        # Session-scoped progress heartbeat (human-readable).
        # Updated alongside PROGRESS_FILE to aid debugging and watchdog recovery.
        "progress": os.path.join(session_dir, "progress.txt"),
        # Session-scoped progress in machine-readable JSONL (one event per line).
        # Useful for dashboards/post-processing; best-effort only.
        "progress_jsonl": os.path.join(session_dir, "progress.jsonl"),
    }


def get_checkpoint_prefix() -> str:
    """Return a sanitized checkpoint prefix from environment, or empty string.

    Uses `TINKERER_CHECKPOINT_PREFIX` and keeps only `[A-Za-z0-9_-]`.
    """
    raw = (os.environ.get(CHECKPOINT_PREFIX_ENV, "") or "").strip()
    if not raw:
        return ""
    sanitized = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in raw).strip("_")
    return sanitized.lower()


def apply_checkpoint_prefix(name: str) -> str:
    """Apply environment checkpoint prefix to `name` when present.

    No-op when prefix is unset or name already starts with `<prefix>_`.
    """
    prefix = get_checkpoint_prefix()
    if not prefix or not name:
        return name
    with_sep = f"{prefix}_"
    if name.startswith(with_sep):
        return name
    return f"{with_sep}{name}"


# Progress file for external monitoring (Modal watchdog can check this)
# Use PID to avoid collisions between concurrent processes in the same container
# IMPORTANT: modal runner watchdog scans /tmp/tinkerer_progress*.txt patterns
PROGRESS_FILE = f"/tmp/tinkerer_progress_{os.getpid()}.txt"


def debug_log(log_file: str, msg: str, force: bool = False) -> None:
    """Write debug message to specified log file if debug mode is on.

    Thread-safe: Uses _progress_lock to prevent concurrent write corruption.

    Args:
        log_file: Path to the log file
        msg: Message to write
        force: If True, write even if debug mode is off (for critical info)
    """
    if not (DEBUG_MODE or force):
        return

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _progress_lock:
        try:
            with open(log_file, "a") as f:
                f.write(f"[{timestamp}] {msg}\n")
                f.flush()  # Explicit flush to prevent log loss on crash
        except Exception as e:
            print(f"Debug log error: {e}", file=sys.stderr, flush=True)


def _log_message(msg: str, level: str, log_file: str | None = None) -> None:
    """Internal helper for logging messages to stderr and debug file."""
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {level}: {msg}"
    print(formatted, file=sys.stderr, flush=True)

    if log_file is None:
        try:
            log_paths = get_log_paths()
            log_file = log_paths["training"]
        except Exception:
            return  # Session not initialized, just print to stderr

    # Type guard: log_file is guaranteed to be str here
    if log_file is None:
        return

    with _progress_lock:
        try:
            with open(log_file, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg}\n")
                f.flush()
        except Exception:
            pass  # Best effort, already printed to stderr


def log_warning(msg: str, log_file: str | None = None) -> None:
    """Log warning message to both stderr and debug file.

    NEVER silent.
    """
    _log_message(msg, "WARNING", log_file)


def log_error(msg: str, log_file: str | None = None) -> None:
    """Log error message to both stderr and debug file.

    NEVER silent.
    """
    _log_message(msg, "ERROR", log_file)


def clear_debug_logs() -> None:
    """Clear all debug log files at session start.

    NOTE: PROGRESS_FILE is intentionally NOT cleared here because it's used
    for watchdog heartbeat monitoring. The progress file is written to at
    init time and must persist to prevent watchdog timeout during slow
    model loading operations.
    """
    log_paths = get_log_paths()
    log_files = [
        log_paths["init"],
        log_paths["training"],
        log_paths["sampling"],
        log_paths["reward"],
        log_paths["grading_function"],
        log_paths["grading_validation"],
        log_paths["wandb"],
        log_paths["save_load"],
        # PROGRESS_FILE intentionally excluded - needed for watchdog heartbeat.
        # log_paths["progress"] intentionally excluded too: it's created before
        # slow init operations and used for post-mortem inspection.
    ]
    for log_file in log_files:
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
        except Exception as e:
            print(f"Warning: Failed to clear log file {log_file}: {e}", file=sys.stderr, flush=True)


# =============================================================================
# Progress Tracking
# =============================================================================


def write_progress_sync(msg: str, include_timestamp: bool = True) -> None:
    """Write progress message to file (synchronous, thread-safe).

    Modal's watchdog monitors stdout from Claude CLI, but can't see
    MCP server's stdout. This file provides an alternative heartbeat.

    Args:
        msg: Progress message to write
        include_timestamp: Whether to prefix with timestamp
    """
    timestamp = time.strftime("%H:%M:%S")
    progress_msg = f"[{timestamp}] {msg}" if include_timestamp else msg

    with _progress_lock:
        try:
            with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
                f.write(progress_msg + "\n")
            # Also write to a session-scoped progress file for easy debugging.
            log_paths = get_log_paths()
            with open(log_paths["progress"], "a", encoding="utf-8") as f:
                f.write(progress_msg + "\n")
            # Best-effort: write structured progress as JSONL for easier tooling.
            try:
                event = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "unix": time.time(),
                    "pid": os.getpid(),
                    "message": msg,
                }
                with open(log_paths["progress_jsonl"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=True) + "\n")
            except Exception:
                pass
        except Exception as e:
            # Use stderr to avoid breaking MCP JSON-RPC protocol on stdout
            print(f"Warning: Could not write progress: {e}", file=sys.stderr, flush=True)

    # NOTE: Do NOT print to stdout in MCP servers!
    # MCP uses stdout for JSON-RPC communication. Any non-JSON output
    # will cause "Unexpected number in JSON" parsing errors.
    # Progress is written to file and debug log for external monitoring.


async def write_progress(msg: str, include_timestamp: bool = True) -> None:
    """Write progress message to file for external monitoring (async wrapper).

    Args:
        msg: Progress message to write
        include_timestamp: Whether to prefix with timestamp
    """
    await asyncio.to_thread(write_progress_sync, msg, include_timestamp)


async def wait_with_heartbeat(
    awaitable,
    *,
    timeout: float,
    progress_msg: str,
    interval: float = 60.0,
    debug_log_file: str | None = None,
) -> object:
    """Await an async operation while periodically updating the progress
    heartbeat.

    This prevents the Modal watchdog from killing long-running API calls that
    do not emit intermediate logs (e.g., model provisioning, saving weights).

    Args:
        awaitable: Awaitable to execute.
        timeout: Total timeout in seconds for the operation.
        progress_msg: Message to write to the progress file periodically.
        interval: Heartbeat interval in seconds.
        debug_log_file: Optional debug log file to append heartbeat info.

    Returns:
        Result of the awaitable if it completes within timeout.

    Raises:
        asyncio.TimeoutError: If the operation exceeds the timeout.
    """
    start = time.time()
    task = asyncio.create_task(awaitable)
    try:
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                task.cancel()
                raise asyncio.TimeoutError()
            step = min(interval, remaining)
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=step)
            except asyncio.TimeoutError:
                elapsed = time.time() - start
                await write_progress(f"{progress_msg} (elapsed {elapsed:.0f}s)")
                if debug_log_file:
                    debug_log(debug_log_file, f"[INFO] {progress_msg} (elapsed {elapsed:.0f}s)", force=True)
    finally:
        if task.done():
            _ = task.exception() if task.cancelled() else None


# =============================================================================
# Learning Rate Scheduling
# =============================================================================


def get_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float, scheduler: str) -> float:
    """Compute learning rate with warmup and decay scheduling.

    Args:
        step: Current step (0-indexed)
        total_steps: Total training steps
        base_lr: Base learning rate
        warmup_ratio: Fraction of steps for linear warmup (e.g., 0.1 = 10%)
        scheduler: "constant", "linear", or "cosine"

    Returns:
        Learning rate for current step
    """
    total_steps = max(int(total_steps), 1)
    warmup_steps = int(total_steps * warmup_ratio)

    if step < warmup_steps:
        # Linear warmup from 0 to base_lr
        return base_lr * (step + 1) / max(warmup_steps, 1)

    # After warmup, apply decay schedule
    if scheduler == "constant":
        return base_lr
    elif scheduler == "linear":
        # Linear decay from base_lr to 0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return base_lr * (1 - progress)
    elif scheduler == "cosine":
        # Cosine decay from base_lr to 0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # Unknown scheduler, fall back to constant
        return base_lr


# =============================================================================
# Timeout Constants
# =============================================================================

# Timeout settings for Tinker API calls (in seconds)
# Large models (8B+) require longer timeouts for loading, sampling, and checkpointing
# NOTE: Keep these below WATCHDOG_KILL_TIMEOUT (2100s) to leave buffer for error propagation
#
# All timeout constants are centralized here for easy tuning and consistency.
# If you need to adjust timeouts, modify these values rather than
# hardcoding elsewhere.

TIMEOUT_SHORT = 300  # 5 min - sampling, simple ops
TIMEOUT_MEDIUM = 900  # 15 min - training steps
TIMEOUT_LONG = 1500  # 25 min - checkpoints, large batches

# Sampling timeout scaling for GRPO
# Used to dynamically adjust sampling timeout based on max_tokens
SAMPLING_TIMEOUT_BASE = 60  # API overhead in seconds
SAMPLING_TOKENS_PER_SECOND = 80  # Conservative estimate for token generation rate

# W&B operation timeouts
WANDB_INIT_TIMEOUT = 60  # Timeout for wandb.init()
# Timeout for wandb.finish() - increased for large artifacts
WANDB_FINISH_TIMEOUT = 120


# =============================================================================
# Tinker ApiFuture Resolution
# =============================================================================


def resolve_api_future(future, timeout: float | None = None):
    """Resolve a Tinker ApiFuture to get the actual response.

    **CRITICAL**: All Tinker API methods (sample, forward_backward, optim_step, etc.)
    return ApiFuture objects. You MUST call .result() to get the actual response.

    From Tinker docs:
        future = sampling_client.sample(prompt=prompt, sampling_params=params)
        result = future.result()  # Required!

    Args:
        future: The ApiFuture from a Tinker API call
        timeout: Optional timeout for future resolution (seconds)

    Returns:
        The resolved response object

    Raises:
        Exception: If future resolution fails
    """
    if timeout:
        return future.result(timeout=timeout)
    return future.result()


def get_sample_sequences(sample_response, log_file: str | None = None):
    """Return the list of sampled sequences from a Tinker SampleResponse.

    Args:
        sample_response: The SampleResponse from Tinker API
        log_file: Optional log file path for debug logging

    Returns:
        List of SampledSequence objects.
    """
    if sample_response is None:
        raise ValueError("Sampling failed: SampleResponse is None.")
    # Tinker training/sampling docs use .sequences; SDK 0.10.0+ exposes it.
    # Enforce .sequences only (no fallbacks) to match the documented behavior.
    if hasattr(sample_response, "sequences"):
        return sample_response.sequences
    attrs = [a for a in dir(sample_response) if not a.startswith("_")]
    if log_file:
        debug_log(
            log_file,
            f"[ERROR] SampleResponse missing required .sequences field. " f"Available attrs: {attrs[:10]}",
            force=True,
        )
    raise ValueError(
        "Sampling failed: SampleResponse missing required .sequences field. "
        "This indicates an SDK mismatch. Please upgrade to the documented Tinker SDK version."
    )


# =============================================================================
# Environment Validation
# =============================================================================


MIN_RUNTIME_VERSIONS = {
    "torch": (2, 10, 0),
    "transformers": (4, 56, 2),
}


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse a version string into an integer tuple (best effort).

    Examples:
        2.10.0 -> (2, 10, 0)
        2.10.0+cu121 -> (2, 10, 0)
        4.56.2rc1 -> (4, 56, 2)
    """
    core = version_str.split("+", 1)[0]
    parsed: list[int] = []
    for part in core.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parsed.append(int(digits))
    return tuple(parsed)


def _version_at_least(installed: str, minimum: tuple[int, ...]) -> bool:
    """Return True if installed version is >= minimum."""
    installed_tuple = _parse_version_tuple(installed)
    if not installed_tuple:
        return False

    max_len = max(len(installed_tuple), len(minimum))
    installed_padded = installed_tuple + (0,) * (max_len - len(installed_tuple))
    minimum_padded = minimum + (0,) * (max_len - len(minimum))
    return installed_padded >= minimum_padded


def validate_environment() -> None:
    """Validate runtime compatibility with required APIs."""
    if not os.environ.get("TINKER_API_KEY"):
        print("WARNING: TINKER_API_KEY not set.", file=sys.stderr)

    errors: list[str] = []

    # Tinker SDK contract validation: SampleResponse.sequences is required.
    try:
        from tinker import types as tinker_types

        sample_fields = getattr(tinker_types.SampleResponse, "model_fields", {})
        if "sequences" not in sample_fields:
            errors.append(
                "Incompatible tinker SDK: SampleResponse.sequences is missing. "
                "Install the pinned SDK from requirements/pyproject."
            )
    except Exception as exc:
        errors.append(f"Could not validate tinker SDK compatibility: {type(exc).__name__}: {exc}")

    # Runtime dependency validation: fail-fast on missing/incompatible core deps.
    for pkg, min_version in MIN_RUNTIME_VERSIONS.items():
        try:
            installed_version = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing required runtime dependency: {pkg}>={'.'.join(map(str, min_version))}")
            continue
        except Exception as exc:
            errors.append(f"Could not read {pkg} version: {type(exc).__name__}: {exc}")
            continue

        if not _version_at_least(installed_version, min_version):
            errors.append(
                f"Incompatible {pkg} version: found {installed_version}, "
                f"require >= {'.'.join(map(str, min_version))}"
            )

    # Renderer API validation: renderer-first harness depends on these symbols.
    try:
        from tinker_cookbook import model_info as tinker_model_info
        from tinker_cookbook import renderers as renderer_helpers
        from tinker_mcp.models import validate_renderer_mappings, validate_renderer_override_consistency

        required_renderer_attrs = ("get_renderer", "get_text_content")
        missing_renderer_attrs = [name for name in required_renderer_attrs if not hasattr(renderer_helpers, name)]
        if missing_renderer_attrs:
            errors.append(
                "Incompatible renderer dependency: missing renderers API(s): " + ", ".join(missing_renderer_attrs)
            )

        if not hasattr(tinker_model_info, "get_recommended_renderer_name"):
            errors.append("Incompatible renderer dependency: missing model_info.get_recommended_renderer_name")

        unmapped_models = validate_renderer_mappings()
        if unmapped_models:
            preview = ", ".join(unmapped_models[:5])
            suffix = "" if len(unmapped_models) <= 5 else f" (+{len(unmapped_models) - 5} more)"
            errors.append(f"Renderer mapping missing for models: {preview}{suffix}")

        override_mismatches = validate_renderer_override_consistency()
        if override_mismatches:
            preview = ", ".join(
                f"{model}={configured} (recommended={recommended})"
                for model, configured, recommended in override_mismatches[:5]
            )
            suffix = "" if len(override_mismatches) <= 5 else f" (+{len(override_mismatches) - 5} more)"
            errors.append(f"Renderer override mismatch vs recommended mapping: {preview}{suffix}")
    except Exception as exc:
        errors.append(f"Could not validate renderer dependency compatibility: {type(exc).__name__}: {exc}")

    if errors:
        message = "Environment compatibility check failed:\n- " + "\n- ".join(errors)
        message += "\nFix: reinstall dependencies from pinned files " "(pip install -r requirements.txt) and retry."
        raise RuntimeError(message)


def get_tinker_client():
    """Get or create the Tinker service client.

    Returns:
        Tinker ServiceClient instance

    Raises:
        RuntimeError: If TINKER_API_KEY is not set
    """
    session = get_session()
    if session.service_client is None:
        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise RuntimeError("TINKER_API_KEY not set.")
        import tinker

        session.service_client = tinker.ServiceClient()
    return session.service_client


async def get_service_client_async():
    """Get or create the Tinker service client (async version).

    Returns:
        Tinker ServiceClient instance
    """
    return get_tinker_client()


# =============================================================================
# Progress File Cleanup
# =============================================================================


def cleanup_progress_file() -> bool:
    """Remove the current session's progress file.

    Call this when finishing a session to prevent progress file accumulation.

    Returns:
        True if file was cleaned up or didn't exist, False if cleanup failed.

    Note:
        Uses atomic check-and-remove pattern to handle race conditions where
        another process might delete the file between exists() and remove().
    """
    try:
        os.remove(PROGRESS_FILE)
        return True
    except FileNotFoundError:
        # File already deleted (possibly by another process) - this is fine
        return True
    except PermissionError as e:
        print(f"Warning: Permission denied cleaning up progress file {PROGRESS_FILE}: {e}", file=sys.stderr, flush=True)
        return False
    except OSError as e:
        print(f"Warning: Failed to cleanup progress file {PROGRESS_FILE}: {e}", file=sys.stderr, flush=True)
        return False


def cleanup_stale_progress_files(max_age_hours: float = 6) -> int:
    """Clean up stale progress files from crashed or orphaned sessions.

    Progress files are created per-process with pattern /tmp/tinkerer_progress_*.txt.
    This function removes files older than max_age_hours to prevent indefinite
    accumulation from crashed sessions.

    Args:
        max_age_hours: Maximum age in hours before a file is considered stale.
                       Default 6 hours.

    Returns:
        Number of stale files cleaned up.

    Note:
        Uses race-condition-safe patterns. Files may be deleted by other processes
        between listing and removal - these cases are handled gracefully.
    """
    import glob

    cleaned = 0
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()

    # Pattern matches /tmp/tinkerer_progress_*.txt
    pattern = "/tmp/tinkerer_progress_*.txt"

    try:
        for filepath in glob.glob(pattern):
            # Don't delete our own progress file
            if filepath == PROGRESS_FILE:
                continue

            try:
                file_mtime = os.path.getmtime(filepath)
                age_seconds = current_time - file_mtime

                if age_seconds > max_age_seconds:
                    try:
                        os.remove(filepath)
                        cleaned += 1
                        if DEBUG_MODE:
                            print(
                                f"Cleaned stale progress file: {filepath} (age: {age_seconds / 3600:.1f}h)",
                                file=sys.stderr,
                                flush=True,
                            )
                    except FileNotFoundError:
                        # File was deleted by another process between check and
                        # remove - this is fine
                        pass
                    except PermissionError as e:
                        if DEBUG_MODE:
                            print(f"Warning: Permission denied removing {filepath}: {e}", file=sys.stderr, flush=True)
            except FileNotFoundError:
                # File was deleted between glob and getmtime - this is fine
                pass
            except OSError as e:
                # Other OS errors (could be transient)
                if DEBUG_MODE:
                    print(f"Warning: Could not process {filepath}: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Warning: Failed to scan for stale progress files: {e}", file=sys.stderr, flush=True)

    return cleaned
