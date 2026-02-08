"""Tinkerer Modal Runner.

Launches an agent (Claude Code or OpenAI Codex) inside a Modal sandbox, wired up
to MCP servers for:
- Dataset access (`hf-datasets`)
- Training/eval/save/finish via the Tinker API (`python -m tinker_mcp`)

Usage:
    # Required for all runs. You can also add optional keys here (WANDB_API_KEY, HF_TOKEN).
    modal secret create tinker-api-key TINKER_API_KEY=... WANDB_API_KEY=... HF_TOKEN=...

    # Required for Claude Code runs (default agent).
    modal secret create anthropic-api-key ANTHROPIC_API_KEY=...

    # Required for Codex runs.
    modal secret create openai-api-key OPENAI_API_KEY=...

    # Quick infra check
    modal run modal/run_tinkerer.py --test-only

    # Run a demo
    MODAL_FORCE_BUILD=1 TINKERER_DEBUG=1 \\
    modal run modal/run_tinkerer.py \\
      --model "vendor/model-base" \\
      --prompt "$(cat prompts/arithmetic_quick_win.txt)"
"""

from importlib import metadata
from pathlib import Path
from functools import lru_cache
import glob
import os
import pty
import select
import subprocess
import sys
import threading
import time

import modal

PROJECT_ROOT = Path(__file__).parent.parent

TINKER_MODEL_LINEUP_SNAPSHOT = """
Qwen/Qwen3-VL-235B-A22B-Instruct
Qwen/Qwen3-VL-30B-A3B-Instruct
Qwen/Qwen3-235B-A22B-Instruct-2507
Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen/Qwen3-30B-A3B
Qwen/Qwen3-30B-A3B-Base
Qwen/Qwen3-32B
Qwen/Qwen3-8B
Qwen/Qwen3-8B-Base
Qwen/Qwen3-4B-Instruct-2507
openai/gpt-oss-120b
openai/gpt-oss-20b
deepseek-ai/DeepSeek-V3.1
deepseek-ai/DeepSeek-V3.1-Base
meta-llama/Llama-3.1-70B
meta-llama/Llama-3.3-70B-Instruct
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.2-3B
meta-llama/Llama-3.2-1B
moonshotai/Kimi-K2-Thinking
moonshotai/Kimi-K2.5
""".strip()

# Define the image with all dependencies and local files
image = (
    modal.Image.debian_slim(python_version="3.13")
    # Keep the base image "agent-friendly": many agent workflows assume basic
    # unix tooling exists (e.g., `rg` for search, `ps` for process inspection).
    .apt_install("curl", "git", "nodejs", "npm", "ripgrep", "procps")
    # Install Python dependencies from the repo-pinned requirements.txt
    # uv_pip_install expects a local path, not the in-image /app path.
    .uv_pip_install(requirements=[str(PROJECT_ROOT / "requirements.txt")])
    .add_local_file(str(PROJECT_ROOT / "requirements.txt"), remote_path="/app/requirements.txt", copy=True)
    .run_commands(
        # Install Claude Code CLI
        "npm install -g @anthropic-ai/claude-code",
        # Install Codex CLI
        "npm install -g @openai/codex",
        # Create Claude Code settings dir. We copy the actual settings file later
        # from a repo-tracked JSON to avoid shell-escaping/quoting issues.
        "mkdir -p /app/.claude",
        # Create Codex config dir. We copy the actual config later from a
        # repo-tracked TOML to avoid shell-escaping/quoting issues.
        "mkdir -p /root/.codex",
    )
    # Add local files to image (replaces deprecated modal.Mount)
    # Note: copy=True is required when running build steps after add_local_*
    .add_local_dir(str(PROJECT_ROOT / "mcp-servers"), remote_path="/app/mcp-servers", copy=True)
    .add_local_file(str(PROJECT_ROOT / "pyproject.toml"), remote_path="/app/pyproject.toml", copy=True)
    .add_local_file(str(PROJECT_ROOT / ".mcp.json"), remote_path="/app/.mcp.json", copy=True)
    .add_local_file(
        str(PROJECT_ROOT / "modal" / "claude_settings.json"),
        remote_path="/app/claude_settings.json",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "modal" / "codex_config.toml"),
        remote_path="/app/codex_config.toml",
        copy=True,
    )
    # Add sandbox instructions file and mirror it to the agent-convention filenames.
    .add_local_file(
        str(PROJECT_ROOT / "modal" / "sandbox_agent_instructions.md"),
        remote_path="/app/sandbox_agent_instructions.md",
        copy=True,
    )
    # Install tinker_mcp package in editable mode so python -m tinker_mcp works
    .run_commands(
        "cp /app/claude_settings.json /app/.claude/settings.json",
        "cp /app/codex_config.toml /root/.codex/config.toml",
        # Claude Code reads CLAUDE.md, Codex reads AGENTS.md. Keep them in sync.
        "cp /app/sandbox_agent_instructions.md /root/CLAUDE.md",
        "cp /app/sandbox_agent_instructions.md /app/CLAUDE.md",
        "cp /app/sandbox_agent_instructions.md /app/AGENTS.md",
        # Dependencies are installed from requirements.txt above; keep the
        # editable install fast and avoid dependency resolution work.
        "pip install -e /app --no-deps",
    )
)

app = modal.App("tinkerer-test")


# Output memory management - prevent unbounded growth.
# This only affects the *returned* text for `modal run`; live stdout is still streamed.
# Bump the default for open-source "max observability" demos; still capped and
# truncation is explicitly warned.
DEFAULT_OUTPUT_CAPTURE_MB = 200


def _slugify(text: str) -> str:
    out = []
    last_was_dash = False
    for ch in (text or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_was_dash = False
            continue
        if not last_was_dash:
            out.append("-")
            last_was_dash = True
    return "".join(out).strip("-")[:60] or "run"


def _default_wandb_group(*, prompt: str, model: str) -> str:
    # Deterministic, non-colliding across agent types:
    # - Same day + same model + same prompt => same group (Codex vs Claude comparable)
    # - Different day/prompt/model => different group
    import hashlib

    day = time.strftime("%Y%m%d")
    model_slug = _slugify(model.split("/")[-1] if model else "")
    prompt_hash = hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()[:10]  # nosec - not for security
    return f"tinkerer-{day}-{model_slug}-{prompt_hash}"


def _print_missing_prompt_help() -> None:
    print("Error: Missing required --prompt.\n", flush=True)
    print(
        "Example:\n"
        '  modal run modal/run_tinkerer.py --model "meta-llama/Llama-3.2-3B" '
        '--prompt "$(cat prompts/arithmetic_quick_win.txt)"\n',
        flush=True,
    )


def _print_missing_model_help(*, supported_models: list[str]) -> None:
    print("Error: Missing required --model.\n", flush=True)
    print(
        "Model selection guidance (best practices):\n"
        "- In general, use MoE models, which are more cost effective than dense models.\n"
        "- Use Base models only if you're doing research or running the full post-training pipeline.\n"
        "- For a task/domain: start from an existing post-trained model and fine-tune it.\n"
        "- If you care about latency: use Instruction models.\n"
        "- If you care about intelligence/robustness: use Hybrid or Reasoning models.\n",
        flush=True,
    )
    print("Common Tinker model IDs (snapshot):\n  " + TINKER_MODEL_LINEUP_SNAPSHOT.replace("\n", "\n  "), flush=True)
    print("\nKnown-good models in this repo (renderer-mapped):\n  " + "\n  ".join(sorted(supported_models)), flush=True)


# WATCHDOG CONFIGURATION
# Kill process if no activity (stdout OR progress file) for this duration.
WATCHDOG_TIMEOUT_SECONDS = 2100  # 35 min
WATCHDOG_WARNING_RATIO = 0.8  # Warn at 80% of timeout

# Codex runtime defaults (override via environment variables).
# This harness targets API-key Codex usage via OPENAI_API_KEY.
DEFAULT_CODEX_MODEL = "gpt-5.2-codex"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"
DEFAULT_CODEX_REASONING_SUMMARY = "detailed"
SUPPORTED_AGENTS = frozenset({"claude", "codex"})

# Progress files written by MCP server.
# Global watchdog heartbeat (all methods): /tmp/tinkerer_progress_{pid}.txt
# Session-scoped progress (all methods): /tmp/tinkerer_*/progress.txt
PROGRESS_FILE_PATTERNS = (
    "/tmp/tinkerer_progress*.txt",
    "/tmp/tinkerer_*/progress.txt",
)

RUNTIME_PACKAGES = (
    "tinker",
    "mcp",
    "huggingface_hub",
    "datasets",
    "transformers",
    "numpy",
    "torch",
    "wandb",
    "modal",
)

# Security hardening: only pass an allowlist of env vars to agent subprocesses.
ALLOWED_AGENT_ENV_PASSTHROUGH = {
    "PATH",
    "HOME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
}

_CLAUDE_REQUIRED_SECRETS = [
    modal.Secret.from_name("anthropic-api-key"),
    modal.Secret.from_name("tinker-api-key"),
]
_CODEX_REQUIRED_SECRETS = [
    modal.Secret.from_name("openai-api-key"),
    modal.Secret.from_name("tinker-api-key"),
]


def _print_runtime_versions() -> None:
    """Print runtime package versions for reproducibility/debugging."""
    print("[SANITY] Python:", sys.version.replace("\n", " "), flush=True)
    for pkg in RUNTIME_PACKAGES:
        try:
            ver = metadata.version(pkg)
            print(f"[SANITY] {pkg}=={ver}", flush=True)
        except metadata.PackageNotFoundError:
            print(f"[SANITY] {pkg} not installed", flush=True)


def _build_claude_env() -> dict[str, str]:
    """Build a restricted environment for Claude subprocess execution."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY secret not found - add with: "
            "modal secret create anthropic-api-key ANTHROPIC_API_KEY=..."
        )
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY secret not found - add with: "
            "modal secret create tinker-api-key TINKER_API_KEY=..."
        )

    if not os.environ.get("HF_TOKEN"):
        print("[WARNING] HF_TOKEN not set - gated datasets will be inaccessible", flush=True)
    if not os.environ.get("WANDB_API_KEY"):
        print("[WARNING] WANDB_API_KEY not set - training metrics will not be logged to W&B", flush=True)

    env = {key: os.environ[key] for key in ALLOWED_AGENT_ENV_PASSTHROUGH if key in os.environ}

    def _maybe_set_env_var(key: str, value: str | None) -> None:
        # Never set empty strings for WANDB_*: wandb validates env vars strictly and
        # will error if variables like WANDB_MODE/WANDB_RUN_ID exist but are "".
        if value is None:
            return
        v = value.strip()
        if not v:
            return
        env[key] = v

    env.update(
        {
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "TINKER_API_KEY": os.environ["TINKER_API_KEY"],
            "TINKERER_DEBUG": os.environ.get("TINKERER_DEBUG", "1"),
            "IS_SANDBOX": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "100000"),
        }
    )

    # Optional secrets / logging configuration.
    _maybe_set_env_var("TINKERER_CHECKPOINT_PREFIX", os.environ.get("TINKERER_CHECKPOINT_PREFIX"))
    _maybe_set_env_var("HF_TOKEN", os.environ.get("HF_TOKEN"))
    _maybe_set_env_var("WANDB_API_KEY", os.environ.get("WANDB_API_KEY"))
    _maybe_set_env_var("WANDB_PROJECT", os.environ.get("WANDB_PROJECT"))
    _maybe_set_env_var("WANDB_ENTITY", os.environ.get("WANDB_ENTITY"))
    _maybe_set_env_var("WANDB_GROUP", os.environ.get("WANDB_GROUP"))
    return env


def _build_codex_env() -> dict[str, str]:
    """Build a restricted environment for Codex subprocess execution."""
    codex_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
    if not codex_api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY (recommended) or CODEX_API_KEY "
            "in the Modal environment (e.g., via `modal secret create openai-api-key OPENAI_API_KEY=...`)."
        )
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY secret not found - add with: "
            "modal secret create tinker-api-key TINKER_API_KEY=..."
        )

    if not os.environ.get("HF_TOKEN"):
        print("[WARNING] HF_TOKEN not set - gated datasets will be inaccessible", flush=True)
    if not os.environ.get("WANDB_API_KEY"):
        print("[WARNING] WANDB_API_KEY not set - training metrics will not be logged to W&B", flush=True)

    env = {key: os.environ[key] for key in ALLOWED_AGENT_ENV_PASSTHROUGH if key in os.environ}

    def _maybe_set_env_var(key: str, value: str | None) -> None:
        # Never set empty strings for WANDB_*: wandb validates env vars strictly and
        # will error if variables like WANDB_MODE/WANDB_RUN_ID exist but are "".
        if value is None:
            return
        v = value.strip()
        if not v:
            return
        env[key] = v

    env.update(
        {
            "TINKER_API_KEY": os.environ["TINKER_API_KEY"],
            "TINKERER_DEBUG": os.environ.get("TINKERER_DEBUG", "1"),
            "IS_SANDBOX": "1",
            "CODEX_MODEL": os.environ.get("CODEX_MODEL", DEFAULT_CODEX_MODEL),
            "CODEX_REASONING_EFFORT": os.environ.get("CODEX_REASONING_EFFORT", DEFAULT_CODEX_REASONING_EFFORT),
            "CODEX_REASONING_SUMMARY": os.environ.get("CODEX_REASONING_SUMMARY", DEFAULT_CODEX_REASONING_SUMMARY),
        }
    )

    # Provide both names for maximum compatibility across Codex CLI versions.
    env["OPENAI_API_KEY"] = codex_api_key
    env["CODEX_API_KEY"] = codex_api_key

    # Optional secrets / logging configuration.
    _maybe_set_env_var("TINKERER_CHECKPOINT_PREFIX", os.environ.get("TINKERER_CHECKPOINT_PREFIX"))
    _maybe_set_env_var("HF_TOKEN", os.environ.get("HF_TOKEN"))
    _maybe_set_env_var("WANDB_API_KEY", os.environ.get("WANDB_API_KEY"))
    _maybe_set_env_var("WANDB_PROJECT", os.environ.get("WANDB_PROJECT"))
    _maybe_set_env_var("WANDB_ENTITY", os.environ.get("WANDB_ENTITY"))
    _maybe_set_env_var("WANDB_GROUP", os.environ.get("WANDB_GROUP"))
    return env


def _launch_claude_process(prompt: str, slave_fd: int, env: dict[str, str]) -> subprocess.Popen:
    """Launch Claude Code process attached to a PTY."""
    try:
        return subprocess.Popen(
            [
                "claude",
                "-p",
                prompt,
                "--model",
                "opus",  # Use Opus 4.6 for best reasoning/agentic capabilities
                "--print",
                "--output-format",
                "stream-json",  # Real-time streaming
                "--verbose",  # Show thinking/trace
                "--mcp-config",
                "/app/.mcp.json",
                "--dangerously-skip-permissions",
            ],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Claude Code CLI binary `claude` was not found in PATH. "
            "Verify @anthropic-ai/claude-code is installed in the Modal image."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to launch Claude Code process: {exc}") from exc


def _launch_codex_process(prompt: str, slave_fd: int, env: dict[str, str]) -> subprocess.Popen:
    """Launch Codex CLI process attached to a PTY."""
    try:
        codex_model = env.get("CODEX_MODEL") or DEFAULT_CODEX_MODEL
        reasoning_effort = env.get("CODEX_REASONING_EFFORT", DEFAULT_CODEX_REASONING_EFFORT)
        reasoning_summary = env.get("CODEX_REASONING_SUMMARY", DEFAULT_CODEX_REASONING_SUMMARY)
        auth_method = "apikey"
        cmd = [
            "codex",
            "exec",
            "--cd",
            "/app",
            "--model",
            codex_model,
            "--config",
            f"preferred_auth_method={auth_method}",
            "--config",
            "cli_auth_credentials_store=file",
            "--config",
            f"model_reasoning_effort={reasoning_effort}",
            "--config",
            f"model_reasoning_summary={reasoning_summary}",
            "--skip-git-repo-check",
        ]

        # Require absolute max permissions for non-interactive `codex exec`.
        # This must be supported by the installed CLI; otherwise, fail fast with a clear message.
        help_text = _get_codex_exec_help()
        if "--dangerously-bypass-approvals-and-sandbox" in help_text:
            perm_flag = "--dangerously-bypass-approvals-and-sandbox"
        elif "--yolo" in help_text:
            # Older CLI versions used `--yolo`; treat it as equivalent max permissions.
            perm_flag = "--yolo"
        else:
            raise RuntimeError(
                "Installed Codex CLI does not support a max-permissions flag "
                "(`--dangerously-bypass-approvals-and-sandbox` or `--yolo`). "
                "Upgrade the Codex CLI in the Modal image (npm install -g @openai/codex)."
            )
        print(f"[CODEX] Using max-permissions flag: {perm_flag}", flush=True)
        cmd.append(perm_flag)

        cmd.append(prompt)

        return subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Codex CLI binary `codex` was not found in PATH. "
            "Verify @openai/codex is installed in the Modal image."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to launch Codex process: {exc}") from exc


@lru_cache(maxsize=1)
def _get_codex_exec_help() -> str:
    """Return cached `codex exec --help` output for flag compatibility checks."""
    try:
        result = subprocess.run(
            ["codex", "exec", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return (result.stdout or "") + "\n" + (result.stderr or "")
    except Exception:
        return ""


def _get_latest_progress_mtime() -> float | None:
    """Return the latest mtime from known progress files, if any."""
    try:
        progress_files: list[str] = []
        for pattern in PROGRESS_FILE_PATTERNS:
            progress_files.extend(glob.glob(pattern))
        if not progress_files:
            return None

        mtimes: list[float] = []
        for progress_file in progress_files:
            try:
                mtimes.append(os.path.getmtime(progress_file))
            except (FileNotFoundError, OSError):
                continue
        return max(mtimes) if mtimes else None
    except Exception:
        return None


def _close_fd_safely(fd: int | None) -> None:
    """Close a file descriptor, ignoring close errors."""
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _get_supported_models() -> list[str] | None:
    """Safely get SUPPORTED_MODELS, returning None if package not available locally.
    
    This allows validation to work locally when package is installed, but gracefully
    skips validation when running Modal without local package installation.
    Validation will still occur in the Modal sandbox where the package is guaranteed.
    """
    try:
        from tinker_mcp.models import SUPPORTED_MODELS
        return SUPPORTED_MODELS
    except ImportError:
        return None


def _normalize_agent_name(agent: str) -> str:
    """Normalize and validate the requested agent runtime name.

    Args:
        agent: Agent name (will be normalized to lowercase, whitespace stripped)
               Must be a non-empty string.

    Returns:
        Normalized agent name ("claude" or "codex")

    Raises:
        ValueError: If agent is empty or not in SUPPORTED_AGENTS
    """
    if not agent or not agent.strip():
        allowed = ", ".join(sorted(SUPPORTED_AGENTS))
        raise ValueError(f"Agent cannot be empty. Use one of: {allowed}.")
    normalized = agent.strip().lower()
    if normalized not in SUPPORTED_AGENTS:
        allowed = ", ".join(sorted(SUPPORTED_AGENTS))
        raise ValueError(f"Unsupported agent '{agent}'. Use one of: {allowed}.")
    return normalized


def _run_agent_process(
    prompt: str,
    env: dict[str, str],
    launch_process,
) -> str:
    """Run an agent CLI in a PTY with watchdog and bounded output capture."""
    capture_mb_raw = os.environ.get("TINKERER_OUTPUT_CAPTURE_MB")
    if capture_mb_raw is None:
        output_capture_bytes = DEFAULT_OUTPUT_CAPTURE_MB * 1024 * 1024
    else:
        try:
            capture_mb = int(capture_mb_raw)
        except ValueError as exc:
            raise ValueError(f"TINKERER_OUTPUT_CAPTURE_MB must be an int, got {capture_mb_raw!r}") from exc
        if capture_mb <= 0:
            raise ValueError(f"TINKERER_OUTPUT_CAPTURE_MB must be > 0, got {capture_mb}")
        output_capture_bytes = capture_mb * 1024 * 1024

    total_output_bytes = 0
    output_truncated = False

    # Set working directory where MCP/server config is located
    os.chdir("/app")
    _print_runtime_versions()

    master_fd, slave_fd = None, None
    process = None
    try:
        master_fd, slave_fd = pty.openpty()

        try:
            process = launch_process(prompt, slave_fd, env)
        except RuntimeError as exc:
            return f"Error: {exc}"

        _close_fd_safely(slave_fd)
        slave_fd = None

        output_lock = threading.Lock()
        output_lines_lock = threading.Lock()
        last_output_time = time.time()
        watchdog_triggered = False
        last_watchdog_log_time = time.time()

        def watchdog_thread():
            nonlocal watchdog_triggered, last_watchdog_log_time
            warning_logged = False
            warning_threshold = WATCHDOG_TIMEOUT_SECONDS * WATCHDOG_WARNING_RATIO

            print(
                f"[WATCHDOG] Started: timeout={WATCHDOG_TIMEOUT_SECONDS}s, "
                f"warn_at={int(warning_threshold)}s",
                flush=True,
            )

            while process.poll() is None:
                now = time.time()

                with output_lock:
                    stdout_activity = last_output_time

                progress_mtime = _get_latest_progress_mtime()
                progress_activity = progress_mtime if progress_mtime else 0

                last_activity = max(stdout_activity, progress_activity)
                silence_duration = now - last_activity

                if progress_activity > stdout_activity:
                    activity_source = "progress_file"
                else:
                    activity_source = "stdout"

                if silence_duration > 60 and now - last_watchdog_log_time > 60:
                    print(
                        f"[WATCHDOG] No activity for {int(silence_duration)}s "
                        f"(last: {activity_source}, timeout: {WATCHDOG_TIMEOUT_SECONDS}s)",
                        flush=True,
                    )
                    last_watchdog_log_time = now

                if silence_duration < warning_threshold:
                    warning_logged = False

                if silence_duration > warning_threshold and not warning_logged:
                    time_remaining = WATCHDOG_TIMEOUT_SECONDS - silence_duration
                    print(f"\n{'=' * 60}", flush=True)
                    print("[WATCHDOG WARNING] Approaching timeout!", flush=True)
                    print(
                        f"[WATCHDOG WARNING] No activity for {int(silence_duration)}s "
                        f"(last: {activity_source})",
                        flush=True,
                    )
                    print(
                        f"[WATCHDOG WARNING] Kill in {int(time_remaining)}s "
                        "unless activity detected",
                        flush=True,
                    )
                    print(f"{'=' * 60}\n", flush=True)
                    warning_logged = True

                if silence_duration > WATCHDOG_TIMEOUT_SECONDS:
                    print(f"\n\n{'=' * 60}", flush=True)
                    print(f"[WATCHDOG KILL] No activity for {int(silence_duration)}s!", flush=True)
                    print(f"[WATCHDOG KILL] Last activity source: {activity_source}", flush=True)
                    print("[WATCHDOG KILL] The process appears to be hung. Terminating...", flush=True)
                    print("[WATCHDOG KILL] This usually means a Tinker API call timed out.", flush=True)
                    log_files = (
                        glob.glob("/tmp/tinkerer_*/*.log")
                        + glob.glob("/tmp/tinkerer_*/progress.txt")
                        + glob.glob("/tmp/tinkerer_progress*.txt")
                    )
                    if log_files:
                        print(f"[WATCHDOG KILL] Debug files: {log_files[:5]}", flush=True)
                    else:
                        print("[WATCHDOG KILL] No debug logs found - training may not have started", flush=True)
                    print(f"{'=' * 60}\n", flush=True)
                    watchdog_triggered = True
                    process.kill()
                    break

                time.sleep(2)

        watchdog = threading.Thread(target=watchdog_thread, daemon=True)
        watchdog.start()

        output_lines = []
        truncation_warned = False

        def append_output_chunk(text: str) -> None:
            nonlocal total_output_bytes, output_truncated, truncation_warned
            text_bytes = len(text.encode("utf-8"))

            with output_lines_lock:
                output_lines.append(text)
                total_output_bytes += text_bytes

                while total_output_bytes > output_capture_bytes and len(output_lines) > 1:
                    removed = output_lines.pop(0)
                    total_output_bytes -= len(removed.encode("utf-8"))
                    output_truncated = True
                    if not truncation_warned:
                        print(
                            f"\n[OUTPUT CAPTURE WARNING] Exceeded capture limit "
                            f"({output_capture_bytes // (1024 * 1024)}MB). "
                            "Dropping oldest captured logs. Increase via env "
                            "TINKERER_OUTPUT_CAPTURE_MB.\n",
                            flush=True,
                        )
                        truncation_warned = True

        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="replace")
                    print(text, end="", flush=True)
                    append_output_chunk(text)

                    with output_lock:
                        last_output_time = time.time()
                except OSError:
                    break
            if process.poll() is not None:
                for _ in range(10):
                    ready, _, _ = select.select([master_fd], [], [], 0.2)
                    if not ready:
                        break
                    try:
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        text = data.decode("utf-8", errors="replace")
                        print(text, end="", flush=True)
                        append_output_chunk(text)
                    except OSError:
                        break
                break

        if watchdog_triggered:
            append_output_chunk("\n\n[WATCHDOG TERMINATION: Process was killed due to inactivity]\n")
        else:
            returncode = process.poll()
            if returncode not in (None, 0):
                # Some agent CLIs can exit non-zero despite printing a full result.
                # Surface it as an observability signal without failing the Modal run.
                append_output_chunk(f"\n\n[AGENT EXIT WARNING] Process exit code: {returncode}\n")

        with output_lines_lock:
            output_text = "".join(output_lines)
        if output_truncated:
            output_text = (
                "[OUTPUT TRUNCATED] Oldest logs were dropped after reaching the capture limit.\n"
                f"Set TINKERER_OUTPUT_CAPTURE_MB to increase (current: {output_capture_bytes // (1024 * 1024)}MB).\n"
                + output_text
            )

        return output_text

    finally:
        _close_fd_safely(slave_fd)
        _close_fd_safely(master_fd)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def _run_claude_agent(prompt: str, *, wandb_project: str = "", wandb_group: str = "") -> str:
    env = _build_claude_env()
    # Default checkpoint prefix must be unique per Modal run to avoid name
    # collisions when you run multiple jobs in parallel (even with the same agent).
    if not env.get("TINKERER_CHECKPOINT_PREFIX"):
        import uuid

        env["TINKERER_CHECKPOINT_PREFIX"] = f"claude-{uuid.uuid4().hex[:6]}"
    print(f"[RUN] checkpoint_prefix={env['TINKERER_CHECKPOINT_PREFIX']}", flush=True)
    if wandb_project and "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = wandb_project
    if wandb_group and "WANDB_GROUP" not in env:
        env["WANDB_GROUP"] = wandb_group
    return _run_agent_process(
        prompt=prompt,
        env=env,
        launch_process=_launch_claude_process,
    )


def _run_codex_agent(
    prompt: str,
    *,
    wandb_project: str = "",
    wandb_group: str = "",
) -> str:
    env = _build_codex_env()
    if not env.get("TINKERER_CHECKPOINT_PREFIX"):
        import uuid

        env["TINKERER_CHECKPOINT_PREFIX"] = f"codex-{uuid.uuid4().hex[:6]}"
    print(f"[RUN] checkpoint_prefix={env['TINKERER_CHECKPOINT_PREFIX']}", flush=True)
    if wandb_project and "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = wandb_project
    if wandb_group and "WANDB_GROUP" not in env:
        env["WANDB_GROUP"] = wandb_group
    return _run_agent_process(
        prompt=prompt,
        env=env,
        launch_process=_launch_codex_process,
    )


@app.function(
    image=image,
    secrets=_CLAUDE_REQUIRED_SECRETS,
    timeout=86400,  # 24 hours (maximum)
    # Default to no automatic retries: agent-driven long training jobs should
    # fail visibly instead of silently restarting from scratch.
    retries=modal.Retries(
        max_retries=0,
        initial_delay=10.0,  # Wait 10s before first retry
        backoff_coefficient=2.0,  # Double delay each retry (10s, 20s, 40s)
    ),
    # Note: GPU not strictly required - Tinker API handles GPU remotely.
    # Uncomment below if local GPU acceleration is needed for future features.
    # gpu="any",
)
def run_claude_code(prompt: str, wandb_project: str = "", wandb_group: str = "") -> str:
    """Run Claude Code CLI with the given prompt.

    Claude Code has access to these MCP tools:

    hf-datasets MCP:
    - search_datasets(query, ...) -> str

    tinker MCP (training + evaluation):
    - init_grpo(base_model, ...) -> str
    - train_grpo_step(prompts, reward_function, ...) -> str
    - init_sft(base_model, ...) -> str
    - train_sft_step(examples, ...) -> str
    - sample(prompt, ...) -> str
    - save(name) -> str
    - load(path_or_alias) -> str
    - finish() -> str

    Notes:
    - Prefer passing file paths for large JSON inputs (prompts/examples) to avoid
      CLI quoting/escaping issues.
    - The agent playbook is mounted at /root/CLAUDE.md inside the sandbox.
    """
    return _run_claude_agent(prompt, wandb_project=wandb_project, wandb_group=wandb_group)


@app.function(
    image=image,
    secrets=_CODEX_REQUIRED_SECRETS,
    timeout=86400,
    retries=modal.Retries(
        max_retries=0,
        initial_delay=10.0,
        backoff_coefficient=2.0,
    ),
)
def run_codex_code(
    prompt: str,
    wandb_project: str = "",
    wandb_group: str = "",
) -> str:
    """Run Codex CLI with the given prompt and MCP tools."""
    return _run_codex_agent(
        prompt,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("tinker-api-key")],
    timeout=600,
)
def test_mcp_servers() -> str:
    """Test that MCP servers start correctly."""
    import subprocess
    import os

    results = []

    # Test HF datasets MCP
    print("Testing hf-datasets MCP server...")
    try:
        # Use importlib to import the standalone hf-datasets server
        import importlib.util
        hf_datasets_path = Path("/app/mcp-servers/hf-datasets/server.py")
        spec = importlib.util.spec_from_file_location("hf_datasets_server", hf_datasets_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load hf-datasets server")
        hf_server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hf_server)
        if hasattr(hf_server, "mcp"):
            results.append("hf-datasets: HF datasets MCP OK")
        else:
            raise ImportError("hf-datasets server missing 'mcp' attribute")
    except Exception as e:
        results.append(f"hf-datasets: ERROR - {e}")

    # Test Tinker MCP
    print("Testing tinker MCP server...")
    try:
        result = subprocess.run(
            ["python", "-c", "from tinker_mcp.server import mcp; print('Tinker MCP OK')"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ.copy(), "TINKER_API_KEY": os.environ.get("TINKER_API_KEY", "")},
        )
        results.append(f"tinker: {result.stdout.strip() or result.stderr.strip()}")
    except Exception as e:
        results.append(f"tinker: ERROR - {e}")

    return "\n".join(results)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("tinker-api-key")],
    timeout=120,
)
def test_tinker_connection() -> str:
    """Test direct connection to Tinker API."""
    import tinker

    try:
        sc = tinker.ServiceClient()
        capabilities = sc.get_server_capabilities()
        models = capabilities.supported_models[:5]  # First 5 models
        return f"Tinker connection OK. Sample models: {models}"
    except Exception as e:
        return f"Tinker connection ERROR: {e}"


@app.local_entrypoint()
def main(
    prompt: str | None = "",
    model: str | None = "",
    agent: str | None = "claude",
    test_only: bool = False,
):
    """Main entrypoint for running Tinkerer in Modal.

    Args:
        prompt: The training prompt (task description / instructions for the agent).
                Required if test_only=False. Cannot be empty.
        model: The base model to use for fine-tuning (prefer base checkpoints for GRPO).
               Required if test_only=False. Must be in SUPPORTED_MODELS registry.
        agent: Agent runtime to use: "claude" or "codex". Defaults to "claude" if not provided.
               Must be one of: "claude", "codex".
        test_only: If True, only test MCP servers without running an agent

    Raises:
        SystemExit: If validation fails (invalid agent, missing prompt/model, or model not in registry)
    """
    import time

    start_time = time.time()

    # Normalize and validate inputs (handle None, empty strings, whitespace)
    prompt = (prompt or "").strip()
    model = (model or "").strip()
    # Default agent to "claude" if not provided
    agent = (agent or "claude").strip()

    # Validate prompt (required for non-test runs)
    if not test_only:
        if not prompt:
            _print_missing_prompt_help()
            return

    # Validate model (required for non-test runs)
    # Note: Validation happens locally if package available, otherwise deferred to Modal sandbox
    if not test_only:
        supported_models = _get_supported_models()
        
        if not model:
            if supported_models is not None:
                _print_missing_model_help(supported_models=list(supported_models))
            else:
                print("ERROR: Model is required for non-test runs.", flush=True)
            return

        # Error if model not in registry (strict validation)
        # Skip if package unavailable locally - validation will occur in Modal sandbox
        if supported_models is not None and model not in supported_models:
            print("=" * 60, flush=True)
            print(f"ERROR: Model '{model}' is not in the MODEL_INFO registry.", flush=True)
            print("\nSupported models:", flush=True)
            for supported_model in sorted(supported_models):
                print(f"  - {supported_model}", flush=True)
            print(
                "\nIf you need to use an unlisted model, add it to "
                "mcp-servers/tinker_mcp/models.py with proper renderer mapping.",
                flush=True,
            )
            print("=" * 60, flush=True)
            return

    # Validate agent (must be "claude" or "codex")
    try:
        agent = _normalize_agent_name(agent)
    except ValueError as exc:
        print("=" * 60, flush=True)
        print(f"ERROR: {exc}", flush=True)
        print("=" * 60, flush=True)
        return
    template_path = PROJECT_ROOT / "modal" / "agent_prompt_template.md"
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: Failed to read agent prompt template at {template_path}: {exc}")
        return

    if "<<PROMPT>>" not in template or "<<MODEL>>" not in template:
        print(f"Error: agent prompt template missing placeholders (<<PROMPT>> and/or <<MODEL>>): {template_path}")
        return

    agent_prompt = template.replace("<<PROMPT>>", prompt).replace("<<MODEL>>", model)

    print("=" * 60)
    print("TINKERER MODAL RUNNER")
    print("=" * 60)

    if test_only:
        print("\n[1/2] Testing MCP servers...")
        mcp_result = test_mcp_servers.remote()
        print(mcp_result)

        print("\n[2/2] Testing Tinker connection...")
        tinker_result = test_tinker_connection.remote()
        print(tinker_result)

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"CHECK COMPLETE - Run completed in {elapsed_time:.1f} seconds")
        print("=" * 60)
    else:
        print(f"\nPrompt: {prompt}")
        print(f"Model: {model}\n")
        print(f"Agent: {agent}\n")
        wandb_project = (os.environ.get("WANDB_PROJECT") or "tinkerer").strip() or "tinkerer"
        wandb_group = (os.environ.get("WANDB_GROUP") or "").strip() or _default_wandb_group(prompt=prompt, model=model)
        print(f"W&B defaults: project={wandb_project}, group={wandb_group}\n")

        print("=" * 60)
        if agent == "claude":
            _ = run_claude_code.remote(agent_prompt, wandb_project=wandb_project, wandb_group=wandb_group)
        else:
            codex_model = os.environ.get("CODEX_MODEL", DEFAULT_CODEX_MODEL)
            codex_effort = os.environ.get("CODEX_REASONING_EFFORT", DEFAULT_CODEX_REASONING_EFFORT)
            codex_summary = os.environ.get("CODEX_REASONING_SUMMARY", DEFAULT_CODEX_REASONING_SUMMARY)
            print(
                f"Codex config: model={codex_model}, reasoning_effort={codex_effort}, reasoning_summary={codex_summary}"
            )
            _ = run_codex_code.remote(agent_prompt, wandb_project=wandb_project, wandb_group=wandb_group)
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"COMPLETE - Run completed in {elapsed_time:.1f} seconds")
