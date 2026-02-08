"""Tinker Fine-Tuning MCP Server.

This MCP server provides a small set of tools for running post-training against
the Tinker API from an agent loop (for example Claude Code or Codex) or a human
operator.

Design note (intentional statefulness):
Post-training is not an idempotent workflow. Model weights evolve during
training, sampling clients track the currently-exported policy, optional W&B
logging persists across steps, and counters accumulate for scheduling. This
server therefore maintains a single in-process session and serializes tool calls
with a lock to protect shared state.

Tools:
- SFT: init_sft, train_sft_step
- GRPO/RL: init_grpo, train_grpo_step
- Common: sample, save, load, finish

Typical workflow:
1. init_*()
2. (train_*_step() -> sample()) repeated until satisfied
3. save()
4. finish()

Environment:
- TINKER_API_KEY: required
- WANDB_API_KEY: optional (recommended)

Installation:
    pip install -r requirements.txt
"""

import asyncio
import functools
import os
import sys
import time

from mcp.server.fastmcp import FastMCP
from tinker_cookbook import renderers as renderer_helpers

# Internal modules
from tinker_mcp.state import (
    get_session,
    try_acquire_tool_call_lock,
    release_tool_call_lock,
)
from tinker_mcp.utils import (
    DEBUG_MODE,
    TIMEOUT_LONG,
    SAMPLING_TIMEOUT_BASE,
    SAMPLING_TOKENS_PER_SECOND,
    apply_checkpoint_prefix,
    debug_log,
    get_log_paths,
    validate_environment,
    get_service_client_async,
    resolve_api_future,
    get_sample_sequences,
    cleanup_progress_file,
)
from tinker_mcp.training import (
    init_sft_session,
    train_sft_step_impl,
    init_grpo_session,
    train_grpo_step_impl,
)

# Validate environment on import
validate_environment()

# Create MCP server
mcp = FastMCP("tinker")

# Serialize tool calls because this harness uses one mutable global session.


def _resolve_tool_lock_timeout_seconds() -> float:
    # Default is intentionally >0 so accidental concurrent calls fail fast-ish,
    # but not so small that UIs or humans hit spurious "busy" errors.
    raw = os.environ.get("TINKERER_TOOL_LOCK_TIMEOUT_SECONDS", "10.0")
    try:
        timeout = float(raw)
    except ValueError:
        print(
            "WARNING: Invalid TINKERER_TOOL_LOCK_TIMEOUT_SECONDS="
            f"{raw!r}; defaulting to 10.0s",
            file=sys.stderr,
            flush=True,
        )
        return 10.0
    if timeout < 0:
        return -1.0
    return timeout


TOOL_LOCK_TIMEOUT_SECONDS = _resolve_tool_lock_timeout_seconds()


def _session_busy_error(tool_name: str) -> str:
    """Consistent error for concurrent calls against the singleton session."""
    return (
        "Error: Another tool call is already running for this training session. "
        "Tinkerer serializes tool execution to protect shared state. "
        f"Retry `{tool_name}` once the current call finishes. "
        "If you need tool calls to wait (instead of returning busy), set "
        "`TINKERER_TOOL_LOCK_TIMEOUT_SECONDS=-1` in the MCP server environment."
    )


def serialize_session_tool(fn):
    """Decorator that gates stateful MCP tools behind a global lock."""

    @functools.wraps(fn)
    async def _wrapped(*args, **kwargs):
        acquired = await try_acquire_tool_call_lock(TOOL_LOCK_TIMEOUT_SECONDS)
        if not acquired:
            return _session_busy_error(fn.__name__)
        try:
            return await fn(*args, **kwargs)
        finally:
            release_tool_call_lock()

    return _wrapped


# =============================================================================
# SFT TOOLS
# =============================================================================


@mcp.tool()
@serialize_session_tool
async def init_sft(
    base_model: str,
    lora_rank: int = 16,
    wandb_project: str = "tinkerer",
    wandb_run_name: str | None = None,
    debug: bool = False,
) -> str:
    """Initialize a supervised fine-tuning session.

    SFT is best when you have high-quality input/output pairs and you want the
    model to imitate style, tone, formatting, or domain-specific phrasing.

    Model selection (high-level guidance):
    - Prefer a base checkpoint for post-training when possible. It tends to be
      more stable and avoids instruction-following priors fighting your data.
    - Start small/cheap to validate your data pipeline, then scale up.
    - If your task is multimodal, ensure your inputs and renderer support it.

    WHEN TO USE SFT:
    - Style/tone transfer (subjective quality)
    - Creative writing (no ground truth)
    - Tasks where "good" is a spectrum, not binary
    For verifiable tasks (code, math, SQL), use init_grpo() instead.

    PURPOSE: SFT teaches a model to imitate examples. Use when you have quality
    input-output pairs and want the model to learn patterns, style, or format.
    Best for subjective tasks where "good" means "similar to these examples."

    WORKFLOW: After init, call train_sft_step() in small increments and check
    quality with sample() between calls. This gives you full control.

    PARAMETERS:
    - base_model: Model to fine-tune. Base models are recommended but not required.
    - lora_rank: Adapter capacity. Higher ranks can capture more complex behavior.
    - wandb_project/wandb_run_name: For experiment tracking.

    RETURNS: Confirmation that model is loaded and ready for training.
    """
    return await init_sft_session(base_model, lora_rank, wandb_project, wandb_run_name, debug)


@mcp.tool()
@serialize_session_tool
async def train_sft_step(
    examples: str,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    batch_size: int = 128,
    warmup_ratio: float = 0.1,
    lr_scheduler: str = "constant",
    scheduler_total_steps: int = 0,
    validation_split: float = 0.1,
    debug: bool = False,
) -> str:
    """Train the model on examples (one incremental SFT step).

    You can call this repeatedly; progress is cumulative within the active
    session. Prefer small, inspectable steps and use sample() between calls to
    verify behavior and stop before overfitting.

    Practical tuning heuristics:
    - If loss is unstable/spiky: lower learning_rate and consider a constant
      schedule.
    - If loss is flat and outputs are not changing: increase learning_rate or
      improve data quality/coverage.
    - If validation loss worsens while training loss improves: stop and save
      (classic overfitting signal).

    LR scheduling across repeated calls:
    If lr_scheduler is "cosine" or "linear" and you train in many small calls,
    set scheduler_total_steps to a fixed horizon so decay does not collapse
    early. Warmup is cumulative; for continued training, warmup_ratio should
    typically be 0.

    PARAMETERS:
    - examples: JSON array [{"prompt": "...", "response": "..."}] OR a file path
      (recommended for larger datasets).
    - num_epochs: Passes through your dataset for this call.
    - learning_rate/batch_size/warmup_ratio/lr_scheduler: Optimizer controls.
    - validation_split: Optional holdout split for early overfitting detection.

    RETURNS: A training report with metrics and decision guidance.
    """
    return await train_sft_step_impl(
        examples,
        num_epochs,
        learning_rate,
        batch_size,
        warmup_ratio,
        lr_scheduler,
        scheduler_total_steps if scheduler_total_steps > 0 else None,
        validation_split,
        debug,
    )


# =============================================================================
# GRPO TOOLS
# =============================================================================


@mcp.tool()
@serialize_session_tool
async def init_grpo(
    base_model: str,
    lora_rank: int = 32,
    group_size: int = 4,
    wandb_project: str = "tinkerer",
    wandb_run_name: str | None = None,
    debug: bool = False,
) -> str:
    """Initialize a GRPO training session for reward-based learning.

    GRPO is best for verifiable tasks where you can score completions with a
    reward function (tests, parsers, execution, exact match, etc.).

    IMPORTANT: Base models are usually the safest starting point for GRPO
    because they often provide more rollout diversity (reward variance). Non-
    base models can work, but may require more careful learning-rate / KL
    control.

    GROUP SIZE:
    group_size is the number of independent completions sampled per prompt.
    Larger groups can reduce advantage noise and improve ranking signal, but
    increase sampling cost. Start small; increase if learning is noisy and you
    can afford the extra sampling.

    WHEN TO USE GRPO:
    - Code generation (tests verify correctness)
    - Math problems (answers can be checked)
    - Format compliance (schemas can be validated)
    For subjective tasks (style, creative), use init_sft() instead.

    HOW IT WORKS: For each prompt, generate group_size completions, score each with
    your reward function, then update the model to favor higher-scoring strategies.
    The key insight: GRPO needs VARIANCE in rewards to learn - if all completions
    get the same score, there's no signal about which is better.

    WORKFLOW: After init, call train_grpo_step() one iteration at a time, checking
    metrics between calls. This mini-batch pattern gives you full control to stop
    early, adjust hyperparameters, or investigate issues.

    PARAMETERS:
    - base_model: Model to fine-tune. Base models are recommended but not required.
    - group_size: Completions per prompt. More = better signal but slower.
    - lora_rank: Adapter capacity. Higher ranks can help RL explore strategies.

    RETURNS: Confirmation that model and RL infrastructure are ready.
    """
    return await init_grpo_session(base_model, lora_rank, group_size, wandb_project, wandb_run_name, debug)


@mcp.tool()
@serialize_session_tool
async def train_grpo_step(
    prompts: str,
    reward_function: str,
    num_iterations: int = 1,
    learning_rate: float = 4e-5,
    warmup_ratio: float = 0.1,
    lr_scheduler: str = "constant",
    scheduler_total_steps: int = 0,
    temperature: float = 0.7,
    max_tokens: int = 24576,
    auto_checkpoint: bool = True,
    debug: bool = False,
    sampling_debug_prompt_limit: int | None = None,
    auto_checkpoint_reward_threshold: float = 0.3,
    auto_checkpoint_min_iterations: int = 3,
) -> str:
    """Run GRPO training iterations with reward-based learning.

    GRPO is reward-based learning for verifiable tasks. Each iteration:
    1. Samples group_size completions per prompt from the current sampling
       policy.
    2. Computes rewards via your reward function.
    3. Centers rewards within each group to produce advantages (GRPO-style).
    4. Updates the policy using the importance_sampling loss.

    GRPO needs variance within a group to learn. If all completions for a prompt
    receive the same reward, that prompt contributes no gradient and is skipped.
    Uniform all-1 often means "already solved" (no learning signal). Uniform
    all-0 often means "too hard" or a reward bug (also no learning signal).

    Reward function guidelines:
    - Deterministic, fast, and side-effect free.
    - Robust to failures (timeouts/exceptions) and always returns a float.
    - Prefer rewards that correlate smoothly with quality when possible.

    Iteration strategy:
    - Use num_iterations=1 for evaluation-driven tuning.
    - Keep lr_scheduler="constant" while debugging. If you use "cosine"/"linear"
      across many small calls, set scheduler_total_steps to the planned total so
      decay does not collapse early.
    - If training becomes unstable or KL grows quickly, lower learning_rate
      and/or increase effective batch (more prompts and/or larger group_size).

    Monitoring:
    - reward_mean trend, skip causes (uniform rewards vs token/logprob issues),
      and KL divergence (when available). Use sample() periodically to verify
      real behavior changes.

    PARAMETERS:
    - prompts: JSON array [{"prompt": "...", "ground_truth": "..."}] OR a file
      path (recommended for larger prompt sets).
    - reward_function: Python defining compute_reward(completion,
      ground_truth) -> float (can be negative).
    - sampling_debug_prompt_limit: -1 logs all prompts, 0 disables per-prompt
      sampling debug details.

    RETURNS: A training report with metrics and decision guidance.
    """
    return await train_grpo_step_impl(
        prompts=prompts,
        reward_function=reward_function,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler=lr_scheduler,
        scheduler_total_steps=scheduler_total_steps if scheduler_total_steps > 0 else None,
        temperature=temperature,
        max_tokens=max_tokens,
        auto_checkpoint=auto_checkpoint,
        debug=debug,
        sampling_debug_prompt_limit=sampling_debug_prompt_limit,
        auto_checkpoint_reward_threshold=auto_checkpoint_reward_threshold,
        auto_checkpoint_min_iterations=auto_checkpoint_min_iterations,
    )


# =============================================================================
# COMMON TOOLS
# =============================================================================


@mcp.tool()
@serialize_session_tool
async def sample(prompt: str, max_tokens: int = 24576, temperature: float = 0.7, debug: bool = False) -> str:
    """Generate a completion from the model in its current state.

    PURPOSE: See what the model produces RIGHT NOW. This is your window into
    whether training is working.

    WHEN TO CALL:
    - After every train_*_step to verify quality is improving
    - When metrics look concerning but you're not sure why
    - Before save() to confirm the model is worth keeping
    - To compare early vs late training outputs (the "wow" moment)

    INTERPRETING OUTPUT:
    - Looks like pre-training -> Training hasn't taken effect yet
    - Matches training examples too closely -> Possible overfitting
    - Gibberish or loops -> Training may have destabilized (reduce LR)

    PARAMETERS:
    - prompt: What you want the model to respond to. Make it representative.
    - temperature: Lower = more deterministic; higher = more diverse.
    - max_tokens: Set high enough for complete responses. The default is high to
      reduce truncation; lower it for faster quick checks.
    """
    session = get_session()
    log_paths = get_log_paths()

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    if session.training_client is None:
        return (
            "Error: No training session active. Initialize a session first:\n"
            "- For verifiable tasks: init_grpo(base_model, ...)\n"
            "- For example-based tasks: init_sft(base_model, ...)"
        )
    if session.renderer is None:
        return "Error: Renderer not initialized. Re-initialize your session with init_sft() or init_grpo()."

    if do_debug:
        prompt_preview = prompt[:100].replace("\n", "\\n") if prompt else ""
        debug_log(
            log_paths["sampling"],
            f'[SAMPLE] prompt="{prompt_preview}..." (max_tokens={max_tokens}, temp={temperature})',
            force=True,
        )

    try:
        from tinker import types as tinker_types

        messages = [{"role": "user", "content": prompt}]
        model_input = session.renderer.build_generation_prompt(messages)
        stop_sequences = session.renderer.get_stop_sequences()

        if do_debug:
            debug_log(
                log_paths["sampling"],
                f"[SAMPLE] Renderer: {session.renderer_name}, prompt length: {model_input.length}",
                force=True,
            )

        sampling_params = tinker_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
        )

        # Get or create sampling client
        if session.sampling_client is not None:
            sampling_client = session.sampling_client
            if do_debug:
                debug_log(
                    log_paths["sampling"],
                    f"[SAMPLE] Using cached {session.session_method} sampling client",
                    force=True,
                )
        else:
            if do_debug:
                debug_log(log_paths["sampling"], "[SAMPLE] Creating temporary sampling client...", force=True)
            temp_checkpoint = apply_checkpoint_prefix("sample_temp")
            sampling_client = await asyncio.wait_for(
                session.training_client.save_weights_and_get_sampling_client_async(temp_checkpoint),
                timeout=TIMEOUT_LONG,
            )

        # Tinker API: sample() returns ApiFuture, must call .result() to get
        # response
        start_time = time.time()
        # Scale timeout based on max_tokens (same formula as GRPO training)
        # This prevents timeout errors when sampling with large max_tokens
        sampling_timeout = SAMPLING_TIMEOUT_BASE + (max_tokens / SAMPLING_TOKENS_PER_SECOND)
        sample_future = await asyncio.wait_for(
            asyncio.to_thread(
                sampling_client.sample, prompt=model_input, sampling_params=sampling_params, num_samples=1
            ),
            timeout=sampling_timeout,
        )
        # Resolve the ApiFuture to get actual response
        sample_response = resolve_api_future(sample_future)

        # Validate API response - fail fast instead of returning empty strings
        # silently
        sampling_log_file: str | None = log_paths.get("sampling") if do_debug else None
        sequences = get_sample_sequences(sample_response, log_file=sampling_log_file)
        if len(sequences) != 1:
            raise ValueError(f"Sampling failed: expected 1 sample (num_samples=1), got {len(sequences)}.")

        # Extract generated text
        seq = sequences[0]
        generated_tokens = list(seq.tokens) if seq.tokens else []

        # Validate tokens - don't silently return empty output
        if not generated_tokens or len(generated_tokens) == 0:
            raise ValueError(
                "Model generated 0 tokens. This usually means:\n"
                "1. Prompt formatting is wrong for this model\n"
                "2. Model immediately output EOS token\n"
                "3. API error occurred silently\n"
                f"Model: {session.current_model}\n"
                f"Prompt tokens: {model_input.length}"
            )

        parsed_message, parse_ok = session.renderer.parse_response(generated_tokens)
        generated_text = renderer_helpers.get_text_content(parsed_message)
        if not generated_text or not generated_text.strip():
            raise ValueError("Renderer parsed response but extracted empty text content.")
        if do_debug and not parse_ok:
            debug_log(
                log_paths["sampling"],
                "[SAMPLE] Renderer parse marked response as partial; using parsed assistant text.",
                force=True,
            )

        elapsed = time.time() - start_time
        if do_debug:
            debug_log(
                log_paths["sampling"],
                f"[SAMPLE] Generated {len(generated_tokens)} tokens in {elapsed:.1f}s",
                force=True,
            )

        return f"""
GENERATED OUTPUT:
{generated_text}
"""

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        # Always capture tracebacks for post-mortems; sample() is often used as
        # the first debugging signal when training looks suspicious.
        debug_log(log_paths["sampling"], f"[SAMPLE] ERROR: {e}\n{tb}", force=True)
        if do_debug:
            return f"Error generating sample: {e}\n{tb}"
        return f"Error generating sample: {e}"


@mcp.tool()
@serialize_session_tool
async def save(name: str, debug: bool = False) -> str:
    """Checkpoint the current adapter weights so you can return to this state
    later.

    PURPOSE: Training is exploratory - you try things, some work, some don't.
    Save lets you bookmark promising states. If further training degrades quality,
    you can load() a saved checkpoint and try a different direction.

    WHEN TO CALL:
    - After seeing good sample() quality - lock in that progress
    - Before trying aggressive hyperparameters - create a fallback
    - When reward_mean plateaus at a good level - that's likely optimal
    - At regular intervals during long training - insurance against issues

    WHAT'S SAVED: Just the LoRA adapter weights (small), not the full base model.
    """
    session = get_session()
    log_paths = get_log_paths()

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    if session.training_client is None:
        return (
            "Error: No training session active. You must train first before "
            "saving.\nWorkflow: init_grpo/init_sft -> "
            "train_grpo_step/train_sft_step -> save(name)"
        )

    if do_debug:
        debug_log(log_paths["save_load"], f'[SAVE] name="{name}"', force=True)

    try:
        checkpoint_name = apply_checkpoint_prefix(name)
        start_time = time.time()
        # Tinker `save_state()` returns an awaitable APIFuture; `save_state_async()`
        # is an async wrapper that returns the APIFuture (so would require a
        # double-await). Use `save_state()` here to keep the flow simple:
        # future -> await -> response.
        save_future = session.training_client.save_state(checkpoint_name)
        save_result = await asyncio.wait_for(save_future, timeout=TIMEOUT_LONG)
        elapsed = time.time() - start_time

        # Store the canonical checkpoint path returned by the API (e.g., tinker://run-id/weights/name).
        adapter_path = getattr(save_result, "path", None) or getattr(save_result, "checkpoint_path", None)
        if not adapter_path:
            raise RuntimeError(
                "Tinker save_state response missing `path`. "
                f"Got: {type(save_result).__name__} attrs={dir(save_result)[:25]}"
            )

        session.saved_adapters[checkpoint_name] = adapter_path
        session.saved_adapters[name] = adapter_path

        # Save scheduler state metadata
        session.save_checkpoint_metadata(checkpoint_name)
        session.saved_metadata[name] = session.saved_metadata[checkpoint_name]

        if do_debug:
            debug_log(log_paths["save_load"], f"[SAVE] Saved successfully in {elapsed:.1f}s", force=True)

        alias_note = ""
        if checkpoint_name != name:
            alias_note = f"\nAlias requested: {name}"

        return f"""
ADAPTER SAVED
=============
Name: {checkpoint_name}{alias_note}
Path: {adapter_path}
State: epochs={session.sft_epochs}, grpo_iters={session.grpo_iterations}, sft_steps={session.sft_steps}

To load later: load("{name}") or load("{checkpoint_name}")
"""

    except Exception as e:
        if do_debug:
            debug_log(log_paths["save_load"], f"[SAVE] ERROR: {e}", force=True)
        return f"Error saving adapter: {e}"


@mcp.tool()
@serialize_session_tool
async def load(path: str, debug: bool = False) -> str:
    """Restore the model to a previously saved state.

    PURPOSE: Return to a known-good checkpoint. Use when:
    - Further training degraded quality and you want to backtrack
    - You want to branch: load a checkpoint, try different hyperparameters
    - You're resuming work from a previous session

    WHAT HAPPENS: The model's learned weights are replaced with the saved weights.
    Any training since that save is discarded.

    IMPORTANT: Loading resets to saved state. If you want to keep current progress,
    save() it first before loading a different checkpoint.
    """
    session = get_session()
    log_paths = get_log_paths()

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    if session.training_client is None:
        return (
            "Error: No training session active. Initialize a session first "
            "before loading adapters.\nExample: init_grpo(base_model) or "
            "init_sft(base_model)"
        )

    # Check known adapter names first, then accept explicit Tinker paths.
    prefixed_path = apply_checkpoint_prefix(path)
    if path in session.saved_adapters:
        adapter_path = session.saved_adapters[path]
    elif prefixed_path in session.saved_adapters:
        adapter_path = session.saved_adapters[prefixed_path]
    elif path.startswith("tinker://"):
        adapter_path = path
    else:
        return (
            "Error: Unknown checkpoint alias.\n\n"
            "Pass either:\n"
            "- A name returned by save() during THIS session (e.g., load(\"my_checkpoint\")), or\n"
            "- A full Tinker *state* path returned by save_state/save() (e.g., tinker://run-id/state/checkpoint-001).\n"
        )

    if do_debug:
        debug_log(log_paths["save_load"], f'[LOAD] path="{path}"', force=True)
        debug_log(log_paths["save_load"], f"[LOAD] Resolved to: {adapter_path}", force=True)

    try:
        start_time = time.time()
        client = await get_service_client_async()
        # save_state() checkpoints include optimizer state; restore it so
        # continued training resumes stably (no optimizer reset).
        session.training_client = await client.create_training_client_from_state_with_optimizer_async(adapter_path)

        # Recreate sampling client
        if do_debug:
            debug_log(log_paths["save_load"], "[LOAD] Creating sampling client from loaded weights...", force=True)
        # `save_weights_and_get_sampling_client_async()` ignores the name in
        # newer SDKs, but keep it safe for older/backwards-compatible behavior.
        safe_hint = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in (path or "")).strip("_")[:80]
        load_refresh_name = apply_checkpoint_prefix(f"loaded_{safe_hint}" if safe_hint else "loaded")
        if session.training_client is None:
            raise RuntimeError("Training client is None after loading checkpoint")
        session.sampling_client = await asyncio.wait_for(
            session.training_client.save_weights_and_get_sampling_client_async(load_refresh_name),
            timeout=TIMEOUT_LONG,
        )

        elapsed = time.time() - start_time

        # Restore scheduler state if metadata available
        metadata = (
            session.get_checkpoint_metadata(path)
            or session.get_checkpoint_metadata(prefixed_path)
            or session.get_checkpoint_metadata(adapter_path)
        )
        if metadata:
            session.restore_from_metadata(metadata)
            restored_state = True
        else:
            # Unknown legacy checkpoint state: keep counters non-zero to avoid
            # warmup restarts that can spike LR on resumed training.
            session.sft_epochs = max(session.sft_epochs, 1)
            session.grpo_iterations = max(session.grpo_iterations, 1)
            session.sft_steps = max(session.sft_steps, 1)
            restored_state = False

        if do_debug:
            debug_log(log_paths["save_load"], f"[LOAD] Loaded successfully in {elapsed:.1f}s", force=True)

        if restored_state:
            return f"""
ADAPTER LOADED
==============
Path: {adapter_path}

State restored: epochs={session.sft_epochs}, grpo_iters={session.grpo_iterations}, sft_steps={session.sft_steps}
LR scheduling will continue from saved step (no warmup restart).
Ready for training or inference.
"""
        else:
            return f"""
ADAPTER LOADED
==============
Path: {adapter_path}

WARNING: No saved state metadata found.
This checkpoint may have been created before metadata tracking was added.
Applied conservative non-zero counters (epochs={session.sft_epochs}, grpo_iters={session.grpo_iterations},
sft_steps={session.sft_steps}) to avoid warmup restart spikes.
If this checkpoint was actually from step 0, you can manually set warmup_ratio>0.
Ready for training or inference.
"""

    except Exception as e:
        if do_debug:
            debug_log(log_paths["save_load"], f"[LOAD] ERROR: {e}", force=True)
        return f"Error loading adapter: {e}"


@mcp.tool()
@serialize_session_tool
async def finish(debug: bool = False) -> str:
    """End the training session and finalize all tracking.

    PURPOSE: Clean up resources and close out the W&B run so metrics are fully
    recorded and the run appears as complete (not crashed/interrupted).

    WHEN TO CALL:
    - After saving your final adapter - training is done
    - Before starting a completely different training run
    - At the end of a session, even if training didn't complete successfully

    IMPORTANT: Call save() before finish() if you want to keep your trained weights!
    Calling finish() without save() means your training progress is lost.

    RETURNS: Final summary with total epochs/iterations and W&B run URL.
    """
    session = get_session()
    log_paths = get_log_paths()

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    if do_debug:
        debug_log(log_paths["save_load"], "[FINISH] Finalizing session...", force=True)

    summary = []
    finished_session_dir = session.session_dir

    if session.session_method == "sft":
        summary.append(f"Total SFT epochs: {session.sft_epochs}")
    elif session.session_method == "grpo":
        summary.append(f"Total GRPO iterations: {session.grpo_iterations}")

    # Import centralized timeout constants
    from tinker_mcp.utils import WANDB_FINISH_TIMEOUT

    # Finalize W&B with timeout
    wandb_url = ""
    if session.wandb_run is not None:
        try:
            wandb_url = session.wandb_run.url
            if do_debug:
                debug_log(log_paths["wandb"], "[W&B] finish: exit_code=0", force=True)

            # Force summary update before finish to ensure data is queued
            try:
                session.wandb_run.summary.update()
            except Exception:
                pass  # Non-critical

            await asyncio.wait_for(
                asyncio.to_thread(session.wandb_run.finish, exit_code=0),
                timeout=WANDB_FINISH_TIMEOUT,
            )
            summary.append(f"W&B run: {wandb_url}")
        except asyncio.TimeoutError:
            print(f"W&B finish timed out after {WANDB_FINISH_TIMEOUT}s.", file=sys.stderr, flush=True)
            print("Run data preserved locally. Manual sync: wandb sync wandb/latest-run", file=sys.stderr, flush=True)
            if wandb_url:
                summary.append(f"W&B run (may need manual sync): {wandb_url}")
        except Exception as e:
            print(f"W&B finish warning: {e}", file=sys.stderr, flush=True)

    # Reset session
    session.reset()

    # Clean up progress file for this session
    cleanup_progress_file()

    if do_debug:
        # session.reset() replaces session_dir, so recompute log paths to avoid
        # writing into a deleted directory.
        reset_log_paths = get_log_paths()
        debug_log(reset_log_paths["save_load"], "[FINISH] All global state reset complete", force=True)

    summary_text = "\n".join(summary) if summary else "No training recorded."

    return f"""
SESSION FINALIZED
=================
{summary_text}

Logs preserved at: {finished_session_dir}

Training session cleaned up. Start a new session with init_sft() or init_grpo().
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
