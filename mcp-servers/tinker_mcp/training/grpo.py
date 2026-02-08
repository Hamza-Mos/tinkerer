"""GRPO (Group Relative Policy Optimization) implementation for the Tinker MCP
Server.

GRPO is a reward-based learning method for tasks with verifiable correctness.
For each prompt, generate group_size completions, score each with the reward
function, then update the model to favor higher-scoring strategies.

Key insight: GRPO needs VARIANCE in rewards to learn - if all completions get
the same score, there's no signal about which is better.

This module contains the implementation logic. The MCP tool docstrings
remain in server.py as they are the "interface" for agent runtimes.
"""

import ast
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tinker import types as tinker_types
from tinker_cookbook import renderers as renderer_helpers
from tinker_mcp.models import (
    MODEL_INFO,
    SUPPORTED_MODELS,
    create_renderer_for_model,
)
from tinker_mcp.state import get_session
from tinker_mcp.training.common import (
    load_json_from_file_or_string,
    validate_input_schema,
    validate_warmup_for_continued_training,
    validate_lr_scheduler_name,
    resolve_scheduler_total_steps,
    cleanup_wandb_on_error,
    maybe_init_wandb_run,
    compute_importance_sampling_loss,
    extract_metrics,
    validate_reward_output,
)
from tinker_mcp.utils import (
    DEBUG_MODE,
    TIMEOUT_MEDIUM,
    TIMEOUT_LONG,
    WANDB_INIT_TIMEOUT,
    SAMPLING_TIMEOUT_BASE,
    SAMPLING_TOKENS_PER_SECOND,
    apply_checkpoint_prefix,
    debug_log,
    clear_debug_logs,
    get_log_paths,
    write_progress,
    wait_with_heartbeat,
    get_service_client_async,
    cleanup_stale_progress_files,
    get_sample_sequences,
    get_lr,
    log_warning,
    log_error,
)


@dataclass
class RewardBatchResult:
    """Result from batch reward computation, including error tracking."""

    rewards: list
    errors: list = field(default_factory=list)
    error_count: int = 0


@dataclass
class FailureStats:
    """Tracks failure types for root cause analysis."""

    uniform_reward_samples: int = 0
    # Samples from prompts where all rewards = 1.0 (model succeeding)
    uniform_all_ones: int = 0
    # Samples from prompts where all rewards = 0.0 (model failing)
    uniform_all_zeros: int = 0
    token_issue_samples: int = 0
    zero_token_samples: int = 0
    reward_error_samples: int = 0
    # API-level errors (e.g., token/length mismatches)
    api_validation_errors: int = 0
    prompts_with_no_signal: int = 0
    prompts_processed: int = 0
    samples_included: int = 0

    @property
    def dominant_failure_type(self) -> str:
        """Return the most common failure type."""
        failures = {
            "uniform_rewards": self.uniform_reward_samples,
            "token_issues": self.token_issue_samples,
            "zero_token": self.zero_token_samples,
            "reward_errors": self.reward_error_samples,
            "api_validation": self.api_validation_errors,
        }
        if sum(failures.values()) == 0:
            return "none"
        # Type-safe max: use lambda to avoid dict.get() type inference issues
        return max(failures, key=lambda k: failures[k])


def _get_root_cause_analysis(failures: FailureStats) -> tuple[str, str]:
    """Return (analysis, fix_suggestion) based on dominant failure type with
    percentages."""
    total_failures = (
        failures.uniform_reward_samples
        + failures.token_issue_samples
        + failures.zero_token_samples
        + failures.reward_error_samples
        + failures.api_validation_errors
    )

    if total_failures == 0:
        return "Training is healthy - low data loss.", "CONTINUE training normally"

    # Calculate sample-level skip rate
    total_attempted = failures.samples_included + total_failures
    sample_skip_rate = total_failures / total_attempted if total_attempted > 0 else 0

    # Calculate prompt-level skip rate (prompts that contributed nothing)
    prompt_skip_rate = (
        failures.prompts_with_no_signal / failures.prompts_processed if failures.prompts_processed > 0 else 0
    )

    dominant = failures.dominant_failure_type

    # Severity based on both metrics.
    #
    # Important nuance: for GRPO, "skips" from uniform ALL-1.0 rewards are often a
    # *success* signal (model consistently correct => no learning signal), not a
    # failure mode. We avoid labeling that as CRITICAL.
    if dominant == "uniform_rewards" and failures.uniform_all_ones > failures.uniform_all_zeros:
        if sample_skip_rate > 0.5 or prompt_skip_rate > 0.5:
            severity = "SUCCESS"
        elif sample_skip_rate > 0.3 or prompt_skip_rate > 0.3:
            severity = "GOOD"
        else:
            severity = "OK"
    else:
        if sample_skip_rate > 0.5 or prompt_skip_rate > 0.5:
            severity = "CRITICAL"
        elif sample_skip_rate > 0.3 or prompt_skip_rate > 0.3:
            severity = "WARNING"
        else:
            severity = "ACCEPTABLE"

    # Build breakdown string
    breakdown_parts = []
    if failures.uniform_reward_samples > 0:
        pct = failures.uniform_reward_samples / total_failures * 100
        breakdown_parts.append(f"uniform_rewards: {failures.uniform_reward_samples} ({pct:.0f}%)")
    if failures.token_issue_samples > 0:
        pct = failures.token_issue_samples / total_failures * 100
        breakdown_parts.append(f"token_issues: {failures.token_issue_samples} ({pct:.0f}%)")
    if failures.zero_token_samples > 0:
        pct = failures.zero_token_samples / total_failures * 100
        breakdown_parts.append(f"zero_token: {failures.zero_token_samples} ({pct:.0f}%)")
    if failures.reward_error_samples > 0:
        pct = failures.reward_error_samples / total_failures * 100
        breakdown_parts.append(f"reward_errors: {failures.reward_error_samples} ({pct:.0f}%)")
    if failures.api_validation_errors > 0:
        pct = failures.api_validation_errors / total_failures * 100
        breakdown_parts.append(f"api_errors: {failures.api_validation_errors} ({pct:.0f}%)")

    breakdown = " | ".join(breakdown_parts)

    # Stats summary
    stats = (
        f"Samples: {failures.samples_included}/{total_attempted} used ({sample_skip_rate:.0%} skipped). "
        f"Prompts: {failures.prompts_processed - failures.prompts_with_no_signal}/{failures.prompts_processed} "
        f"contributed ({prompt_skip_rate:.0%} empty)."
    )

    if dominant == "uniform_rewards":
        # Distinguish between uniform 1s (success) vs uniform 0s (failure)
        ones_pct = failures.uniform_all_ones / max(failures.uniform_reward_samples, 1) * 100
        zeros_pct = failures.uniform_all_zeros / max(failures.uniform_reward_samples, 1) * 100
        uniform_detail = (
            f"All 1s samples: {failures.uniform_all_ones} ({ones_pct:.0f}%), "
            f"All 0s samples: {failures.uniform_all_zeros} ({zeros_pct:.0f}%)"
        )

        if failures.uniform_all_ones > failures.uniform_all_zeros:
            # More all-1s than all-0s = model is succeeding consistently
            analysis = (
                f"[{severity}] {stats} "
                f"Uniform rewards dominated by ALL 1s (model succeeding consistently). "
                f"{uniform_detail}. Breakdown: {breakdown}"
            )
            fix = (
                "GOOD: Model is solving problems consistently. "
                "May be reaching ceiling. Call sample() to verify quality."
            )
        elif failures.uniform_all_zeros > failures.uniform_all_ones:
            # More all-0s than all-1s = model is failing consistently
            analysis = (
                f"[{severity}] {stats} "
                f"Uniform rewards dominated by ALL 0s (model failing consistently). "
                f"{uniform_detail}. Breakdown: {breakdown}"
            )
            if sample_skip_rate > 0.5:
                fix = (
                    "URGENT: Model stuck at 0.0. Try: "
                    "(1) Increase temperature to 0.9+, "
                    "(2) Check if task is too hard, "
                    "(3) Add partial credit"
                )
            else:
                fix = (
                    "FIX: Model struggling. Increase temperature OR simplify "
                    "task OR add partial credit to reward function."
                )
        else:
            # Mixed or unknown
            analysis = (
                f"[{severity}] {stats} "
                f"Uniform rewards: All completions got same score. "
                f"{uniform_detail}. Breakdown: {breakdown}"
            )
            fix = (
                "FIX: Increase temperature OR add partial credit to reward "
                "function. Some uniform rewards are normal."
            )
    elif dominant == "token_issues":
        analysis = (
            f"[{severity}] {stats} "
            f"Token/logprob mismatches: API didn't return valid logprobs. "
            f"Breakdown: {breakdown}"
        )
        fix = (
            "FIX: Reduce max_tokens OR check renderer prompt formatting/stop "
            "sequences. Model may be generating EOS immediately."
        )
    elif dominant == "zero_token":
        analysis = (
            f"[{severity}] {stats} "
            "Short-token generations (<2 completion tokens): cannot form shifted GRPO datum. "
            f"Breakdown: {breakdown}"
        )
        fix = (
            "FIX: Check renderer prompt formatting, confirm tokenizer "
            "compatibility, and verify max_tokens is reasonable."
        )
    elif dominant == "api_validation":
        analysis = (
            f"[{severity}] {stats} " "API validation errors: Tinker API rejected batch. " f"Breakdown: {breakdown}"
        )
        fix = (
            "FIX: Check Datum construction - model_input, target_tokens, " "logprobs, advantages must have same length"
        )
    else:  # reward_errors
        analysis = (
            f"[{severity}] {stats} "
            f"Reward function errors: Exceptions during reward computation. "
            f"Breakdown: {breakdown}"
        )
        fix = "FIX: Debug reward function - check for None handling, type errors, edge cases"

    return analysis, fix


def _extract_text_content(parsed_message) -> str:
    """Extract assistant text content from renderer.parse_response output."""
    return renderer_helpers.get_text_content(parsed_message)


def _is_api_validation_error(exc: Exception) -> bool:
    """Detect API validation errors from Tinker that indicate malformed Datum
    inputs."""
    msg = str(exc).lower()
    return ("input sequence" in msg and "target_tokens" in msg and "logprobs" in msg and "advantages" in msg) or (
        "token_count" in msg and "must have the same length" in msg
    )


def _format_exception(exc: Exception) -> str:
    """Return stable exception text, even when str(exc) is empty."""
    exc_type = type(exc).__name__
    detail = str(exc).strip()
    return f"{exc_type}: {detail}" if detail else exc_type


def _is_trainable_completion(tokens: Optional[list]) -> bool:
    """Return whether completion tokens can form a shifted GRPO datum.

    We append `tokens[:-1]` to the prompt and train against `tokens`, so
    completions shorter than 2 tokens cannot produce a valid training sample.
    """
    return bool(tokens) and len(tokens) >= 2


# Debug sampling logging controls.
# Default to a small number of prompts to keep logs useful but bounded.
# Set TINKERER_GRPO_SAMPLING_DEBUG_PROMPT_LIMIT=-1 for full per-prompt debug.
# Use 0 to disable per-prompt sampling debug details.
DEFAULT_SAMPLING_DEBUG_PROMPT_LIMIT = 3
SAMPLING_DEBUG_PROMPT_LIMIT_ENV = "TINKERER_GRPO_SAMPLING_DEBUG_PROMPT_LIMIT"

# Auto-checkpoint policy controls.
DEFAULT_AUTOCHECKPOINT_REWARD_THRESHOLD = 0.3
DEFAULT_AUTOCHECKPOINT_MIN_ITERATIONS = 3
AUTOCHECKPOINT_REWARD_THRESHOLD_ENV = "TINKERER_GRPO_AUTOCHECKPOINT_REWARD_THRESHOLD"
AUTOCHECKPOINT_MIN_ITERATIONS_ENV = "TINKERER_GRPO_AUTOCHECKPOINT_MIN_ITERATIONS"


def _resolve_sampling_debug_prompt_limit(requested_limit: Optional[int]) -> int:
    """Resolve sampling debug prompt limit from arg/env with validation."""
    if requested_limit is not None:
        limit = requested_limit
    else:
        raw_limit = os.environ.get(SAMPLING_DEBUG_PROMPT_LIMIT_ENV, str(DEFAULT_SAMPLING_DEBUG_PROMPT_LIMIT))
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(
                f"{SAMPLING_DEBUG_PROMPT_LIMIT_ENV} must be an integer >= -1, got: {raw_limit!r}"
            ) from exc
    if limit < -1:
        raise ValueError(f"sampling_debug_prompt_limit must be >= -1, got: {limit}")
    return limit


def _should_log_sampling_debug(
    *,
    do_debug: bool,
    prompt_index: int,
    sampling_debug_prompt_limit: int,
) -> bool:
    """Return whether per-prompt sampling debug should be emitted."""
    if not do_debug:
        return False
    if sampling_debug_prompt_limit < 0:
        return True
    return prompt_index < sampling_debug_prompt_limit


def _resolve_auto_checkpoint_policy(
    reward_threshold: Optional[float],
    min_iterations: Optional[int],
) -> tuple[float, int]:
    """Resolve auto-checkpoint policy from args/env with validation."""
    if reward_threshold is None:
        raw_threshold = os.environ.get(
            AUTOCHECKPOINT_REWARD_THRESHOLD_ENV,
            str(DEFAULT_AUTOCHECKPOINT_REWARD_THRESHOLD),
        )
        try:
            resolved_threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError(
                f"{AUTOCHECKPOINT_REWARD_THRESHOLD_ENV} must be a float >= 0, got: {raw_threshold!r}"
            ) from exc
    else:
        resolved_threshold = float(reward_threshold)

    if not np.isfinite(resolved_threshold) or resolved_threshold < 0:
        raise ValueError(
            f"auto_checkpoint_reward_threshold must be a finite float >= 0, got: {resolved_threshold}"
        )

    if min_iterations is None:
        raw_min_iters = os.environ.get(
            AUTOCHECKPOINT_MIN_ITERATIONS_ENV,
            str(DEFAULT_AUTOCHECKPOINT_MIN_ITERATIONS),
        )
        try:
            resolved_min_iterations = int(raw_min_iters)
        except ValueError as exc:
            raise ValueError(
                f"{AUTOCHECKPOINT_MIN_ITERATIONS_ENV} must be an int >= 0, got: {raw_min_iters!r}"
            ) from exc
    else:
        resolved_min_iterations = int(min_iterations)

    if resolved_min_iterations < 0:
        raise ValueError(f"auto_checkpoint_min_iterations must be >= 0, got: {resolved_min_iterations}")

    return resolved_threshold, resolved_min_iterations


# Checkpoint interval: update sampler weights every iteration to stay on-policy.
CHECKPOINT_INTERVAL = 1

# Maximum retry timeout (30 min) to stay below Modal watchdog kill threshold (35 min)
# This leaves 5 min buffer for error propagation and cleanup
MAX_RETRY_TIMEOUT = 1800


async def _sample_batch_async(sampling_client, model_input, sampling_params, num_samples: int, timeout):
    """Sample multiple completions in a single API call (batch sampling).

    Best practice: use num_samples=N in one call to encourage diversity among
    completions. Making N separate calls with num_samples=1 can reduce diversity
    (e.g., correlated randomness / similar seeds).

    Args:
        sampling_client: The Tinker sampling client
        model_input: Tokenized prompt
        sampling_params: Temperature, max_tokens, etc.
        num_samples: Number of independent samples to generate (group_size for GRPO)
        timeout: Maximum time to wait for sampling

    Returns:
        SampleResponse with num_samples samples
    """
    # CRITICAL: Tinker API methods return ApiFuture objects that need .result() called!
    # Pattern: Run sync method in thread, then resolve the ApiFuture.
    # This matches the working pattern from server.py sample() function.
    from tinker_mcp.utils import resolve_api_future

    # Run the sync sampling method in a thread to avoid blocking
    sample_future = await asyncio.wait_for(
        asyncio.to_thread(
            sampling_client.sample,  # Use sync .sample(), not .sample_async()
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=num_samples,  # Batch sample all completions at once
        ),
        timeout=timeout,
    )
    # Resolve the ApiFuture to get actual SampleResponse
    return resolve_api_future(sample_future, timeout=timeout)


def validate_reward_function_syntax(reward_function_code: str) -> tuple[bool, str]:
    """Validate reward function syntax and required callable contract.

    This catches syntax errors before execution, providing clearer error messages
    and preventing arbitrary code with syntax errors from being written to disk.

    Args:
        reward_function_code: Python code defining compute_reward(completion, ground_truth)

    Returns:
        Tuple of (is_valid, error_message). error_message is empty string if valid.

    Security Note:
        This validates syntax + function signature presence only. The reward
        function still executes arbitrary Python code in a subprocess.
    """
    try:
        tree = ast.parse(reward_function_code)
    except SyntaxError as e:
        # Provide helpful context about the syntax error
        error_msg = f"Invalid Python syntax in reward function at line {e.lineno}"
        if e.text:
            error_msg += f": {e.text.strip()}"
        if e.msg:
            error_msg += f"\n  {e.msg}"
        return False, error_msg
    except Exception as e:
        return False, f"Failed to parse reward function: {e}"

    compute_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "compute_reward"
    ]
    if not compute_defs:
        return (
            False,
            "Missing required function `compute_reward(completion, ground_truth)`.",
        )
    if len(compute_defs) > 1:
        return False, "Multiple `compute_reward` definitions found. Define exactly one."

    compute_def = compute_defs[0]
    if isinstance(compute_def, ast.AsyncFunctionDef):
        return False, "`compute_reward` must be a regular function, not `async def`."

    positional_args = list(compute_def.args.posonlyargs) + list(compute_def.args.args)
    if len(positional_args) < 2:
        return (
            False,
            "`compute_reward` must accept at least two positional args: `(completion, ground_truth)`.",
        )

    required_positional = len(positional_args) - len(compute_def.args.defaults)
    if required_positional > 2 and compute_def.args.vararg is None:
        return (
            False,
            "`compute_reward` cannot require more than two positional args. "
            "Only `(completion, ground_truth)` are provided by GRPO.",
        )

    required_kwonly = [
        arg.arg
        for arg, default in zip(compute_def.args.kwonlyargs, compute_def.args.kw_defaults)
        if default is None
    ]
    if required_kwonly:
        return (
            False,
            "`compute_reward` has required keyword-only args "
            f"({', '.join(required_kwonly)}), but GRPO only passes positional args.",
        )

    return True, ""


def compute_rewards_batch(
    reward_function_code: str, completions: list[str], ground_truth: str, timeout: int = 30, do_debug: bool = False
) -> RewardBatchResult:
    """Compute rewards for ALL completions in ONE subprocess (batch mode).

    This is much faster than spawning one subprocess per completion.
    All completions are processed in a single Python invocation.

    Args:
        reward_function_code: Python code defining compute_reward(completion, ground_truth)
        completions: List of completions to evaluate
        ground_truth: Expected answer for all completions
        timeout: Maximum seconds for the entire batch
        do_debug: Whether to log debug info

    Returns:
        RewardBatchResult with rewards list, errors list, and error_count
    """
    if not completions:
        return RewardBatchResult(rewards=[], errors=[], error_count=0)

    session = get_session()
    log_paths = get_log_paths()
    unique_id = uuid.uuid4().hex[:8]
    script_path = os.path.join(session.session_dir, f"batch_reward_{unique_id}.py")
    data_path = os.path.join(session.session_dir, f"batch_data_{unique_id}.json")

    # Build batch reward script that reads data from file (avoids CLI arg length limits)
    # Now tracks individual errors for better debugging
    script_content = f"""
import sys
import json
import io

# Capture any stdout from user code to prevent JSON parsing failures
_orig_stdout = sys.stdout
sys.stdout = io.StringIO()

{reward_function_code}

if __name__ == "__main__":
    sys.stdout = _orig_stdout
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    completions = data["completions"]
    ground_truth = data["ground_truth"]
    rewards = []
    errors = []
    for idx, comp in enumerate(completions):
        try:
            rewards.append(float(compute_reward(comp, ground_truth)))
        except Exception as e:
            rewards.append(0.0)
            errors.append({{"index": idx, "error": type(e).__name__, "msg": str(e)[:100]}})
    print(json.dumps({{"rewards": rewards, "errors": errors, "error_count": len(errors)}}))
"""

    start_time = time.time()

    if do_debug:
        debug_log(log_paths["reward"], f"[REWARD] Batch processing {len(completions)} completions", force=True)

    try:
        # Write script and data to files (avoids [Errno 36] File name too long)
        with open(script_path, "w") as f:
            f.write(script_content)
        with open(data_path, "w") as f:
            json.dump({"completions": completions, "ground_truth": ground_truth}, f)

        result = subprocess.run(
            ["python", script_path, data_path], capture_output=True, text=True, timeout=timeout, cwd=session.session_dir
        )

        elapsed = time.time() - start_time

        if result.returncode == 0 and result.stdout.strip():
            try:
                raw_output = json.loads(result.stdout.strip())
                validated = validate_reward_output(
                    raw_output, num_completions=len(completions), context="compute_rewards_batch"
                )
                rewards = validated["rewards"]
                errors = validated["errors"]
                error_count = validated["error_count"]
            except (TypeError, KeyError, ValueError, json.JSONDecodeError) as e:
                if do_debug:
                    debug_log(log_paths["reward"], f"[REWARD] Output validation FAILED: {e}", force=True)
                print(f"Reward output validation error: {e}", file=sys.stderr, flush=True)
                raise RuntimeError(f"Reward subprocess invalid output: {e}") from e

            if errors and do_debug:
                for err in errors:
                    debug_log(
                        log_paths["reward"],
                        f"[REWARD ERROR] Index {err['index']}: {err['error']}: {err['msg']}",
                        force=True,
                    )

            elapsed = time.time() - start_time
            if do_debug:
                debug_log(
                    log_paths["reward"],
                    f"[REWARD] Batch complete: {len(rewards)} rewards, {error_count} errors in {elapsed:.1f}s",
                    force=True,
                )
            return RewardBatchResult(rewards=rewards, errors=errors, error_count=error_count)
        else:
            if do_debug:
                debug_log(
                    log_paths["reward"],
                    f"[REWARD] Batch failed: returncode={result.returncode}, stderr={result.stderr[:200]}",
                    force=True,
                )
            print(f"Reward batch error: {result.stderr[:200]}", file=sys.stderr, flush=True)
            return RewardBatchResult(rewards=[0.0] * len(completions), errors=[], error_count=len(completions))

    except subprocess.TimeoutExpired:
        if do_debug:
            debug_log(log_paths["reward"], f"[REWARD] Batch timeout after {timeout}s", force=True)
        print(f"Reward batch timed out after {timeout}s", file=sys.stderr, flush=True)
        return RewardBatchResult(rewards=[0.0] * len(completions), errors=[], error_count=len(completions))
    except Exception as e:
        if do_debug:
            debug_log(log_paths["reward"], f"[REWARD] Batch error: {e}", force=True)
        print(f"Reward batch error: {e}", file=sys.stderr, flush=True)
        return RewardBatchResult(rewards=[0.0] * len(completions), errors=[], error_count=len(completions))
    finally:
        # Clean up temporary files
        for path in [script_path, data_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


async def init_grpo_session(
    base_model: str,
    lora_rank: int = 32,
    group_size: int | None = None,
    wandb_project: str = "tinkerer",
    wandb_run_name: Optional[str] = None,
    debug: bool = False,
) -> str:
    """Initialize a GRPO training session for reward-based learning.

    Args:
        base_model: Model to fine-tune
        lora_rank: Adapter capacity (32+ often needed for RL)
        group_size: Completions per prompt (None = use model-specific default from MODEL_INFO, fallback to 4)
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        debug: Enable debug logging

    Returns:
        Status message with session info
    """
    session = get_session()
    log_paths = get_log_paths()

    # Model-aware group_size default: use MODEL_INFO if available, else 4
    if group_size is None:
        model_info = MODEL_INFO.get(base_model, {})
        group_size = model_info.get("recommended_group_size", 4)

    # Clean up stale progress files from crashed sessions (non-blocking)
    cleanup_stale_progress_files(max_age_hours=6)

    # Write progress immediately for watchdog heartbeat (before any slow operations)
    # This ensures the progress file exists during model loading which can
    # take 5-15+ minutes
    await write_progress(f"Initializing GRPO session: {base_model}")

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    # Clear debug logs at session start (but NOT the progress file - it was
    # just created above)
    if do_debug:
        clear_debug_logs()
        debug_log(
            log_paths["init"],
            f"[START] init_grpo: model={base_model}, lora_rank={lora_rank}, group_size={group_size}",
            force=True,
        )

    # Model registry is "known good", not a hard gate. Unlisted models can work
    # as long as the backend supports the model ID and the renderer mapping can
    # be resolved.
    if base_model not in SUPPORTED_MODELS:
        note_msg = (
            f"Model '{base_model}' is not in the harness MODEL_INFO registry. "
            "Proceeding anyway; defaults will be used for metadata and group_size, "
            "and renderer mapping will be resolved automatically."
        )
        log_warning(note_msg)
        if do_debug:
            debug_log(log_paths["init"], f"[NOTE] {note_msg}", force=True)
        await write_progress(f"NOTE: {note_msg}")

    # Non-Base models are supported, but Base models typically provide stronger
    # exploration diversity for GRPO.
    model_info = MODEL_INFO.get(base_model, {})
    training_type = model_info.get("training_type")
    if training_type and training_type != "Base":
        warning_msg = (
            f"Model '{base_model}' is tagged as '{training_type}'. "
            "GRPO is allowed, but Base models usually provide more stable variance signals."
        )
        log_warning(warning_msg)
        if do_debug:
            debug_log(
                log_paths["init"],
                f"[WARNING] {warning_msg}",
                force=True,
            )
        await write_progress(f"WARNING: {warning_msg}")

    try:
        start_time = time.time()

        if do_debug:
            debug_log(log_paths["init"], "[INFO] Creating service client...", force=True)
        client_start = time.time()
        client = await get_service_client_async()
        if do_debug:
            client_elapsed = time.time() - client_start
            debug_log(log_paths["init"], f"[INFO] Service client created ({client_elapsed:.1f}s)", force=True)

        if do_debug:
            debug_log(log_paths["init"], "[INFO] Creating LoRA training client...", force=True)
        training_start = time.time()

        # LoRA training client creation with timeout
        # Large models (e.g., 30B) can take 15-30 minutes to provision
        # Use MAX_RETRY_TIMEOUT (30 min) to stay under Modal watchdog (35 min)
        try:
            if do_debug:
                debug_log(
                    log_paths["init"],
                    f"[INFO] LoRA client creation (timeout: {MAX_RETRY_TIMEOUT}s / 30 min)...",
                    force=True,
                )

            session.training_client = await wait_with_heartbeat(
                client.create_lora_training_client_async(base_model=base_model, rank=lora_rank),
                timeout=MAX_RETRY_TIMEOUT,
                progress_msg="Creating LoRA training client",
                interval=60.0,
                debug_log_file=log_paths["init"] if do_debug else None,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - training_start
            error_msg = (
                "LoRA training client creation timed out after "
                f"{elapsed:.0f}s. The model '{base_model}' may require longer "
                "provisioning time or the Tinker API may be experiencing "
                "issues."
            )
            log_warning(error_msg)
            if do_debug:
                debug_log(log_paths["init"], f"[ERROR] {error_msg}", force=True)
            return (
                f"Error: {error_msg}\nTry a smaller model or check Tinker API status."
            )

        if do_debug:
            train_elapsed = time.time() - training_start
            debug_log(log_paths["init"], f"[INFO] Training client created ({train_elapsed:.1f}s)", force=True)

        if do_debug:
            debug_log(log_paths["init"], "[INFO] Loading tokenizer...", force=True)
        session.tokenizer = session.training_client.get_tokenizer()
        if do_debug:
            debug_log(log_paths["init"], f"[INFO] Tokenizer loaded: {type(session.tokenizer).__name__}", force=True)

        # Initialize renderer (prompt/stop/parse handling).
        session.renderer, session.renderer_name, session.image_processor = create_renderer_for_model(
            base_model, session.tokenizer
        )
        if do_debug:
            debug_log(
                log_paths["init"],
                f"[INFO] Renderer loaded: {session.renderer_name}",
                force=True,
            )

        session.current_model = base_model
        session.session_method = "grpo"
        session.grpo_iterations = 0
        # group_size defaults to 4 if not provided
        session.grpo_group_size = group_size if group_size is not None else 4
        session.grpo_scheduler_total_steps = 0
        session.sampling_client = None

        # Eagerly create sampling client so sample() works immediately after init.
        # Best practice: keep a sampling client available before the first rollout.
        init_sampler_name = apply_checkpoint_prefix(f"grpo_init_{uuid.uuid4().hex[:8]}")
        try:
            if do_debug:
                debug_log(
                    log_paths["init"],
                    f"[INFO] Creating initial sampling client: {init_sampler_name}",
                    force=True,
                )
            session.sampling_client = await wait_with_heartbeat(
                session.training_client.save_weights_and_get_sampling_client_async(init_sampler_name),
                timeout=TIMEOUT_LONG,
                progress_msg="Creating initial sampling client",
                interval=60.0,
                debug_log_file=log_paths["init"] if do_debug else None,
            )
        except Exception as e:
            # Non-fatal: train_grpo_step will retry sampling client creation if needed.
            session.sampling_client = None
            await write_progress(
                "WARNING: Initial sampling client creation failed; sample() may be slower until first train_grpo_step."
            )
            if do_debug:
                debug_log(log_paths["init"], f"[WARNING] Initial sampling client creation failed: {e}", force=True)

        # Initialize W&B (optional).
        wandb_status = await maybe_init_wandb_run(
            session=session,
            method="grpo",
            base_model=base_model,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            group_default=f"{base_model.split('/')[-1]}-grpo-{time.strftime('%Y%m%d')}",
            name_components=[f"g{group_size}", f"r{lora_rank}"],
            step_metric="iteration",
            config={
                "lora_rank": lora_rank,
                "group_size": group_size,
            },
            extra_tags=None,
            do_debug=do_debug,
            log_paths=log_paths,
            timeout_s=WANDB_INIT_TIMEOUT,
        )

        total_time = time.time() - start_time
        if do_debug:
            debug_log(log_paths["init"], f"[SUCCESS] init_grpo complete (total: {total_time:.1f}s)", force=True)

        # Get model info for enhanced output
        model_info = MODEL_INFO.get(base_model, {})
        arch = model_info.get("architecture", "unknown")
        active = model_info.get("active_params", model_info.get("total_params", "unknown"))
        rec_gs = model_info.get("recommended_group_size", "N/A")
        sampling_client_status = "ready" if session.sampling_client is not None else "deferred (created on first train)"

        return f"""
GRPO SESSION INITIALIZED
========================
Model: {base_model}
  Architecture: {arch}
  Active params: {active}
  Recommended group_size: {rec_gs} (you chose: {group_size})

Config:
  lora_rank: {lora_rank}
  group_size: {group_size}
  sampling_client: {sampling_client_status}
{wandb_status}

GENERAL GUIDANCE:
- Prefer constant learning rates unless you explicitly plan a decay horizon.
  If you use linear/cosine across many small calls, pass scheduler_total_steps.
- Set max_tokens to comfortably fit a full solution AND a final answer.
  If rewards are all 0.0 and outputs look truncated, increase max_tokens.
- Temperature controls exploration: raise it if samples are too similar (low variance),
  lower it if outputs are chaotic and rewards are noisy.

NEXT: Call train_grpo_step(prompts, reward_function, num_iterations=1) to train.
      Use num_iterations=1 for full control over the training loop.
"""

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        # Always log the full traceback for init failures: without it, users
        # have to guess whether the issue was provisioning, auth, renderer, etc.
        debug_log(log_paths["init"], f"[ERROR] init_grpo failed: {e}\n{tb}", force=True)
        # Init failure is fatal for this run; close W&B cleanly.
        cleanup_wandb_on_error(session, do_debug, log_paths, fatal=True, reason="init_grpo failed")
        if do_debug:
            return f"Error initializing GRPO session: {e}\n{tb}"
        return f"Error initializing GRPO session: {e}"


@dataclass
class PromptProcessingOutcome:
    """Outcome of processing one prompt within a GRPO iteration."""

    datums: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    skipped_samples: int = 0
    uniform_all_ones: int = 0
    uniform_all_zeros: int = 0
    token_logprob_mismatches: int = 0
    zero_token_samples: int = 0
    reward_errors: int = 0
    included_samples: int = 0
    prompt_failed: bool = False
    sample_debug_item: Optional[dict] = None


def _to_float(val):
    """Best-effort float conversion for optional metric fields."""
    try:
        return float(val)
    except Exception:
        return None


async def _ensure_grpo_sampling_client(session, do_debug: bool, log_paths: dict) -> Optional[str]:
    """Create initial sampling client if missing.

    Returns an error message if creation fails, else None.
    """
    if session.sampling_client is not None:
        return None

    if do_debug:
        debug_log(log_paths["training"], "[INFO] Creating initial sampling client...", force=True)

    sampling_client_created = False
    init_checkpoint_name = apply_checkpoint_prefix(f"grpo_init_{uuid.uuid4().hex[:8]}")
    for attempt in range(3):
        try:
            retry_timeout = min(TIMEOUT_LONG * (1 + attempt * 0.3), MAX_RETRY_TIMEOUT)
            session.sampling_client = await wait_with_heartbeat(
                session.training_client.save_weights_and_get_sampling_client_async(init_checkpoint_name),
                timeout=retry_timeout,
                progress_msg="Creating initial sampling client",
                interval=60.0,
                debug_log_file=log_paths["training"] if do_debug else None,
            )
            sampling_client_created = True
            break
        except asyncio.TimeoutError:
            if attempt < 2:
                retry_msg = f"Sampling client init attempt {attempt + 1} timed out after {retry_timeout}s, retrying..."
                await write_progress(retry_msg)
                if do_debug:
                    debug_log(log_paths["training"], f"[WARNING] {retry_msg}", force=True)
                await asyncio.sleep(5 * (attempt + 1))
            else:
                raise

    if not sampling_client_created:
        return "Error: Failed to create sampling client after 3 attempts. Check API connectivity."

    if do_debug:
        debug_log(log_paths["training"], "[INFO] Sampling client created", force=True)
    return None


async def _process_grpo_prompt(
    *,
    session,
    prompt_text: str,
    ground_truth: str,
    prompt_index: int,
    prompts_total: int,
    iteration_index: int,
    num_iterations: int,
    reward_function_code: str,
    temperature: float,
    max_tokens: int,
    do_debug: bool,
    log_paths: dict,
    include_sample_debug: bool,
    sampling_debug_prompt_limit: int,
) -> PromptProcessingOutcome:
    """Generate, score, and construct training datums for one GRPO prompt."""
    outcome = PromptProcessingOutcome()
    renderer = session.renderer
    prompt_num = prompt_index + 1
    sample_debug_enabled = _should_log_sampling_debug(
        do_debug=do_debug,
        prompt_index=prompt_index,
        sampling_debug_prompt_limit=sampling_debug_prompt_limit,
    )

    await write_progress(f"GRPO iter {iteration_index + 1}/{num_iterations} prompt {prompt_num}/{prompts_total}")

    if sample_debug_enabled:
        debug_log(
            log_paths["sampling"],
            f'[SAMPLE] Prompt {prompt_num}: "{prompt_text[:80]}..." ({session.grpo_group_size} completions)',
            force=True,
        )

    try:
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)
    except Exception as e:
        log_warning(f"SKIPPING prompt {prompt_num}: renderer.build_generation_prompt failed: {_format_exception(e)}")
        outcome.prompt_failed = True
        return outcome

    if model_input.length < 1:
        log_warning(f"SKIPPING prompt {prompt_num}: prompt encoded to 0 tokens; cannot build training datum.")
        outcome.prompt_failed = True
        return outcome

    ob_len = model_input.length - 1
    stop_sequences = renderer.get_stop_sequences()
    sampling_params = tinker_types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )
    sampling_timeout = min(SAMPLING_TIMEOUT_BASE + (max_tokens / SAMPLING_TOKENS_PER_SECOND), TIMEOUT_LONG)

    await write_progress(
        f"GRPO iter {iteration_index + 1} prompt {prompt_num} "
        f"sampling {session.grpo_group_size} completions (batch)"
    )

    sample_start = time.time()
    try:
        sample_response = await _sample_batch_async(
            session.sampling_client,
            model_input,
            sampling_params,
            num_samples=session.grpo_group_size,
            timeout=sampling_timeout,
        )
        batch_time = time.time() - sample_start

        if sample_debug_enabled:
            sequences = get_sample_sequences(sample_response, log_file=log_paths["sampling"])
            num_samples = len(sequences)
            debug_log(
                log_paths["sampling"],
                f"[SAMPLE] Batch sampling complete: {num_samples} samples in {batch_time:.1f}s",
                force=True,
            )
    except Exception as e:
        batch_time = time.time() - sample_start
        err_detail = _format_exception(e)
        await write_progress(f"Batch sampling failed for prompt {prompt_num}: {err_detail}")
        if do_debug:
            debug_log(
                log_paths["sampling"],
                f"[SAMPLE] Batch FAILED after {batch_time:.1f}s for prompt {prompt_num}: {err_detail}",
                force=True,
            )
        outcome.prompt_failed = True
        return outcome

    sampling_log_file: str | None = log_paths.get("sampling") if do_debug else None
    sequences = get_sample_sequences(sample_response, log_file=sampling_log_file)
    if len(sequences) != session.grpo_group_size:
        raise ValueError(
            f"SampleResponse.sequences length {len(sequences)} "
            f"!= requested num_samples {session.grpo_group_size}"
        )

    completions = []
    tokens_list = []
    logprobs_list = []
    zero_token_samples = 0

    for comp_idx, seq in enumerate(sequences):
        generated_tokens = list(seq.tokens) if seq.tokens else []
        if not generated_tokens:
            if sample_debug_enabled:
                debug_log(log_paths["sampling"], f"  Completion {comp_idx}: 0 tokens (skipped)", force=True)
            zero_token_samples += 1
            outcome.zero_token_samples += 1
            continue

        try:
            parsed_message, parse_ok = renderer.parse_response(generated_tokens)
            generated_text = _extract_text_content(parsed_message)
            if not generated_text or not generated_text.strip():
                raise ValueError("parse_response produced empty assistant text")
            if sample_debug_enabled and not parse_ok:
                debug_log(
                    log_paths["sampling"],
                    f"  Completion {comp_idx}: parse_ok=False; using parsed assistant text",
                    force=True,
                )
        except Exception as e:
            log_warning(f"SKIPPING completion {comp_idx}: renderer.parse_response failed: {e}")
            outcome.reward_errors += 1
            continue

        if seq.logprobs is None:
            raise ValueError(f"Completion {comp_idx}: missing logprobs (required by importance_sampling)")
        sample_logprobs = list(seq.logprobs)
        if len(sample_logprobs) != len(generated_tokens):
            raise ValueError(
                f"Completion {comp_idx}: logprobs length {len(sample_logprobs)} "
                f"!= tokens length {len(generated_tokens)}"
            )
        if any(lp is None for lp in sample_logprobs):
            raise ValueError(f"Completion {comp_idx}: logprobs contains None values")

        completions.append(generated_text)
        tokens_list.append(generated_tokens)
        logprobs_list.append(sample_logprobs)

        if sample_debug_enabled:
            tokens_count = len(generated_tokens)
            logprobs_count = len(sample_logprobs)
            match_status = "MATCH" if tokens_count == logprobs_count else "MISMATCH"
            debug_log(
                log_paths["sampling"],
                f"  Completion {comp_idx}: {tokens_count} tokens, {logprobs_count} logprobs ({match_status})",
                force=True,
            )

    reward_timeout = 30 + (session.grpo_group_size * 15)
    reward_start = time.time()
    reward_result = compute_rewards_batch(
        reward_function_code,
        completions,
        ground_truth,
        timeout=reward_timeout,
        do_debug=do_debug,
    )
    rewards = reward_result.rewards
    outcome.reward_errors += reward_result.error_count
    reward_time = time.time() - reward_start

    if do_debug:
        debug_log(
            log_paths["training"],
            f"[REWARD] Prompt {prompt_num}: reward computation took {reward_time:.1f}s",
            force=True,
        )
        debug_log(
            log_paths["training"],
            f"[REWARD] Prompt {prompt_num}: rewards={[f'{r:.2f}' for r in rewards]}, "
            f"mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}",
            force=True,
        )

    if include_sample_debug:
        outcome.sample_debug_item = {
            "prompt": prompt_text[:100],
            "completion": completions[0][:200] if completions else "",
            "reward": rewards[0] if rewards else 0.0,
        }

    outcome.rewards = rewards

    if len(rewards) == 0:
        log_warning(
            f"SKIPPING prompt {prompt_num}: All {session.grpo_group_size} completions failed to generate. "
            f'Prompt: "{prompt_text[:50]}..."'
        )
        outcome.prompt_failed = True
        if do_debug:
            debug_log(log_paths["training"], f"[SKIP] Prompt {prompt_num}: 0 valid completions", force=True)
        return outcome

    if len(rewards) == 1:
        outcome.skipped_samples += 1
        outcome.prompt_failed = True
        log_warning(
            f"SKIPPING prompt {prompt_num}: Only 1 valid completion (need >=2 for variance). "
            f'Reward={rewards[0]:.2f}. Prompt: "{prompt_text[:50]}..."'
        )
        if do_debug:
            debug_log(
                log_paths["training"],
                f"[SKIP] Prompt {prompt_num}: only 1 completion (reward={rewards[0]:.2f}), need >=2 for variance",
                force=True,
            )
        return outcome

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    if std_r < 1e-6:
        outcome.skipped_samples += len(rewards)
        outcome.prompt_failed = True
        if mean_r > 0.9:
            outcome.uniform_all_ones += len(rewards)
        elif mean_r < 0.1:
            outcome.uniform_all_zeros += len(rewards)
        log_warning(
            f"SKIPPING prompt {prompt_num}: Zero variance in rewards (all {rewards[0]:.2f}). "
            f'No learning signal. Prompt: "{prompt_text[:50]}..."'
        )
        if do_debug:
            rewards_preview = [f"{r:.2f}" for r in rewards]
            debug_log(
                log_paths["training"],
                f"[SKIP] Prompt {prompt_num}: uniform rewards (std={std_r:.6f}), rewards={rewards_preview}",
                force=True,
            )
        return outcome

    std_r = std_r + 1e-8
    advantages = [max(-5.0, min(5.0, (r - mean_r) / std_r)) for r in rewards]
    if sample_debug_enabled:
        debug_log(
            log_paths["training"],
            f"[ADVANTAGE] advantages={[f'{a:.2f}' for a in advantages[:4]]}..., pre_loop_batch_size=0",
            force=True,
        )

    for toks, lp, adv in zip(tokens_list, logprobs_list, advantages):
        if not _is_trainable_completion(toks):
            zero_token_samples += 1
            outcome.zero_token_samples += 1
            if do_debug:
                debug_log(
                    log_paths["training"],
                    "[SKIP] Sample has <2 completion tokens; cannot build shifted GRPO datum",
                    force=True,
                )
            continue
        if lp is None or len(lp) == 0:
            outcome.token_logprob_mismatches += 1
            log_warning(f"SKIPPING sample: No logprobs returned for {len(toks)} tokens. API may have failed.")
            continue
        if len(toks) != len(lp):
            outcome.token_logprob_mismatches += 1
            log_warning(f"SKIPPING sample: tokens/logprobs length mismatch: {len(toks)} tokens vs {len(lp)} logprobs")
            continue

        model_input_full = model_input.append(tinker_types.EncodedTextChunk(tokens=toks[:-1]))
        target_tokens = [0] * ob_len + list(toks)
        padded_logprobs = [0.0] * ob_len + list(lp)
        padded_advantages = [0.0] * ob_len + [adv] * (model_input_full.length - ob_len)
        expected_len = model_input_full.length
        if not (len(target_tokens) == len(padded_logprobs) == len(padded_advantages) == expected_len):
            outcome.token_logprob_mismatches += 1
            log_warning(
                "SKIPPING sample: length mismatch after padding "
                f"(model_input={expected_len}, target={len(target_tokens)}, "
                f"logprobs={len(padded_logprobs)}, advantages={len(padded_advantages)})"
            )
            continue

        # Type ignore: numpy arrays are compatible with TensorData at runtime
        # but type checker doesn't recognize the conversion
        datum = tinker_types.Datum(
            model_input=model_input_full,
            loss_fn_inputs={  # type: ignore[arg-type]
                "target_tokens": np.asarray(target_tokens, dtype=np.int64),
                "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
                "advantages": np.asarray(padded_advantages, dtype=np.float32),
            },
        )
        outcome.datums.append(datum)
        outcome.included_samples += 1

    if outcome.token_logprob_mismatches > 0 or zero_token_samples > 0:
        total_attempted = len(sequences)
        await write_progress(
            f"Prompt {prompt_num}: {outcome.included_samples}/{total_attempted} included, "
            f"{outcome.token_logprob_mismatches} logprob issues, {zero_token_samples} short-token(<2)"
        )
        complete_failures = outcome.token_logprob_mismatches + zero_token_samples
        failure_ratio = complete_failures / total_attempted if total_attempted > 0 else 0
        if failure_ratio > 0.5:
            warning_msg = (
                f"WARNING: Prompt {prompt_num} has {failure_ratio:.0%} sample failures "
                f"({outcome.token_logprob_mismatches} logprob issues, {zero_token_samples} short-token(<2)). "
                f"Skipping this prompt. If many prompts fail, check: "
                f"(1) max_tokens={max_tokens} may be too high, "
                f"(2) renderer prompt formatting issues for model {session.current_model}."
            )
            await write_progress(warning_msg)
            if do_debug:
                debug_log(log_paths["training"], f"[WARNING] {warning_msg}", force=True)
            outcome.datums = []
            outcome.included_samples = 0

    if outcome.included_samples == 0:
        outcome.prompt_failed = True
        if do_debug:
            debug_log(
                log_paths["training"],
                f"[SKIP] Prompt {prompt_num}: no valid samples (all failed token/logprob checks)",
                force=True,
            )

    if do_debug:
        debug_log(
            log_paths["training"],
            f"[BATCH] Prompt {prompt_num}: added {outcome.included_samples} samples",
            force=True,
        )

    return outcome


async def _run_grpo_optimization_step(
    *,
    session,
    batch_data: list,
    num_iterations: int,
    iteration_index: int,
    learning_rate: float,
    warmup_ratio: float,
    lr_scheduler: str,
    scheduler_total_steps: int,
    do_debug: bool,
    log_paths: dict,
    failure_stats: FailureStats,
):
    """Execute forward_backward + optim_step and return optimization metrics."""
    # Guard against empty batch - API will error if called with empty list
    if not batch_data:
        raise ValueError("Cannot run optimization step with empty batch_data")

    if do_debug:
        debug_log(
            log_paths["training"],
            f"[TRAIN] forward_backward starting with {len(batch_data)} samples...",
            force=True,
        )

    train_start = time.time()
    estimated_total_steps = scheduler_total_steps
    current_cumulative_step = session.grpo_iterations + iteration_index
    current_lr = get_lr(
        step=current_cumulative_step,
        total_steps=estimated_total_steps,
        base_lr=learning_rate,
        warmup_ratio=warmup_ratio,
        scheduler=lr_scheduler,
    )

    try:
        fwd_bwd_future = await asyncio.wait_for(
            session.training_client.forward_backward_async(batch_data, loss_fn="importance_sampling"),
            timeout=TIMEOUT_MEDIUM,
        )
        optim_start = time.time()
        optim_future = await asyncio.wait_for(
            session.training_client.optim_step_async(
                tinker_types.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95)
            ),
            timeout=TIMEOUT_MEDIUM,
        )
        fwd_bwd_result = await wait_with_heartbeat(
            asyncio.to_thread(fwd_bwd_future.result, TIMEOUT_MEDIUM),
            timeout=TIMEOUT_MEDIUM,
            progress_msg=f"GRPO iter {iteration_index + 1}: waiting for forward_backward",
            interval=60.0,
            debug_log_file=log_paths["training"] if do_debug else None,
        )
    except Exception as e:
        if _is_api_validation_error(e):
            failure_stats.api_validation_errors += len(batch_data)
            api_msg = (
                "Error: Tinker API validation error during forward_backward. "
                "This usually means len(model_input) == len(target_tokens) == len(logprobs) == "
                "len(advantages) was violated. Check Datum construction and padding."
            )
            await write_progress(api_msg)
            if do_debug:
                debug_log(log_paths["training"], f"[ERROR] {api_msg} Original: {e}", force=True)
            # Keep W&B run open for retryable API-shape/data errors.
            cleanup_wandb_on_error(
                session,
                do_debug,
                log_paths,
                fatal=False,
                reason="train_grpo_step api validation error",
            )
            return None, None, current_lr, f"{api_msg}\nOriginal error: {e}"
        raise

    fwdbwd_time = time.time() - train_start
    if fwd_bwd_result is None:
        raise RuntimeError("forward_backward returned None - API call failed unexpectedly")

    # Type ignore: fwd_bwd_result is ForwardBackwardOutput but type checker sees object
    # Runtime checks ensure attributes exist
    if do_debug:
        lfo = fwd_bwd_result.loss_fn_outputs  # type: ignore[attr-defined]
        lfo_type = type(lfo).__name__
        lfo_len = len(lfo) if hasattr(lfo, "__len__") else "N/A"
        lfo_sample = repr(lfo[:2] if isinstance(lfo, list) and len(lfo) >= 2 else lfo)[:500]
        debug_log(
            log_paths["training"],
            f"[DEBUG] loss_fn_outputs: type={lfo_type}, len={lfo_len}, sample={lfo_sample}",
            force=True,
        )

    loss_sum = compute_importance_sampling_loss(
        fwd_bwd_result.loss_fn_outputs, batch_data, context="GRPO forward_backward"  # type: ignore[attr-defined]
    )
    if do_debug:
        debug_log(
            log_paths["training"],
            f"[TRAIN] forward_backward complete: {fwdbwd_time:.1f}s, "
            f"batch_size={len(batch_data)}, loss:sum={loss_sum:.4f}",
            force=True,
        )

    kl_v1 = None
    kl_v2 = None
    metrics = extract_metrics(fwd_bwd_result.metrics, context="GRPO forward_backward")  # type: ignore[attr-defined]
    kl_v1 = _to_float(metrics.get("kl_sample_train_v1"))
    kl_v2 = _to_float(metrics.get("kl_sample_train_v2"))

    await wait_with_heartbeat(
        asyncio.to_thread(optim_future.result, TIMEOUT_MEDIUM),
        timeout=TIMEOUT_MEDIUM,
        progress_msg=f"GRPO iter {iteration_index + 1}: waiting for optim_step",
        interval=60.0,
        debug_log_file=log_paths["training"] if do_debug else None,
    )
    optim_time = time.time() - optim_start
    if do_debug:
        debug_log(
            log_paths["training"],
            f"[TRAIN] optim_step: {optim_time:.1f}s, lr={current_lr:.2e} "
            f"(step {current_cumulative_step + 1}/{estimated_total_steps}, scheduler={lr_scheduler})",
            force=True,
        )

    return kl_v1, kl_v2, current_lr, None


async def _maybe_refresh_grpo_sampling_client(
    *,
    session,
    iteration_index: int,
    num_iterations: int,
    do_debug: bool,
    log_paths: dict,
) -> Optional[str]:
    """Checkpoint and refresh sampling client per configured interval."""
    should_checkpoint = (iteration_index + 1) % CHECKPOINT_INTERVAL == 0 or iteration_index == num_iterations - 1
    if not should_checkpoint:
        if do_debug:
            debug_log(
                log_paths["training"],
                f"[WEIGHTS] Skipping checkpoint (interval={CHECKPOINT_INTERVAL}, iter={iteration_index + 1})",
                force=True,
            )
        return None

    checkpoint_name = apply_checkpoint_prefix(f"grpo_iter_{session.grpo_iterations}")
    if do_debug:
        debug_log(
            log_paths["training"],
            f"[WEIGHTS] Saving checkpoint {checkpoint_name}...",
            force=True,
        )

    checkpoint_saved = False
    checkpoint_start = time.time()
    for attempt in range(3):
        try:
            retry_timeout = min(TIMEOUT_LONG * (1 + attempt * 0.3), MAX_RETRY_TIMEOUT)
            session.sampling_client = await wait_with_heartbeat(
                session.training_client.save_weights_and_get_sampling_client_async(checkpoint_name),
                timeout=retry_timeout,
                progress_msg=f"Saving checkpoint {checkpoint_name}",
                interval=60.0,
                debug_log_file=log_paths["training"] if do_debug else None,
            )
            checkpoint_saved = True
            break
        except asyncio.TimeoutError:
            if attempt < 2:
                retry_msg = f"Checkpoint save attempt {attempt + 1} timed out after {retry_timeout}s, retrying..."
                await write_progress(retry_msg)
                await asyncio.sleep(5 * (attempt + 1))
            else:
                raise

    if not checkpoint_saved:
        return f"CRITICAL: Checkpoint failed after 3 attempts at iteration {iteration_index + 1}"

    if do_debug:
        checkpoint_elapsed = time.time() - checkpoint_start
        debug_log(log_paths["training"], f"[WEIGHTS] Checkpoint saved ({checkpoint_elapsed:.1f}s)", force=True)
    return None


async def train_grpo_step_impl(
    prompts: str,
    reward_function: str,
    num_iterations: int = 1,
    learning_rate: float = 4e-5,
    warmup_ratio: float = 0.1,
    lr_scheduler: str = "constant",
    scheduler_total_steps: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 24576,
    auto_checkpoint: bool = True,
    debug: bool = False,
    sampling_debug_prompt_limit: Optional[int] = None,
    auto_checkpoint_reward_threshold: Optional[float] = None,
    auto_checkpoint_min_iterations: Optional[int] = None,
) -> str:
    """Run GRPO training iterations with reward-based learning. Implementation
    function.

    Args:
        prompts: JSON array of {"prompt": "...", "ground_truth": "..."} OR a file path to a JSON file
        reward_function: Python code defining compute_reward(completion, ground_truth)
        num_iterations: Number of training iterations
        learning_rate: Training learning rate
        warmup_ratio: Warmup fraction for this call
        lr_scheduler: "constant", "linear", or "cosine"
        scheduler_total_steps: Optional fixed horizon for linear/cosine schedules.
            Recommended when calling num_iterations=1 repeatedly.
        temperature: Sampling temperature
        max_tokens: Maximum tokens per completion
        auto_checkpoint: Enable auto-checkpoint policy.
        debug: Enable debug logging
        sampling_debug_prompt_limit: Number of prompts with detailed sampling debug.
            Use -1 to log all prompts, 0 to disable per-prompt sampling debug details.
        auto_checkpoint_reward_threshold: Reward threshold for auto-checkpoint policy.
        auto_checkpoint_min_iterations: Minimum cumulative iterations before auto-checkpoint.

    Returns:
        Training report with metrics and guidance
    """
    session = get_session()
    log_paths = get_log_paths()

    # Write progress immediately for watchdog heartbeat (before
    # validation/setup)
    await write_progress(f"Starting GRPO training step: {num_iterations} iterations")

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    if session.training_client is None or session.session_method != "grpo":
        return (
            "Error: No GRPO session active. Call init_grpo(base_model) first "
            "before training.\nExample: "
            'init_grpo("vendor/model-base")'
        )
    if session.renderer is None:
        return "Error: Renderer not initialized. Call init_grpo(base_model) again to set up renderer."
    try:
        sampling_debug_prompt_limit = _resolve_sampling_debug_prompt_limit(sampling_debug_prompt_limit)
    except ValueError as exc:
        return f"Error: {exc}"
    try:
        auto_checkpoint_reward_threshold, auto_checkpoint_min_iterations = _resolve_auto_checkpoint_policy(
            auto_checkpoint_reward_threshold,
            auto_checkpoint_min_iterations,
        )
    except ValueError as exc:
        return f"Error: {exc}"

    if do_debug:
        # Log prompt parameter type: file path or JSON string
        prompt_type = (
            "file"
            if (prompts and (prompts.strip().startswith("/") or prompts.strip().endswith(".json")))
            else "json_string"
        )
        debug_log(
            log_paths["training"],
            f"[START] train_grpo_step: prompts_input={prompt_type}, iters={num_iterations}, "
            f"group_size={session.grpo_group_size}, "
            f"lr_scheduler={lr_scheduler}, scheduler_total_steps={scheduler_total_steps}, "
            f"sampling_debug_prompt_limit={sampling_debug_prompt_limit}, "
            f"auto_checkpoint_threshold={auto_checkpoint_reward_threshold}, "
            f"auto_checkpoint_min_iterations={auto_checkpoint_min_iterations}",
            force=True,
        )

    scheduler_error = validate_lr_scheduler_name(lr_scheduler)
    if scheduler_error:
        return f"Error: {scheduler_error}"

    try:
        resolved_scheduler_total_steps, scheduler_warning = resolve_scheduler_total_steps(
            scheduler=lr_scheduler,
            explicit_total_steps=scheduler_total_steps,
            cumulative_steps=session.grpo_iterations,
            steps_this_call=num_iterations,
            persisted_total_steps=session.grpo_scheduler_total_steps,
            default_floor_steps=20,
            unit_name="iteration",
        )
    except ValueError as exc:
        return f"Error: {exc}"
    session.grpo_scheduler_total_steps = max(session.grpo_scheduler_total_steps, resolved_scheduler_total_steps)

    # Check for warmup_ratio on continued training - warn user about potential
    # LR instability
    warmup_ratio, warmup_warning = validate_warmup_for_continued_training(
        warmup_ratio, session.grpo_iterations, step_name="iteration"
    )
    if warmup_warning:
        print(f"\n[WARNING] {warmup_warning}\n", file=sys.stderr, flush=True)
        await write_progress(f"WARNING: {warmup_warning}")
        if do_debug:
            debug_log(log_paths["training"], f"[WARNING] {warmup_warning}", force=True)
    if scheduler_warning:
        print(f"\n[WARNING] {scheduler_warning}\n", file=sys.stderr, flush=True)
        await write_progress(f"WARNING: {scheduler_warning}")
        if do_debug:
            debug_log(log_paths["training"], f"[WARNING] {scheduler_warning}", force=True)

    # Load prompts from file or parse JSON string
    prompts_list, load_error = load_json_from_file_or_string(prompts, "prompts", do_debug, log_paths)
    if load_error:
        return load_error

    if not prompts_list:
        if do_debug:
            debug_log(log_paths["training"], "[ERROR] No prompts provided", force=True)
        return "Error: No prompts provided."

    # Validate JSON schema
    schema_error = validate_input_schema(prompts_list, ["prompt", "ground_truth"], "Prompt")
    if schema_error:
        return schema_error

    if do_debug:
        debug_log(log_paths["training"], f"[INFO] Parsed {len(prompts_list)} prompts with ground_truth", force=True)

    # Validate reward function syntax before using it
    # This catches syntax errors early with clear messages instead of cryptic
    # subprocess failures
    is_valid, syntax_error = validate_reward_function_syntax(reward_function)
    if not is_valid:
        if do_debug:
            debug_log(log_paths["training"], f"[ERROR] Reward function syntax invalid: {syntax_error}", force=True)
        return f"""Error: Invalid reward function definition.

{syntax_error}

Your reward function must be valid Python code defining:

```python
def compute_reward(completion: str, ground_truth: str) -> float:
    # Return a value between 0.0 and 1.0
    ...
```

Common issues:
- Missing `compute_reward` function
- Missing colons after function definitions
- Indentation errors
- Unclosed brackets or quotes
- Invalid Python syntax

Fix the syntax error and try again."""

    reward_function_code = reward_function

    if do_debug:
        debug_log(log_paths["training"], "[INFO] Reward function syntax validated successfully", force=True)
        try:
            with open(log_paths["grading_function"], "w") as f:
                f.write(reward_function_code)
            debug_log(
                log_paths["grading_validation"],
                f"Reward function written to {log_paths['grading_function']}",
                force=True,
            )
        except Exception:
            pass

    try:
        sampling_client_error = await _ensure_grpo_sampling_client(session, do_debug, log_paths)
        if sampling_client_error:
            return sampling_client_error

        iteration_results = []
        total_skipped = 0
        total_token_issues = 0
        total_zero_token_samples = 0
        total_reward_errors = 0
        total_samples = 0
        sample_rewards_debug = []
        auto_checkpoints_saved = []
        failure_stats = FailureStats()
        kl_metrics = []

        for iteration in range(num_iterations):
            iter_start = time.time()
            await write_progress(
                f"GRPO iter {iteration + 1}/{num_iterations} starting (cumulative: {session.grpo_iterations})"
            )
            if do_debug:
                debug_log(log_paths["training"], f"[ITER {iteration + 1}/{num_iterations}] Starting...", force=True)

            iteration_rewards = []
            iteration_skipped = 0
            iteration_uniform_all_ones = 0
            iteration_uniform_all_zeros = 0
            iteration_token_logprob_mismatches = 0
            iteration_zero_token_samples = 0
            iteration_reward_errors = 0
            iteration_prompt_failures = 0
            batch_data = []
            kl_v1 = None
            kl_v2 = None
            iteration_lr = learning_rate

            for p_idx, p in enumerate(prompts_list):
                prompt_text = p.get("prompt", "")
                ground_truth = p.get("ground_truth", "")

                outcome = await _process_grpo_prompt(
                    session=session,
                    prompt_text=prompt_text,
                    ground_truth=ground_truth,
                    prompt_index=p_idx,
                    prompts_total=len(prompts_list),
                    iteration_index=iteration,
                    num_iterations=num_iterations,
                    reward_function_code=reward_function_code,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    do_debug=do_debug,
                    log_paths=log_paths,
                    include_sample_debug=(len(sample_rewards_debug) < 5 and iteration == 0),
                    sampling_debug_prompt_limit=sampling_debug_prompt_limit,
                )

                if outcome.sample_debug_item is not None:
                    sample_rewards_debug.append(outcome.sample_debug_item)

                iteration_rewards.extend(outcome.rewards)
                iteration_skipped += outcome.skipped_samples
                iteration_uniform_all_ones += outcome.uniform_all_ones
                iteration_uniform_all_zeros += outcome.uniform_all_zeros
                iteration_token_logprob_mismatches += outcome.token_logprob_mismatches
                iteration_zero_token_samples += outcome.zero_token_samples
                iteration_reward_errors += outcome.reward_errors
                if outcome.prompt_failed:
                    iteration_prompt_failures += 1
                batch_data.extend(outcome.datums)
                total_samples += outcome.included_samples

            total_skipped += iteration_skipped
            total_token_issues += iteration_token_logprob_mismatches
            total_zero_token_samples += iteration_zero_token_samples
            total_reward_errors += iteration_reward_errors

            failure_stats.uniform_reward_samples += iteration_skipped
            failure_stats.uniform_all_ones += iteration_uniform_all_ones
            failure_stats.uniform_all_zeros += iteration_uniform_all_zeros
            failure_stats.token_issue_samples += iteration_token_logprob_mismatches
            failure_stats.zero_token_samples += iteration_zero_token_samples
            failure_stats.reward_error_samples += iteration_reward_errors
            failure_stats.samples_included += len(batch_data)
            failure_stats.prompts_processed += len(prompts_list)
            failure_stats.prompts_with_no_signal += iteration_prompt_failures

            if iteration_skipped > 0 or iteration_token_logprob_mismatches > 0 or iteration_zero_token_samples > 0:
                skip_msg = (
                    f"Iter {iteration + 1}: Skipped {iteration_skipped} samples (uniform rewards) + "
                    f"{iteration_token_logprob_mismatches} samples (token/logprob issues) + "
                    f"{iteration_zero_token_samples} samples (short-token<2)"
                )
                await write_progress(skip_msg)

            if batch_data:
                kl_v1, kl_v2, iteration_lr, optimization_error = await _run_grpo_optimization_step(
                    session=session,
                    batch_data=batch_data,
                    num_iterations=num_iterations,
                    iteration_index=iteration,
                    learning_rate=learning_rate,
                    warmup_ratio=warmup_ratio,
                    lr_scheduler=lr_scheduler,
                    scheduler_total_steps=resolved_scheduler_total_steps,
                    do_debug=do_debug,
                    log_paths=log_paths,
                    failure_stats=failure_stats,
                )
                if optimization_error:
                    return optimization_error

                if kl_v1 is not None or kl_v2 is not None:
                    kl_metrics.append({"iteration": session.grpo_iterations + 1, "kl_v1": kl_v1, "kl_v2": kl_v2})
                    if do_debug:
                        debug_log(
                            log_paths["training"],
                            f"[KL] iter={session.grpo_iterations + 1} v1={kl_v1} v2={kl_v2}",
                            force=True,
                        )

                session.consecutive_empty_batches = 0
                successful = len(prompts_list) - iteration_prompt_failures
                if successful < len(prompts_list):
                    await write_progress(f"Iter {iteration + 1}: {successful}/{len(prompts_list)} prompts contributed")
            else:
                session.consecutive_empty_batches += 1
                log_error(
                    f"Iteration {iteration + 1}: batch_data empty after processing all prompts "
                    f"(consecutive: {session.consecutive_empty_batches}). "
                    f"All samples skipped due to uniform rewards or token/logprob issues."
                )
                await write_progress(
                    f"ERROR: Iteration {iteration + 1}: batch_data empty "
                    f"(consecutive: {session.consecutive_empty_batches})"
                )

                threshold = 2 if len(prompts_list) == 1 else 5
                if session.consecutive_empty_batches >= threshold:
                    detail = (
                        "Single-prompt failing."
                        if len(prompts_list) == 1
                        else f"All {len(prompts_list)} prompts failing."
                    )
                    log_error(f"FATAL: {session.consecutive_empty_batches} consecutive empty batches. {detail}")
                    return (
                        f"Error: {session.consecutive_empty_batches} consecutive empty batches. {detail}\n"
                        "Check reward function or sampling - all completions "
                        "are being skipped due to uniform rewards or "
                        "token/logprob mismatches."
                    )

            checkpoint_error = await _maybe_refresh_grpo_sampling_client(
                session=session,
                iteration_index=iteration,
                num_iterations=num_iterations,
                do_debug=do_debug,
                log_paths=log_paths,
            )
            if checkpoint_error:
                await write_progress(checkpoint_error)
                return f"Error: {checkpoint_error}\nTraining state may be inconsistent. Check logs and retry."

            session.grpo_iterations += 1

            reward_mean = np.mean(iteration_rewards) if iteration_rewards else 0.0
            reward_std = np.std(iteration_rewards) if iteration_rewards else 0.0
            iteration_results.append(
                {
                    "iteration": session.grpo_iterations,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "skipped": iteration_skipped,
                    "token_logprob_mismatches": iteration_token_logprob_mismatches,
                    "zero_token_samples": iteration_zero_token_samples,
                    "kl_v1": kl_v1,
                    "kl_v2": kl_v2,
                    "lr_used": iteration_lr,
                }
            )

            total_attempted_this_iter = len(prompts_list) * session.grpo_group_size
            iter_skip_rate = iteration_skipped / max(total_attempted_this_iter, 1)
            iter_token_issue_rate = iteration_token_logprob_mismatches / max(total_attempted_this_iter, 1)
            iter_zero_token_rate = iteration_zero_token_samples / max(total_attempted_this_iter, 1)
            iter_reward_error_rate = iteration_reward_errors / max(total_attempted_this_iter, 1)
            iter_hard_failure_rate = (
                iteration_token_logprob_mismatches + iteration_zero_token_samples + iteration_reward_errors
            ) / max(total_attempted_this_iter, 1)
            iter_data_loss_rate = (
                iteration_skipped + iteration_token_logprob_mismatches + iteration_zero_token_samples
            ) / max(total_attempted_this_iter, 1)

            if session.wandb_run is not None:
                try:
                    log_data = {
                        "iteration": int(session.grpo_iterations),
                        "reward_mean": float(reward_mean),
                        "reward_std": float(reward_std),
                        "skip_rate": float(iter_skip_rate),
                        "token_issue_rate": float(iter_token_issue_rate),
                        "zero_token_rate": float(iter_zero_token_rate),
                        # NOTE: "data_loss_rate" historically included uniform reward skips.
                        # Keep it for backwards compatibility, but also log explicit rates below.
                        "data_loss_rate": float(iter_data_loss_rate),
                        # Observability: why are samples being skipped?
                        "samples_generated": int(total_attempted_this_iter),
                        "samples_used": int(len(batch_data)),
                        "samples_used_rate": float(len(batch_data) / max(total_attempted_this_iter, 1)),
                        "uniform_reward_samples": int(iteration_skipped),
                        "uniform_all_ones": int(iteration_uniform_all_ones),
                        "uniform_all_zeros": int(iteration_uniform_all_zeros),
                        "uniform_all_ones_rate": float(
                            iteration_uniform_all_ones / max(iteration_skipped, 1) if iteration_skipped else 0.0
                        ),
                        "uniform_all_zeros_rate": float(
                            iteration_uniform_all_zeros / max(iteration_skipped, 1) if iteration_skipped else 0.0
                        ),
                        "token_issue_samples": int(iteration_token_logprob_mismatches),
                        "zero_token_samples": int(iteration_zero_token_samples),
                        "reward_error_samples": int(iteration_reward_errors),
                        "reward_error_rate": float(iter_reward_error_rate),
                        "hard_failure_rate": float(iter_hard_failure_rate),
                        "no_signal_rate": float(iter_skip_rate),
                        "prompts_processed": int(len(prompts_list)),
                        "prompts_contributed": int(len(prompts_list) - iteration_prompt_failures),
                        "prompts_no_signal": int(iteration_prompt_failures),
                        "learning_rate": float(iteration_lr),
                        "learning_rate_base": float(learning_rate),
                        "scheduler_total_steps": int(resolved_scheduler_total_steps),
                    }
                    if kl_v1 is not None:
                        log_data["kl_sample_train_v1"] = float(kl_v1)
                    if kl_v2 is not None:
                        log_data["kl_sample_train_v2"] = float(kl_v2)
                    session.wandb_run.log(log_data, step=session.grpo_iterations)
                    if do_debug:
                        debug_log(log_paths["wandb"], f"[W&B] log: {log_data}", force=True)
                except Exception as e:
                    print(f"W&B metric log warning: {e}", file=sys.stderr, flush=True)

            iter_time = time.time() - iter_start
            await write_progress(
                f"GRPO iter {iteration + 1}/{num_iterations} done: reward_mean={reward_mean:.3f}, "
                f"hard_fail={iter_hard_failure_rate:.1%}, no_signal={iter_skip_rate:.1%}"
            )

            if do_debug:
                debug_log(
                    log_paths["training"],
                    f"[ITER {iteration + 1}/{num_iterations}] DONE: "
                    f"reward_mean={reward_mean:.3f}, skip_rate={iter_skip_rate:.1%}, time={iter_time:.1f}s",
                    force=True,
                )

            if iter_skip_rate > 0.3:
                if iteration_uniform_all_ones > iteration_uniform_all_zeros:
                    await write_progress(
                        f"NOTE: High no-signal rate {iter_skip_rate:.1%} in iter {iteration + 1} "
                        "(mostly ALL 1.0) - model may be converged or data too easy"
                    )
                elif iteration_uniform_all_zeros > iteration_uniform_all_ones:
                    await write_progress(
                        f"WARNING: High no-signal rate {iter_skip_rate:.1%} in iter {iteration + 1} "
                        "(mostly ALL 0.0) - data may be too hard or reward too strict"
                    )
                else:
                    await write_progress(
                        f"WARNING: High no-signal rate {iter_skip_rate:.1%} in iter {iteration + 1} "
                        "- uniform rewards dominated; consider more diverse prompts or partial credit"
                    )

            if (
                auto_checkpoint
                and reward_mean > auto_checkpoint_reward_threshold
                and session.grpo_iterations >= auto_checkpoint_min_iterations
            ):
                reward_str = f"{reward_mean:.2f}".replace(".", "_")
                checkpoint_name = apply_checkpoint_prefix(f"auto_iter{session.grpo_iterations}_r{reward_str}")
                try:
                    # Tinker `save_state()` returns an awaitable APIFuture; keep
                    # auto-checkpointing single-await and robust.
                    save_future = session.training_client.save_state(checkpoint_name)
                    save_result = await asyncio.wait_for(save_future, timeout=TIMEOUT_MEDIUM)
                    adapter_path = getattr(save_result, "path", None) or getattr(save_result, "checkpoint_path", None)
                    if not adapter_path:
                        raise RuntimeError(
                            "Tinker save_state response missing `path`. "
                            f"Got: {type(save_result).__name__}"
                        )
                    session.saved_adapters[checkpoint_name] = adapter_path
                    session.save_checkpoint_metadata(checkpoint_name)
                    auto_checkpoints_saved.append(checkpoint_name)
                    await write_progress(f"AUTO-CHECKPOINT: Saved '{checkpoint_name}' (reward_mean={reward_mean:.3f})")
                    if do_debug:
                        debug_log(log_paths["save_load"], f"[AUTO-CHECKPOINT] Saved {checkpoint_name}", force=True)
                except Exception as e:
                    await write_progress(f"Auto-checkpoint failed (non-fatal): {e}")

        # Check cumulative data loss before building report
        # Total attempts = skipped + token_issues + reward_errors +
        # samples_used
        total_attempted = (
            total_skipped + total_token_issues + total_zero_token_samples + total_reward_errors + total_samples
        )
        cumulative_skip_rate = total_skipped / max(total_attempted, 1)  # Uniform rewards only
        cumulative_token_issue_rate = total_token_issues / max(total_attempted, 1)  # Token/logprob issues
        # Includes both empty outputs and one-token outputs (<2 cannot form shifted datum).
        cumulative_zero_token_rate = total_zero_token_samples / max(total_attempted, 1)
        cumulative_reward_error_rate = total_reward_errors / max(total_attempted, 1)  # Reward function exceptions
        hard_failures = total_token_issues + total_zero_token_samples + total_reward_errors
        cumulative_hard_failure_rate = hard_failures / max(total_attempted, 1)
        cumulative_no_signal_rate = cumulative_skip_rate

        # Get root cause analysis
        root_cause, fix_suggestion = _get_root_cause_analysis(failure_stats)

        # Fail fast ONLY on hard failures (API/sampling/reward errors), not on
        # uniform-reward "no-signal" skips (which are expected in GRPO, and can
        # even indicate success when dominated by ALL-1.0).
        if cumulative_hard_failure_rate > 0.5 and total_attempted > 0:
            return f"""Error: Too many unusable samples ({cumulative_hard_failure_rate:.1%}).

Over half of samples were unusable due to token/logprob issues, short-token generations, or reward errors.
This indicates a functional problem (API/sampling/reward function), not just GRPO no-signal skipping.

BREAKDOWN:
- Total attempted: {total_attempted}
- Token/logprob issues: {total_token_issues} ({cumulative_token_issue_rate:.1%}) - API/sampling issues
- Short-token generations (<2 tokens): {total_zero_token_samples} ({cumulative_zero_token_rate:.1%})
- Reward function errors: {total_reward_errors} ({cumulative_reward_error_rate:.1%}) - exceptions thrown
- Uniform no-signal skips: {total_skipped} ({cumulative_no_signal_rate:.1%}) - expected when rewards uniform
- Used for training: {total_samples}

ROOT CAUSE: {root_cause}

SUGGESTED FIX: {fix_suggestion}

NEXT STEPS:
1. If token/logprob issues: Reduce max_tokens or verify renderer/stop sequences
2. If short-token: Check prompt formatting; model may be EOS'ing immediately
3. If reward errors: Debug reward function; verify compute_reward exists and is robust
4. Inspect logs: {log_paths['reward']}, {log_paths['training']}, {log_paths['sampling']}
"""

        # Build report - use data_loss_rate for comprehensive tracking
        iter_report = "\n".join(
            [
                (
                    f"  Iter {r['iteration']}: reward={r['reward_mean']:.3f} +/- {r['reward_std']:.3f}, "
                    f"lr={r['lr_used']:.2e}"
                    + (f", kl_v1={r['kl_v1']:.4f}" if r.get("kl_v1") is not None else "")
                    + (f", kl_v2={r['kl_v2']:.4f}" if r.get("kl_v2") is not None else "")
                )
                for r in iteration_results
            ]
        )

        sample_report = "\n".join(
            [
                f"  [{i + 1}] reward={s['reward']:.2f} | {s['completion'][:80]}..."
                for i, s in enumerate(sample_rewards_debug)
            ]
        )

        warnings = []
        if warmup_warning:
            warnings.append(warmup_warning)
        if scheduler_warning:
            warnings.append(scheduler_warning)
        if cumulative_hard_failure_rate > 0.3:
            warnings.append(
                f"HIGH UNUSABLE SAMPLE RATE: {cumulative_hard_failure_rate:.1%} "
                f"(token_issues={cumulative_token_issue_rate:.1%}, zero_token={cumulative_zero_token_rate:.1%}, "
                f"reward_errors={cumulative_reward_error_rate:.1%})"
            )
        if cumulative_no_signal_rate > 0.7:
            if failure_stats.uniform_all_ones > failure_stats.uniform_all_zeros:
                warnings.append(
                    f"HIGH NO-SIGNAL RATE: {cumulative_no_signal_rate:.1%} "
                    "(mostly ALL 1.0) - model may be converged or data too easy"
                )
            elif failure_stats.uniform_all_zeros > failure_stats.uniform_all_ones:
                warnings.append(
                    f"HIGH NO-SIGNAL RATE: {cumulative_no_signal_rate:.1%} "
                    "(mostly ALL 0.0) - data may be too hard or reward too strict"
                )
            else:
                warnings.append(f"HIGH NO-SIGNAL RATE: {cumulative_no_signal_rate:.1%} (uniform rewards dominated)")
        if cumulative_reward_error_rate > 0.1:
            warnings.append(
                f"HIGH REWARD ERROR RATE: {cumulative_reward_error_rate:.1%}. " "Check reward function for exceptions."
            )
        if iteration_results and iteration_results[-1]["reward_mean"] < iteration_results[0]["reward_mean"]:
            warnings.append("DECLINING REWARDS - reduce learning_rate or check reward function")
        # KL stability heuristic from Tinker docs: stable < 0.01
        if kl_metrics:
            last_kl = kl_metrics[-1]
            for key in ("kl_v1", "kl_v2"):
                val = last_kl.get(key)
                if val is not None and val > 0.01:
                    warnings.append(
                        f"HIGH KL ({key}={val:.4f}) - potential instability, consider reducing learning_rate"
                    )

        warning_text = "\n".join(warnings) if warnings else "None"

        # Compute decision guidance
        if iteration_results:
            first_reward = iteration_results[0]["reward_mean"]
            last_reward = iteration_results[-1]["reward_mean"]
            reward_delta = last_reward - first_reward

            if reward_delta > 0.02:
                reward_trend = "improving"
            elif reward_delta < -0.02:
                reward_trend = "declining"
            else:
                reward_trend = "flat"

            if cumulative_hard_failure_rate > 0.5:
                loss_status = "CRITICAL (>50%)"
            elif cumulative_hard_failure_rate > 0.3:
                loss_status = "HIGH (>30%)"
            else:
                loss_status = "OK (<30%)"

            if cumulative_hard_failure_rate > 0.5:
                decision_text = "CRITICAL: Too many unusable samples; training is not learning effectively."
                next_step = (
                    "INVESTIGATE: Check sampling_debug.log + reward_debug.log for token/logprob and reward errors"
                )
            elif cumulative_hard_failure_rate > 0.3:
                decision_text = "WARNING: Many unusable samples suggests token handling or reward execution issues."
                next_step = "INVESTIGATE: Check if errors are token/logprob, short-token, or reward exceptions"
            elif cumulative_no_signal_rate > 0.7 and failure_stats.uniform_all_ones > failure_stats.uniform_all_zeros:
                decision_text = "SUCCESS: Most prompts are ALL 1.0 (uniform). Model likely converged or data too easy."
                next_step = "EVALUATE: Call sample(); if quality is good, save+finish. Otherwise increase difficulty."
            elif cumulative_no_signal_rate > 0.7 and failure_stats.uniform_all_zeros > failure_stats.uniform_all_ones:
                decision_text = (
                    "WARNING: Most prompts are ALL 0.0 (uniform). Data may be too hard or reward too strict."
                )
                next_step = "ADJUST: Use easier prompts or add partial credit to the reward function"
            elif reward_delta < -0.02:
                decision_text = "WARNING: Rewards declining indicates learning instability."
                next_step = "ADJUST: Reduce learning_rate by 2-5x and try again"
            elif abs(reward_delta) < 0.01 and session.grpo_iterations > 5:
                decision_text = "NOTE: Rewards flat - may have reached ceiling or learning rate too low."
                next_step = "EVALUATE: Call sample() to check quality. If good, save. If not, try 2x learning_rate"
            else:
                decision_text = "HEALTHY: Rewards improving, training is working."
                next_step = "CONTINUE: Call train_grpo_step(num_iterations=1) to continue training"
        else:
            reward_trend = "N/A"
            loss_status = "N/A"
            decision_text = "No iteration data available."
            next_step = "Check for errors above"

        if do_debug:
            final_reward = iteration_results[-1]["reward_mean"] if iteration_results else 0.0
            debug_log(
                log_paths["training"],
                f"[SUCCESS] train_grpo_step complete ({num_iterations} iters, final_reward={final_reward:.3f})",
                force=True,
            )

        auto_checkpoint_policy_text = (
            f"reward_mean > {auto_checkpoint_reward_threshold} and "
            f"cumulative_iterations >= {auto_checkpoint_min_iterations}"
        )
        if auto_checkpoints_saved:
            auto_checkpoint_summary = "  " + "\n".join(auto_checkpoints_saved)
        elif auto_checkpoint:
            auto_checkpoint_summary = f"  None ({auto_checkpoint_policy_text} not met)"
        else:
            auto_checkpoint_summary = "  None (auto_checkpoint disabled)"

        wandb_line = ""
        if session.wandb_run is not None:
            url = getattr(session.wandb_run, "url", "") or ""
            if url:
                wandb_line = f"\nW&B run: {url}\n"

        return f"""
GRPO TRAINING STEP COMPLETE
===========================
Iterations this step: {num_iterations}
Cumulative iterations: {session.grpo_iterations}
Prompts: {len(prompts_list)}
Group size: {session.grpo_group_size}
{wandb_line}

HYPERPARAMETERS USED:
  learning_rate: {learning_rate}
  warmup_ratio: {warmup_ratio}
  lr_scheduler: {lr_scheduler}
  scheduler_total_steps: {resolved_scheduler_total_steps if lr_scheduler != "constant" else "N/A (constant)"}
  max_tokens: {max_tokens}
  temperature: {temperature}
  auto_checkpoint: {auto_checkpoint}
  auto_checkpoint_reward_threshold: {auto_checkpoint_reward_threshold}
  auto_checkpoint_min_iterations: {auto_checkpoint_min_iterations}

AUTO-CHECKPOINTS SAVED:
{auto_checkpoint_summary}

PER-ITERATION METRICS:
{iter_report}

KL METRICS:
If you don't see kl_sample_train_v1/v2 above, check W&B or training_debug.log.
Some backends may omit KL in the immediate tool output.

SAMPLE UTILIZATION:
- Generated: {total_attempted}
- Used for training: {total_samples} ({total_samples / max(total_attempted, 1):.0%})

FAILURE BREAKDOWN:
  Uniform rewards: {total_skipped} ({cumulative_skip_rate:.1%}) - no variance in rewards
  Token/logprob:   {total_token_issues} ({cumulative_token_issue_rate:.1%}) - API/sampling issues
  Short-token<2:   {total_zero_token_samples} ({cumulative_zero_token_rate:.1%})
  Reward errors:   {total_reward_errors} ({cumulative_reward_error_rate:.1%}) - exceptions in reward function

ROOT CAUSE: {root_cause}
SUGGESTION: {fix_suggestion}

SAMPLE REWARD COMPUTATIONS:
{sample_report}

WARNINGS:
{warning_text}

-------------------------------------
DECISION GUIDANCE
-------------------------------------
reward_mean: {reward_trend}
sample_quality: {loss_status}
no_signal_rate: {cumulative_no_signal_rate:.1%}

{decision_text}

SUGGESTED NEXT STEP: {next_step}

DEBUG LOGS (if investigating):
- {log_paths["reward"]} - Per-completion reward scores
- {log_paths["sampling"]} - Model outputs
- {log_paths["grading_function"]} - Your reward function code
- {log_paths["training"]} - Training loop details
"""

    except Exception as e:
        import traceback

        if do_debug:
            debug_log(log_paths["training"], f"[ERROR] train_grpo_step failed: {e}", force=True)
        # Keep W&B run open on recoverable step errors so retries continue logging.
        cleanup_wandb_on_error(session, do_debug, log_paths, fatal=False, reason="train_grpo_step failed")
        return f"Error in GRPO training: {e}\n{traceback.format_exc()}"
