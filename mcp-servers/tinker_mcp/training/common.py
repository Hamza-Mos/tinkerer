"""Shared utilities for SFT and GRPO training modules.

This module contains common functions used by both training implementations to
reduce code duplication and ensure consistent behavior.
"""

import json
import os
import sys
import asyncio
import time
import uuid
import re
from typing import Optional

import numpy as np

from tinker_mcp.utils import debug_log


_UNRESOLVED_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{[A-Za-z0-9_]+\}$")


def load_json_from_file_or_string(
    data: str, entity_name: str, do_debug: bool = False, log_paths: dict = None
) -> tuple[Optional[list], Optional[str]]:
    """Load JSON/JSONL from file path or parse string.

    Supports:
    - JSON arrays/objects via `json.loads(...)`
    - JSONL (one JSON object per line) as a convenience for agent-generated datasets
    - File paths (starting with '/' or ending with '.json' / '.jsonl')

    Args:
        data: Either a JSON string or a file path
        entity_name: Name for error messages (e.g., "prompts", "examples")
        do_debug: Whether to log debug info
        log_paths: Dict with log file paths (needs "training" key)

    Returns:
        Tuple of (parsed_data, error_message). error_message is None if successful.
    """
    def _parse_jsonl_text(text: str) -> tuple[Optional[list], Optional[str]]:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        items: list[dict] = []
        for line_no, line in enumerate(lines, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                return None, f"Error: Failed to parse {entity_name} as JSONL (line {line_no}): {exc}"
            if not isinstance(obj, dict):
                return (
                    None,
                    f"Error: {entity_name.capitalize()} JSONL line {line_no} must be an object/dict, got {type(obj).__name__}.",
                )
            items.append(obj)
        return items, None

    data_to_parse = data or ""
    source_file_path: str | None = None
    source_is_jsonl_file = False

    # Check if data is a file path
    if data and (data.strip().startswith("/") or data.strip().endswith(".json") or data.strip().endswith(".jsonl")):
        file_path = data.strip()
        source_file_path = file_path
        source_is_jsonl_file = file_path.lower().endswith(".jsonl")
        if do_debug and log_paths:
            debug_log(log_paths["training"], f"[INFO] Loading {entity_name} from file: {file_path}", force=True)
        try:
            if source_is_jsonl_file:
                # Stream parse JSONL to avoid reading large files into memory.
                items: list[dict] = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_no, raw_line in enumerate(f, start=1):
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as exc:
                            return (
                                None,
                                f"Error: Failed to parse {entity_name} JSONL file {file_path} (line {line_no}): {exc}",
                            )
                        if not isinstance(obj, dict):
                            return (
                                None,
                                (
                                    f"Error: {entity_name.capitalize()} JSONL file {file_path} line {line_no} "
                                    f"must be an object/dict, got {type(obj).__name__}."
                                ),
                            )
                        items.append(obj)
                return items, None

            with open(file_path, "r", encoding="utf-8") as f:
                data_to_parse = f.read()
            if do_debug and log_paths:
                debug_log(log_paths["training"], f"[INFO] Loaded {len(data_to_parse)} bytes from {file_path}", force=True)
        except FileNotFoundError:
            return (
                None,
                (
                    f"Error: {entity_name.capitalize()} file not found: "
                    f"{file_path}\nEither pass a JSON string directly or "
                    "a valid file path."
                ),
            )
        except Exception as e:
            return None, f"Error: Failed to read {entity_name} file {file_path}: {e}"

    # Parse JSON
    try:
        parsed = json.loads(data_to_parse)
        return parsed, None
    except json.JSONDecodeError as e:
        if do_debug and log_paths:
            debug_log(log_paths["training"], f"[ERROR] JSON parse failed: {e}", force=True)

        # Fallback: many agent flows naturally produce JSONL (one object per line).
        # Try parsing as JSONL if it looks like JSONL (or the user provided a .jsonl path
        # but content was loaded as text because it didn't end with .jsonl / start with '/').
        stripped = (data_to_parse or "").lstrip()
        looks_like_jsonl = bool(stripped.startswith("{") and "\n" in stripped)
        if looks_like_jsonl:
            jsonl_parsed, jsonl_error = _parse_jsonl_text(data_to_parse)
            if jsonl_error is None:
                return jsonl_parsed, None

        tip_suffix = (
            f"TIP: You can also pass a file path (e.g., '/tmp/{entity_name}.json' or '/tmp/{entity_name}.jsonl') "
            "and the tool will read it."
        )
        source_note = f" (from file {source_file_path})" if source_file_path else ""
        return None, (
            f"Error: Failed to parse {entity_name} as JSON{source_note}: {e}\n"
            "Expected a JSON array (e.g., `[{...}, {...}]`).\n"
            "If you intended JSONL (one JSON object per line), use a `.jsonl` file or pass JSONL text.\n\n"
            f"{tip_suffix}"
        )


def validate_input_schema(items: list, required_fields: list[str], entity_name: str) -> Optional[str]:
    """Validate items have required string fields.

    Args:
        items: List of dicts to validate
        required_fields: List of field names that must be present (as strings)
        entity_name: Name for error messages (e.g., "Prompt", "Example")

    Returns:
        Error message if validation fails, None if successful.
    """
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return f"Error: {entity_name} {i} must be a dict, got {type(item).__name__}."

        for field in required_fields:
            value = item.get(field)
            if value is not None and not isinstance(value, str):
                return f"Error: {entity_name} {i} '{field}' must be a string, got {type(value).__name__}"

    return None


def validate_warmup_for_continued_training(
    warmup_ratio: float, current_step: int, step_name: str = "step"
) -> tuple[float, Optional[str]]:
    """Auto-adjust warmup and return (adjusted_ratio, warning_msg).

    Warmup is typically only needed on the first training call. Using warmup
    on continued training can cause LR instability (spike back up to full LR).

    Args:
        warmup_ratio: Requested warmup ratio
        current_step: Current cumulative step/iteration count
        step_name: Name for the step unit (e.g., "step", "iteration")

    Returns:
        Tuple of (adjusted_warmup_ratio, warning_message).
        warning_message is None if no adjustment was needed.
    """
    is_first_call = current_step == 0

    if warmup_ratio > 0 and not is_first_call:
        warning = (
            f"warmup_ratio={warmup_ratio} on continued training ({step_name} {current_step}). "
            f"Warmup is typically only needed on the FIRST training call. "
            f"Auto-adjusting to warmup_ratio=0.0 to prevent LR spike."
        )
        return 0.0, warning

    return warmup_ratio, None


def validate_lr_scheduler_name(lr_scheduler: str) -> Optional[str]:
    """Validate scheduler name used by get_lr.

    Returns:
        Error string if invalid, otherwise None.
    """
    valid = {"constant", "linear", "cosine"}
    if lr_scheduler not in valid:
        valid_list = ", ".join(sorted(valid))
        return f"Invalid lr_scheduler '{lr_scheduler}'. Expected one of: {valid_list}."
    return None


def resolve_scheduler_total_steps(
    *,
    scheduler: str,
    explicit_total_steps: Optional[int],
    cumulative_steps: int,
    steps_this_call: int,
    persisted_total_steps: int = 0,
    default_floor_steps: int = 20,
    unit_name: str = "step",
) -> tuple[int, Optional[str]]:
    """Resolve a stable total_steps horizon for linear/cosine schedules.

    Why this exists:
    - Repeated calls with a single step and no fixed horizon make linear/cosine
      decay collapse too quickly (e.g., 1.0x -> 0.5x -> 0.25x ...).
    - We infer a floor horizon by default and allow explicit override.

    Returns:
        (resolved_total_steps, warning_msg_or_none)
    """
    min_required = cumulative_steps + steps_this_call
    if min_required <= 0:
        min_required = 1

    if explicit_total_steps is not None:
        if explicit_total_steps <= 0:
            raise ValueError("scheduler_total_steps must be > 0 when provided.")
        if explicit_total_steps < min_required:
            raise ValueError(
                "scheduler_total_steps must be >= cumulative steps after this call "
                f"({min_required}), got {explicit_total_steps}."
            )
        return explicit_total_steps, None

    if scheduler == "constant":
        return min_required, None

    inferred_floor = max(default_floor_steps, steps_this_call)
    resolved_total_steps = max(min_required, persisted_total_steps, inferred_floor)

    warning = None
    # Warn on the repeated single-step pattern that causes rapid decay.
    if steps_this_call == 1 and resolved_total_steps > min_required:
        warning = (
            f"{scheduler} scheduler requested without scheduler_total_steps during iterative "
            f"single-{unit_name} training. Using inferred scheduler_total_steps={resolved_total_steps} "
            "to avoid rapid LR collapse. For deterministic decay, pass scheduler_total_steps explicitly "
            "or use lr_scheduler='constant'."
        )

    return resolved_total_steps, warning


def cleanup_wandb_on_error(
    session,
    do_debug: bool,
    log_paths: dict,
    *,
    fatal: bool = False,
    reason: str = "",
) -> None:
    """Handle W&B on errors without breaking recoverable training loops.

    For recoverable errors (default), keep the run open so retries can continue
    logging to the same run. Only close the run for fatal/session-ending errors.

    Args:
        session: The training session object
        do_debug: Whether to log debug info
        log_paths: Dict with log file paths
        fatal: Whether this error should terminate the W&B run
        reason: Optional short reason for debug logs
    """
    if session.wandb_run is None:
        return

    log_file = log_paths.get("wandb", log_paths.get("training")) if log_paths else None
    reason_suffix = f" ({reason})" if reason else ""

    if not fatal:
        if do_debug and log_file:
            debug_log(log_file, f"[W&B] recoverable error, keeping run open{reason_suffix}", force=True)
        return

    try:
        session.wandb_run.finish(exit_code=1)
        session.wandb_run = None
        if do_debug and log_file:
            debug_log(log_file, f"[W&B] finish: exit_code=1 (fatal cleanup){reason_suffix}", force=True)
    except Exception as cleanup_error:
        print(f"WARNING: W&B cleanup failed: {cleanup_error}", file=sys.stderr, flush=True)


def sanitize_wandb_env_vars(*, do_debug: bool, log_paths: dict | None = None) -> None:
    """Remove invalid empty-string W&B env vars.

    wandb validates some env vars (e.g., WANDB_MODE, WANDB_RUN_ID) strictly; if they
    exist but are empty strings, wandb.init() can fail even if we pass mode/id=None.
    """
    keys = (
        "WANDB_MODE",
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_GROUP",
    )
    removed: list[str] = []
    for key in keys:
        raw = os.environ.get(key)
        if raw is None:
            continue
        val = raw.strip()
        if not val or _UNRESOLVED_ENV_PLACEHOLDER_RE.match(val):
            os.environ.pop(key, None)
            removed.append(key)

    if not removed:
        return

    if not do_debug or not log_paths:
        return

    log_file = log_paths.get("wandb") or log_paths.get("init") or log_paths.get("training")
    if log_file:
        debug_log(log_file, f"[W&B] Removed empty env vars: {', '.join(removed)}", force=True)


def _env_nonempty(name: str) -> Optional[str]:
    """Return stripped env var value, or None if missing/empty."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    val = raw.strip()
    if _UNRESOLVED_ENV_PLACEHOLDER_RE.match(val):
        return None
    return val or None


async def maybe_init_wandb_run(
    *,
    session,
    method: str,
    base_model: str,
    wandb_project: str,
    wandb_run_name: Optional[str],
    group_default: str,
    name_components: list[str],
    step_metric: str,
    config: dict,
    extra_tags: Optional[list[str]],
    do_debug: bool,
    log_paths: dict,
    timeout_s: float,
) -> str:
    """Initialize a W&B run if WANDB_API_KEY is set; return a status string.

    This keeps the W&B integration centralized and consistent across SFT/GRPO.
    The intent is:
    - Simple defaults (project/group/run name auto-generated)
    - No empty-string env var footguns (sanitize_wandb_env_vars)
    - Max observability (tags + config + stable step axis)
    """
    if session.wandb_run is not None:
        # Already initialized for this session.
        return ""

    if _env_nonempty("WANDB_API_KEY") is None:
        return ""

    try:
        sanitize_wandb_env_vars(do_debug=do_debug, log_paths=log_paths)
        import wandb

        base_model_short = base_model.split("/")[-1] if base_model else "model"
        agent_prefix = (_env_nonempty("TINKERER_CHECKPOINT_PREFIX") or "tinkerer").strip() or "tinkerer"
        # For W&B filtering, keep a stable "agent kind" tag even when checkpoint
        # prefixes include per-run random suffixes (e.g., "claude-acde12").
        agent_kind = agent_prefix.split("-", 1)[0].split("_", 1)[0] or agent_prefix

        effective_project = _env_nonempty("WANDB_PROJECT") or (wandb_project or "").strip() or "tinkerer"
        effective_entity = _env_nonempty("WANDB_ENTITY")
        effective_group = _env_nonempty("WANDB_GROUP") or group_default

        # Advanced env overrides (kept optional; not required for normal usage).
        wandb_run_id = _env_nonempty("WANDB_RUN_ID")
        wandb_resume = _env_nonempty("WANDB_RESUME")
        wandb_mode = _env_nonempty("WANDB_MODE")
        if wandb_resume is not None and wandb_run_id is None:
            # W&B resume without an explicit run id is usually accidental and can
            # trigger strict validation errors depending on wandb version/config.
            # Keep integration simple: only resume when the run id is provided.
            if do_debug:
                debug_log(
                    log_paths["init"],
                    "[W&B] Ignoring WANDB_RESUME because WANDB_RUN_ID is not set",
                    force=True,
                )
            wandb_resume = None

        ts = time.strftime("%Y%m%d_%H%M%S")
        run_suffix = uuid.uuid4().hex[:6]
        if wandb_run_name:
            run_name = wandb_run_name
        else:
            parts = [agent_prefix, base_model_short, method] + list(name_components) + [ts, run_suffix]
            run_name = "-".join(p for p in parts if p)

        if do_debug:
            debug_log(
                log_paths["init"],
                f"[INFO] Initializing W&B: project={effective_project}, group={effective_group}, run={run_name}",
                force=True,
            )

        base_model_org = base_model.split("/")[0] if base_model and "/" in base_model else "model"
        tags = [method, base_model_org, f"agent:{agent_kind}"]
        if extra_tags:
            tags.extend(extra_tags)

        wandb_kwargs = {
            "project": effective_project,
            "group": effective_group,
            "name": run_name,
            "tags": tags,
            "reinit": "finish_previous",
            "save_code": True,
            "job_type": method,
            "config": {
                "method": method,
                "base_model": base_model,
                "agent_prefix": agent_prefix,
                "agent_kind": agent_kind,
                **(config or {}),
            },
        }
        # Only pass optional args when present; passing "" or placeholders can
        # trigger strict validation errors in some wandb versions.
        if effective_entity is not None:
            wandb_kwargs["entity"] = effective_entity

        # Only pass optional args when present; passing "" can trigger strict validation errors.
        if wandb_run_id is not None:
            wandb_kwargs["id"] = wandb_run_id
        if wandb_resume is not None:
            wandb_kwargs["resume"] = wandb_resume
        if wandb_mode is not None:
            wandb_kwargs["mode"] = wandb_mode

        session.wandb_run = await asyncio.wait_for(
            asyncio.to_thread(wandb.init, **wandb_kwargs),
            timeout=timeout_s,
        )

        # Prefer the provided step metric as the x-axis for all logged metrics.
        try:
            wandb.define_metric(step_metric)
            wandb.define_metric("*", step_metric=step_metric)
        except Exception:
            pass

        url = getattr(session.wandb_run, "url", "")
        if do_debug and url:
            debug_log(log_paths.get("wandb", log_paths["init"]), f"[W&B] init SUCCESS: {url}", force=True)
        return f"\nW&B: {url} (project={effective_project}, name={run_name})" if url else "\nW&B: initialized"

    except Exception as e:
        if do_debug:
            debug_log(log_paths["init"], f"[WARNING] W&B init failed: {e}", force=True)
        return f"\nW&B init failed: {e}"


def compute_weighted_nll(loss_fn_outputs, weights_list, *, context: str) -> float:
    """Compute weighted negative log-likelihood from loss_fn_outputs.

    Tinker training docs show loss_fn_outputs as a list of dicts containing
    per-token logprobs for each Datum. We compute the weighted mean loss
    exactly as documented.

    Args:
        loss_fn_outputs: List of dicts, each with "logprobs".
        weights_list: List of numpy arrays with per-token weights (aligned to each output).
        context: Short context string for error messages.

    Returns:
        Weighted mean NLL (scalar).
    """
    if not isinstance(loss_fn_outputs, list):
        raise TypeError(f"{context}: loss_fn_outputs must be a list, got {type(loss_fn_outputs).__name__}")
    if not loss_fn_outputs:
        raise ValueError(f"{context}: loss_fn_outputs is an empty list")
    if not isinstance(weights_list, list):
        raise TypeError(f"{context}: weights_list must be a list, got {type(weights_list).__name__}")
    if len(loss_fn_outputs) != len(weights_list):
        raise ValueError(
            f"{context}: loss_fn_outputs length {len(loss_fn_outputs)} != weights_list length {len(weights_list)}"
        )

    logprobs_chunks = []
    weight_chunks = []
    for i, (output, weights) in enumerate(zip(loss_fn_outputs, weights_list)):
        if not isinstance(output, dict):
            raise TypeError(
                f"{context}: loss_fn_outputs[{i}] must be a dict, got {type(output).__name__}. "
                f"Value: {repr(output)[:200]}"
            )
        if "logprobs" not in output:
            available_keys = list(output.keys())[:10]
            raise KeyError(
                f"{context}: loss_fn_outputs[{i}] missing required key 'logprobs'. "
                f"Available keys: {available_keys}. Value: {repr(output)[:200]}"
            )
        lp = output["logprobs"]
        if hasattr(lp, "tolist"):
            lp = lp.tolist()
        lp_arr = np.asarray(lp, dtype=float)
        w_arr = np.asarray(weights, dtype=float)
        if lp_arr.shape != w_arr.shape:
            raise ValueError(f"{context}: logprobs shape {lp_arr.shape} != weights shape {w_arr.shape} for item {i}")
        logprobs_chunks.append(lp_arr)
        weight_chunks.append(w_arr)

    logprobs_all = np.concatenate(logprobs_chunks) if logprobs_chunks else np.asarray([], dtype=float)
    weights_all = np.concatenate(weight_chunks) if weight_chunks else np.asarray([], dtype=float)
    if weights_all.size == 0 or float(weights_all.sum()) <= 0:
        raise ValueError(f"{context}: total weights sum to 0; cannot compute loss")
    loss = -float(np.dot(logprobs_all, weights_all)) / float(weights_all.sum())
    return loss


def compute_importance_sampling_loss(loss_fn_outputs, datums, *, context: str) -> float:
    """Compute importance_sampling loss from outputs + inputs.

    Per docs, loss_fn_outputs is a list of dicts with "logprobs" (target logprobs).
    Each corresponding Datum provides loss_fn_inputs["logprobs"] (sampling logprobs)
    and loss_fn_inputs["advantages"] (per-token advantages).

    Args:
        loss_fn_outputs: List of dicts with "logprobs".
        datums: List of Datum objects used for forward_backward.
        context: Short context string for error messages.

    Returns:
        Scalar loss:sum (sum over all tokens in batch).
    """
    if not isinstance(loss_fn_outputs, list):
        raise TypeError(f"{context}: loss_fn_outputs must be a list, got {type(loss_fn_outputs).__name__}")
    if not loss_fn_outputs:
        raise ValueError(f"{context}: loss_fn_outputs is an empty list")
    if not isinstance(datums, list):
        raise TypeError(f"{context}: datums must be a list, got {type(datums).__name__}")
    if len(loss_fn_outputs) != len(datums):
        raise ValueError(f"{context}: loss_fn_outputs length {len(loss_fn_outputs)} != datums length {len(datums)}")

    total_loss = 0.0
    for i, (output, datum) in enumerate(zip(loss_fn_outputs, datums)):
        if not isinstance(output, dict):
            raise TypeError(
                f"{context}: loss_fn_outputs[{i}] must be a dict, got {type(output).__name__}. "
                f"Value: {repr(output)[:200]}"
            )
        if "logprobs" not in output:
            available_keys = list(output.keys())[:10]
            raise KeyError(
                f"{context}: loss_fn_outputs[{i}] missing required key 'logprobs'. "
                f"Available keys: {available_keys}. Value: {repr(output)[:200]}"
            )
        if not hasattr(datum, "loss_fn_inputs") or not isinstance(datum.loss_fn_inputs, dict):
            raise TypeError(f"{context}: datum[{i}] missing loss_fn_inputs dict")
        if "logprobs" not in datum.loss_fn_inputs or "advantages" not in datum.loss_fn_inputs:
            available_keys = list(datum.loss_fn_inputs.keys())[:10]
            raise KeyError(
                f"{context}: datum[{i}].loss_fn_inputs missing required keys "
                f"'logprobs' and/or 'advantages'. Available keys: {available_keys}"
            )

        target_lp = output["logprobs"]
        sampling_lp = datum.loss_fn_inputs["logprobs"]
        advantages = datum.loss_fn_inputs["advantages"]

        if hasattr(target_lp, "tolist"):
            target_lp = target_lp.tolist()
        if hasattr(sampling_lp, "tolist"):
            sampling_lp = sampling_lp.tolist()
        if hasattr(advantages, "tolist"):
            advantages = advantages.tolist()

        target_lp_arr = np.asarray(target_lp, dtype=float)
        sampling_lp_arr = np.asarray(sampling_lp, dtype=float)
        adv_arr = np.asarray(advantages, dtype=float)

        if target_lp_arr.shape != sampling_lp_arr.shape:
            raise ValueError(
                f"{context}: target_logprobs shape {target_lp_arr.shape} "
                f"!= sampling_logprobs shape {sampling_lp_arr.shape}"
            )
        if target_lp_arr.shape != adv_arr.shape:
            raise ValueError(
                f"{context}: target_logprobs shape {target_lp_arr.shape} != advantages shape {adv_arr.shape}"
            )

        prob_ratio = np.exp(target_lp_arr - sampling_lp_arr)
        total_loss += float(-(prob_ratio * adv_arr).sum())

    return total_loss


def extract_metrics(metrics, *, context: str) -> dict:
    """Validate and extract metrics from ForwardBackwardOutput."""
    if metrics is None:
        return {}
    if not isinstance(metrics, dict):
        raise TypeError(
            f"{context}: metrics must be a dict, got {type(metrics).__name__}. Value: {repr(metrics)[:100]}"
        )
    return metrics


def validate_reward_output(output, *, num_completions: int, context: str) -> dict:
    """Validate reward subprocess JSON output."""
    if not isinstance(output, dict):
        raise TypeError(
            f"{context}: output must be a dict, got {type(output).__name__}. " f"Value: {repr(output)[:200]}"
        )
    if "rewards" not in output:
        raise KeyError(f"{context}: missing required key 'rewards'. Keys: {list(output.keys())}")

    rewards = output["rewards"]
    if not isinstance(rewards, list):
        raise TypeError(f"{context}: 'rewards' must be list, got {type(rewards).__name__}")
    if len(rewards) != num_completions:
        raise ValueError(f"{context}: rewards length {len(rewards)} != expected {num_completions}")

    # Validate reward values are numeric and finite (negative values are
    # allowed)
    validated_rewards = []
    for i, r in enumerate(rewards):
        try:
            r_float = float(r)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{context}: rewards[{i}] must be numeric, got {type(r).__name__}. " f"Value: {repr(r)[:200]}"
            ) from exc
        if not np.isfinite(r_float):
            raise ValueError(f"{context}: rewards[{i}] must be finite, got {r_float}.")
        validated_rewards.append(r_float)

    return {
        "rewards": validated_rewards,
        "errors": output.get("errors", []) if isinstance(output.get("errors"), list) else [],
        "error_count": int(output.get("error_count", 0)) if isinstance(output.get("error_count"), (int, float)) else 0,
    }
