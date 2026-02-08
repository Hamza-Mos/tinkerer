"""SFT (Supervised Fine-Tuning) implementation for the Tinker MCP Server.

SFT teaches a model to imitate examples. Use when you have quality input-output
pairs and want the model to learn patterns, style, or format. Best for
subjective tasks where "good" means "similar to these examples."

This module contains the implementation logic. The MCP tool docstrings remain
in server.py as they are the "interface" for agent runtimes.
"""

import asyncio
import random
import sys
import time
from typing import Optional

import numpy as np
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from tinker_mcp.state import get_session
from tinker_mcp.models import (
    MODEL_INFO,
    SUPPORTED_MODELS,
    create_renderer_for_model,
)
from tinker_mcp.utils import (
    DEBUG_MODE,
    TIMEOUT_MEDIUM,
    TIMEOUT_LONG,
    WANDB_INIT_TIMEOUT,
    apply_checkpoint_prefix,
    debug_log,
    clear_debug_logs,
    get_log_paths,
    get_lr,
    write_progress,
    wait_with_heartbeat,
    get_service_client_async,
    cleanup_stale_progress_files,
    log_warning,
    log_error,
)
from tinker_mcp.training.common import (
    load_json_from_file_or_string,
    validate_warmup_for_continued_training,
    validate_lr_scheduler_name,
    resolve_scheduler_total_steps,
    cleanup_wandb_on_error,
    maybe_init_wandb_run,
    compute_weighted_nll,
)

# Checkpoint interval: refresh sampler every epoch to keep sample() on
# latest weights.
SFT_CHECKPOINT_INTERVAL = 1


def _tensor_like_to_1d_numpy(tensor_like, dtype) -> np.ndarray:
    """Convert TensorData / torch tensor / list-like input to a flat numpy
    array."""
    if isinstance(tensor_like, np.ndarray):
        arr = tensor_like.astype(dtype, copy=False)
    elif hasattr(tensor_like, "detach") and hasattr(tensor_like, "cpu"):
        arr = tensor_like.detach().cpu().numpy().astype(dtype, copy=False)
    elif hasattr(tensor_like, "data"):
        arr = np.asarray(tensor_like.data, dtype=dtype)
    elif hasattr(tensor_like, "tolist"):
        arr = np.asarray(tensor_like.tolist(), dtype=dtype)
    else:
        arr = np.asarray(tensor_like, dtype=dtype)
    return arr.reshape(-1)


async def init_sft_session(
    base_model: str,
    lora_rank: int = 16,
    wandb_project: str = "tinkerer",
    wandb_run_name: Optional[str] = None,
    debug: bool = False,
) -> str:
    """Initialize a supervised fine-tuning session.

    Args:
        base_model: Model to fine-tune
        lora_rank: Adapter capacity (16 for simple, 32+ for complex)
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        debug: Enable debug logging

    Returns:
        Status message with session info
    """
    session = get_session()
    log_paths = get_log_paths()

    # Clean up stale progress files from crashed sessions (non-blocking)
    cleanup_stale_progress_files(max_age_hours=6)

    # Write progress immediately for watchdog heartbeat (before any slow operations)
    # This ensures the progress file exists during model loading which can
    # take 5-15+ minutes
    await write_progress(f"Initializing SFT session: {base_model}")

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    # Clear debug logs at session start (but NOT the progress file - it was
    # just created above)
    if do_debug:
        clear_debug_logs()
        debug_log(log_paths["init"], f"[START] init_sft: model={base_model}, lora_rank={lora_rank}", force=True)

    # Model registry is "known good", not a hard gate. Unlisted models can work
    # as long as the backend supports the model ID and the renderer mapping can
    # be resolved.
    if base_model not in SUPPORTED_MODELS:
        note_msg = (
            f"Model '{base_model}' is not in the harness MODEL_INFO registry. "
            "Proceeding anyway; defaults will be used for metadata, and renderer "
            "mapping will be resolved automatically."
        )
        log_warning(note_msg)
        if do_debug:
            debug_log(log_paths["init"], f"[NOTE] {note_msg}", force=True)
        await write_progress(f"NOTE: {note_msg}")

    # Non-Base models are supported, but Base models are usually easier to
    # post-train and less instruction-biased.
    model_info = MODEL_INFO.get(base_model, {})
    training_type = model_info.get("training_type")
    if training_type and training_type != "Base":
        warning_msg = (
            f"Model '{base_model}' is tagged as '{training_type}'. "
            "SFT is allowed, but Base models are usually preferred for post-training."
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
        # Use 30 min timeout to stay under Modal watchdog (35 min)
        MAX_RETRY_TIMEOUT = 1800  # 30 min cap
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

        # Initialize renderer (prompt/supervised formatting).
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
        session.session_method = "sft"
        session.sft_epochs = 0
        session.sft_steps = 0
        session.sft_scheduler_total_steps = 0

        # Initialize sampling client for SFT (avoids checkpoint save on every
        # sample() call)
        if do_debug:
            debug_log(log_paths["init"], "[INFO] Creating initial sampling client for SFT...", force=True)
        init_checkpoint_name = apply_checkpoint_prefix("sft_init")
        session.sampling_client = await wait_with_heartbeat(
            session.training_client.save_weights_and_get_sampling_client_async(init_checkpoint_name),
            timeout=TIMEOUT_LONG,
            progress_msg="Creating initial SFT sampling client",
            interval=60.0,
            debug_log_file=log_paths["init"] if do_debug else None,
        )
        if do_debug:
            debug_log(log_paths["init"], "[INFO] SFT sampling client created", force=True)

        # Initialize W&B (optional).
        wandb_status = await maybe_init_wandb_run(
            session=session,
            method="sft",
            base_model=base_model,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            group_default=f"{base_model.split('/')[-1]}-sft-{time.strftime('%Y%m%d')}",
            name_components=[f"r{lora_rank}"],
            # SFT logs per-batch (step) for maximum observability.
            step_metric="step",
            config={
                "lora_rank": lora_rank,
            },
            extra_tags=None,
            do_debug=do_debug,
            log_paths=log_paths,
            timeout_s=WANDB_INIT_TIMEOUT,
        )

        total_time = time.time() - start_time
        if do_debug:
            debug_log(log_paths["init"], f"[SUCCESS] init_sft complete (total: {total_time:.1f}s)", force=True)

        # Get model info for enhanced output
        model_info = MODEL_INFO.get(base_model, {})
        arch = model_info.get("architecture", "unknown")
        active = model_info.get("active_params", model_info.get("total_params", "unknown"))

        return f"""
SFT SESSION INITIALIZED
=======================
Model: {base_model}
  Architecture: {arch}
  Active params: {active}

Config:
  lora_rank: {lora_rank}
  renderer: {session.renderer_name}
{wandb_status}

SFT HYPERPARAMETER GUIDANCE:
- Prefer training one epoch at a time so you can stop early.
- If loss spikes or outputs degrade, reduce the learning rate.
- Keep a validation split so you can detect overfitting (val_loss up while train_loss down).

Watch for overfitting: val_loss up while train_loss down = STOP!

NEXT: Call train_sft_step(examples, num_epochs=1) to train.
"""

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        # Always log the full traceback for init failures: without it, users
        # have to guess whether the issue was provisioning, auth, renderer, etc.
        debug_log(log_paths["init"], f"[ERROR] init_sft failed: {e}\n{tb}", force=True)
        # Init failure is fatal for this run; close W&B cleanly.
        cleanup_wandb_on_error(session, do_debug, log_paths, fatal=True, reason="init_sft failed")
        if do_debug:
            return f"Error initializing SFT session: {e}\n{tb}"
        return f"Error initializing SFT session: {e}"


def _validate_sft_example_schema(examples_list: list) -> Optional[str]:
    """Validate SFT example JSON schema."""
    for i, ex in enumerate(examples_list):
        if not isinstance(ex, dict):
            return (
                f"Error: Example {i} must be a dict, got {type(ex).__name__}. "
                "Expected format: {'prompt': '...', 'response': '...'}"
            )
        prompt = ex.get("prompt") or ex.get("input")
        response = ex.get("response") or ex.get("output")
        if prompt is not None and not isinstance(prompt, str):
            return f"Error: Example {i} prompt must be a string, got {type(prompt).__name__}"
        if response is not None and not isinstance(response, str):
            return f"Error: Example {i} response must be a string, got {type(response).__name__}"
    return None


def _tokenize_sft_examples(examples_list: list, renderer, renderer_name: str, do_debug: bool, log_paths: dict) -> tuple:
    """Tokenize examples with renderer and filter invalid rows."""
    tokenized = []
    filtered_count = 0
    token_lengths = []

    for idx, ex in enumerate(examples_list):
        prompt = ex.get("prompt", ex.get("input", ""))
        response = ex.get("response", ex.get("output", ""))
        if not prompt or not response:
            filtered_count += 1
            missing = []
            if not prompt:
                missing.append("prompt")
            if not response:
                missing.append("response")
            log_warning(f"SKIPPING example {idx}: Empty {' and '.join(missing)}")
            continue

        try:
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            model_input, weights_tensor = renderer.build_supervised_example(conversation)
            weights_arr = _tensor_like_to_1d_numpy(weights_tensor, np.float32)
            if do_debug and idx == 0:
                debug_log(
                    log_paths["training"],
                    f"[TOKENIZE] Renderer={renderer_name}, tokens={model_input.length}",
                    force=True,
                )
        except Exception as e:
            log_warning(f"SKIPPING example {idx}: renderer.build_supervised_example failed: {e}")
            if do_debug:
                debug_log(log_paths["training"], f"[TOKENIZE] Renderer failed for example {idx}: {e}", force=True)
            filtered_count += 1
            continue

        if model_input.length < 2:
            log_warning(f"SKIPPING example {idx}: example tokenized to {model_input.length} token(s), need >=2")
            filtered_count += 1
            continue
        if model_input.length != weights_arr.shape[0]:
            log_warning(
                "SKIPPING example "
                f"{idx}: renderer produced length mismatch (tokens={model_input.length}, "
                f"weights={weights_arr.shape[0]})"
            )
            filtered_count += 1
            continue

        tokenized.append(
            {
                "model_input": model_input,
                "weights": weights_tensor,
                "example_key": f"{prompt}\n{response}",
            }
        )
        token_lengths.append(model_input.length)

    return tokenized, filtered_count, token_lengths


def _split_tokenized_sft_examples(tokenized: list, validation_split: float) -> tuple[list, list]:
    """Split tokenized examples deterministically into train/validation sets."""
    if validation_split <= 0 or len(tokenized) <= 1:
        return tokenized, []

    n_val = max(1, int(len(tokenized) * validation_split))
    shuffled = tokenized.copy()

    import hashlib

    data_str = str(sorted(ex["example_key"] for ex in tokenized))
    data_hash = int(hashlib.md5(data_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(data_hash)
    rng.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


def _build_cross_entropy_batch_datums(batch_items: list, context_label: str) -> tuple[list, list[np.ndarray], float]:
    """Create Datum objects and validate shape contracts."""
    batch_datums = []
    batch_weights_list = []
    batch_weight_sum = 0.0

    for item in batch_items:
        datum = datum_from_model_input_weights(item["model_input"], item["weights"])
        target_tokens_arr = _tensor_like_to_1d_numpy(datum.loss_fn_inputs["target_tokens"], np.int64)
        target_weights_arr = _tensor_like_to_1d_numpy(datum.loss_fn_inputs["weights"], np.float32)

        if target_tokens_arr.shape[0] != target_weights_arr.shape[0]:
            raise ValueError(
                f"{context_label} length mismatch: "
                f"target_tokens={target_tokens_arr.shape[0]}, weights={target_weights_arr.shape[0]}"
            )
        if datum.model_input.length != target_tokens_arr.shape[0]:
            raise ValueError(
                f"{context_label}/model_input mismatch: model_input={datum.model_input.length}, "
                f"target_tokens={target_tokens_arr.shape[0]}"
            )

        batch_weight_sum += float(target_weights_arr.sum())
        batch_weights_list.append(target_weights_arr)
        batch_datums.append(datum)

    return batch_datums, batch_weights_list, batch_weight_sum


async def _run_cross_entropy_train_batch(session, batch_items: list, lr: float, types_module) -> float:
    """Run one train batch: forward_backward + optim_step + weighted NLL."""
    batch_datums, batch_weights_list, batch_weight_sum = _build_cross_entropy_batch_datums(
        batch_items, "SFT batch datum"
    )

    fwdbwd_future = await asyncio.wait_for(
        session.training_client.forward_backward_async(batch_datums, loss_fn="cross_entropy"),
        timeout=TIMEOUT_MEDIUM,
    )
    optim_future = await asyncio.wait_for(
        session.training_client.optim_step_async(
            types_module.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95)
        ),
        timeout=TIMEOUT_MEDIUM,
    )
    fwdbwd_result = await asyncio.to_thread(fwdbwd_future.result, TIMEOUT_MEDIUM)
    await asyncio.to_thread(optim_future.result, TIMEOUT_MEDIUM)

    if batch_weight_sum <= 0:
        raise ValueError("Training failed: batch_weight_sum is 0. No valid weights to compute loss.")
    return compute_weighted_nll(fwdbwd_result.loss_fn_outputs, batch_weights_list, context="SFT train forward_backward")


async def _run_cross_entropy_validation(
    session,
    val_tokenized: list,
    batch_size: int,
    types_module,
    do_debug: bool,
    log_paths: dict,
) -> Optional[float]:
    """Run validation forward passes and clear gradients after validation."""
    if not val_tokenized:
        return None

    val_total_loss = 0.0
    val_steps = 0
    consecutive_zero_val_losses = 0

    for i in range(0, len(val_tokenized), batch_size):
        val_batch = val_tokenized[i : i + batch_size]
        val_batch_datums, val_batch_weights_list, val_batch_weight_sum = _build_cross_entropy_batch_datums(
            val_batch, "SFT val datum"
        )

        val_fwd_future = await asyncio.wait_for(
            session.training_client.forward_backward_async(val_batch_datums, loss_fn="cross_entropy"),
            timeout=TIMEOUT_MEDIUM,
        )
        val_fwd_result = await asyncio.to_thread(val_fwd_future.result, TIMEOUT_MEDIUM)

        if val_batch_weight_sum <= 0:
            raise ValueError("Validation failed: val_batch_weight_sum is 0. No valid weights to compute loss.")

        val_batch_loss = compute_weighted_nll(
            val_fwd_result.loss_fn_outputs,
            val_batch_weights_list,
            context="SFT validation forward_backward",
        )

        if val_batch_loss == 0.0:
            consecutive_zero_val_losses += 1
            log_warning(
                f"val_batch_loss is exactly 0.0 (consecutive: {consecutive_zero_val_losses}) - "
                f"API may have failed silently."
            )
            if consecutive_zero_val_losses >= 3:
                log_error(
                    "FATAL: 3 consecutive zero validation losses detected. "
                    "API is failing silently or returning invalid results."
                )
                raise ValueError(
                    "Validation failed: 3 consecutive zero validation losses detected. "
                    "This indicates the API is failing silently or returning invalid results. "
                    "Check API connectivity and model state."
                )
        else:
            consecutive_zero_val_losses = 0

        val_total_loss += val_batch_loss
        val_steps += 1

    clear_grad_future = await asyncio.wait_for(
        session.training_client.optim_step_async(
            types_module.AdamParams(learning_rate=0.0, beta1=0.9, beta2=0.95)
        ),
        timeout=TIMEOUT_MEDIUM,
    )
    await asyncio.to_thread(clear_grad_future.result, TIMEOUT_MEDIUM)
    if do_debug:
        debug_log(log_paths["training"], "[VAL] Cleared validation gradients (optim_step lr=0)", force=True)

    return val_total_loss / max(val_steps, 1)


async def train_sft_step_impl(
    examples: str,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    batch_size: int = 128,
    warmup_ratio: float = 0.1,
    lr_scheduler: str = "constant",
    scheduler_total_steps: Optional[int] = None,
    validation_split: float = 0.1,
    debug: bool = False,
) -> str:
    """Train the model on examples. Implementation function.

    Args:
        examples: JSON array of {"prompt": "...", "response": "..."} OR a file path to a JSON file
        num_epochs: Number of passes through data
        learning_rate: Base learning rate
        batch_size: Training batch size
        warmup_ratio: Fraction of steps for warmup
        lr_scheduler: "constant", "linear", or "cosine"
        scheduler_total_steps: Optional fixed horizon for linear/cosine schedules.
        validation_split: Fraction for validation
        debug: Enable debug logging

    Returns:
        Training report with metrics and guidance
    """
    session = get_session()
    log_paths = get_log_paths()

    # Write progress immediately for watchdog heartbeat (before
    # validation/setup)
    await write_progress(f"Starting SFT training step: {num_epochs} epochs")

    # Enable debug logging if debug=True or TINKERER_DEBUG=1
    do_debug = debug or DEBUG_MODE

    # Collect warnings to include in return value
    warnings_for_output = []

    if session.training_client is None or session.session_method != "sft":
        return (
            "Error: No SFT session active. Call init_sft(base_model) first "
            "before training.\nExample: "
            'init_sft("vendor/model-base")'
        )
    if session.renderer is None:
        return "Error: Renderer not initialized. Call init_sft(base_model) again to set up renderer."

    scheduler_error = validate_lr_scheduler_name(lr_scheduler)
    if scheduler_error:
        return f"Error: {scheduler_error}"

    if do_debug:
        debug_log(
            log_paths["training"],
            f"[START] train_sft_step: examples={len(examples) if examples else 0} chars, "
            f"epochs={num_epochs}, lr={learning_rate}",
            force=True,
        )
        debug_log(
            log_paths["progress"],
            f"[START] train_sft_step: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}",
            force=True,
        )

    # Check for warmup_ratio on continued training - warn user about potential
    # LR instability
    warmup_ratio, warmup_warning = validate_warmup_for_continued_training(
        warmup_ratio, session.sft_steps, step_name="step"
    )
    if warmup_warning:
        warnings_for_output.append(warmup_warning)
        print(f"\n[WARNING] {warmup_warning}\n", file=sys.stderr, flush=True)
        if do_debug:
            debug_log(log_paths["training"], f"[WARNING] {warmup_warning}", force=True)

    # Load examples from file or parse JSON string
    examples_list, load_error = load_json_from_file_or_string(examples, "examples", do_debug, log_paths)
    if load_error:
        return load_error

    if not examples_list:
        if do_debug:
            debug_log(log_paths["training"], "[ERROR] No examples provided", force=True)
        return "Error: No examples provided."

    schema_error = _validate_sft_example_schema(examples_list)
    if schema_error:
        return schema_error

    if do_debug:
        debug_log(log_paths["training"], f"[INFO] Parsed {len(examples_list)} examples", force=True)

    try:
        from tinker import types

        tokenized, filtered_count, token_lengths = _tokenize_sft_examples(
            examples_list,
            session.renderer,
            session.renderer_name,
            do_debug,
            log_paths,
        )

        if not tokenized:
            if do_debug:
                debug_log(log_paths["training"], "[ERROR] No valid examples after tokenization", force=True)
            return "Error: No valid examples after tokenization."

        if do_debug:
            debug_log(
                log_paths["training"],
                f"[INFO] Parsed {len(tokenized)} examples ({filtered_count} filtered)",
                force=True,
            )
            debug_log(
                log_paths["training"],
                f"[INFO] Tokenization: min={min(token_lengths)}, max={max(token_lengths)}, "
                f"mean={sum(token_lengths) // len(token_lengths)} tokens",
                force=True,
            )

        train_tokenized, val_tokenized = _split_tokenized_sft_examples(tokenized, validation_split)

        if do_debug:
            debug_log(
                log_paths["training"],
                f"[INFO] Train/val split: {len(train_tokenized)}/{len(val_tokenized)}",
                force=True,
            )

        # Calculate total steps for LR scheduling
        num_batches = (len(train_tokenized) + batch_size - 1) // batch_size
        steps_this_call = num_epochs * num_batches
        try:
            estimated_total_steps, scheduler_warning = resolve_scheduler_total_steps(
                scheduler=lr_scheduler,
                explicit_total_steps=scheduler_total_steps,
                cumulative_steps=session.sft_steps,
                steps_this_call=steps_this_call,
                persisted_total_steps=session.sft_scheduler_total_steps,
                default_floor_steps=20,
                unit_name="step",
            )
        except ValueError as exc:
            return f"Error: {exc}"
        session.sft_scheduler_total_steps = max(session.sft_scheduler_total_steps, estimated_total_steps)
        if scheduler_warning:
            warnings_for_output.append(scheduler_warning)
            print(f"\n[WARNING] {scheduler_warning}\n", file=sys.stderr, flush=True)
            if do_debug:
                debug_log(log_paths["training"], f"[WARNING] {scheduler_warning}", force=True)

        if do_debug:
            debug_log(log_paths["training"], f"[INFO] Batches: {num_batches} (batch_size={batch_size})", force=True)
            debug_log(
                log_paths["training"],
                f"[INFO] Steps this call: {steps_this_call}, cumulative before: {session.sft_steps}",
                force=True,
            )
            debug_log(
                log_paths["training"],
                f"[INFO] Scheduler horizon: {estimated_total_steps} (requested={scheduler_total_steps})",
                force=True,
            )

        # Training loop
        epoch_losses = []
        val_losses = []
        steps_in_call = 0
        consecutive_zero_losses = 0
        val_cursor = 0
        val_check_every = max(1, num_batches // 4) if val_tokenized else 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            last_lr = learning_rate

            for i in range(0, len(train_tokenized), batch_size):
                batch_start = time.time()
                batch = train_tokenized[i : i + batch_size]
                batch_num = i // batch_size + 1

                # Compute learning rate with warmup and decay using cumulative
                # steps
                current_cumulative_step = session.sft_steps + steps_in_call
                lr = get_lr(current_cumulative_step, estimated_total_steps, learning_rate, warmup_ratio, lr_scheduler)
                last_lr = lr
                warmup_steps = int(estimated_total_steps * warmup_ratio)
                lr_phase = "warmup" if current_cumulative_step < warmup_steps else lr_scheduler

                batch_loss = await _run_cross_entropy_train_batch(session, batch, lr, types)

                # Track consecutive zero losses - indicates API failure
                if batch_loss == 0.0:
                    consecutive_zero_losses += 1
                    # ALWAYS log - never silent
                    log_warning(
                        f"batch_loss is exactly 0.0 (consecutive: {consecutive_zero_losses}) - "
                        f"API may have failed silently. Check API connectivity."
                    )
                    if consecutive_zero_losses >= 3:
                        log_error(
                            "FATAL: 3 consecutive zero losses detected. "
                            "API is failing silently or returning invalid gradients."
                        )
                        raise ValueError(
                            "Training failed: 3 consecutive zero losses detected. "
                            "This indicates the API is failing silently or returning invalid gradients. "
                            "Check API connectivity and model state."
                        )
                else:
                    consecutive_zero_losses = 0

                epoch_loss += batch_loss
                epoch_steps += 1
                steps_in_call += 1

                # Write progress for watchdog heartbeat
                await write_progress(
                    f"SFT epoch {epoch + 1}/{num_epochs}: batch {batch_num}/{num_batches}, loss={batch_loss:.4f}"
                )

                # Periodic mid-epoch validation for early overfitting detection.
                # Use a single validation batch (not the full val set) to keep feedback tight.
                val_loss_batch = None
                if val_tokenized and val_check_every and (batch_num % val_check_every == 0 or batch_num == num_batches):
                    # Rotate through val data deterministically.
                    start = val_cursor
                    end = start + batch_size
                    if end <= len(val_tokenized):
                        val_batch = val_tokenized[start:end]
                    else:
                        val_batch = val_tokenized[start:] + val_tokenized[: max(0, end - len(val_tokenized))]
                    val_cursor = (val_cursor + batch_size) % max(len(val_tokenized), 1)
                    val_loss_batch = await _run_cross_entropy_validation(
                        session,
                        val_batch,
                        batch_size,
                        types,
                        do_debug,
                        log_paths,
                    )

                # Log to W&B per batch (step).
                if session.wandb_run is not None:
                    try:
                        # 1-index step for readability in charts.
                        wandb_step = int(session.sft_steps + steps_in_call)
                        log_data = {
                            "step": wandb_step,
                            "epoch": int(session.sft_epochs + epoch + 1),
                            "batch": int(batch_num),
                            "train_loss": float(batch_loss),
                            "learning_rate": float(lr),
                        }
                        if val_loss_batch is not None:
                            log_data["val_loss_batch"] = float(val_loss_batch)
                        session.wandb_run.log(log_data, step=wandb_step)
                        if do_debug:
                            debug_log(log_paths["wandb"], f"[W&B] log: {log_data}", force=True)
                    except Exception as e:
                        print(f"W&B metric log warning: {e}", file=sys.stderr, flush=True)

                if do_debug:
                    batch_time = time.time() - batch_start
                    debug_log(
                        log_paths["training"],
                        f"[BATCH {batch_num}/{num_batches}] loss={batch_loss:.4f}, "
                        f"lr={lr:.2e} ({lr_phase}), time={batch_time:.1f}s",
                        force=True,
                    )

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_losses.append(avg_epoch_loss)
            session.sft_epochs += 1

            val_loss = await _run_cross_entropy_validation(
                session,
                val_tokenized,
                batch_size,
                types,
                do_debug,
                log_paths,
            )
            if val_loss is not None:
                val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start

            # Log epoch summary to W&B.
            if session.wandb_run is not None:
                try:
                    wandb_step = int(session.sft_steps + steps_in_call)
                    log_data = {
                        "step": wandb_step,
                        "epoch": int(session.sft_epochs),
                        "train_loss_epoch": float(avg_epoch_loss),
                        "learning_rate": float(last_lr),
                    }
                    if val_loss is not None:
                        log_data["val_loss"] = float(val_loss)
                    session.wandb_run.log(log_data, step=wandb_step)
                    if do_debug:
                        debug_log(log_paths["wandb"], f"[W&B] log: {log_data}", force=True)
                except Exception as e:
                    print(f"W&B metric log warning: {e}", file=sys.stderr, flush=True)

            # Write progress
            progress_msg = f"SFT epoch {epoch + 1}/{num_epochs} done: train_loss={avg_epoch_loss:.4f}"
            if val_loss is not None:
                progress_msg += f", val_loss={val_loss:.4f}"
            progress_msg += f" (cumulative: {session.sft_epochs})"
            await write_progress(progress_msg)

            if do_debug:
                epoch_log = f"[EPOCH {epoch + 1}/{num_epochs}] train_loss={avg_epoch_loss:.4f}"
                if val_loss is not None:
                    epoch_log += f", val_loss={val_loss:.4f}"
                epoch_log += f", time={epoch_time:.1f}s"
                debug_log(log_paths["training"], epoch_log, force=True)

        # Build report
        if val_losses:
            epoch_report = "\n".join(
                [
                    f"  Epoch {session.sft_epochs - num_epochs + i + 1}: train_loss={tl:.4f}, val_loss={vl:.4f}"
                    for i, (tl, vl) in enumerate(zip(epoch_losses, val_losses))
                ]
            )
        else:
            epoch_report = "\n".join(
                [
                    f"  Epoch {session.sft_epochs - num_epochs + i + 1}: loss={loss:.4f}"
                    for i, loss in enumerate(epoch_losses)
                ]
            )

        # Check for overfitting
        overfitting_warning = ""
        is_overfitting = False
        if len(val_losses) >= 2 and val_losses[-1] > val_losses[0] and epoch_losses[-1] < epoch_losses[0]:
            overfitting_warning = "\nWARNING: val_loss increasing while train_loss decreasing - possible overfitting!"
            is_overfitting = True

        train_val_split_info = ""
        if val_tokenized:
            train_val_split_info = (
                f"\nTrain/Val split: {len(train_tokenized)}/{len(val_tokenized)} "
                f"({100 - validation_split * 100:.0f}%/{validation_split * 100:.0f}%)"
            )

        # Compute decision guidance
        if epoch_losses:
            first_loss = epoch_losses[0]
            last_loss = epoch_losses[-1]
            loss_delta = last_loss - first_loss

            if loss_delta < -0.01:
                train_trend = "decreasing (good)"
            elif loss_delta > 0.01:
                train_trend = "increasing (problem)"
            else:
                train_trend = "flat"

            if val_losses:
                first_val = val_losses[0]
                last_val = val_losses[-1]
                val_delta = last_val - first_val
                if val_delta < -0.01:
                    val_trend = "decreasing (good)"
                elif val_delta > 0.01:
                    val_trend = "increasing (caution)"
                else:
                    val_trend = "flat"
            else:
                val_trend = "N/A (no validation split)"

            if is_overfitting:
                decision_text = "WARNING: Overfitting detected. Model is memorizing, not generalizing."
                next_step = "STOP: Save current checkpoint and evaluate quality with sample()"
            elif loss_delta > 0.1:
                decision_text = "WARNING: Loss increasing significantly. Training may be unstable."
                next_step = "ADJUST: Reduce learning_rate by 2-5x and retry"
            elif abs(loss_delta) < 0.001 and session.sft_epochs > 3:
                decision_text = "NOTE: Loss plateaued - model may have converged."
                next_step = "EVALUATE: Call sample() to check quality. If good, save. If not, consider more/better data"
            else:
                decision_text = "HEALTHY: Loss decreasing, training is working."
                next_step = "CONTINUE: Call train_sft_step(num_epochs=1, warmup_ratio=0) to continue"
        else:
            train_trend = "N/A"
            val_trend = "N/A"
            decision_text = "No epoch data available."
            next_step = "Check for errors above"

        # Update cumulative step counter
        session.sft_steps += steps_in_call

        # Only checkpoint every SFT_CHECKPOINT_INTERVAL epochs (or on last
        # epoch of this call) to reduce overhead
        should_checkpoint = (
            session.sft_epochs % SFT_CHECKPOINT_INTERVAL == 0
            or
            # Always checkpoint for single-epoch calls (user expects immediate
            # feedback)
            num_epochs == 1
        )

        if should_checkpoint:
            # Update sampling client with new weights
            checkpoint_name = apply_checkpoint_prefix(f"sft_epoch_{session.sft_epochs}")
            if do_debug:
                debug_log(log_paths["training"], f"[WEIGHTS] Saving SFT checkpoint {checkpoint_name}...", force=True)
            session.sampling_client = await wait_with_heartbeat(
                session.training_client.save_weights_and_get_sampling_client_async(checkpoint_name),
                timeout=TIMEOUT_LONG,
                progress_msg=f"Saving SFT checkpoint {checkpoint_name}",
                interval=60.0,
                debug_log_file=log_paths["training"] if do_debug else None,
            )
            session.save_checkpoint_metadata(checkpoint_name)
        else:
            if do_debug:
                debug_log(
                    log_paths["training"],
                    f"[WEIGHTS] Skipping checkpoint (interval={SFT_CHECKPOINT_INTERVAL}, epoch={session.sft_epochs})",
                    force=True,
                )

        # Build warnings section for output
        warnings_text = ""
        if warnings_for_output:
            warnings_text = "\n\nWARNINGS:\n" + "\n".join(f"- {w}" for w in warnings_for_output)

        wandb_line = ""
        if session.wandb_run is not None:
            url = getattr(session.wandb_run, "url", "") or ""
            if url:
                wandb_line = f"\nW&B run: {url}\n"

        return f"""
SFT TRAINING STEP COMPLETE
==========================
Epochs this step: {num_epochs}
Cumulative epochs: {session.sft_epochs}
Examples: {len(examples_list)}{train_val_split_info}
{wandb_line}

HYPERPARAMETERS USED:
  learning_rate: {learning_rate}
  batch_size: {batch_size}
  lr_scheduler: {lr_scheduler} (warmup: {warmup_ratio:.0%})
  scheduler_total_steps: {estimated_total_steps if lr_scheduler != "constant" else "N/A (constant)"}

LOSS PER EPOCH:
{epoch_report}{overfitting_warning}{warnings_text}

-------------------------------------
DECISION GUIDANCE
-------------------------------------
train_loss: {train_trend}
val_loss: {val_trend}

{decision_text}

SUGGESTED NEXT STEP: {next_step}

DEBUG LOGS (if investigating):
- {log_paths["training"]} - Training loop details
- {log_paths["progress"]} - Progress heartbeat
"""

    except Exception as e:
        import traceback

        if do_debug:
            debug_log(log_paths["training"], f"[ERROR] train_sft_step failed: {e}", force=True)
        # Keep W&B run open on recoverable step errors so retries continue logging.
        cleanup_wandb_on_error(session, do_debug, log_paths, fatal=False, reason="train_sft_step failed")
        return f"Error in SFT training: {e}\n{traceback.format_exc()}"
