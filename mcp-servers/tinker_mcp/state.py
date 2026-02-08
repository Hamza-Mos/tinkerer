"""Session state management for the Tinker MCP Server.

This module encapsulates all mutable training session state into a single
TrainingSession class. The server is INTENTIONALLY STATEFUL - unlike typical
MCP servers that prefer stateless, idempotent tools, fine-tuning requires
maintaining state across multiple tool calls:

- Model weights evolve during training (train_sft_step/train_grpo_step)
- Sampling client tracks current checkpoint state
- W&B run persists for logging across training iterations
- Cumulative counters track epochs/iterations for LR scheduling

This is a deliberate design choice for training workflows where:
1. Each training step builds on previous state
2. The user iteratively trains, samples, and adjusts
3. Session state enables "train -> sample -> train more" patterns

Future consideration: Add session_id parameter for multi-session support.
"""

import asyncio
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

# Session directory permissions - restrict to owner only for security
# Prevents other users on shared systems from reading training data
SESSION_DIR_MODE = 0o700


def _create_session_dir() -> str:
    """Create a session directory with restricted permissions.

    Returns:
        Path to the created directory
    """
    dir_path = tempfile.mkdtemp(prefix="tinkerer_")
    try:
        os.chmod(dir_path, SESSION_DIR_MODE)
    except OSError as e:
        print(f"Warning: Could not set session directory permissions: {e}", file=sys.stderr)
    return dir_path


@dataclass
class TrainingSession:
    """Encapsulates all mutable training session state.

    This class holds all the state that persists across tool calls during
    a training session. It provides a clean interface for resetting state
    and checking session status.

    Attributes:
        service_client: Tinker service client for API calls
        training_client: Tinker training client for forward/backward passes
        sampling_client: Tinker sampling client for inference
        tokenizer: Model tokenizer for encoding/decoding
        current_model: Name of the base model being trained
        wandb_run: Active W&B run for metric logging
        session_method: "sft" or "grpo" - which training method is active
        sft_epochs: Cumulative SFT epochs across train_sft_step calls
        sft_steps: Cumulative SFT steps for LR scheduling
        grpo_iterations: Cumulative GRPO iterations across train_grpo_step calls
        grpo_group_size: Number of completions per prompt in GRPO
        saved_adapters: Maps adapter_name -> full tinker:// path
        saved_metadata: Maps checkpoint name -> training state for restore
        consecutive_empty_batches: Track empty batches for robustness
        session_dir: Temp directory for this session's files
    """

    # API clients
    service_client: Optional[Any] = None
    training_client: Optional[Any] = None
    sampling_client: Optional[Any] = None

    # Model state
    tokenizer: Optional[Any] = None
    current_model: Optional[str] = None
    renderer: Optional[Any] = None
    renderer_name: Optional[str] = None
    image_processor: Optional[Any] = None

    # Logging
    wandb_run: Optional[Any] = None

    # Session type
    session_method: Optional[str] = None  # "sft" or "grpo"

    # Cumulative counters for LR scheduling
    sft_epochs: int = 0
    sft_steps: int = 0
    grpo_iterations: int = 0
    grpo_group_size: int = 4
    # Optional fixed scheduler horizons for multi-call training.
    # These prevent repeated single-step calls from collapsing cosine/linear LR.
    sft_scheduler_total_steps: int = 0
    grpo_scheduler_total_steps: int = 0

    # Checkpoints
    saved_adapters: dict = field(default_factory=dict)  # Maps adapter_name -> path
    saved_metadata: dict = field(default_factory=dict)  # Maps checkpoint -> state

    # Robustness tracking
    consecutive_empty_batches: int = 0

    # Session-specific temp directory with restricted permissions
    session_dir: str = field(default_factory=lambda: _create_session_dir())

    def reset(self) -> None:
        """Reset all session state for a new training run.

        Call this after finish() or when starting a completely new session.
        Creates a fresh temp directory for the new session.

        Returns:
            None. Logs warnings if cleanup fails but does not raise.
        """
        # NOTE: We intentionally preserve the previous session directory on reset.
        # This maximizes observability and enables post-mortems after `finish()`.
        # Modal sandboxes are ephemeral; stale /tmp directories are cleaned up by
        # the container lifecycle.

        # Reset all state
        self.service_client = None
        self.training_client = None
        self.sampling_client = None
        self.tokenizer = None
        self.current_model = None
        self.renderer = None
        self.renderer_name = None
        self.image_processor = None
        self.wandb_run = None
        self.session_method = None
        self.sft_epochs = 0
        self.sft_steps = 0
        self.grpo_iterations = 0
        self.grpo_group_size = 4
        self.sft_scheduler_total_steps = 0
        self.grpo_scheduler_total_steps = 0
        self.saved_adapters = {}
        self.saved_metadata = {}
        self.consecutive_empty_batches = 0
        self.session_dir = _create_session_dir()

    def get_checkpoint_metadata(self, name: str) -> Optional[dict]:
        """Get saved metadata for a checkpoint.

        Args:
            name: Checkpoint name or path

        Returns:
            Metadata dict if found, None otherwise
        """
        return self.saved_metadata.get(name)

    def save_checkpoint_metadata(self, name: str) -> None:
        """Save current state as metadata for a checkpoint.

        Args:
            name: Checkpoint name to associate with current state

        Raises:
            ValueError: If checkpoint name is invalid (empty, contains path separators, or starts with '.')
        """
        # Validate checkpoint name to prevent path traversal
        if not name:
            raise ValueError("Checkpoint name cannot be empty")
        if "/" in name or "\\" in name:
            raise ValueError(f"Invalid checkpoint name: '{name}' (cannot contain path separators)")
        if name.startswith("."):
            raise ValueError(f"Invalid checkpoint name: '{name}' (cannot start with '.')")

        self.saved_metadata[name] = {
            "cumulative_sft_epochs": self.sft_epochs,
            "cumulative_grpo_iterations": self.grpo_iterations,
            "cumulative_sft_steps": self.sft_steps,
            "sft_scheduler_total_steps": self.sft_scheduler_total_steps,
            "grpo_scheduler_total_steps": self.grpo_scheduler_total_steps,
            "session_method": self.session_method,
        }

    def restore_from_metadata(self, metadata: dict) -> bool:
        """Restore state from checkpoint metadata with validation.

        Args:
            metadata: Dict with cumulative counters and session info

        Returns:
            True if metadata was valid and restored, False if metadata was invalid

        Note:
            Invalid metadata results in default values being used rather than
            raising an exception, to allow recovery from corrupted checkpoints.
        """
        # Validate metadata is a dict
        if not isinstance(metadata, dict):
            print(
                f"Warning: Invalid checkpoint metadata type: {type(metadata).__name__}. Using default values.",
                file=sys.stderr,
            )
            return False

        # Expected keys and their types
        expected_keys = {
            "cumulative_sft_epochs": int,
            "cumulative_grpo_iterations": int,
            "cumulative_sft_steps": int,
            "sft_scheduler_total_steps": int,
            "grpo_scheduler_total_steps": int,
            "session_method": (str, type(None)),
        }

        # Validate and restore each field
        validation_errors = []
        for key, expected_type in expected_keys.items():
            value = metadata.get(key)
            if value is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        validation_errors.append(f"{key}: expected {expected_type}, got {type(value).__name__}")
                elif not isinstance(value, expected_type):
                    validation_errors.append(f"{key}: expected {expected_type.__name__}, got {type(value).__name__}")

        if validation_errors:
            print(
                "Warning: Checkpoint metadata validation errors:\n  "
                + "\n  ".join(validation_errors)
                + "\n  Using default values for invalid fields.",
                file=sys.stderr,
            )

        # Restore values, using defaults for invalid/missing fields
        try:
            self.sft_epochs = int(metadata.get("cumulative_sft_epochs", 0))
        except (TypeError, ValueError):
            self.sft_epochs = 0

        try:
            self.grpo_iterations = int(metadata.get("cumulative_grpo_iterations", 0))
        except (TypeError, ValueError):
            self.grpo_iterations = 0

        try:
            self.sft_steps = int(metadata.get("cumulative_sft_steps", 0))
        except (TypeError, ValueError):
            self.sft_steps = 0

        try:
            self.sft_scheduler_total_steps = int(metadata.get("sft_scheduler_total_steps", 0))
        except (TypeError, ValueError):
            self.sft_scheduler_total_steps = 0

        try:
            self.grpo_scheduler_total_steps = int(metadata.get("grpo_scheduler_total_steps", 0))
        except (TypeError, ValueError):
            self.grpo_scheduler_total_steps = 0

        session_method = metadata.get("session_method")
        if session_method in ("sft", "grpo", None):
            self.session_method = session_method
        else:
            self.session_method = None

        return len(validation_errors) == 0


# Global singleton instance
# This is the documented intentional statefulness for training workflows
session = TrainingSession()

# Global lock for serializing stateful MCP tool calls.
# The harness uses a single mutable session, so concurrent writes must be
# gated to avoid state corruption.
_tool_call_lock = threading.Lock()


def get_session() -> TrainingSession:
    """Get the global training session instance.

    Returns:
        The singleton TrainingSession instance
    """
    return session


async def try_acquire_tool_call_lock(timeout_seconds: float) -> bool:
    """Attempt to acquire the global tool-call lock within timeout.

    Args:
        timeout_seconds: Max time to wait. Values < 0 wait indefinitely.

    Returns:
        True if lock acquired, False on timeout.
    """
    timeout = -1.0 if timeout_seconds < 0 else float(timeout_seconds)
    return await asyncio.to_thread(_tool_call_lock.acquire, True, timeout)


def release_tool_call_lock() -> None:
    """Release the global tool-call lock if held."""
    if _tool_call_lock.locked():
        _tool_call_lock.release()
