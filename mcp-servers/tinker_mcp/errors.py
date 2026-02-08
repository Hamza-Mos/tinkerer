"""Error types and structured responses for the Tinker MCP Server.

This module provides a hierarchy of exceptions that produce MCP-compliant
JSON error responses. The error types are designed to fail-fast by default
(raise exceptions, don't return 0.0) to surface issues early in training.

Error Codes:
- TIMEOUT: Operation exceeded time limit
- VALIDATION_FAILED: Input validation error
- API_ERROR: Tinker API communication error
- REWARD_COMPUTATION_FAILED: Reward function execution error
- SAMPLING_FAILED: Model sampling error
- MODEL_NOT_FOUND: Requested model not in registry
"""

import json
from enum import Enum
from typing import Optional


class ErrorCode(Enum):
    """Standard error codes for MCP tool responses."""

    TIMEOUT = "TIMEOUT"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    API_ERROR = "API_ERROR"
    REWARD_FAILED = "REWARD_COMPUTATION_FAILED"
    SAMPLING_FAILED = "SAMPLING_FAILED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"


class MCPToolError(Exception):
    """Base exception with structured MCP-compliant response.

    All training-related errors should inherit from this class to ensure
    consistent error formatting for Claude Code consumption.

    Attributes:
        code: ErrorCode enum value identifying the error type
        message: Human-readable error description
        details: Additional context dict (optional)
    """

    def __init__(self, code: ErrorCode, message: str, details: Optional[dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code.value}: {message}")

    def to_dict(self) -> dict:
        """Return structured error as dictionary."""
        return {"error": {"code": self.code.value, "message": self.message, "details": self.details}}

    def to_response(self) -> str:
        """Return JSON error response for MCP tool output."""
        return json.dumps(self.to_dict(), indent=2)


class RewardComputationError(MCPToolError):
    """Error during reward function execution.

    Raised when the user-provided reward function fails, times out, or produces
    invalid output. This is a critical error for GRPO training since rewards
    are essential for learning.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(ErrorCode.REWARD_FAILED, message, details)


class ModelNotFoundError(MCPToolError):
    """Error when requested model is not in the registry.

    Raised when a user specifies a model that isn't in MODEL_INFO. The error
    message should suggest valid alternatives.
    """

    def __init__(self, model_name: str, available_models: list):
        details = {
            "requested_model": model_name,
            # Truncate for readability
            "available_models": available_models[:10],
            "total_available": len(available_models),
        }
        super().__init__(ErrorCode.MODEL_NOT_FOUND, f"Model '{model_name}' not found in registry", details)


class TinkerTimeoutError(MCPToolError):
    """Error when an operation exceeds time limit.

    Raised when API calls, training steps, or other operations
    take longer than the configured timeout.

    Note: Named TinkerTimeoutError to avoid shadowing Python's built-in TimeoutError.
    """

    def __init__(self, operation: str, timeout_seconds: int, details: Optional[dict] = None):
        message = f"{operation} timed out after {timeout_seconds}s"
        error_details = {"operation": operation, "timeout_seconds": timeout_seconds}
        if details:
            error_details.update(details)
        super().__init__(ErrorCode.TIMEOUT, message, error_details)
