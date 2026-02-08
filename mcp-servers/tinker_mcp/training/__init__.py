"""Training implementations for the Tinker MCP Server.

This package contains the core training logic for:
- SFT (Supervised Fine-Tuning): Learning from examples
- GRPO (Group Relative Policy Optimization): Reward-based learning

The implementations are separated from the MCP tool definitions in server.py
to keep the codebase maintainable. The server.py file retains the full
docstrings which serve as the "interface" for agent runtimes (Codex, Claude
Code) and humans.
"""

from tinker_mcp.training.sft import init_sft_session, train_sft_step_impl
from tinker_mcp.training.grpo import init_grpo_session, train_grpo_step_impl

__all__ = [
    "init_sft_session",
    "train_sft_step_impl",
    "init_grpo_session",
    "train_grpo_step_impl",
]
