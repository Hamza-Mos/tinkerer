"""Tinker MCP Server - Fine-tune LLMs with Tinker's distributed training API."""

# Keep package import side-effect free.
#
# Importing `tinker_mcp.server` performs environment validation and initializes
# an MCP server, which is appropriate for `python -m tinker_mcp` but surprising
# for library-style imports (e.g., importing `tinker_mcp.models`).

__all__ = ["mcp"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "mcp":
        from tinker_mcp.server import mcp as _mcp

        return _mcp
    raise AttributeError(name)
