"""Entry point for: python -m tinker_mcp"""

from tinker_mcp.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
