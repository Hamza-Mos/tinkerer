from __future__ import annotations

import json
from pathlib import Path


def test_mcp_json_does_not_use_unresolved_env_placeholders():
    """Keep `.mcp.json` robust across agent CLIs by avoiding `${VAR}` literals."""
    root = Path(__file__).resolve().parents[1]
    text = (root / ".mcp.json").read_text(encoding="utf-8")
    assert "${" not in text

    config = json.loads(text)
    assert "mcpServers" in config
    for server in config["mcpServers"].values():
        assert "env" not in server
