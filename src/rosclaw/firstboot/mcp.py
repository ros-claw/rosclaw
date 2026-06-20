"""MCP configuration generation for ROSClaw First Boot."""

from __future__ import annotations

import json
from pathlib import Path

from .workspace import backup_file


def generate_mcp_config(home: Path) -> Path:
    """Generate the MCP client config sample at ~/.rosclaw/config/mcp.json."""
    path = home / "config" / "mcp.json"
    backup_file(path)
    config = {
        "mcpServers": {
            "rosclaw": {
                "command": "rosclaw-mcp",
                "args": [],
                "env": {
                    "ROSCLAW_HOME": str(home),
                    "ROSCLAW_CONFIG": str(home / "config" / "rosclaw.yaml"),
                },
            }
        }
    }
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
