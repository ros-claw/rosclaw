"""ROSClaw built-in MCP server for Intel RealSense D405."""
from __future__ import annotations

from rosclaw.mcp.servers import main, run_tool

if __name__ == "__main__":
    main("realsense_d405")
