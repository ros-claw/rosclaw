"""P0 ROSClaw MCP server.

Exposes seven safety-gated tools:
  S0 read-only      : get_robot_state, list_skills, query_memory, practice_query
  S1 simulation-only: sandbox_run
  S2 validated-plan : validate_trajectory
  S4 emergency      : emergency_stop

Supports stdio and streamable-http transports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from rosclaw.agent.detectors import build_project_profile
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.schemas.common import make_response
from rosclaw.mcp.tools import P0_TOOLS, set_client

logger = logging.getLogger("rosclaw.mcp.server")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _redact_for_audit(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of arguments safe for the audit log."""
    sensitive = {"token", "password", "secret", "api_key", "apikey", "auth"}
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(key, str) and key.lower() in sensitive:
            out[key] = "<REDACTED>"
        else:
            out[key] = value
    return out


def _audit(
    trace_id: str,
    tool: str,
    arguments: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """Append one JSON line to ~/.rosclaw/logs/mcp/audit.jsonl."""
    log_dir = Path.home() / ".rosclaw" / "logs" / "mcp"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "trace_id": trace_id,
                "timestamp": make_response({})["timestamp"],
                "tool": tool,
                "arguments": _redact_for_audit(arguments),
                "ok": response.get("ok", False),
            },
            ensure_ascii=False,
            default=str,
        )
        with (log_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.debug("audit logging failed: %s", exc)


def serve(
    *,
    transport: str,
    host: str,
    port: int,
    robot_id: str | None,
    project_root: str | None,
    log_level: str,
) -> None:
    """Start the P0 MCP server."""
    _setup_logging(log_level)

    root = Path(project_root) if project_root else Path.cwd()
    profile = build_project_profile(project_root=root, robot=robot_id)

    client = RuntimeClient(
        project_root=root,
        robot_id=robot_id or profile.robot_id,
        runtime_profile=profile.runtime_profile,
    )
    set_client(client)

    instructions = (
        "ROSClaw P0 physical-AI runtime. Read-only, simulation, validated-plan, "
        "and emergency tools only. No real-execution tool is available."
    )

    mcp = FastMCP(
        "rosclaw-p0",
        instructions=instructions,
        host=host,
        port=port,
        log_level=log_level.upper(),  # type: ignore[arg-type]
    )

    for tool_func in P0_TOOLS:
        mcp.add_tool(tool_func)

    logger.info("Starting ROSClaw P0 MCP server (%s:%s via %s)", host, port, transport)
    # FastMCP.run accepts "stdio", "sse", or "streamable-http"; map the CLI
    # convenience name "http" to the official streamable HTTP transport.
    mcp_transport = {"http": "streamable-http"}.get(transport, transport)
    mcp.run(transport=mcp_transport)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosclaw-mcp-serve",
        description="ROSClaw P0 MCP server",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default=os.environ.get("ROSCLAW_MCP_TRANSPORT", "stdio"),
        help="MCP transport",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("ROSCLAW_MCP_HOST", "127.0.0.1"),
        help="HTTP/SSE host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ROSCLAW_MCP_PORT", "9090")),
        help="HTTP/SSE port",
    )
    parser.add_argument(
        "--robot-id",
        default=os.environ.get("ROSCLAW_ROBOT_ID"),
        help="Robot identifier",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("ROSCLAW_PROJECT_ROOT", "."),
        help="Project root path",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("ROSCLAW_LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    args = parser.parse_args(argv)

    try:
        serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            robot_id=args.robot_id,
            project_root=args.project_root,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down on interrupt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
