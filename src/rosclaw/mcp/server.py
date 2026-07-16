"""P0 ROSClaw MCP server.

Exposes the P0 safety-gated tool surface:
  S0 read-only/body-context: robot state, skills, memory, practice, body context
  S1 simulation-only       : sandbox_run
  S2 validated-plan        : validate_trajectory
  S4 emergency             : emergency_stop

Supports stdio and streamable-http transports.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from rosclaw.agent.detectors import build_project_profile
from rosclaw.mcp.adapters.runtime_client import RuntimeClient
from rosclaw.mcp.tools import P0_TOOLS, set_client, set_context

logger = logging.getLogger("rosclaw.mcp.server")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _profile_name(profile: Any) -> str:
    """Derive a short runtime profile name for the response envelope."""
    if profile.profile_path is not None:
        return profile.profile_path.stem
    return "default"


def serve(
    *,
    transport: str,
    host: str,
    port: int,
    robot_id: str | None,
    profile: str | None,
    project_root: str | None,
    log_level: str,
    fixture_mode: bool = False,
) -> None:
    """Start the P0 MCP server."""
    _setup_logging(log_level)

    root = Path(project_root) if project_root else Path.cwd()
    project_profile = build_project_profile(project_root=root, profile=profile, robot=robot_id)

    client = RuntimeClient(
        project_root=root,
        robot_id=robot_id or project_profile.robot_id,
        runtime_profile=project_profile.runtime_profile,
        fixture_mode=fixture_mode,
    )
    set_client(client)
    set_context(
        project_root=str(root),
        runtime_profile=_profile_name(project_profile),
        agent_client=os.environ.get("ROSCLAW_AGENT_CLIENT", "claude-code"),
    )

    instructions = (
        "ROSClaw P0 physical-AI runtime. Read-only, simulation, validated-plan, "
        "and emergency tools only. No real-execution tool is available."
    )

    mcp = FastMCP(
        "rosclaw",
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
        "--profile",
        default=os.environ.get("ROSCLAW_PROFILE", "default"),
        help="ROSClaw runtime profile name",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("ROSCLAW_PROJECT_ROOT", "."),
        help="Project root path",
    )
    parser.add_argument(
        "--project",
        default=None,
        dest="project_root",
        help="Alias for --project-root",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("ROSCLAW_LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        default=os.environ.get("ROSCLAW_MCP_FIXTURE", "").lower() in {"1", "true", "yes", "on"},
        help="Explicitly enable synthetic fixture responses; never valid for real execution",
    )
    args = parser.parse_args(argv)

    try:
        serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            robot_id=args.robot_id,
            profile=args.profile,
            project_root=args.project_root,
            log_level=args.log_level,
            fixture_mode=args.fixture,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down on interrupt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
