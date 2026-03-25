"""UR RTDE MCP Server - Universal Robots control via RTDE protocol.

This module provides the FastMCP server instance for UR robot control.
"""

import argparse
import sys
from typing import Optional

from fastmcp import FastMCP

from rosclaw_mcp.servers.ur_rtde.config import URRTDEConfig
from rosclaw_mcp.servers.ur_rtde.tools import register_ur_tools


def create_ur_rtde_server(
    robot_ip: str = "192.168.1.100",
    robot_model: str = "ur5e",
    safety_limits: Optional[dict] = None,
) -> FastMCP:
    """Create and configure a UR RTDE MCP server.

    Args:
        robot_ip: IP address of the UR robot controller
        robot_model: Robot model (ur5, ur5e, ur10, ur10e, ur16e, ur20, ur30)
        safety_limits: Optional custom safety limits override

    Returns:
        Configured FastMCP server instance
    """
    config = URRTDEConfig(
        robot_ip=robot_ip,
        robot_model=robot_model,
        safety_limits=safety_limits or {},
    )

    mcp = FastMCP("ur-rtde-server")
    register_ur_tools(mcp, config)

    return mcp


def main():
    """Main entry point for the UR RTDE MCP server."""
    parser = argparse.ArgumentParser(
        description="UR RTDE MCP Server - Control Universal Robots via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rosclaw_mcp.servers.ur_rtde.server
  python -m rosclaw_mcp.servers.ur_rtde.server --robot-ip 192.168.1.50 --model ur10e
  python -m rosclaw_mcp.servers.ur_rtde.server --transport http --port 9000
        """,
    )

    parser.add_argument(
        "--robot-ip",
        default="192.168.1.100",
        help="UR robot controller IP address (default: 192.168.1.100)",
    )

    parser.add_argument(
        "--model",
        default="ur5e",
        choices=["ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20", "ur30"],
        help="UR robot model (default: ur5e)",
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for HTTP transports (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port for HTTP transports (default: 9000)",
    )

    args = parser.parse_args()

    # Create server
    mcp = create_ur_rtde_server(
        robot_ip=args.robot_ip,
        robot_model=args.model,
    )

    # Run with appropriate transport
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport in {"http", "streamable-http"}:
        print(
            f"UR RTDE Server: {args.transport} -> http://{args.host}:{args.port}",
            file=sys.stderr,
        )
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
