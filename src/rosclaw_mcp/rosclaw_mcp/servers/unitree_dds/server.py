"""Unitree DDS MCP Server - Control Unitree humanoid robots via DDS.

This module provides the FastMCP server instance for Unitree G1/Go2 robots.
"""

import argparse
import sys
from typing import Optional

from fastmcp import FastMCP

from rosclaw_mcp.servers.unitree_dds.config import UnitreeDDSConfig
from rosclaw_mcp.servers.unitree_dds.tools import register_unitree_tools


def create_unitree_dds_server(
    robot_ip: str = "192.168.123.161",
    robot_model: str = "g1",
    interface: str = "eth0",
) -> FastMCP:
    """Create and configure a Unitree DDS MCP server.

    Args:
        robot_ip: IP address of the Unitree robot
        robot_model: Robot model (g1, go2, h1)
        interface: Network interface for DDS communication

    Returns:
        Configured FastMCP server instance
    """
    config = UnitreeDDSConfig(
        robot_ip=robot_ip,
        robot_model=robot_model,
        interface=interface,
    )

    mcp = FastMCP("unitree-dds-server")
    register_unitree_tools(mcp, config)

    return mcp


def main():
    """Main entry point for the Unitree DDS MCP server."""
    parser = argparse.ArgumentParser(
        description="Unitree DDS MCP Server - Control Unitree robots via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rosclaw_mcp.servers.unitree_dds.server
  python -m rosclaw_mcp.servers.unitree_dds.server --robot-ip 192.168.123.162 --model go2
  python -m rosclaw_mcp.servers.unitree_dds.server --interface wlan0
        """,
    )

    parser.add_argument(
        "--robot-ip",
        default="192.168.123.161",
        help="Unitree robot IP address (default: 192.168.123.161)",
    )

    parser.add_argument(
        "--model",
        default="g1",
        choices=["g1", "go2", "h1", "b2"],
        help="Unitree robot model (default: g1)",
    )

    parser.add_argument(
        "--interface",
        default="eth0",
        help="Network interface for DDS (default: eth0)",
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
        default=9001,
        help="Port for HTTP transports (default: 9001)",
    )

    args = parser.parse_args()

    # Create server
    mcp = create_unitree_dds_server(
        robot_ip=args.robot_ip,
        robot_model=args.model,
        interface=args.interface,
    )

    # Run with appropriate transport
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport in {"http", "streamable-http"}:
        print(
            f"Unitree DDS Server: {args.transport} -> http://{args.host}:{args.port}",
            file=sys.stderr,
        )
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
