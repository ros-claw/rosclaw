"""FT Sensor MCP Server - Force/Torque sensor control via MCP.

This module provides the FastMCP server instance for force/torque sensor access.
"""

import argparse
import sys

from fastmcp import FastMCP

from rosclaw_mcp.servers.ft_sensor.config import FTSensorConfig
from rosclaw_mcp.servers.ft_sensor.tools import register_ft_tools


def create_ft_sensor_server(
    sensor_type: str = "ati",
    connection_type: str = "ros2",
    ros_topic: str = "/ft_sensor/data",
    ip_address: str = "192.168.1.1",
) -> FastMCP:
    """Create and configure an FT Sensor MCP server.

    Args:
        sensor_type: Sensor type (ati, robotiq, wacoh, mock)
        connection_type: Connection type (ros2, ethercat, tcp, mock)
        ros_topic: ROS2 topic for sensor data
        ip_address: IP address for TCP-connected sensors

    Returns:
        Configured FastMCP server instance
    """
    config = FTSensorConfig(
        sensor_type=sensor_type,
        connection_type=connection_type,
        ros_topic=ros_topic,
        ip_address=ip_address,
    )

    mcp = FastMCP("ft-sensor-server")
    register_ft_tools(mcp, config)

    return mcp


def main():
    """Main entry point for the FT Sensor MCP server."""
    parser = argparse.ArgumentParser(
        description="FT Sensor MCP Server - Force/Torque sensor access via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rosclaw_mcp.servers.ft_sensor.server
  python -m rosclaw_mcp.servers.ft_sensor.server --sensor-type ati --topic /ft_sensor/data
  python -m rosclaw_mcp.servers.ft_sensor.server --connection-type tcp --ip 192.168.1.50
        """,
    )

    parser.add_argument(
        "--sensor-type",
        default="ati",
        choices=["ati", "robotiq", "wacoh", "optoforce", "mock"],
        help="FT sensor type (default: ati)",
    )

    parser.add_argument(
        "--connection-type",
        default="ros2",
        choices=["ros2", "ethercat", "tcp", "mock"],
        help="Connection type (default: ros2)",
    )

    parser.add_argument(
        "--topic",
        default="/ft_sensor/data",
        help="ROS2 topic for sensor data (default: /ft_sensor/data)",
    )

    parser.add_argument(
        "--ip",
        default="192.168.1.1",
        help="Sensor IP address for TCP connection (default: 192.168.1.1)",
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
        default=9003,
        help="Port for HTTP transports (default: 9003)",
    )

    args = parser.parse_args()

    # Create server
    mcp = create_ft_sensor_server(
        sensor_type=args.sensor_type,
        connection_type=args.connection_type,
        ros_topic=args.topic,
        ip_address=args.ip,
    )

    # Run with appropriate transport
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport in {"http", "streamable-http"}:
        print(
            f"FT Sensor Server: {args.transport} -> http://{args.host}:{args.port}",
            file=sys.stderr,
        )
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
