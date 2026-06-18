# Third-Party Attributions

This project incorporates ideas and interface patterns from the following open-source projects while retaining independent implementations.

## robotmcp/ros-mcp-server

- **License:** Apache-2.0
- **Source:** https://github.com/robotmcp/ros-mcp-server
- **Referenced ideas:**
  - rosbridge WebSocket transport decoupling between MCP clients and ROS/ROS2 runtimes.
  - ROS graph discovery conventions (`/rosapi/*` services, topic/service/action enumeration).
  - ROS1/ROS2 version detection and `rosapi` prefix fallback patterns.
  - FastMCP tool annotations (`readOnlyHint`, `destructiveHint`) for ROS primitives.
  - Robot specification YAMLs (e.g., Unitree Go2) for preferred/discouraged interfaces.
  - turtlesim as a low-cost integration test target.

No source code from `ros-mcp-server` was copied verbatim. All implementations in `src/rosclaw/connectors/ros/` were written from scratch for the ROSClaw architecture.

## fishros/install

- **License:** BSD-3-Clause (per project repository)
- **Source:** https://github.com/fishros/install
- **Referenced idea:** one-command ROS installation automation referenced as a deployment option in container documentation.

The production container images in this repository use the upstream `osrf/ros:humble-desktop` base image and standard `apt` packages for reproducibility.
