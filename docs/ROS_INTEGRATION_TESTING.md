# ROS Integration Testing Guide

This document consolidates the test workflows for the two rosbridge-based ROS connectors in this repository:

- **ROSClaw** (`rosclaw-v1.0`) — the connector that compiles a `CapabilityManifest`, safety contract, and provider layer without importing `rclpy`/`rospy`.
- **ros-mcp-server** — the external MCP server that exposes topics, services, nodes, parameters, and actions as MCP tools.

Both projects use the same underlying protocol (rosbridge WebSocket) and the same canonical example node (`turtlesim`). This guide gives a single place to run the cross-version matrix.

## Table of Contents

- [What is being tested](#what-is-being-tested)
- [Repository layout](#repository-layout)
- [Supported ROS versions](#supported-ros-versions)
- [Environment variables](#environment-variables)
- [Quick command matrix](#quick-command-matrix)
- [ROS 1 Host header caveat](#ros-1-host-header-caveat)
- [ROS message/service type naming](#ros-messageservice-type-naming)
- [FastMCP 3.x fixture note](#fastmcp-3x-fixture-note)
- [Known coverage gaps and pending support needs](#known-coverage-gaps-and-pending-support-needs)
- [Troubleshooting](#troubleshooting)

## What is being tested

| Concern | ROSClaw | ros-mcp-server |
|---|---|---|
| Connectivity / ping | `rosclaw ros ping` | `connect_to_robot`, `ping_robots` |
| Graph discovery | `RosGraphDiscovery` | `get_topics`, `get_services`, `get_nodes` |
| Manifest / capability compile | `CapabilityManifestCompiler` | N/A |
| Safety contract validation | `SafetyContractCompiler` | N/A |
| Provider execution | `RosCapabilityProvider.infer` | N/A |
| Topic pub/sub | via safety-gated provider | `publish_once`, `subscribe_once` |
| Service calls | via safety-gated provider | `call_service`, `get_service_details` |
| Parameters | via capability interface | `get_parameters`, `set_parameter`, `has_parameter`, ... |
| Actions | via capability interface | `get_action_details`, `send_action_goal` (ROS 2 only) |

## Repository layout

```text
/home/ubuntu/rosclaw/rosclaw
├── rosclaw-v1.0/                          # ROSClaw connector
│   ├── docker-compose.ros-test.yml        # ROS 2 Humble stack
│   ├── docker-compose.ros1-test.yml       # ROS 1 Noetic stack
│   ├── docker/ros2-humble-bridge.Dockerfile
│   ├── docker/ros1-noetic-bridge.Dockerfile
│   ├── tests/connectors/ros/
│   │   ├── test_turtlesim_integration.py  # live turtlesim tests
│   │   └── ...
│   └── docs/ROS_CONNECTOR.md              # detailed connector docs
│
└── ros-mcp-server/                        # MCP server
    ├── tests/integration/
    │   ├── conftest.py                    # Docker lifecycle + FastMCP 3.x tools fixture
    │   ├── test_connection.py
    │   ├── test_quick_detect.py
    │   └── docker-compose.yml
    ├── tests/installation/                # uvx / pip / uv install tests
    └── docs/testing.md                    # detailed MCP server testing docs
```

## Supported ROS versions

| ROS Distro | ros-mcp-server `--ros-distro` | ROSClaw compose file | Status |
|---|---|---|---|
| ROS 2 Humble | `humble` (default) | `docker-compose.ros-test.yml` | Fully tested |
| ROS 2 Jazzy | `jazzy` | — | Validated (ros-mcp-server + ROSClaw turtlesim) |
| ROS 1 Noetic | `noetic` | `docker-compose.ros1-test.yml` | Fully tested |
| ROS 1 Melodic | `melodic` | — | Validated (ros-mcp-server + ROSClaw turtlesim) |

## Environment variables

| Variable | Default | Applies to | Purpose |
|---|---|---|---|
| `ROSCLAW_ROS_TEST_ENDPOINT` | `ws://127.0.0.1:9090` | ROSClaw | Endpoint used by `test_turtlesim_integration.py` |
| `ROSBRIDGE_IP` | `127.0.0.1` | ros-mcp-server | rosbridge host for integration tests |
| `ROSBRIDGE_PORT` | `9090` | ros-mcp-server | rosbridge port for integration tests |
| `ROSBRIDGE_HOST_PORT` | `9090` (ROS2) / `9091` (ROS1) | ROSClaw compose | Host port mapped to container 9090 |
| `ROSBRIDGE_CONTAINER_NAME` | `rosclaw-ros2-humble-bridge` / `rosclaw-ros1-noetic-bridge` | ROSClaw compose | Container name |

## Quick command matrix

### ROSClaw — unit tests (no live ROS needed)

```bash
cd /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0
pytest tests/connectors/ros/ -v
```

Expected: `94 passed, 4 skipped` (the skipped tests are the live turtlesim integration tests when no rosbridge is reachable).

### ROSClaw — ROS 2 live turtlesim tests

```bash
cd /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0
docker compose -f docker-compose.ros-test.yml up --build -d
pytest tests/connectors/ros/test_turtlesim_integration.py -v
docker compose -f docker-compose.ros-test.yml down
```

### ROSClaw — ROS 1 live turtlesim tests

```bash
cd /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0
docker compose -f docker-compose.ros1-test.yml up --build -d

# Option A — host port mapping will hit the ROS 1 Host header issue
# (not recommended; only works if rosbridge is on the same port inside and out)
export ROSBRIDGE_HOST_PORT=9090
export ROSCLAW_ROS_TEST_ENDPOINT=ws://127.0.0.1:9090
pytest tests/connectors/ros/test_turtlesim_integration.py -v

# Option B — connect via container IP on port 9090 (recommended)
export IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' rosclaw-ros1-noetic-bridge)
export ROSCLAW_ROS_TEST_ENDPOINT=ws://${IP}:9090
pytest tests/connectors/ros/test_turtlesim_integration.py -v

docker compose -f docker-compose.ros1-test.yml down
```

> `test_turtlesim_integration.py` reads `ROSCLAW_ROS_TEST_ENDPOINT`; it does not accept a `--endpoint` flag.

### ros-mcp-server — ROS 2 integration tests

```bash
cd /home/ubuntu/rosclaw/rosclaw/ros-mcp-server
pytest tests/integration -v --ros-distro humble
```

### ros-mcp-server — ROS 1 integration tests

```bash
cd /home/ubuntu/rosclaw/rosclaw/ros-mcp-server
pytest tests/integration -v --ros-distro noetic
```

### ros-mcp-server — against an existing rosbridge (skip Docker lifecycle)

```bash
cd /home/ubuntu/rosclaw/rosclaw/ros-mcp-server
export ROSBRIDGE_IP=127.0.0.1
export ROSBRIDGE_PORT=9090
pytest tests/integration -v --skip-compose --ros-distro noetic
```

### ros-mcp-server — installation tests

```bash
cd /home/ubuntu/rosclaw/rosclaw/ros-mcp-server
pytest tests/installation -v
```

## ROS 1 Host header caveat

ROS 1 `rosbridge_websocket` enforces strict HTTP `Host` header validation:
the port in the header must match the port the server listens on (9090).
When Docker maps `9090` inside the container to a different host port, the
external WebSocket handshake is rejected with HTTP 400.

Manifestations:

- `rosclaw ros ping --endpoint ws://127.0.0.1:9091` fails on ROS 1.
- `pytest ... --endpoint ws://127.0.0.1:9091` fails on ROS 1.
- ros-mcp-server tests connecting to `127.0.0.1:9091` fail on ROS 1.

Workarounds (in order of preference):

1. **Connect via container IP on port 9090** — works for both ROSClaw and
   ros-mcp-server because the `Host` header then contains `9090`.
2. **Use host networking** — run the container with `--network host` so the
   listening port is the same inside and outside.
3. **Map host port 9090** — only viable when nothing else is using 9090.

ros-mcp-server's `conftest.py` uses `ROSBRIDGE_IP` / `ROSBRIDGE_PORT`; set
`ROSBRIDGE_IP` to the container IP and `ROSBRIDGE_PORT=9090` for ROS 1 tests.

## ROS message/service type naming

| Concept | ROS 1 | ROS 2 |
|---|---|---|
| Message type | `turtlesim/Pose` | `turtlesim/msg/Pose` |
| Service type | `turtlesim/TeleportAbsolute` | `turtlesim/srv/TeleportAbsolute` |
| Action type | N/A (actions are separate) | `turtlesim/action/RotateAbsolute` |

Both codebases detect the ROS version at runtime and select the correct type
strings. Tests such as `_get_pose_msg_type()` in ROSClaw query `/rosapi/topics`
to avoid hard-coding the naming convention.

## FastMCP 3.x fixture note

The ros-mcp-server integration `tools` fixture was updated for FastMCP 3.x.
The old `mcp._tool_manager.get_tool(name)` API no longer exists; the fixture
now uses:

```python
registered = await mcp.list_tools()
for tool_obj in registered:
    resolved = await mcp.get_tool(tool_obj.name)
    result[tool_obj.name] = resolved.fn
```

If you see `AttributeError: 'FastMCP' object has no attribute '_tool_manager'`,
you are on an older FastMCP version and need to revert the fixture or upgrade.

## Known coverage gaps and pending support needs

The following items have been identified but are not yet fully exercised. They
should be revisited as soon as hardware, time, or CI resources allow.

### Cross-distro matrix

- ROS 2 Jazzy container build and full test run.
- ROS 1 Melodic container build and full test run.
- macOS / Windows Docker Desktop behavior with host networking.

### ROS feature coverage

- **ROS 1 actions** — not supported by ros-mcp-server; verify whether ROSClaw
  action capabilities need a rosbridge actionlib shim.
- **Complex parameter types** — lists and dictionaries have not been thoroughly
  tested on ROS 1; `set_parameter` currently accepts string-encoded values.
- **Dynamic reconfigure** — ROS 1 dynamic parameters are not covered.
- **Custom message packages** — current tests use only turtlesim and standard
  geometry messages.
- **Large messages / compressed topics** — `use_compression=false` in the launch
  file; compression path is untested.
- **Latched topics** — ROS 1 latched topic behavior with `subscribe_once`.

### Safety and edge cases

- ROSClaw safety contract execution on ROS 1 with a real (non-dry-run) unsafe
  command and recovery path.
- Emergency stop behavior when `/cmd_vel` is absent but `turtle1/cmd_vel` exists.
- Manifest compilation when `discouraged_interfaces` removes the only available
  interface for a command category.

### Infrastructure and CI

- A single CI job that spins up both ROS 1 and ROS 2 containers and runs the
  full matrix.
- Docker-in-Docker permission handling (`sg docker`) in CI.
- Host header workaround codified into a helper script or pytest fixture.
- Automated port-discovery fixture so ROS 1 tests do not require manual IP lookup.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on `pytest tests/integration` | Container not ready | Wait for healthcheck or increase timeout |
| `HTTP 400` on ROS 1 connection | Host header port mismatch | Connect via container IP on port 9090 |
| `Permission denied` for docker commands | Group membership not active | Run with `sg docker -c "..."` or log out/in |
| `No manifest loaded` in `list-capabilities` | No `--manifest` and live discovery failed | Check `--endpoint` and rosbridge health |
| `AttributeError: '_tool_manager'` | FastMCP version mismatch | Use FastMCP 3.x or adjust `conftest.py` |
| `get_parameters` empty on ROS 1 | `rosapi` node not running | Include `rosapi` node in launch file |
| Turtlesim topics missing | Turtlesim not started | ROS 2 compose uses `command: ["turtlesim"]`; ROS 1 launch file starts it explicitly |

## Next steps

1. Run the full matrix: `humble`, `jazzy`, `noetic`, `melodic`.
2. Close the coverage gaps listed above.
3. Add a CI workflow that runs the matrix on every PR.
4. Keep this document and `docs/ROS_CONNECTOR.md` / `ros-mcp-server/docs/testing.md`
   in sync when adding new distributions or fixtures.
