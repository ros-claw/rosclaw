# ROSClaw ROS Connector

The ROS connector lets ROSClaw control and observe ROS 1 / ROS 2 robots
through [rosbridge](https://github.com/RobotWebTools/rosbridge_suite)
without importing `rclpy` or `rospy`. This means ROSClaw can be installed
and run on machines that do not have ROS installed.

## Quick start

1. Start a rosbridge server on the robot / ROS host:

   ```bash
   # ROS 2
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml

   # ROS 1
   roslaunch rosbridge_server rosbridge_websocket.launch
   ```

2. Use the ROSClaw CLI:

   ```bash
   rosclaw ros ping
   rosclaw ros discover --robot-id turtlesim
   rosclaw ros compile --robot-id turtlesim --output turtlesim_manifest.json
   rosclaw ros list-capabilities --robot-id turtlesim
   rosclaw ros validate-capability turtlesim.base.velocity_command --args '{"linear":{"x":0.1},"duration":0.5}'
   ```

   Override the default endpoint (`ws://127.0.0.1:9090`) with `--endpoint`:

   ```bash
   rosclaw ros ping --endpoint ws://172.21.0.2:9090
   rosclaw doctor --ros --endpoint ws://172.21.0.2:9090
   ```

## Docker test stack

Pre-built Docker Compose stacks are provided for integration testing with
turtlesim on both ROS 1 and ROS 2.

### ROS 2 (default)

```bash
docker compose -f docker-compose.ros-test.yml up --build -d
pytest tests/connectors/ros/test_turtlesim_integration.py -v
docker compose -f docker-compose.ros-test.yml down
```

The container exposes rosbridge on `ws://127.0.0.1:9090`.

### ROS 1

```bash
docker compose -f docker-compose.ros1-test.yml up --build -d
pytest tests/connectors/ros/test_turtlesim_integration.py -v --endpoint ws://127.0.0.1:9091
docker compose -f docker-compose.ros1-test.yml down
```

The ROS 1 compose file maps the container port `9090` to host port `9091`
by default so it can run alongside the ROS 2 stack.

### ROS 1 Host header caveat

ROS 1 `rosbridge_websocket` validates that the port in the HTTP `Host`
header matches its internal listening port (`9090`). When the container
port is mapped to a different host port (e.g. `9091`), external clients
connecting to `ws://127.0.0.1:9091` are rejected with HTTP 400.

Recommended workarounds:

- Connect from inside the Docker network using the container IP and port
  `9090`:

  ```bash
  docker network inspect rosclaw-rosclaw-ros-test
  pytest tests/connectors/ros/test_turtlesim_integration.py -v \
      --endpoint ws://<container-ip>:9090
  ```

- Use host networking for the rosbridge service.
- Map the host port to `9090:9090` if no other stack is using it.

### Using a custom endpoint in tests

The integration test file reads `ROSCLAW_ROS_TEST_ENDPOINT`:

```bash
export ROSCLAW_ROS_TEST_ENDPOINT=ws://172.21.0.2:9090
pytest tests/connectors/ros/test_turtlesim_integration.py -v
```

## Architecture

```text
┌─────────────┐      WebSocket       ┌─────────────────┐
│   ROSClaw   │  ──────────────────▶ │    rosbridge    │
│  connector  │  (rosbridge protocol)│    server       │
└─────────────┘                      └────────┬────────┘
                                              │
                                       ROS graph / topics
```

Key layers:

- **transport** – WebSocket transport, no ROS Python imports.
- **discovery** – Queries `/rosapi/*` services to build a graph snapshot.
- **compiler** – Converts the graph snapshot into a `CapabilityManifest` with
  safety metadata.
- **safety contract** – Decides `ALLOW` / `MODIFY` / `BLOCK` for each request.
- **provider** – Implements the ROSClaw `Provider` ABC and routes execution
  through the safety contract.
- **MCP tools** – Exposes safe tools only (`ros_ping`, `ros_discover`,
  `ros_execute_capability`, ...); raw `publish_once` / `call_any_service` are
  intentionally not exposed.
- **CLI** – Human-facing commands under `rosclaw ros ...`.

## Safety model

- Read-only topics are low risk.
- Command topics such as `/cmd_vel` are high risk and require a duration,
  velocity bounds, and a stop guard.
- Services and actions are medium/high risk depending on the interface.
- Forbidden patterns (e.g. `torque_control`, `disable_safety`) are blocked.

Robot embodiment cards in `src/rosclaw/connectors/ros/specs/` encode
preferred interfaces, discouraged interfaces, and safety defaults for
specific robots.

## CLI reference

| Command | Purpose |
|---|---|
| `rosclaw ros ping` | Check rosbridge connectivity |
| `rosclaw ros discover` | Discover ROS graph |
| `rosclaw ros compile` | Compile `CapabilityManifest` |
| `rosclaw ros list-capabilities` | List compiled capabilities |
| `rosclaw ros inspect-capability ID` | Show capability schema/risk |
| `rosclaw ros validate-capability ID --args JSON` | Validate args dry-run |
| `rosclaw ros execute-capability ID --args JSON` | Execute through provider |
| `rosclaw ros emergency-stop` | Request E-Stop through `rosclawd` and report evidence |
| `rosclaw doctor --ros --endpoint URL` | Health check without `rclpy` |

All subcommands that talk to rosbridge accept `--endpoint`.

## Integration tests

- `tests/connectors/ros/test_no_ros_import_in_core.py` – Verifies no ROS
  Python imports in the connector core.
- `tests/connectors/ros/test_turtlesim_integration.py` – Live tests against
  turtlesim + rosbridge; skipped unless a server is reachable.

Run the live tests against a specific endpoint via the environment variable:

```bash
export ROSCLAW_ROS_TEST_ENDPOINT=ws://172.21.0.2:9090
pytest tests/connectors/ros/test_turtlesim_integration.py -v
```

The integration tests do not accept `--endpoint`; they read `ROSCLAW_ROS_TEST_ENDPOINT`.

## Example requests

See `examples/ros/`:

- `turtlesim_move_safe.json` – Within velocity/duration limits.
- `turtlesim_move_too_fast.json` – Should be blocked by the safety contract.

## Practice / Memory / KNOW / HOW / Auto integration

The ROS connector is a first-class citizen of the ROSClaw grounding plane:

- **Practice** – `RosCapabilityProvider` emits `rosclaw.practice.event.created`
  and `rosclaw.sandbox.episode.failed` events. `RosPracticeAdapter` converts
  these into `praxis.recorded` so `EpisodeRecorder` can build artifacts.
- **Memory** – The provider also publishes `rosclaw.runtime.execution.completed`
  / `.failed`, which `MemoryInterface` ingests as experiences.
- **KNOW** – Discovered capabilities can be seeded into the knowledge graph via
  `rosclaw.connectors.ros.know.ros_knowledge_seed`.
- **HOW** – ROS-specific recovery rules (rosbridge disconnects, topic/service
  not found, velocity blocks) are seeded into `heuristic_rules` via
  `rosclaw.connectors.ros.how.ros_recovery_rules`.
- **Auto** – Practice events feed the self-evolution control plane the same way
  other skills do.

To enable the integration when constructing a provider, pass `event_bus`,
`knowledge_interface`, and/or `seekdb_client` in the manifest `extra` dict.
