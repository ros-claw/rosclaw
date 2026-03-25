# ROSClaw-OpenClaw Integration: Phase 1

This document describes how ROSClaw (production-ready embodied MCP layer) integrates with OpenClaw's Agent Event Loop via the mcporter bridge.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OpenClaw Agent Layer (TypeScript/Node.js)            │
│                        - Agent Event Loop (src/agents/acp-spawn.ts)         │
│                        - LLM Integration                                    │
│                        - ACP Protocol                                       │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    mcporter Bridge                                     │ │
│  │  (OpenClaw's MCP Integration Bridge)                                   │ │
│  │  - Spawns external MCP servers as subprocesses                         │ │
│  │  - Manages connection lifecycle                                        │ │
│  │  - Routes tool calls to appropriate MCP servers                        │ │
│  └─────────────────────────────┬─────────────────────────────────────────┘ │
│                                │                                            │
│                                │ stdio transport                            │
│                                ▼                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ stdio communication
                                        │ (JSON-RPC 2.0 / MCP protocol)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ROSClaw MCP Server (Python)                          │
│                        - UR5MCPServer (Layers 1-4)                          │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    Layer 4: Embodiment MCP                             │ │
│  │  - MCP Tools: move_robot, get_robot_state, execute_trajectory          │ │
│  │  - Digital Twin validation before execution                            │ │
│  └─────────────────────────────┬─────────────────────────────────────────┘ │
│                                │                                            │
│  ┌─────────────────────────────▼─────────────────────────────────────────┐ │
│  │                    Layer 3: Digital Twin (MuJoCo)                      │ │
│  │  - Physics validation of trajectories                                  │ │
│  │  - Collision detection                                                 │ │
│  │  - SafetyViolationError on validation failure                          │ │
│  └─────────────────────────────┬─────────────────────────────────────────┘ │
│                                │                                            │
│  ┌─────────────────────────────▼─────────────────────────────────────────┐ │
│  │                    Layer 2: Semantic-HAL                               │ │
│  │  - Fast Lane: ROS 2 topics (30Hz)                                      │ │
│  │  - Slow Lane: MCP tools (1Hz)                                          │ │
│  └─────────────────────────────┬─────────────────────────────────────────┘ │
│                                │                                            │
│  ┌─────────────────────────────▼─────────────────────────────────────────┐ │
│  │                    Layer 1: ROS 2 Runtime                              │ │
│  │  - UR5ROSNode (actual rclpy implementation)                            │ │
│  │  - Real Subscribers, Publishers, ActionClients                         │ │
│  └─────────────────────────────┬─────────────────────────────────────────┘ │
│                                │                                            │
│                                │ ROS 2 DDS (FastDDS/CycloneDDS)             │
│                                ▼                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Ethernet/IP
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Physical Robot (UR5)                                 │
│                        - RTDE/URScript                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Integration Flow

### 1. OpenClaw Agent Event Loop

The Agent Event Loop in OpenClaw (`src/agents/acp-spawn.ts`) manages the lifecycle of agent subprocesses. When an agent needs robot control capabilities:

```typescript
// OpenClaw agent configuration
const agentConfig = {
  model: "claude-sonnet-4-1-20250514",
  servers: [
    {
      name: "rosclaw-ur5",
      command: "rosclaw-ur5-mcp",  // Entry point from pyproject.toml
      env: {
        "ROBOT_IP": "192.168.1.100",
        "DIGITAL_TWIN_ENABLED": "true"
      }
    }
  ]
};
```

### 2. mcporter Bridge

mcporter spawns the ROSClaw MCP server as a subprocess:

```typescript
// mcporter creates subprocess
const process = spawn('rosclaw-ur5-mcp', [], {
  env: { ...process.env, ROBOT_IP: '192.168.1.100' }
});

// Communication via stdio (JSON-RPC 2.0)
process.stdout.on('data', (data) => {
  // Parse MCP protocol messages
  // Route to Agent Event Loop
});
```

### 3. ROSClaw MCP Server Initialization

When spawned, `rosclaw-ur5-mcp` (defined in `pyproject.toml`):

```python
# src/rosclaw/mcp/ur5_server.py
async def main():
    # 1. Initialize ROS 2 node
    rclpy.init()
    ros_node = UR5ROSNode(
        robot_ip=os.environ.get("ROBOT_IP", "192.168.1.100"),
        use_digital_twin=os.environ.get("DIGITAL_TWIN_ENABLED", "true") == "true"
    )

    # 2. Start ROS 2 node in separate thread
    ros_thread = threading.Thread(target=ros_node.run)
    ros_thread.start()

    # 3. Initialize MCP server
    mcp = UR5MCPServer(ros_node)

    # 4. Start MCP stdio transport (blocks here)
    await mcp.run()
```

### 4. Tool Call Flow

When OpenClaw agent calls a tool:

```
OpenClaw Agent
     │
     │ 1. LLM decides to move robot
     │    Tool Call: move_robot(joint_positions=[...])
     ▼
mcporter Bridge
     │
     │ 2. Serialize to MCP protocol
     │    JSON-RPC: {"method": "tools/call", ...}
     ▼
ROSClaw UR5MCPServer
     │
     │ 3. Validate via Digital Twin
     │    - Load MuJoCo model
     │    - Simulate trajectory
     │    - Check collisions/limits
     ▼
     │ 4a. Validation Failed?
     │    → Return SafetyViolationError
     ▼
     │ 4b. Validation Passed
     │    → Send ROS 2 Action Goal
     ▼
UR5ROSNode
     │
     │ 5. FollowJointTrajectory Action
     │    - Send goal to /follow_joint_trajectory
     ▼
ROS 2 DDS
     │
     │ 6. Network transport
     ▼
UR5 Controller
     │
     │ 7. Execute motion
     │    - RTDE feedback
     ▼
ROS 2 DDS (feedback)
     │
     │ 8. Action result
     ▼
UR5ROSNode
     │
     │ 9. Return result
     ▼
ROSClaw UR5MCPServer
     │
     │ 10. Format MCP response
     ▼
mcporter
     │
     │ 11. Deserialize
     ▼
OpenClaw Agent
     │ 12. LLM observes result
```

## Safety Architecture

### Digital Twin Firewall

Before any physical motion, the Digital Twin validates:

```python
from rosclaw.firewall import DigitalTwinFirewall, SafetyViolationError

# Initialize firewall with MuJoCo model
firewall = DigitalTwinFirewall(
    model_path="src/rosclaw/specs/ur5e.xml",
    check_collision=True,
    check_joint_limits=True,
    check_torque_limits=True
)

# Validate trajectory before execution
trajectory = [[0.0, -1.57, 1.57, 0.0, 0.0, 0.0], ...]
result = firewall.validate_trajectory(
    trajectory=trajectory,
    time_step=0.001,
    max_sim_time=5.0
)

if not result.is_valid:
    raise SafetyViolationError(
        f"Trajectory unsafe: {result.violations}"
    )
```

### Decorator Pattern

Functions can be wrapped for automatic validation:

```python
from rosclaw.firewall import mujoco_firewall

@mujoco_firewall(
    model_path="src/rosclaw/specs/ur5e.xml",
    safety_level=SafetyLevel.STRICT
)
def execute_move(trajectory_points: list[list[float]]):
    """Execute motion - automatically validated by Digital Twin."""
    # Only runs if validation passes
    ...
```

## MCP Tools Reference

### Tool: `move_robot`

Move the robot to a specific joint configuration.

**Input:**
```json
{
  "joint_positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
  "time_from_start": 5.0,
  "validate": true
}
```

**Output:**
```json
{
  "success": true,
  "message": "Motion completed successfully",
  "actual_positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
  "execution_time": 4.98
}
```

### Tool: `execute_trajectory`

Execute a multi-point trajectory.

**Input:**
```json
{
  "trajectory_points": [
    {"positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0], "time": 0.0},
    {"positions": [0.5, -1.0, 1.0, 0.5, 0.0, 0.0], "time": 2.0},
    {"positions": [1.0, -0.5, 0.5, 1.0, 0.0, 0.0], "time": 4.0}
  ],
  "validate": true
}
```

### Tool: `get_robot_state`

Get current robot state.

**Output:**
```json
{
  "joint_positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
  "joint_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "tcp_pose": {
    "position": [0.4, 0.2, 0.5],
    "orientation": [0.0, 0.0, 0.0, 1.0]
  },
  "is_moving": false,
  "digital_twin_status": "active"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOT_IP` | `192.168.1.100` | IP address of UR5 robot |
| `ROBOT_PORT` | `50002` | RTDE port |
| `DIGITAL_TWIN_ENABLED` | `true` | Enable MuJoCo validation |
| `MUJOCO_MODEL_PATH` | `src/rosclaw/specs/ur5e.xml` | Path to MJCF model |
| `SAFETY_LEVEL` | `strict` | Safety validation level |

### OpenClaw Configuration

Add to OpenClaw agent config:

```json
{
  "mcpServers": {
    "rosclaw-ur5": {
      "command": "rosclaw-ur5-mcp",
      "env": {
        "ROBOT_IP": "192.168.1.100",
        "DIGITAL_TWIN_ENABLED": "true"
      }
    }
  }
}
```

## Development

### Setup

```bash
# Install ROSClaw
cd /root/workspace/rosclaw/rosclaw
pip install -e ".[ros2,dev]"

# Test Digital Twin
python -c "
from rosclaw.firewall import DigitalTwinFirewall
fw = DigitalTwinFirewall('src/rosclaw/specs/ur5e.xml')
print('Digital Twin loaded successfully')
"

# Test MCP Server (stdio mode)
rosclaw-ur5-mcp

# Test MCP Server (HTTP mode for debugging)
rosclaw-ur5-mcp --transport http --port 9000
```

### Running with OpenClaw

```bash
# 1. Start OpenClaw with mcporter pointing to ROSClaw
# In OpenClaw config, add rosclaw-ur5-mcp as MCP server

# 2. Agent Event Loop automatically spawns and manages ROSClaw
# Agent can now call move_robot, execute_trajectory, etc.
```

## Next Steps: Phase 2

1. **Multi-Agent Support**: Extend to support G1 + UR5 coordination
2. **Event-Driven Ring Buffer**: Implement 60s ring buffer with event triggers
3. **OpenClaw-RL Integration**: Connect to `slime` for RL training
4. **VLA Policy Server**: Add OpenVLA/π0 integration

## References

- **OpenClaw**: `/root/workspace/rosclaw/openclaw/` - TypeScript agent framework
- **mcporter**: OpenClaw's MCP bridge (spawns external MCP servers)
- **MuJoCo**: Physics engine for Digital Twin
- **ROS 2**: Robot Operating System for hardware control
- **MCP Protocol**: Model Context Protocol specification
