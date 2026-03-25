# ROSClaw V4

Production-Ready Embodied Multi-Agent Operating System

## Overview

ROSClaw V4 is an embodied intelligence operating system that bridges AI agents with physical robots through the Model Context Protocol (MCP). It integrates:

- **RoboClaw**: Procedure-based robot control
- **OpenClaw-RL**: Asynchronous RL training framework
- **mjlab**: MuJoCo Warp physics simulation
- **VLA Models**: Vision-Language-Action policies (OpenVLA)

## Architecture

```
Layer 7: RosClaw-RL Training (OpenClaw-RL integration)
Layer 6: Cognitive & Planning (Confidence Router, Behavior Trees)
Layer 5: Skill & Policy (VLA, Skill Library)
Layer 4: Embodiment MCP (RoboClaw Procedure system)
Layer 3: Digital Twin (mjlab simulation)
Layer 2: Physical Data (Event-Driven Ring Buffer)
Layer 1: Real-Time Runtime (ROS 2)
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Start simulation environment
python -m rosclaw_v4.simulation.start

# Run single robot demo
python -m rosclaw_v4.demo.so101
```

## Project Structure

```
rosclaw_v4/
├── src/rosclaw_core/        # Core framework (RoboClaw-based)
├── src/rosclaw_mcp/         # MCP servers
├── src/rosclaw_vla/         # VLA policy service
├── src/rosclaw_sim/         # mjlab integration
├── src/rosclaw_rl/          # OpenClaw-RL integration
├── configs/                 # Robot and assembly configs
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## License

Apache 2.0
