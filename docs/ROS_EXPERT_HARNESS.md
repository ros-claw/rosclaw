# ROS Expert Harness

This extension lives inside the existing ROS Connector. It adds read-only system understanding, deterministic diagnostics, semantic readiness, technology resolution, TaskGraph proposals and independent coverage calculations. The isolated Nav2/Gazebo complete-cleaning loop has passed at 98.04% independent coverage with zero physics contacts and canonical SIM evidence; it does not establish hardware capability. See the [reproducible acceptance fixture](../integrations/ros_probe/acceptance/README.md).

## Read-only workflow

```bash
PYTHONPATH=src .venv/bin/python -m rosclaw.entrypoint ros inspect-system --robot-id my_robot --deep --json
PYTHONPATH=src .venv/bin/python -m rosclaw.entrypoint ros diagnose --robot-id my_robot --deep --profile navigation --json
PYTHONPATH=src .venv/bin/python -m rosclaw.entrypoint ros resolve --robot-id my_robot --deep --task '完成整个房间清扫' --json
PYTHONPATH=src .venv/bin/python -m rosclaw.entrypoint ros mission plan --robot-id my_robot --deep --task '清扫整个房间' --json
```

Deep inspection requires the native sidecar on the ROS host. Missing native observations are reported as errors/UNKNOWN. Use `--graph`, `--native` and `--body` for offline capture composition, or `--snapshot` for a sealed snapshot replay. `--output` writes an artifact; `ros context --output ROS_CONTEXT.md` writes a derived summary with capture time/hash/freshness. Historical snapshots do not become fresh when replayed.

## Native probe

On the ROS host, use its ROS-compatible Python:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 integrations/ros_probe/ros2/probe.py
```

The sidecar publishes only `/rosclaw_probe/snapshot` and `/rosclaw_probe/status`; `/rosclaw_probe/refresh` schedules read-only collection. Its RPC clients are restricted to lifecycle GetState and parameter ListParameters/GetParameters. It never publishes commands, sets parameters, sends action goals or invokes motion services. Core imports do not require ROS.

## MCP and context

Set `ROSCLAW_ROS_EXPERT=1` when starting the existing canonical MCP server to register exactly four additional read-only tools: `ros_inspect_system`, `ros_diagnose_system`, `ros_resolve_task`, `ros_verify_mission`. The old connector registration remains compatible; the canonical extension does not enable legacy actuation tools. Discovery/diagnostic results grant no physical authority.

The Python ContextCompiler accepts a validated model in `SourceBundle.extra['ros_system_model']`, checks its hash/body binding/freshness and augments L2/L3 using the existing source protocols. The context bundle schema remains unchanged. Opt-in Native Chat collects live observations through a bounded read-only worker and binds the configured Body hash. Raw snapshot timestamps remain in evidence while stable readiness facts participate in context leases. Physical capabilities use the existing approval UI and rosclawd; actual-model single-input dynamic-obstacle cleaning is accepted at 98.01% coverage, zero contacts, verified Memory persistence and TaskKernel SUCCEEDED.

## Evidence

See [implementation report](reports/ros-expert-harness/FINAL_IMPLEMENTATION_REPORT.md). Native DDS probe acceptance and fixture replay are distinct from Nav2/Gazebo task acceptance. A 100% synthetic coverage mask is not evidence that a robot cleaned a room.
