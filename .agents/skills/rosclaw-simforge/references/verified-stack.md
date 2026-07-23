# Verified stack and evidence rules

## Historical operator run (2026-07-22)

This section records an operator-reported environment. Its raw logs are kept in
the external evidence store, not in Git. Re-run the bundled verification scripts
before making a claim for the current checkout.

- ROSClaw checkout used by that run: `1ce343a369750bef0449c1bf750cf7b4be18704f`
- Ubuntu 22.04, four NVIDIA RTX A6000 GPUs, driver 595.71.05
- ROS 2 Humble, rosbridge 2.0.7, turtlesim 1.4.3
- Gazebo Fortress / Ignition Gazebo 6 through `ros-humble-ros-gz` 0.244.25
- Isaac Lab `develop` at `b634245535dd7572f13a5699e0ff2fd2542b33c7`
- Python 3.12, PyTorch 2.11.0+cu128, Newton 1.5.0.dev0,
  MuJoCo 3.10, MuJoCo Warp 3.10.0.2
- Isaac Sim container `nvcr.io/nvidia/isaac-sim:6.0.0-dev2`
- Community Isaac Sim MCP `whats2000/isaacsim-mcp-server` at
  `4704503bfa4ba4a7fd8a4f7d0e8eb036b6d85d31`

Re-check all versions before a new report; this file records evidence, not a
promise that these remain latest.

## Reported evidence from that run

- ROS 2: live `/rosapi/topics`, turtlesim command/pose discovery, pose
  subscription, capability compilation, and fail-closed direct velocity calls.
- Gazebo: headless server runs and a `ros_gz_bridge` `/clock` sample reaches
  ROS 2.
- Isaac Lab: bounded single-GPU and four-rank RSL-RL Cartpole training exits 0
  using Newton/MJWarp.
- MCP: protocol initialize, 42-tool enumeration, and `get_scene_info` call
  complete against the simulator extension.
- Hub: signed publish and remote install round trip returns to zero installed
  assets.

## Known limits

- Isaac Sim 6.0.0-dev2 can advance physics but may abort in normal shutdown
  with a busy TaskGroup. Keep this as a lifecycle defect; use kitless Isaac Lab
  for a clean bounded training pass.
- The community Isaac MCP test suite assumes a robot key named `franka`; Isaac
  Sim 6 exposes `frankapanda`, `frankafr3`, and related keys. Its full robot
  suite therefore needs a version adapter update.
- ROSClaw generic Hub is currently local/file-backed and separate from the
  legacy MCP discovery endpoint. It is not a production public registry.
- ClawHub ROS skills must be audited before use. Prefer read-only introspection;
direct ROS publish skills conflict with ROSClaw's daemon/firewall boundary.

## Phase 3 GuardedBase acceptance (2026-07-23)

The current verifier supersedes the historical `/clock`-only Gazebo check. It
requires a real Fortress diff-drive body, `Odometry`, `LaserScan`, rosbridge,
the daemon-owned guarded executor, and a ROS-side command deadman.

`launch_testing` supervises the Gazebo server, command/odometry/scan bridges,
deadman, and command worker. It injects real `SIGKILL` / `SIGTERM` faults.
The product CLI additionally kills and restarts rosbridge while an action is
running, verifies that missing observation cannot become `TASK_VERIFIED`,
rebinds to the new connection generation, uses a new Action ID, and checks that
the old action produces no duplicate physical effect.

Use `scripts/verify_gazebo.sh` for the bounded process test and
`rosclaw chaos run gazebo-guarded-base ...` for the complete canonical path.

## Primary documentation

- ROS 2 turtlesim: https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Introducing-Turtlesim/Introducing-Turtlesim.html
- Gazebo / ROS installation: https://gazebosim.org/docs/fortress/ros_installation/
- Isaac Lab installation: https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html
- Isaac Sim containers: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html
