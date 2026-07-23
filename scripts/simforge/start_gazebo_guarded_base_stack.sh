#!/usr/bin/env bash
set -euo pipefail

repo_root="${ROSCLAW_REPO_ROOT:-/workspace}"
evidence_dir="${ROSCLAW_GAZEBO_EVIDENCE_DIR:-/evidence}"
world="${repo_root}/benchmarks/simforge/suites/core_v1/guarded_base/gazebo_guarded_base.sdf"
node_script="${repo_root}/scripts/simforge/gazebo_guarded_base_node.py"
mkdir -p "$evidence_dir"

declare -a pids=()

start_process() {
  local name="$1"
  shift
  "$@" >"${evidence_dir}/${name}.log" 2>&1 &
  local pid=$!
  pids+=("$pid")
  printf '%s %s\n' "$name" "$pid" >>"${evidence_dir}/processes.txt"
}

cleanup() {
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

: >"${evidence_dir}/processes.txt"
start_process gazebo ign gazebo -r -s "$world"
start_process cmd_bridge \
  /opt/ros/humble/lib/ros_gz_bridge/parameter_bridge \
  "/guarded_base/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist"
start_process odom_bridge \
  /opt/ros/humble/lib/ros_gz_bridge/parameter_bridge \
  "/guarded_base/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry"
start_process scan_bridge \
  /opt/ros/humble/lib/ros_gz_bridge/parameter_bridge \
  "/guarded_base/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan"
start_process deadman \
  python3 "$node_script" deadman \
  --timeout-sec "${ROSCLAW_DEADMAN_TIMEOUT_SEC:-0.35}" \
  --event-log "${evidence_dir}/stack-deadman-events.jsonl"
start_process rosapi /opt/ros/humble/lib/rosapi/rosapi_node
start_process rosbridge \
  /opt/ros/humble/lib/rosbridge_server/rosbridge_websocket \
  --ros-args -p port:="${ROSCLAW_ROSBRIDGE_PORT:-9090}"

gazebo_pid="${pids[0]}"
deadman_pid="${pids[4]}"
cmd_bridge_pid="${pids[1]}"
while kill -0 "$gazebo_pid" 2>/dev/null \
  && kill -0 "$deadman_pid" 2>/dev/null \
  && kill -0 "$cmd_bridge_pid" 2>/dev/null; do
  sleep 0.2
done

echo "a safety-critical GuardedBase process exited" >&2
exit 1
