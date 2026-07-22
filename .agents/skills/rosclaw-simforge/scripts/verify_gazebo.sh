#!/usr/bin/env bash
set -euo pipefail

repo_root="${ROSCLAW_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$repo_root"

proxy_args=()
build_network=default
host_http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
host_https_proxy="${https_proxy:-${HTTPS_PROXY:-}}"
if [[ -n "$host_http_proxy" ]]; then
  proxy_args+=(--build-arg "http_proxy=$host_http_proxy")
fi
if [[ -n "$host_https_proxy" ]]; then
  proxy_args+=(--build-arg "https_proxy=$host_https_proxy")
fi
if [[ "$host_http_proxy" == *"127.0.0.1"* || "$host_https_proxy" == *"127.0.0.1"* ]]; then
  build_network=host
fi

docker build --network "$build_network" --pull=false "${proxy_args[@]}" \
  -t rosclaw/ros2-humble-gazebo:latest \
  -f docker/ros2-humble-gazebo.Dockerfile .

timeout 120 docker run --rm rosclaw/ros2-humble-gazebo:latest bash -lc '
source /opt/ros/humble/setup.bash
set -eo pipefail
ign gazebo -r -s /usr/share/ignition/ignition-gazebo6/worlds/grid.sdf >/tmp/gazebo.log 2>&1 &
gazebo_pid=$!
bridge_pid=
trap "kill $gazebo_pid ${bridge_pid:-} 2>/dev/null || true" EXIT
for attempt in {1..30}; do
  if ign topic -l | grep -qx /clock; then break; fi
  sleep 1
done
ign topic -l | grep -qx /clock
ros2 run ros_gz_bridge parameter_bridge "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock" >/tmp/bridge.log 2>&1 &
bridge_pid=$!
for attempt in {1..30}; do
  if ros2 topic list | grep -qx /clock; then break; fi
  sleep 1
done
ros2 topic list | grep -qx /clock
timeout 20 ros2 topic echo /clock --once
'

echo "ROSCLAW_GAZEBO_VERIFY_OK simulator=fortress bridge=/clock"
