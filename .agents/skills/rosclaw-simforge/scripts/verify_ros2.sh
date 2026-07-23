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
if [[ "$host_http_proxy" =~ (127\.0\.0\.1|localhost|\[::1\]) || "$host_https_proxy" =~ (127\.0\.0\.1|localhost|\[::1\]) ]]; then
  build_network=host
fi

docker build --network "$build_network" --pull=false "${proxy_args[@]}" \
  -t rosclaw/ros2-humble-bridge:latest \
  -f docker/ros2-humble-bridge.Dockerfile .

.venv/bin/pytest -q -m "deployment and integration" \
  tests/connectors/ros/integration/test_ros2_container_smoke.py

port="${ROSCLAW_ROS_TEST_PORT:-$(.venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "ROSCLAW_ROS_TEST_PORT must be an integer from 1 to 65535" >&2
  exit 2
fi
export ROSBRIDGE_HOST_PORT="$port"
export ROSBRIDGE_CONTAINER_NAME=rosclaw-ros2-turtlesim-simforge
export COMPOSE_PROJECT_NAME=rosclaw-turtlesim-simforge
trap 'docker compose -f docker-compose.ros-test.yml down -v --remove-orphans >/dev/null 2>&1 || true' EXIT
docker compose -f docker-compose.ros-test.yml up --no-build -d --wait
ROSCLAW_ROS_TEST_ENDPOINT="ws://127.0.0.1:$port" \
  .venv/bin/pytest -q -m integration tests/connectors/ros/test_turtlesim_integration.py

echo "ROSCLAW_ROS2_VERIFY_OK endpoint=ws://127.0.0.1:$port"
