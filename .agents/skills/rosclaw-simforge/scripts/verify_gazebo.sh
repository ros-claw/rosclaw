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
  -t rosclaw/ros2-humble-gazebo:latest \
  -f docker/ros2-humble-gazebo.Dockerfile .

evidence_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$evidence_dir"
}
trap cleanup EXIT

timeout 120 docker run --rm --network host --ipc host \
  -e ROS_DOMAIN_ID=187 \
  -e ROS_LOCALHOST_ONLY=1 \
  -e ROSCLAW_REPO_ROOT=/workspace \
  -e ROSCLAW_GAZEBO_EVIDENCE_DIR=/evidence \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$repo_root:/workspace:ro" \
  -v "$evidence_dir:/evidence:rw" \
  rosclaw/ros2-humble-gazebo:latest bash -lc '
source /opt/ros/humble/setup.bash
set -eo pipefail
ign sdf -k /workspace/benchmarks/simforge/suites/core_v1/guarded_base/gazebo_guarded_base.sdf
launch_test --junit-xml=/evidence/launch-testing-junit.xml \
  /workspace/tests/simforge/launch/test_gazebo_guarded_base_launch.py
'

test -s "$evidence_dir/launch-testing-result.json"
grep -q '"passed": true' "$evidence_dir/launch-testing-result.json"
echo "ROSCLAW_GAZEBO_VERIFY_OK simulator=fortress model=diff_drive odom=true laser=true launch_testing=true deadman=true"
