#!/usr/bin/env bash
# Stop a running ROS2 RealSense camera node started by run_rosclaw_rps.sh.
#
# The launcher intentionally leaves the camera node running across demo runs
# to avoid D435i stop/start failures on Jetson. Run this when you want to
# shut the camera down cleanly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

pids=$(pgrep -f '[r]ealsense2_camera_node' || true)
if [ -z "${pids}" ]; then
    echo "No realsense2_camera_node is running."
    exit 0
fi

echo "Stopping realsense2_camera_node (${pids})..."
echo "${pids}" | xargs -r kill -INT 2>/dev/null || true

waited=0
while [ "${waited}" -lt 10 ]; do
    pids=$(pgrep -f '[r]ealsense2_camera_node' || true)
    if [ -z "${pids}" ]; then
        echo "Camera node stopped."
        exit 0
    fi
    sleep 1
    waited=$((waited + 1))
    if [ "${waited}" -eq 5 ]; then
        echo "${pids}" | xargs -r kill -TERM 2>/dev/null || true
    fi
done

echo "WARNING: realsense2_camera_node did not stop gracefully."
echo "         It may be stuck in USB I/O. If you need to start the camera again,"
echo "         try ./scripts/reset_realsense_usb.sh or reseat the USB cable."
exit 1
