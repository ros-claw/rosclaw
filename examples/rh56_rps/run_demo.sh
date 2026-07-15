#!/usr/bin/env bash
# One-command launcher for the RH56 RPS dual-hand demo.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Source ROS 2 so we can inspect topics.
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Camera node PID we started (if any).  Tracked so we can stop it cleanly on exit.
CAMERA_PID=""

cleanup() {
    if [ -n "${CAMERA_PID}" ] && kill -0 "${CAMERA_PID}" 2>/dev/null; then
        echo "Stopping camera node (PID ${CAMERA_PID})..."
        kill -INT "${CAMERA_PID}" 2>/dev/null || true
        wait "${CAMERA_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Start the D405 camera if it is not already publishing.
if ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
    echo "D405 camera topic already available."
else
    echo "Starting D405 camera..."
    ./launch_camera.sh &
    CAMERA_PID=$!

    # Wait up to 20 seconds for the topic to appear.
    for i in $(seq 1 40); do
        if ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
            echo "D405 camera is up."
            break
        fi
        sleep 0.5
    done

    if ! ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
        echo "ERROR: D405 camera topic did not appear.  Check USB and try ./launch_camera.sh manually."
        exit 1
    fi
fi

# Run the actual demo.
./run_ui.sh --config-dir configs/dual "$@"
