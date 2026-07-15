#!/usr/bin/env bash
# Gracefully stop the RealSense D405 ROS2 node launched by launch_camera.sh.
# SIGINT lets the node close streams cleanly; SIGKILL can leave the device in a
# bad state and require a re-plug.
set -e

PIDS=$(ps aux | grep 'realsense2_camera/realsense2_camera_node' | grep -v grep | awk '{print $2}')
if [ -z "$PIDS" ]; then
    echo "No RealSense camera node is running."
    exit 0
fi

echo "Stopping RealSense camera node(s): $PIDS"
for PID in $PIDS; do
    kill -INT "$PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 0.2
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Node $PID did not stop gracefully, using SIGKILL..."
        kill -9 "$PID" 2>/dev/null || true
    fi
done
echo "Stopped."
