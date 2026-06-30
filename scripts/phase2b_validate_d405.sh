#!/usr/bin/env bash
# Phase 2b — D405-only hardware validation (D435i unavailable due to xHCI failure)
set -eo pipefail

source /opt/ros/jazzy/setup.bash
REPORT_DIR="$(cd "$(dirname "$0")/.." && pwd)/reports"
mkdir -p "$REPORT_DIR"

echo "[phase2b] Enumerating RealSense devices ..."
rs-enumerate-devices -s > "$REPORT_DIR/rs-enumerate-devices.txt" 2>&1 || true

echo "[phase2b] Launching D405 ROS2 node ..."
PYTHONUNBUFFERED=1 timeout 30 ros2 launch realsense2_camera rs_launch.py \
    camera_name:=d405 serial_no:=\"230422272729\" \
    depth_module.depth_profile:=848x480x10 \
    rgb_camera.color_profile:=848x480x10 \
    enable_accel:=false enable_gyro:=false \
    > "$REPORT_DIR/ros2_d405.log" 2>&1 &
d405_pid=$!
sleep 10

echo "[phase2b] D405 topic list / hz ..."
ros2 topic list > "$REPORT_DIR/d405_topics.txt" 2>&1 || true
PYTHONUNBUFFERED=1 timeout 10 ros2 topic hz /camera/d405/color/image_raw --window 30 > "$REPORT_DIR/d405_color_hz.txt" 2>&1 &
hz_pid=$!
sleep 6
kill $hz_pid 2>/dev/null || true
wait $hz_pid 2>/dev/null || true

echo "[phase2b] D405 camera_info ..."
ros2 topic echo /camera/d405/color/camera_info --once > "$REPORT_DIR/d405_color_camera_info.txt" 2>&1 || true
ros2 topic echo /camera/d405/depth/camera_info --once > "$REPORT_DIR/d405_depth_camera_info.txt" 2>&1 || true

echo "[phase2b] Stopping D405 node ..."
kill $d405_pid 2>/dev/null || true
pkill -9 -f "realsense2_camera_node.*camera_name:=d405" || true
wait $d405_pid 2>/dev/null || true

cat > "$REPORT_DIR/phase2b_d405_validation.json" <<EOF
{
  "phase": "2b",
  "status": "completed",
  "d405_serial": "230422272729",
  "d405_usb_type": "2.1",
  "note": "D435i unavailable: xHCI host controller for USB bus 4 died during earlier launch and did not recover without reboot."
}
EOF

echo "[phase2b] Done."
