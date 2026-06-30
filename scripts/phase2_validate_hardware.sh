#!/usr/bin/env bash
# Phase 2 — RealSense driver / ROS2 Jazzy hardware validation
# Discovered hardware:
#   D435i serial: 231122070092, USB 3.2
#   D405  serial: 230422272729, USB 2.1 (limited to 848x480@10)
set -eo pipefail

source /opt/ros/jazzy/setup.bash
REPORT_DIR="$(cd "$(dirname "$0")/.." && pwd)/reports"
mkdir -p "$REPORT_DIR"
REPORT="$REPORT_DIR/phase2_hardware_validation.json"

kill_launch() {
  local pid="$1"
  local name="$2"
  if [[ -z "$pid" ]]; then return; fi
  set +e
  # Kill the launch process and its entire process group.
  local pgid
  pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')
  if [[ -n "$pgid" ]]; then
    kill -TERM -"$pgid" 2>/dev/null || true
  fi
  kill "$pid" 2>/dev/null || true
  pkill -f "realsense2_camera_node.*$name" 2>/dev/null || true
  pkill -f "ros2 launch realsense2_camera.*$name" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 2
}

cleanup() {
  set +e
  kill_launch "${d405_pid:-}" "d405"
  kill_launch "${d435i_pid:-}" "d435i"
  kill_launch "${dual_pid:-}" "d405\|d435i"
  if [[ -n "${hz_pid:-}" ]]; then kill $hz_pid 2>/dev/null || true; wait $hz_pid 2>/dev/null || true; fi
  if [[ -n "${hz2_pid:-}" ]]; then kill $hz2_pid 2>/dev/null || true; wait $hz2_pid 2>/dev/null || true; fi
}
trap cleanup EXIT

# ---- D405 (USB 2.1, low bandwidth profile) ----
echo "[phase2] Launching D405 ROS2 node ..."
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=d405 device_type:=d405 \
    depth_module.depth_profile:=848x480x10 \
    rgb_camera.color_profile:=848x480x10 \
    enable_accel:=false enable_gyro:=false \
    > "$REPORT_DIR/ros2_d405.log" 2>&1 &
d405_pid=$!
sleep 10

echo "[phase2] D405 topic list / hz ..."
ros2 topic list > "$REPORT_DIR/d405_topics.txt" 2>&1 || true
ros2 topic hz /camera/d405/color/image_raw --window 30 > "$REPORT_DIR/d405_color_hz.txt" 2>&1 &
hz_pid=$!
sleep 5
kill $hz_pid || true
wait $hz_pid 2>/dev/null || true

echo "[phase2] D405 camera_info ..."
ros2 topic echo /camera/d405/color/camera_info --once > "$REPORT_DIR/d405_color_camera_info.txt" 2>&1 || true
ros2 topic echo /camera/d405/depth/camera_info --once > "$REPORT_DIR/d405_depth_camera_info.txt" 2>&1 || true

echo "[phase2] Stopping D405 node ..."
kill_launch "$d405_pid" "d405"

# ---- D435i (USB 3.2) ----
echo "[phase2] Launching D435i ROS2 node ..."
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=d435i serial_no:=\"231122070092\" \
    depth_module.depth_profile:=1280x720x30 \
    rgb_camera.color_profile:=1280x720x30 \
    enable_accel:=true enable_gyro:=true \
    > "$REPORT_DIR/ros2_d435i.log" 2>&1 &
d435i_pid=$!
sleep 10

echo "[phase2] D435i topic list / hz / IMU ..."
ros2 topic list > "$REPORT_DIR/d435i_topics.txt" 2>&1 || true
ros2 topic hz /camera/d435i/color/image_raw --window 30 > "$REPORT_DIR/d435i_color_hz.txt" 2>&1 &
hz_pid=$!
ros2 topic hz /camera/d435i/imu --window 100 > "$REPORT_DIR/d435i_imu_hz.txt" 2>&1 &
hz2_pid=$!
sleep 5
kill $hz_pid $hz2_pid || true
wait $hz_pid $hz2_pid 2>/dev/null || true

echo "[phase2] D435i camera_info ..."
ros2 topic echo /camera/d435i/color/camera_info --once > "$REPORT_DIR/d435i_color_camera_info.txt" 2>&1 || true
ros2 topic echo /camera/d435i/depth/camera_info --once > "$REPORT_DIR/d435i_depth_camera_info.txt" 2>&1 || true

echo "[phase2] Stopping D435i node ..."
kill_launch "$d435i_pid" "d435i"

# ---- Dual-camera launch (both simultaneously) ----
echo "[phase2] Launching dual-camera ROS2 node ..."
ros2 launch realsense2_camera rs_multi_camera_launch.py \
    camera_name1:=d405 serial_no1:=\"230422272729\" device_type1:=d405 \
    depth_module.depth_profile1:=848x480x10 rgb_camera.color_profile1:=848x480x10 \
    camera_name2:=d435i serial_no2:=\"231122070092\" \
    depth_module.depth_profile2:=1280x720x30 rgb_camera.color_profile2:=1280x720x30 \
    enable_accel2:=true enable_gyro2:=true \
    > "$REPORT_DIR/ros2_dual.log" 2>&1 &
dual_pid=$!
sleep 12
ros2 topic list > "$REPORT_DIR/dual_topics.txt" 2>&1 || true
kill_launch "$dual_pid" "d405\|d435i"

cat > "$REPORT" <<EOF
{
  "phase": 2,
  "status": "completed",
  "report_dir": "$REPORT_DIR",
  "d435i_serial": "231122070092",
  "d405_serial": "230422272729",
  "d405_usb_type": "2.1",
  "d435i_usb_type": "3.2",
  "artifacts": [
    "ros2_d405.log",
    "d405_topics.txt",
    "d405_color_hz.txt",
    "d405_color_camera_info.txt",
    "d405_depth_camera_info.txt",
    "ros2_d435i.log",
    "d435i_topics.txt",
    "d435i_color_hz.txt",
    "d435i_imu_hz.txt",
    "d435i_color_camera_info.txt",
    "d435i_depth_camera_info.txt",
    "ros2_dual.log",
    "dual_topics.txt"
  ]
}
EOF

echo "[phase2] Done. Report written to $REPORT"
