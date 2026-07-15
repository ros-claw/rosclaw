#!/usr/bin/env bash
# Launch the RealSense camera ROS2 node for the RH56 RPS demo.
# Defaults to D435i; set RH56_CAMERA_DEVICE=d405 to use the D405.
set -e

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Allow --device d405|d435i or env override.
DEVICE="${RH56_CAMERA_DEVICE:-d435i}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        *)
            break
            ;;
    esac
done

PROFILE="640x480x30"

# Color-only.  Depth/infra are disabled because the RPS demo only needs the
# color feed; this lowers USB bandwidth and CPU load.
#
# D405 exposes color through the depth module, while D435i has a dedicated
# rgb_camera module, so the profile parameter must match the device.
case "${DEVICE}" in
    d405)
        exec ros2 launch realsense2_camera rs_launch.py \
            device_type:=d405 \
            enable_depth:=false \
            enable_infra1:=false \
            enable_infra2:=false \
            depth_module.color_profile:=${PROFILE} \
            "$@"
        ;;
    d435|d435i|d456|d455)
        exec ros2 launch realsense2_camera rs_launch.py \
            device_type:=${DEVICE} \
            enable_depth:=false \
            enable_infra1:=false \
            enable_infra2:=false \
            rgb_camera.color_profile:=${PROFILE} \
            "$@"
        ;;
    *)
        echo "Unknown RealSense device type: ${DEVICE}" >&2
        echo "Use --device d405|d435i or set RH56_CAMERA_DEVICE" >&2
        exit 1
        ;;
esac
