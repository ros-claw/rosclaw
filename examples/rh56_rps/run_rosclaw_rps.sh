#!/usr/bin/env bash
# One-command launcher for the ROSClaw-native RH56 RPS demo.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Source ROS 2 if available (needed for the RealSense camera bridge).
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Make the demo package and its ROSClaw / RH56 dependencies importable.
DEMO_SRC="${SCRIPT_DIR}/src"
ROSCLAW_SRC="/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src"
RH56_SRC="/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src"
export PYTHONPATH="${DEMO_SRC}:${ROSCLAW_SRC}:${RH56_SRC}:${PYTHONPATH}"

# Camera node PID we started (if any). We intentionally leave the RealSense
# node running after the demo exits because repeated stop/start cycles on this
# Jetson + D435i combination frequently leave the camera firmware stuck in a
# UVC timeout state. Use ./scripts/stop_camera.sh to stop it manually.
CAMERA_PID=""

_kill_realsense_node_gracefully() {
    # Avoid matching this grep command itself by using the '[r]' trick.
    local pids
    pids=$(pgrep -f '[r]ealsense2_camera_node' || true)
    if [ -z "${pids}" ]; then
        return 0
    fi
    echo "Stopping existing realsense2_camera_node (${pids})..."
    echo "${pids}" | xargs -r kill -INT 2>/dev/null || true
    local waited=0
    while [ "${waited}" -lt 8 ]; do
        pids=$(pgrep -f '[r]ealsense2_camera_node' || true)
        if [ -z "${pids}" ]; then
            echo "Camera node stopped."
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        if [ "${waited}" -eq 4 ]; then
            echo "${pids}" | xargs -r kill -TERM 2>/dev/null || true
        fi
    done
    echo "WARNING: realsense2_camera_node did not stop gracefully."
    echo "         The camera may need a USB bus reset or reseat before it works again."
    return 1
}

cleanup() {
    # Intentionally a no-op for the camera node. We keep it alive across runs.
    # Serial guardian is handled by run_with_guardian.sh.
    :
}
trap cleanup EXIT INT TERM

# Pick a Python interpreter that can import ROS2, OpenCV, and serial.
SYSTEM_PYTHON="/usr/bin/python3"
PYTHON="${SYSTEM_PYTHON}"

VENV="/home/nvidia/workspace/rosclaw/rosclaw_test/.venv"
if [ -x "${VENV}/bin/python" ] && \
   "${VENV}/bin/python" -c "import rclpy, cv2, serial" 2>/dev/null; then
    PYTHON="${VENV}/bin/python"
fi

# Parse our own arguments first so we know which config/mode is requested.
MODE="mock"
CONFIG_DIR="configs/dual"

# Use a proper shift-based parser so --config-dir configs/single works anywhere.
args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
    a="${args[$i]}"
    case "$a" in
        --mode=*) MODE="${a#*=}" ;;
        --mode) i=$((i+1)); MODE="${args[$i]:-mock}" ;;
        --config-dir=*) CONFIG_DIR="${a#*=}" ;;
        --config-dir) i=$((i+1)); CONFIG_DIR="${args[$i]:-configs/dual}" ;;
    esac
    i=$((i+1))
done

# Determine which camera source/device the selected config uses.
# Only start the ROS2 RealSense node when the source is "ros2".
CAMERA_SOURCE="ros2"
CAMERA_DEVICE="d435i"
CAMERA_CONFIG="${SCRIPT_DIR}/${CONFIG_DIR}/rps_demo.yaml"
if [ -f "${CAMERA_CONFIG}" ]; then
    _cam_cfg=$(python3 - <<PY
import yaml
try:
    cfg = yaml.safe_load(open('${CAMERA_CONFIG}'))['camera']
    print(cfg.get('source','ros2'), cfg.get('device_type','d435i'))
except Exception as e:
    print('ros2 d435i')
PY
    )
    CAMERA_SOURCE=$(echo "${_cam_cfg}" | awk '{print $1}')
    CAMERA_DEVICE=$(echo "${_cam_cfg}" | awk '{print $2}')
fi

# Quick helper: return true if the color topic is publishing frames.
_camera_streaming() {
    local timeout_s="${1:-3}"
    local rate
    rate=$(timeout "${timeout_s}" ros2 topic hz /camera/camera/color/image_raw --window 20 2>/dev/null \
        | grep 'average rate:' | tail -1 | awk '{print $3}')
    [ -n "${rate}" ] && awk "BEGIN{exit !(${rate} > 0)}" 2>/dev/null
}

if [ "${MODE}" = "full" ] && [ "${CAMERA_SOURCE}" = "ros2" ]; then
    if ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
        echo "RealSense camera topic already available; verifying frames..."
        if _camera_streaming 5; then
            echo "RealSense ${CAMERA_DEVICE} camera is streaming."
        else
            echo "RealSense topic exists but is not streaming. Restarting camera node..."
            if ! _kill_realsense_node_gracefully; then
                echo "ERROR: Cannot restart the camera node because the old one is stuck."
                echo "       Try ./scripts/reset_realsense_usb.sh or reseat the USB cable."
                exit 1
            fi
            # Wait for the stale topic to disappear so we don't mistakenly reuse it.
            for _ in $(seq 1 10); do
                if ! ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
                    break
                fi
                sleep 0.5
            done
        fi
    fi

    if ! ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
        echo "Starting RealSense ${CAMERA_DEVICE} camera..."
        # Reset the camera firmware between sessions: on this D435i a previous
        # session's teardown leaves the firmware in a bad state and the next
        # pipe.start() wedges (UVC GET_CUR -110 -> USB disconnect). A
        # hardware_reset restores the freshly-plugged condition. Best-effort.
        if [ -x "${VENV}/bin/python" ]; then
            "${VENV}/bin/python" - <<'PY' || echo "WARNING: camera hardware_reset failed; continuing"
import time
try:
    import pyrealsense2 as rs
except Exception as exc:
    print(f"pyrealsense2 unavailable, skipping reset: {exc}")
else:
    devs = rs.context().query_devices()
    if devs:
        devs[0].hardware_reset()
        for _ in range(20):
            time.sleep(1)
            if rs.context().query_devices():
                break
        time.sleep(1.5)
        print("Camera firmware reset OK")
    else:
        print("No RealSense device to reset")
PY
        fi
        RH56_CAMERA_DEVICE="${CAMERA_DEVICE}" ./launch_camera.sh &
        CAMERA_PID=$!

        for i in $(seq 1 40); do
            if ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
                echo "RealSense ${CAMERA_DEVICE} camera topic appeared."
                break
            fi
            sleep 0.5
        done

        if ! ros2 topic list 2>/dev/null | grep -q '/camera/camera/color/image_raw'; then
            echo "ERROR: RealSense camera topic did not appear. Check USB and try ./launch_camera.sh --device ${CAMERA_DEVICE} manually."
            exit 1
        fi
    fi

    # Final check: make sure frames are actually flowing before handing off to
    # the UI.  If not, the UI would open on a black feed.
    if ! _camera_streaming 8; then
        echo "WARNING: RealSense camera is not publishing frames. The UI may show 'No camera feed'."
        echo "         Try restarting the camera node or reseating the USB cable."
    fi
fi

# Default config dir is configs/dual (permanent, not /tmp).
"${PYTHON}" -m rosclaw_rps.rosclaw_cli \
    --config-dir "${CONFIG_DIR}" \
    --rosclaw-config "${CONFIG_DIR}/rps_rosclaw.yaml" \
    "$@"
