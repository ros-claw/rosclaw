#!/usr/bin/env bash
# Launch the RH56 Rock-Paper-Scissors demo.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate ROS 2 if available (required for the ROS2 RealSense camera bridge).
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Make sure the local rosclaw_rps package is importable.
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/src"

# Use the system ROS 2 Python interpreter by default; the project venv does not
# have rclpy/cv2 on this machine.
SYSTEM_PYTHON="/usr/bin/python3"
PYTHON="${SYSTEM_PYTHON}"

# If rosclaw_rh56 is not installed, try the sibling runtime source tree.
if ! "${PYTHON}" -c "import rosclaw_rh56" 2>/dev/null; then
    CANDIDATE="/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src"
    if [ -d "${CANDIDATE}/rosclaw_rh56" ]; then
        export PYTHONPATH="${PYTHONPATH}:${CANDIDATE}"
    fi
fi

cd "${SCRIPT_DIR}"

# Prefer the project venv if it exists AND can import ROS2/OpenCV/serial.
VENV="/home/nvidia/workspace/rosclaw/rosclaw_test/.venv"
if [ -x "${VENV}/bin/python" ] && \
   "${VENV}/bin/python" -c "import rclpy, cv2, serial" 2>/dev/null; then
    PYTHON="${VENV}/bin/python"
fi

exec "${PYTHON}" -m rosclaw_rps.cli --mode full "$@"
