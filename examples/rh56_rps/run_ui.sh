#!/usr/bin/env bash
# Launch the RH56 Rock-Paper-Scissors demo (UI mode).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate ROS 2 if available (required for the ROS2 RealSense camera bridge).
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Path policy (P0-2): never hardcode a developer's private checkout.
#   * Python:       $RPS_PYTHON -> ./.venv/bin/python (repo-local) -> python3
#   * rosclaw_rh56: $RPS_RH56_SRC -> import rosclaw_rh56 -> sibling checkouts
if [ -n "${RPS_PYTHON:-}" ] && [ -x "${RPS_PYTHON}" ]; then
    PYTHON="${RPS_PYTHON}"
else
    # Venv discovery: prefer one that can import rosclaw AND the hardware deps.
    PYTHON=""
    for candidate in \
        "${SCRIPT_DIR}/.venv/bin/python" \
        "${SCRIPT_DIR}/../.venv/bin/python" \
        "${SCRIPT_DIR}/../../.venv/bin/python" \
        "${SCRIPT_DIR}/../../../.venv/bin/python"; do
        if [ -x "${candidate}" ] && \
           "${candidate}" -c "import rosclaw, cv2, serial" 2>/dev/null; then
            PYTHON="$(cd "$(dirname "${candidate}")" && pwd)/python"
            break
        fi
    done
    if [ -z "${PYTHON}" ]; then
        # Fall back: any venv that can import rosclaw.
        for candidate in \
            "${SCRIPT_DIR}/.venv/bin/python" \
            "${SCRIPT_DIR}/../.venv/bin/python" \
            "${SCRIPT_DIR}/../../.venv/bin/python" \
            "${SCRIPT_DIR}/../../../.venv/bin/python"; do
            if [ -x "${candidate}" ] && \
               "${candidate}" -c "import rosclaw" 2>/dev/null; then
                PYTHON="$(cd "$(dirname "${candidate}")" && pwd)/python"
                break
            fi
        done
    fi
    if [ -z "${PYTHON}" ]; then
        PYTHON="python3"
    fi
fi

# Make sure the local rosclaw_rps package is importable.
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/src"

RH56_SRC="${RPS_RH56_SRC:-}"
if [ -z "${RH56_SRC}" ]; then
    RH56_SRC="$("${PYTHON}" -c "import rosclaw_rh56, os; print(os.path.dirname(rosclaw_rh56.__file__))" 2>/dev/null || true)"
fi
if [ -z "${RH56_SRC}" ]; then
    for candidate in \
        "${SCRIPT_DIR}/../../rosclaw-rh56-runtime/src" \
        "${SCRIPT_DIR}/../../../rosclaw-rh56-runtime/src"; do
        if [ -f "${candidate}/rosclaw_rh56/__init__.py" ]; then
            RH56_SRC="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
if [ -n "${RH56_SRC}" ]; then
    export PYTHONPATH="${PYTHONPATH}:${RH56_SRC}"
else
    echo "WARNING: rosclaw_rh56 not importable and no sibling runtime checkout found."
    echo "         Set RPS_RH56_SRC to the runtime src dir."
fi

if ! "${PYTHON}" -c "import rclpy, cv2, serial" 2>/dev/null; then
    echo "WARNING: ${PYTHON} lacks rclpy/cv2/pyserial; full mode may fail."
fi
echo "[run_ui] python: ${PYTHON} | rh56: ${RH56_SRC:-<missing>}"

cd "${SCRIPT_DIR}"
exec "${PYTHON}" -m rosclaw_rps.cli --mode full "$@"
