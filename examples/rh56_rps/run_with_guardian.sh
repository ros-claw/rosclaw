#!/usr/bin/env bash
# Run the ROSClaw RPS demo through the serial guardian.
#
# The guardian keeps the real CH340 serial ports open for the whole session,
# avoiding the close/reopen bug that makes /dev/ttyUSB* die with
# Input/output error after a normal process exit.
#
# Usage (after reseating the CH340 adapters):
#   ./run_with_guardian.sh --mode full --rounds 3 --auto --headless
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

DEMO_SRC="${SCRIPT_DIR}/src"
ROSCLAW_SRC="/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src"
RH56_SRC="/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src"
export PYTHONPATH="${DEMO_SRC}:${ROSCLAW_SRC}:${RH56_SRC}:${PYTHONPATH}"

export RH56_GUARDIAN=1
export RH56_GUARDIAN_DIR="/tmp/rh56_guardian"

rm -rf "${RH56_GUARDIAN_DIR}"
mkdir -p "${RH56_GUARDIAN_DIR}"

echo "Starting serial guardian..."
/usr/bin/python3 scripts/serial_guardian.py --socket-dir "${RH56_GUARDIAN_DIR}" &
GUARDIAN_PID=$!
trap 'kill "${GUARDIAN_PID}" 2>/dev/null || true; rm -rf "${RH56_GUARDIAN_DIR}"' EXIT INT TERM

# Wait for the guardian to discover hands and create sockets.
for i in $(seq 1 40); do
    if [ -S "${RH56_GUARDIAN_DIR}/1.sock" ] && [ -S "${RH56_GUARDIAN_DIR}/2.sock" ]; then
        echo "Guardian ready."
        break
    fi
    if ! kill -0 "${GUARDIAN_PID}" 2>/dev/null; then
        echo "Guardian exited unexpectedly."
        exit 1
    fi
    sleep 0.2
done

if [ ! -S "${RH56_GUARDIAN_DIR}/1.sock" ] || [ ! -S "${RH56_GUARDIAN_DIR}/2.sock" ]; then
    echo "Guardian did not discover both hands. Check USB connections."
    exit 1
fi

# Forward all arguments to the normal launcher.
exec ./run_rosclaw_rps.sh "$@"
