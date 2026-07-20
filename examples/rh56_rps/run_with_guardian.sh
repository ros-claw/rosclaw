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

_import_dir() {
    "${PYTHON}" -c "import $1, os; print(os.path.dirname($1.__file__))" 2>/dev/null || true
}

ROSCLAW_SRC="${RPS_ROSCLAW_SRC:-}"
if [ -z "${ROSCLAW_SRC}" ]; then
    ROSCLAW_SRC="$(_import_dir rosclaw)"
fi
if [ -z "${ROSCLAW_SRC}" ]; then
    echo "ERROR: rosclaw is not importable from ${PYTHON}."
    echo "       Install rosclaw or set RPS_ROSCLAW_SRC."
    exit 1
fi

RH56_SRC="${RPS_RH56_SRC:-}"
if [ -z "${RH56_SRC}" ]; then
    RH56_SRC="$(_import_dir rosclaw_rh56)"
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
if [ -z "${RH56_SRC}" ]; then
    echo "WARNING: rosclaw_rh56 not importable and no sibling runtime checkout found."
    RH56_SRC=""
fi

if [ -n "${RH56_SRC}" ]; then
    export PYTHONPATH="${DEMO_SRC}:${ROSCLAW_SRC}:${RH56_SRC}:${PYTHONPATH}"
else
    export PYTHONPATH="${DEMO_SRC}:${ROSCLAW_SRC}:${PYTHONPATH}"
fi

echo "[$(basename "$0")] python: ${PYTHON} | rosclaw: ${ROSCLAW_SRC} | rh56: ${RH56_SRC:-<missing>}"

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
