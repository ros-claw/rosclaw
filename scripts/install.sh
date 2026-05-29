#!/usr/bin/env bash
# ROSClaw v1.0 One-Click Installer
# Usage: ./scripts/install.sh [--dev] [--ros2]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"
INSTALL_DEV=false
INSTALL_ROS2=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev) INSTALL_DEV=true; shift ;;
        --ros2) INSTALL_ROS2=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  ROSClaw v1.0 Installer"
echo "========================================"
echo ""

# 1. Python version check
echo "[1/7] Checking Python version..."
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$($cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ "$(printf '%s\n' "$version" "3.10" | sort -V | head -n1)" == "3.10" ]]; then
            PYTHON_CMD="$cmd"
            echo "  Found Python $version ($cmd)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "[ERROR] Python >= 3.10 required but not found."
    exit 1
fi

# 2. Create virtual environment
echo ""
echo "[2/7] Creating virtual environment..."
VENV_VALID=true
if [[ -d "$VENV_DIR" ]]; then
    if [[ ! -f "$VENV_DIR/bin/pip" ]]; then
        echo "  venv exists but pip is missing, recreating..."
        VENV_VALID=false
    else
        echo "  venv already exists at $VENV_DIR"
    fi
else
    VENV_VALID=false
fi

if [[ "$VENV_VALID" == false ]]; then
    rm -rf "$VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR" --system-site-packages
    echo "  Created venv at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Ensure pip is available
if [[ ! -f "$VENV_PIP" ]]; then
    echo "  Installing pip into venv..."
    "$VENV_PYTHON" -m ensurepip --upgrade
fi

# 3. Upgrade pip
echo ""
echo "[3/7] Upgrading pip..."
"$VENV_PIP" install --upgrade pip setuptools wheel -q

# 4. Install ROSClaw
echo ""
echo "[4/7] Installing ROSClaw..."
cd "$PROJECT_ROOT"
"$VENV_PIP" install -e "." -q

# Optional extras
if [[ "$INSTALL_DEV" == true ]]; then
    echo "  Installing dev dependencies..."
    "$VENV_PIP" install -e ".[dev]" -q
fi

if [[ "$INSTALL_ROS2" == true ]]; then
    echo "  Installing ROS2 dependencies..."
    "$VENV_PIP" install -e ".[ros2]" -q
fi

# 5. Link e-URDF Zoo
echo ""
echo "[5/7] Setting up e-URDF Zoo..."
EURDF_ZOO="${PROJECT_ROOT}/e-urdf-zoo"
if [[ -d "$EURDF_ZOO" ]]; then
    echo "  e-URDF Zoo found: $EURDF_ZOO"
    echo "  Robots: $(ls -1 "$EURDF_ZOO" | tr '\n' ' ')"
else
    echo "  [WARN] e-URDF Zoo not found at $EURDF_ZOO"
fi

# 6. Initialize workspace
echo ""
echo "[6/7] Initializing ROSClaw workspace..."
"$VENV_PYTHON" -m rosclaw.cli init --force "$PROJECT_ROOT" 2>/dev/null || true
mkdir -p "${PROJECT_ROOT}/practice_data"

# 7. Health check
echo ""
echo "[7/7] Running health check..."
"$VENV_PYTHON" -m rosclaw.cli doctor 2>/dev/null || {
    echo "  [WARN] rosclaw doctor had issues (non-fatal)"
}

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "To activate: source ${VENV_DIR}/bin/activate"
echo "To start:     rosclaw start"
echo "To check:     rosclaw doctor"
echo "To see help:  rosclaw --help"
echo ""
