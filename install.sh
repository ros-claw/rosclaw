#!/usr/bin/env bash
# ROSClaw v1.0 — One-Click Install Script
# Usage: bash install.sh [--break-system-packages]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="ROSClaw"
PROJECT_VERSION="1.0.0"
MIN_PYTHON_VERSION="3.10"
VENV_DIR="${SCRIPT_DIR}/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }
step()  { echo -e "${CYAN}▶ $*${NC}"; }

BREAK_SYSTEM_PACKAGES=""
for arg in "$@"; do
    case "$arg" in
        --break-system-packages) BREAK_SYSTEM_PACKAGES="1" ;;
        -h|--help)
            echo "ROSClaw v1.0 Install Script"
            echo "Usage: bash install.sh [--break-system-packages]"
            exit 0 ;;
    esac
done

cat <<EOF
╔══════════════════════════════════════════════════════════════╗
║   ${PROJECT_NAME} v${PROJECT_VERSION} — Installer                           ║
║   The Universal OS for Software-Defined Embodied AI          ║
╚══════════════════════════════════════════════════════════════╝
EOF

step "Step 1/5: Detecting Python"
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ $(printf '%s\n%s' "$MIN_PYTHON_VERSION" "$version" | sort -V | head -n1) == "$MIN_PYTHON_VERSION" ]]; then
            PYTHON_CMD="$cmd"
            ok "Found Python $version at $(command -v $cmd)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    err "Python >= $MIN_PYTHON_VERSION is required."
    err "  Ubuntu: sudo apt install python3 python3-venv python3-pip"
    err "  macOS:  brew install python@3.12"
    exit 1
fi

step "Step 2/5: Handling PEP668"
USE_VENV=""
if ! $PYTHON_CMD -m pip install --dry-run pip &>/dev/null 2>&1; then
    warn "PEP668 externally-managed environment detected"
    if [[ -z "$BREAK_SYSTEM_PACKAGES" ]]; then
        read -r -p "Use virtual environment? [Y/n]: " response
        response=${response:-Y}
        if [[ "$response" =~ ^[Yy] ]]; then
            USE_VENV="1"
        else
            BREAK_SYSTEM_PACKAGES="1"
            warn "Proceeding with --break-system-packages"
        fi
    fi
fi

step "Step 3/5: Installing ROSClaw"
PIP_INSTALL_ARGS=()
[[ -n "$BREAK_SYSTEM_PACKAGES" ]] && PIP_INSTALL_ARGS+=("--break-system-packages")

if [[ -n "$USE_VENV" ]]; then
    info "Creating virtual environment at $VENV_DIR ..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    ok "Virtual environment activated"
    PIP_CMD="$VENV_DIR/bin/pip"
else
    PIP_CMD="$PYTHON_CMD -m pip"
fi

$PIP_CMD install --upgrade pip "${PIP_INSTALL_ARGS[@]}" &>/dev/null || true

cd "$SCRIPT_DIR"
$PIP_CMD install -e ".[dev]" "${PIP_INSTALL_ARGS[@]}" || {
    err "Installation failed. Try: sudo apt install build-essential python3-dev libgl1-mesa-dev"
    exit 1
}
ok "ROSClaw installed"

step "Step 4/5: Configuring e-URDF-Zoo"
ZOO_SOURCE="${SCRIPT_DIR}/e-urdf-zoo"
ZOO_TARGET="${HOME}/.rosclaw/e-urdf-zoo"
WORKSPACE_DIR="${HOME}/.rosclaw"
mkdir -p "$WORKSPACE_DIR"

if [[ -d "$ZOO_SOURCE" ]]; then
    rm -f "$ZOO_TARGET"
    ln -s "$ZOO_SOURCE" "$ZOO_TARGET" 2>/dev/null || cp -r "$ZOO_SOURCE" "$ZOO_TARGET"
    ok "e-URDF-Zoo linked at $ZOO_TARGET"
fi

if [[ ! -f "${WORKSPACE_DIR}/rosclaw.yaml" ]]; then
    cat > "${WORKSPACE_DIR}/rosclaw.yaml" <<EOF
workspace_dir: ${WORKSPACE_DIR}
eurdf_zoo_path: ${ZOO_TARGET}
runtime:
  default_robot: ur5e
  enable_firewall: true
  enable_sandbox: true
  enable_memory: true
  enable_practice: true
memory:
  backend: seekdb
  data_dir: ${WORKSPACE_DIR}/memory
practice:
  episodes_dir: ${WORKSPACE_DIR}/episodes
  max_episode_history: 1000
logging:
  level: INFO
  dir: ${WORKSPACE_DIR}/logs
EOF
    ok "Created ${WORKSPACE_DIR}/rosclaw.yaml"
fi

step "Step 5/5: Running rosclaw doctor"
if command -v rosclaw &>/dev/null; then
    DOCTOR_OUTPUT=$(rosclaw doctor 2>&1) || true
    echo "$DOCTOR_OUTPUT"
    ISSUE_COUNT=$(echo "$DOCTOR_OUTPUT" | grep -c "❌" || true)
    if [[ "$ISSUE_COUNT" -eq 0 ]]; then
        echo
        ok "══════════════════════════════════════════════════════════════"
        ok "  ${PROJECT_NAME} v${PROJECT_VERSION} installed successfully!"
        ok "  All health checks passed."
        ok "══════════════════════════════════════════════════════════════"
    else
        warn "Installed with $ISSUE_COUNT warnings. Run 'rosclaw doctor' for details."
    fi
else
    err "rosclaw not found in PATH. Try: export PATH=\"$HOME/.local/bin:\$PATH\""
    exit 1
fi

echo
cat <<EOF
${CYAN}Quick Start:${NC}
  rosclaw --version     # Show version
  rosclaw doctor        # Health diagnosis
  rosclaw init          # Initialize workspace
  rosclaw robot list    # List robots
  rosclaw run           # Start runtime

${CYAN}Paths:${NC}
  Workspace:  ${WORKSPACE_DIR}
  e-URDF Zoo: ${ZOO_TARGET}
  Logs:       ${WORKSPACE_DIR}/logs

EOF

if [[ -n "$USE_VENV" ]]; then
    echo -e "${YELLOW}Virtual environment:${NC} ${VENV_DIR}"
    echo -e "${YELLOW}Activate later:${NC} source ${VENV_DIR}/bin/activate"
fi
