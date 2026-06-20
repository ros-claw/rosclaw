#!/usr/bin/env bash
# ROSClaw First Boot Bootstrapper
# Usage: curl -sSL https://rosclaw.io/get | bash
#
# This script only installs the ROSClaw CLI and creates a minimal workspace.
# It does NOT start any runtime, connect to robots, or upload telemetry.

set -euo pipefail

ROSCLAW_VERSION="${ROSCLAW_VERSION:-stable}"
ROSCLAW_HOME="${ROSCLAW_HOME:-$HOME/.rosclaw}"
ROSCLAW_CHANNEL="${ROSCLAW_CHANNEL:-stable}"
ROSCLAW_INSTALL_LOG="$ROSCLAW_HOME/logs/install.log"
ROSCLAW_PIP_SPEC="${ROSCLAW_PIP_SPEC:-rosclaw}"

NO_COLOR="${NO_COLOR:-}"
ASSUME_YES="${ROSCLAW_ASSUME_YES:-0}"
DRY_RUN="${ROSCLAW_DRY_RUN:-0}"

red()    { [ -n "$NO_COLOR" ] && echo "$*" || printf "\033[0;31m%s\033[0m\n" "$*"; }
green()  { [ -n "$NO_COLOR" ] && echo "$*" || printf "\033[0;32m%s\033[0m\n" "$*"; }
yellow() { [ -n "$NO_COLOR" ] && echo "$*" || printf "\033[1;33m%s\033[0m\n" "$*"; }
blue()   { [ -n "$NO_COLOR" ] && echo "$*" || printf "\033[0;34m%s\033[0m\n" "$*"; }

log() {
  mkdir -p "$(dirname "$ROSCLAW_INSTALL_LOG")"
  printf "%s %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >> "$ROSCLAW_INSTALL_LOG" || true
}

fail() {
  local code=1
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    code=$1
    shift
  fi
  red "❌ $*"
  log "ERROR (exit $code) $*"
  echo
  echo "Exit code: $code"
  echo
  case "$code" in
    10)
      echo "Python >= 3.11 is required."
      case "$OS" in
        linux)
          echo "Ubuntu/Debian: sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev"
          echo "Or install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
          ;;
        darwin)
          echo "macOS: brew install python@3.12"
          echo "Or install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
          ;;
      esac
      ;;
    11)
      echo "This platform is not supported by the ROSClaw bootstrapper."
      echo "Supported: Linux, macOS, Windows WSL."
      ;;
    20)
      echo "Failed to install the ROSClaw package."
      echo "Check your network connection and proxy settings, then re-run:"
      echo "  export ROSCLAW_CHANNEL=stable"
      echo "  bash scripts/get.sh"
      ;;
    21)
      echo "pip install failed with an externally-managed-environment error (PEP 668)."
      echo "Use the venv backend or install uv:"
      echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
      echo "Then re-run this script."
      ;;
    30)
      echo "Permission denied while creating the workspace."
      echo "Fix ownership/permissions:"
      echo "  sudo chown -R \"$(id -u):$(id -g)\" \"$ROSCLAW_HOME\""
      echo "Or set ROSCLAW_HOME to a writable directory:"
      echo "  export ROSCLAW_HOME=/path/to/writable/.rosclaw"
      ;;
    40)
      echo "Existing installation conflict or unsupported install backend."
      echo "Remove the existing workspace or force a clean install:"
      echo "  rm -rf \"$ROSCLAW_HOME\" && bash scripts/get.sh"
      ;;
    *)
      echo "Run diagnostics:"
      echo "  rosclaw doctor --bootstrap"
      ;;
  esac
  echo
  echo "If rosclaw is not in PATH, try:"
  echo "  export PATH=\"$ROSCLAW_HOME/bin:\$PATH\""
  echo
  exit "$code"
}

step() {
  blue "▶ $*"
  log "STEP $*"
}

dry_run_notice() {
  if [ "$DRY_RUN" = "1" ]; then
    yellow "[DRY RUN] Would execute: $*"
    return 0
  fi
  return 1
}

detect_platform() {
  OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
  ARCH="$(uname -m)"

  IS_WSL="false"
  if [ -f /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL="true"
  fi

  case "$OS" in
    linux|darwin) ;;
    *) fail 11 "Unsupported OS: $OS. Use Linux, macOS, or Windows WSL." ;;
  esac

  case "$ARCH" in
    x86_64|amd64|arm64|aarch64) ;;
    *) fail 11 "Unsupported architecture: $ARCH" ;;
  esac

  if [ "$IS_WSL" = "true" ]; then
    yellow "Windows WSL detected."
    yellow "ROSClaw will install inside the WSL Linux environment."
    yellow "Native Windows install is not supported by this bootstrapper."
  fi

  if [ "$OS" = "linux" ] && [ "${ROSCLAW_HOME#$HOME}" = "$ROSCLAW_HOME" ] && [ "${ROSCLAW_HOME#"/mnt/"}" != "$ROSCLAW_HOME" ]; then
    yellow "Warning: Installing under /mnt/ may be slower and cause permission issues."
    yellow "Recommended workspace: ~/.rosclaw"
  fi

  green "Platform: os=$OS arch=$ARCH wsl=$IS_WSL"
}

find_python() {
  PYTHON_BIN=""
  for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
      version="$($cmd - <<'PY' 2>/dev/null || true
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
      major="${version%%.*}"
      minor="${version#*.}"
      if [ -n "$major" ] && [ -n "$minor" ] && [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
        PYTHON_BIN="$(command -v "$cmd")"
        PYTHON_VERSION="$version"
        break
      fi
    fi
  done

  if [ -z "$PYTHON_BIN" ]; then
    fail 10 "Python >= 3.11 is required."
  fi

  green "Python: $PYTHON_BIN ($PYTHON_VERSION)"
}

choose_backend() {
  if command -v uv >/dev/null 2>&1; then
    INSTALL_BACKEND="uv_tool"
  elif command -v pipx >/dev/null 2>&1; then
    INSTALL_BACKEND="pipx"
  else
    INSTALL_BACKEND="venv"
  fi
  green "Install backend: $INSTALL_BACKEND"
}

install_cli() {
  step "Installing ROSClaw CLI"

  if dry_run_notice "install_cli"; then
    return 0
  fi

  mkdir -p "$ROSCLAW_HOME/bin" "$ROSCLAW_HOME/logs" "$ROSCLAW_HOME/cache"

  case "$INSTALL_BACKEND" in
    uv_tool)
      if [ "$ROSCLAW_CHANNEL" = "dev" ]; then
        uv tool install --force "git+https://github.com/ros-claw/rosclaw.git" || fail 20 "uv tool install failed"
      else
        uv tool install --force "$ROSCLAW_PIP_SPEC" || fail 20 "uv tool install failed"
      fi
      ;;
    pipx)
      if [ "$ROSCLAW_CHANNEL" = "dev" ]; then
        pipx install --force "git+https://github.com/ros-claw/rosclaw.git" || fail 20 "pipx install failed"
      else
        pipx install --force "$ROSCLAW_PIP_SPEC" || fail 20 "pipx install failed"
      fi
      ;;
    venv)
      "$PYTHON_BIN" -m venv "$ROSCLAW_HOME/venv" || fail 20 "venv creation failed"
      # shellcheck disable=SC1091
      . "$ROSCLAW_HOME/venv/bin/activate"
      python -m pip install --upgrade pip wheel setuptools || fail 21 "pip bootstrap failed (PEP668?)"
      if [ "$ROSCLAW_CHANNEL" = "dev" ]; then
        python -m pip install "git+https://github.com/ros-claw/rosclaw.git" || fail 20 "pip install failed"
      else
        python -m pip install "$ROSCLAW_PIP_SPEC" || fail 20 "pip install failed"
      fi

      cat > "$ROSCLAW_HOME/bin/rosclaw" <<EOF || fail 30 "Cannot write PATH shim"
#!/usr/bin/env bash
exec "$ROSCLAW_HOME/venv/bin/rosclaw" "\$@"
EOF
      chmod +x "$ROSCLAW_HOME/bin/rosclaw"
      ;;
  esac

  if ! command -v rosclaw >/dev/null 2>&1 && [ ! -x "$ROSCLAW_HOME/bin/rosclaw" ]; then
    fail 20 "rosclaw command not available after install"
  fi
}

ensure_path() {
  step "Checking PATH"

  if dry_run_notice "ensure_path"; then
    return 0
  fi

  if command -v rosclaw >/dev/null 2>&1; then
    green "rosclaw found in PATH: $(command -v rosclaw)"
    return 0
  fi

  if [ -x "$ROSCLAW_HOME/bin/rosclaw" ]; then
    export PATH="$ROSCLAW_HOME/bin:$PATH"
  fi

  if command -v rosclaw >/dev/null 2>&1; then
    green "rosclaw found after PATH update: $(command -v rosclaw)"
    return 0
  fi

  yellow "ROSClaw was installed, but rosclaw is not currently in PATH."
  echo
  echo "Add this to your shell profile:"
  echo "  export PATH=\"$ROSCLAW_HOME/bin:\$PATH\""
  echo
  echo "Then run:"
  echo "  rosclaw firstboot"
  echo
}

init_minimal_workspace() {
  step "Creating minimal workspace: $ROSCLAW_HOME"

  if dry_run_notice "init_minimal_workspace"; then
    return 0
  fi

  mkdir -p \
    "$ROSCLAW_HOME/config" \
    "$ROSCLAW_HOME/logs" \
    "$ROSCLAW_HOME/cache" \
    "$ROSCLAW_HOME/state" || fail 30 "Cannot create workspace"

  INSTALL_ID_FILE="$ROSCLAW_HOME/state/install_id"
  if [ ! -f "$INSTALL_ID_FILE" ]; then
    if command -v uuidgen >/dev/null 2>&1; then
      uuidgen > "$INSTALL_ID_FILE" || fail 30 "Cannot write install_id"
    else
      "$PYTHON_BIN" - <<'PY' > "$INSTALL_ID_FILE" || fail 30 "Cannot write install_id"
import uuid
print(uuid.uuid4())
PY
    fi
  fi

  cat > "$ROSCLAW_HOME/state/install.json" <<EOF || fail 30 "Cannot write install.json"
{
  "schema_version": "1.0",
  "install_id": "$(cat "$INSTALL_ID_FILE")",
  "installed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "installer_version": "1.0.0",
  "install_backend": "$INSTALL_BACKEND",
  "install_channel": "$ROSCLAW_CHANNEL",
  "platform": {
    "os": "$OS",
    "arch": "$ARCH",
    "is_wsl": $IS_WSL
  },
  "python": {
    "path": "$PYTHON_BIN",
    "version": "$PYTHON_VERSION"
  },
  "firstboot_completed": false,
  "last_doctor_status": "pending"
}
EOF
}

run_bootstrap_doctor() {
  step "Running bootstrap health check"

  if ! command -v rosclaw >/dev/null 2>&1; then
    yellow "Skipping doctor because rosclaw is not currently in PATH."
    return 0
  fi

  if dry_run_notice "rosclaw doctor --bootstrap"; then
    return 0
  fi

  if rosclaw doctor --bootstrap; then
    green "Bootstrap doctor passed"
  else
    yellow "Bootstrap doctor found issues. You can continue with:"
    echo "  rosclaw doctor --fix"
  fi
}

print_success() {
  echo
  green "══════════════════════════════════════════════════════════════"
  green " ROSClaw CLI installed successfully"
  green "══════════════════════════════════════════════════════════════"
  echo
  echo "Workspace:"
  echo "  $ROSCLAW_HOME"
  echo
  echo "Next step:"
  echo "  rosclaw firstboot"
  echo
  echo "Useful commands:"
  echo "  rosclaw doctor"
  echo "  rosclaw status"
  echo
}

main() {
  step "ROSClaw First Boot Bootstrapper"

  if [ "$DRY_RUN" = "1" ]; then
    yellow "[DRY RUN] No changes will be made."
  fi

  detect_platform
  find_python
  choose_backend
  install_cli
  init_minimal_workspace
  ensure_path
  run_bootstrap_doctor
  print_success
}

main "$@"
