#!/usr/bin/env bash
# Shared config for the SeekDB 1.4.0 tooling (PR-SDB-140-2).
# shellcheck shell=bash

# The ONLY engine version this tooling installs.  Never "latest".
SEEKDB_ENGINE_VERSION="1.4.0"
SEEKDB_RELEASE_TAG="v1.4.0"
SEEKDB_BUILD="100000212026082616"

# Per-(os,arch) asset names from the v1.4.0 GitHub release.
seekdb_asset_name() {
    local os_id="$1" arch="$2"
    case "$arch" in
        x86_64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) echo "unsupported arch: $arch" >&2; return 1 ;;
    esac
    case "$os_id" in
        ubuntu22.04|ubuntu24.04|debian12|debian13)
            echo "seekdb_${SEEKDB_ENGINE_VERSION}-${SEEKDB_BUILD}${os_id}_${arch}.deb"
            ;;
        *)
            echo "unsupported os: $os_id (supported: ubuntu22.04/24.04, debian12/13)" >&2
            return 1
            ;;
    esac
}

seekdb_asset_url() {
    local asset="$1"
    echo "https://github.com/oceanbase/seekdb/releases/download/${SEEKDB_RELEASE_TAG}/${asset}"
}

# Runtime defaults.
SEEKDB_HOME="${SEEKDB_HOME:-$HOME/.rosclaw/seekdb-1.4}"
SEEKDB_PORT="${SEEKDB_PORT:-2881}"
SEEKDB_DATA_DIR="${SEEKDB_DATA_DIR:-$SEEKDB_HOME/data}"
SEEKDB_PID_FILE="${SEEKDB_PID_FILE:-$SEEKDB_HOME/seekdb.pid}"
SEEKDB_LOG_DIR="${SEEKDB_LOG_DIR:-$SEEKDB_HOME/log}"

seekdb_detect_platform() {
    # prints "<os_id> <arch>" for asset selection
    local os_id arch
    arch="$(uname -m)"
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case "$ID" in
            ubuntu) os_id="ubuntu${VERSION_ID}" ;;
            debian) os_id="debian${VERSION_ID%%.*}" ;;
            *) os_id="" ;;
        esac
    fi
    echo "$os_id $arch"
}
