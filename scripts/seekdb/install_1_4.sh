#!/usr/bin/env bash
# Install seekdb Engine 1.4.0 from the pinned GitHub release asset.
#
# PR-SDB-140-2 created this; PR-SDB-140-5 (P0-6) makes the support matrix
# TRUTHFUL:
#   * supported   = Ubuntu 24.04 amd64/arm64 — checksum pinned, install +
#                   doctor proven on real hardware/CI.  No placeholders.
#   * experimental = asset exists upstream, checksum recorded where verified,
#                   but NOT qualified by ROSClaw.  Installs only with
#                   ROSCLAW_SEEKDB_ALLOW_EXPERIMENTAL=1.
#   * anything else = refused.
# Rules from the outline: pin the exact asset (never "latest"); verify
# SHA256 before installing; never point 1.4 at a 1.3 data dir.
#
# Usage: scripts/seekdb/install_1_4.sh [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

read -r OS_ID ARCH <<<"$(seekdb_detect_platform)"
LEVEL="$(seekdb_support_level "$OS_ID" "$ARCH")"
if [ "$LEVEL" = "unsupported" ]; then
    echo "unsupported platform: $OS_ID/$ARCH" >&2
    echo "  supported:    $SEEKDB_SUPPORTED_PLATFORMS" >&2
    echo "  experimental: $SEEKDB_EXPERIMENTAL_PLATFORMS (opt-in: ROSCLAW_SEEKDB_ALLOW_EXPERIMENTAL=1)" >&2
    exit 1
fi

ASSET="$(seekdb_asset_name "$OS_ID" "$ARCH")"
URL="$(seekdb_asset_url "$ASSET")"
DEST_DIR="${SEEKDB_HOME}/packages"
DEST="$DEST_DIR/$ASSET"

# SHA256 recorded from verified downloads.  SUPPORTED entries must always be
# real checksums — a placeholder here is a release blocker (P0-6 test
# enforces).  Experimental entries may be absent; absent = refuse even with
# the experimental opt-in.
declare -A SHA256=(
    ["seekdb_1.4.0-100000212026082616ubuntu24.04_amd64.deb"]="18f6b329d462a8fd841d43a12574cacce75f12b06156bdb0c7700d782478d5b1"
    ["seekdb_1.4.0-100000212026082616ubuntu24.04_arm64.deb"]="1239102f381b0f93b4d1c72bab5101e42b7ab805f12307d4b5a26a5a37af846f"
    # experimental (checksum verified by download, install NOT qualified):
    ["seekdb_1.4.0-100000212026082616ubuntu22.04_arm64.deb"]="d20bb4910e7eff5be9e147057e51c6ef5c6cafa9dce697602d03586d7c68bf78"
)

echo "seekdb engine: $SEEKDB_ENGINE_VERSION ($SEEKDB_RELEASE_TAG)"
echo "platform:      $OS_ID/$ARCH [$LEVEL]"
echo "asset:         $ASSET"
echo "url:           $URL"

if [ "$LEVEL" = "experimental" ] && [ "${ROSCLAW_SEEKDB_ALLOW_EXPERIMENTAL:-}" != "1" ]; then
    echo "REFUSING: $OS_ID/$ARCH is experimental (not ROSClaw-qualified)." >&2
    echo "  Use a supported platform, or opt in: ROSCLAW_SEEKDB_ALLOW_EXPERIMENTAL=1" >&2
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would download + checksum + install ($LEVEL)"
    exit 0
fi

mkdir -p "$DEST_DIR"
if [ ! -f "$DEST" ]; then
    echo "downloading..."
    curl -fL --retry 3 --retry-delay 5 -o "$DEST" "$URL"
fi

EXPECTED="${SHA256[$ASSET]:-}"
if [ -z "$EXPECTED" ]; then
    echo "no SHA256 recorded for $ASSET — refusing to install an unverified" >&2
    echo "artifact.  Record the checksum first (sha256sum $DEST)." >&2
    exit 2
fi
ACTUAL="$(sha256sum "$DEST" | awk '{print $1}')"
if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "SHA256 MISMATCH for $ASSET" >&2
    echo "  expected: $EXPECTED" >&2
    echo "  actual:   $ACTUAL" >&2
    exit 3
fi
echo "sha256 verified: $ACTUAL"

echo "installing (sudo dpkg -i)..."
sudo dpkg -i "$DEST"
echo "installed ($LEVEL platform).  Verify with: scripts/seekdb/doctor.sh"
