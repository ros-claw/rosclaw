#!/usr/bin/env bash
# Install seekdb Engine 1.4.0 from the pinned GitHub release asset.
#
# PR-SDB-140-2.  Rules from the outline:
#   * pin the exact 1.4.0 asset — never "latest";
#   * verify SHA256 before installing;
#   * never point a 1.4 install at a 1.3 data directory (no in-place
#     upgrade upstream — use scripts/seekdb/migrate_1_3_to_1_4.sh).
#
# Usage: scripts/seekdb/install_1_4.sh [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
. "$HERE/_common.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

read -r OS_ID ARCH <<<"$(seekdb_detect_platform)"
if [ -z "$OS_ID" ]; then
    echo "unsupported OS (need ubuntu22.04/24.04 or debian12/13)" >&2
    exit 1
fi

ASSET="$(seekdb_asset_name "$OS_ID" "$ARCH")"
URL="$(seekdb_asset_url "$ASSET")"
DEST_DIR="${SEEKDB_HOME}/packages"
DEST="$DEST_DIR/$ASSET"

# SHA256 recorded from the release assets (fill per platform after first
# verified download; empty = not yet recorded — refuse to install).
declare -A SHA256=(
    ["seekdb_1.4.0-100000212026082616ubuntu22.04_arm64.deb"]="d20bb4910e7eff5be9e147057e51c6ef5c6cafa9dce697602d03586d7c68bf78"
    ["seekdb_1.4.0-100000212026082616ubuntu22.04_amd64.deb"]="__FILL_ME__"
    ["seekdb_1.4.0-100000212026082616ubuntu24.04_arm64.deb"]="1239102f381b0f93b4d1c72bab5101e42b7ab805f12307d4b5a26a5a37af846f"
    ["seekdb_1.4.0-100000212026082616ubuntu24.04_amd64.deb"]="18f6b329d462a8fd841d43a12574cacce75f12b06156bdb0c7700d782478d5b1"
    ["seekdb_1.4.0-100000212026082616debian12_arm64.deb"]="__FILL_ME__"
    ["seekdb_1.4.0-100000212026082616debian12_amd64.deb"]="__FILL_ME__"
    ["seekdb_1.4.0-100000212026082616debian13_arm64.deb"]="__FILL_ME__"
    ["seekdb_1.4.0-100000212026082616debian13_amd64.deb"]="__FILL_ME__"
)

echo "seekdb engine: $SEEKDB_ENGINE_VERSION ($SEEKDB_RELEASE_TAG)"
echo "platform:      $OS_ID/$ARCH"
echo "asset:         $ASSET"
echo "url:           $URL"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would download + checksum + install"
    exit 0
fi

mkdir -p "$DEST_DIR"
if [ ! -f "$DEST" ]; then
    echo "downloading..."
    curl -fL --retry 3 --retry-delay 5 -o "$DEST" "$URL"
fi

EXPECTED="${SHA256[$ASSET]:-}"
if [ -z "$EXPECTED" ] || [ "$EXPECTED" = "__FILL_ME__" ]; then
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
echo "installed.  Verify with: scripts/seekdb/doctor.sh"
