#!/usr/bin/env bash
# Shared config + instance-identity helpers for the SeekDB 1.4.0 tooling.
#
# PR-SDB-140-2 created these scripts; PR-SDB-140-5 adds INSTANCE IDENTITY:
# neither "port answers" nor "pid file pid is alive" proves which engine is
# on the other end (the launcher re-execs; seekdb binds SO_REUSEPORT so two
# engines CAN share one port).  Every lifecycle decision must verify
# /proc/<pid>/cmdline (binary + base-dir + data-dir + port) against the
# recorded runtime.json, and refuse to guess when ambiguous.
# shellcheck shell=bash

# The ONLY engine version this tooling installs.  Never "latest".
SEEKDB_ENGINE_VERSION="1.4.0"
SEEKDB_RELEASE_TAG="v1.4.0"
SEEKDB_BUILD="100000212026082616"

# Version string the engine must report over SQL (SELECT VERSION()).
SEEKDB_VERSION_MARK="seekdb-v1.4.0"

# ---------------------------------------------------------------------------
# Platform support matrix (PR-SDB-140-5, P0-6): TRUTHFUL support levels.
#   supported     = checksum pinned + install proven + doctor proven
#   experimental  = asset exists upstream but not qualified by ROSClaw
# Anything not listed here is unsupported.
# ---------------------------------------------------------------------------
SEEKDB_SUPPORTED_PLATFORMS="ubuntu24.04/amd64 ubuntu24.04/arm64"
SEEKDB_EXPERIMENTAL_PLATFORMS="ubuntu22.04/amd64 ubuntu22.04/arm64 debian12/amd64 debian12/arm64 debian13/amd64 debian13/arm64"

seekdb_support_level() {
    # prints supported | experimental | unsupported for "<os_id> <arch>"
    local os_id="$1" arch="$2" pair
    case "$arch" in
        x86_64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) echo "unsupported"; return 1 ;;
    esac
    pair="$os_id/$arch"
    for p in $SEEKDB_SUPPORTED_PLATFORMS; do
        [ "$p" = "$pair" ] && { echo "supported"; return 0; }
    done
    for p in $SEEKDB_EXPERIMENTAL_PLATFORMS; do
        [ "$p" = "$pair" ] && { echo "experimental"; return 0; }
    done
    echo "unsupported"
    return 1
}

# Per-(os,arch) asset names from the v1.4.0 GitHub release.  Emits the name
# for supported AND experimental platforms; the installer decides whether an
# experimental platform may proceed (only with an explicit opt-in).
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
            echo "unsupported os: $os_id (supported: ubuntu24.04; experimental: ubuntu22.04, debian12/13)" >&2
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
SEEKDB_RUNTIME_JSON="${SEEKDB_RUNTIME_JSON:-$SEEKDB_HOME/runtime.json}"
SEEKDB_BIN="${SEEKDB_BIN:-$(command -v seekdb || echo /usr/bin/seekdb)}"

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

# ---------------------------------------------------------------------------
# Instance identity helpers
# ---------------------------------------------------------------------------

# All live seekdb engine/launcher pids (process name match; both the wrapper
# and the re-exec'd engine carry comm=seekdb).
seekdb_all_pids() {
    pgrep -x seekdb 2>/dev/null || true
}

# Echoes "<pid>" lines for seekdb processes whose cmdline carries this port
# (both "--port 2881" and "--port=2881" forms).
seekdb_pids_for_port() {
    local port="$1" pid args a i
    for pid in $(seekdb_all_pids); do
        [ -r "/proc/$pid/cmdline" ] || continue
        mapfile -d '' -t args < "/proc/$pid/cmdline" 2>/dev/null || continue
        for i in "${!args[@]}"; do
            a="${args[$i]}"
            if [ "$a" = "--port" ] && [ "${args[$((i + 1))]:-}" = "$port" ]; then
                echo "$pid"; break
            fi
            case "$a" in
                --port="$port"|--port="$port "*) echo "$pid"; break ;;
            esac
            case "$a" in
                -P) [ "${args[$((i + 1))]:-}" = "$port" ] && { echo "$pid"; break; } ;;
                -P"$port") echo "$pid"; break ;;
            esac
        done
    done
}

# Verify a pid's /proc cmdline carries EXACTLY our binary/base-dir/data-dir/
# port.  Prints nothing; returns 0 on full match.
seekdb_pid_identity_ok() {
    local pid="$1" base_dir="$2" data_dir="$3" port="$4"
    [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] || return 1
    local args a i have_base=1 have_data=1 have_port=1
    mapfile -d '' -t args < "/proc/$pid/cmdline" 2>/dev/null || return 1
    # binary: argv[0] must BE the seekdb binary (no wrapper scripts)
    [ "${args[0]:-}" = "$SEEKDB_BIN" ] || return 1
    for i in "${!args[@]}"; do
        a="${args[$i]}"
        case "$a" in
            --base-dir) [ "${args[$((i + 1))]:-}" = "$base_dir" ] && have_base=0 ;;
            --base-dir=*) [ "${a#--base-dir=}" = "$base_dir" ] && have_base=0 ;;
            --data-dir) [ "${args[$((i + 1))]:-}" = "$data_dir" ] && have_data=0 ;;
            --data-dir=*) [ "${a#--data-dir=}" = "$data_dir" ] && have_data=0 ;;
            --port) [ "${args[$((i + 1))]:-}" = "$port" ] && have_port=0 ;;
            --port=*) [ "${a#--port=}" = "$port" ] && have_port=0 ;;
            -P) [ "${args[$((i + 1))]:-}" = "$port" ] && have_port=0 ;;
            -P*) [ "${a#-P}" = "$port" ] && have_port=0 ;;
        esac
    done
    [ "$have_base" = 0 ] && [ "$have_data" = 0 ] && [ "$have_port" = 0 ]
}

# /proc start-time (field 22) — PID-reuse guard: a reused pid fails this.
seekdb_pid_starttime() {
    local pid="$1"
    awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true
}

# /proc ppid (field 4).  The engine daemonizes (reparented to 1); the
# launcher's ppid is the calling shell.  This is how we tell the REAL engine
# from the dying launcher when both carry identical cmdlines.
seekdb_pid_ppid() {
    local pid="$1"
    awk '{print $4}' "/proc/$pid/stat" 2>/dev/null || true
}

# Find the REAL engine pid for our instance: identity-matched AND daemonized
# (ppid 1).  Echoes the pid; returns 1 when none.
seekdb_find_engine_pid() {
    local base_dir="$1" data_dir="$2" port="$3" pid fallback=""
    for pid in $(seekdb_pids_for_port "$port"); do
        seekdb_pid_identity_ok "$pid" "$base_dir" "$data_dir" "$port" || continue
        if [ "$(seekdb_pid_ppid "$pid")" = "1" ]; then
            echo "$pid"
            return 0
        fi
        fallback="$pid"
    done
    if [ -n "$fallback" ]; then
        echo "$fallback"
        return 0
    fi
    return 1
}

# runtime.json I/O (python for robust JSON; python3 is always present).
seekdb_runtime_read() {
    [ -f "$SEEKDB_RUNTIME_JSON" ] || return 1
    python3 - "$SEEKDB_RUNTIME_JSON" "$1" <<'PY' 2>/dev/null
import json, sys
try:
    print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
except Exception:
    sys.exit(1)
PY
}

seekdb_runtime_write() {
    # seekdb_runtime_write <pid> <start_time_stat> <started_at_iso>
    python3 - "$SEEKDB_RUNTIME_JSON" "$1" "$2" "$3" \
        "$SEEKDB_ENGINE_VERSION" "$SEEKDB_BIN" "$SEEKDB_HOME" "$SEEKDB_DATA_DIR" "$SEEKDB_PORT" <<'PY'
import json, os, sys
(path, pid, pstart, started, engine, binary, base, data, port) = sys.argv[1:10]
payload = {
    "schema_version": 1,
    "engine_version": engine,
    "pid": int(pid),
    "process_start_time": pstart,
    "binary": binary,
    "base_dir": base,
    "data_dir": data,
    "port": int(port),
    "started_at": started,
}
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(payload, fh, indent=2)
os.replace(tmp, path)
PY
}

# Count live seekdb processes claiming our port (collision detection).
seekdb_port_collision_count() {
    seekdb_pids_for_port "$1" | wc -l
}

# Which local address is our port bound to? prints e.g. "127.0.0.1" /
# "0.0.0.0" / "closed".  Uses /proc/net/tcp{,6} so no ss/netstat dependency.
seekdb_port_bind_addr() {
    python3 - "$1" <<'PY'
import sys
port = int(sys.argv[1])
target = f"{port:04X}"
addrs = []
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(path) as fh:
            next(fh)
            for line in fh:
                parts = line.split()
                if parts[1].endswith(":" + target) and parts[3] == "0A":  # LISTEN
                    addrs.append(parts[1].split(":")[0])
    except FileNotFoundError:
        pass
def decode(h):
    if set(h) <= {"0"}:
        return "::" if len(h) > 8 else "0.0.0.0"
    if len(h) == 8:
        b = bytes.fromhex(h)
        return ".".join(str(x) for x in reversed(b))
    return h
seen = sorted({decode(a) for a in addrs})
print(",".join(seen) if seen else "closed")
PY
}
