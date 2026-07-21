# rosclawd Control Plane

`rosclawd` is the experimental local daemon boundary for ROSClaw physical
actions. CLI, MCP, SDK, and Agent processes are northbound clients. The daemon
owns the action queue, daemon-issued permits, resource leases, driver/executor
registration, emergency-stop latch, execution receipts, and authenticated
restart ledger.

```text
Codex / Claude Code / OpenClaw / operator CLI
                     |
              MCP / DaemonClient
                     |
      protected AF_UNIX socket + SO_PEERCRED
                     |
                  rosclawd
                     |
 Body -> Permit -> Policy -> Lease -> Executor -> ACK -> Verify -> Receipt
```

## Current Boundary

The current implementation provides:

- a versioned, length-bounded Unix-socket protocol;
- a fixed upper bound on concurrent client connections and per-request
  timeouts, preventing an Agent from creating unbounded daemon threads;
- kernel-authenticated peer PID/UID/GID using `SO_PEERCRED`;
- an exact RPC allowlist, with no arbitrary import, ROS publish, serial write,
  driver registration, or executor registration operation;
- a bounded action queue and idempotent action IDs;
- daemon-owned, peer/body/snapshot/capability/action-intent-bound, expiring
  permits with no wildcard capability;
- rejection of caller-forged `authorization.approved=true`;
- emergency stop outside the normal action queue;
- truthful queued-action cancellation and terminal receipts;
- an append-only SQLite event ledger with an HMAC forward chain, a separately
  signed head, and daemon-private file permissions;
- durable permit registration/consumption and action/receipt replay protection
  across process restarts;
- bounded runtime action retention backed by lazy durable-history lookup;
- restart recovery that cancels undispatched work and treats interrupted REAL
  outcomes as unknown, requests E-Stop, and requires operator review;
- refusal to replace files, symlinks, live sockets, or unresponsive sockets;
- MCP action and emergency tools that call the daemon instead of constructing a
  physical Runtime in the Agent process.

An unconfigured daemon loads no hardware pack or REAL executor. A configured,
signed Robot Pack may be loaded only on the daemon side; the current built-in
RealSense path is perception-only and no production actuator Pack is claimed.
REAL actions therefore fail closed unless a trusted daemon-side integration
validates its Body policy, calibration, mapping, and action limits before it
registers both an executor and an exact permit. Permit issuance, Pack loading,
and executor registration are not exposed to Agent RPC.

## Development Mode

Run the daemon in one terminal:

```bash
export ROSCLAW_HOME=/tmp/rosclaw-daemon-dev
rosclawd --robot-id sim_ur5e --log-level INFO
```

Inspect it from another process:

```bash
rosclaw daemon status --json
rosclaw daemon emergency-stop --reason "operator test" --json
rosclaw daemon stop --json
```

This proves process separation, protocol behavior, and fail-closed dispatch.
When both processes use the same Unix UID, it is **not** a privilege boundary
and must not be treated as a non-bypassable real-hardware deployment.

## Durable Ledger and Restart Recovery

By default, rosclawd creates these daemon-private files:

```text
$ROSCLAW_HOME/state/daemon/ledger.sqlite3
$ROSCLAW_HOME/state/daemon/ledger.sqlite3.anchor
$ROSCLAW_HOME/state/daemon/ledger.key
```

The directories must be owned by the daemon UID and deny group/world access;
the files must be regular, non-symlink files with mode `0600`. Paths can be
overridden with `--ledger` / `ROSCLAW_DAEMON_LEDGER` and `--ledger-key` /
`ROSCLAW_DAEMON_LEDGER_KEY`.

The ledger records permit registration and consumption before REAL dispatch,
action submission before scheduling, action start before authorization and
dispatch, and the terminal receipt. Startup verifies SQLite integrity, every
event HMAC, the forward chain, and the signed head before the daemon loads a
Robot Pack or opens its socket. A failed or uncertain ledger write locks out
new actions for that process.

On restart:

- a previously terminal action and receipt remain queryable and action IDs stay
  immutable;
- a queued action is sealed `CANCELLED` without claiming dispatch;
- an interrupted non-REAL action is sealed `FAILED`;
- an interrupted REAL action is sealed `FAILED` with physical outcome unknown,
  E-Stop is requested, and new REAL work is blocked;
- pending REAL recovery requests E-Stop again on every restart;
- pending review action IDs are cumulative across repeated recovery attempts
  and are removed only by one acknowledgement covering the complete set.

After reviewing the retained action, receipt, robot state, and external
evidence, an operator may persist the review **as the rosclawd service UID**:

```bash
sudo -u rosclaw-hw rosclaw daemon acknowledge-recovery \
  --reason "reviewed interrupted action and physical state" --json
```

Acknowledgement removes the recovery gate only. It does not rewrite the unknown
receipt, clear the Runtime E-Stop latch, or prove a physical stop.

This is a machine-local integrity boundary, not an external audit witness. A
root or daemon-state owner that can read the HMAC key can forge history, and an
attacker that replaces the database and signed anchor together can roll both
back. TPM-backed keys, monotonic counters, remote transparency logs, ledger
compaction/materialized snapshots, and long-history startup bounds remain
future work. Keep independent hardware/controller logs for production evidence.

## Production User Boundary

The reference unit is
[`deploy/systemd/rosclawd.service`](../deploy/systemd/rosclawd.service). It
assumes:

- service user: `rosclaw-hw`;
- client group: `rosclaw-agent`;
- agent users are members of `rosclaw-agent`, but not `dialout`;
- only `rosclaw-hw` receives device groups and vendor credentials;
- socket: `/run/rosclaw/rosclawd.sock`, owner `rosclaw-hw`, group
  `rosclaw-agent`, mode `0660`;
- runtime directory mode `0750`, never writable by the client group.
- state directory mode `0700` and service umask `0077`, so daemon evidence and
  credentials are not readable by the client group.

Example host setup, to be reviewed by the robot operator:

```bash
sudo groupadd --system rosclaw-agent
sudo useradd --system --home-dir /var/lib/rosclaw \
  --gid rosclaw-agent --shell /usr/sbin/nologin rosclaw-hw
sudo usermod --append --groups rosclaw-agent AGENT_USER
sudo install -D -m 0644 deploy/systemd/rosclawd.service \
  /etc/systemd/system/rosclawd.service
sudo systemctl daemon-reload
sudo systemctl enable --now rosclawd

# Configure these in the Agent/MCP service environment, not in a prompt.
export ROSCLAW_DAEMON_SOCKET=/run/rosclaw/rosclawd.sock
export ROSCLAW_DAEMON_UID="$(id -u rosclaw-hw)"
rosclaw daemon security-check --json
```

The service starts with `DevicePolicy=closed`; it cannot touch a robot until an
operator allowlists the exact device. For example:

```ini
# /etc/systemd/system/rosclawd.service.d/rh56-device.conf
[Service]
DeviceAllow=/dev/ttyUSB0 rw
```

Apply a stable `/dev/serial/by-id/...` binding in the robot configuration even
when the cgroup rule must name the resolved kernel device. Do not use a broad
`DevicePolicy=auto` override merely to make startup pass.

Store vendor credentials in `/etc/rosclaw/rosclawd.env`, readable only by root
and `rosclaw-hw`. Do not put them in project files, `.mcp.json`, Agent
environment, prompts, or receipts.

## ROS 2 and DDS

A Unix socket does not stop an Agent user from publishing directly to a
network-visible ROS 2 graph. A non-bypassable ROS deployment also requires:

1. a dedicated SROS2 enclave for `rosclawd`;
2. `ROS_SECURITY_ENABLE=true`;
3. `ROS_SECURITY_STRATEGY=Enforce`;
4. a signed, default-deny permissions policy that grants command-topic and
   command-service writes only to the daemon enclave;
5. an Agent enclave that permits only the required read/discovery surface;
6. CA private keys and daemon enclave private keys outside the Agent account.

ROS 2 documents signed topic access controls in its
[official access-control tutorial](https://docs.ros.org/en/ros2_documentation/kilted/Tutorials/Advanced/Security/Access-Controls.html).
Without enforced DDS permissions or equivalent host/network isolation,
`rosclawd` is a preferred path, not a complete anti-bypass boundary.

## Verification

Run:

```bash
scripts/acceptance/daemon_blackbox.sh
rosclaw daemon security-check --json

# These require ROSClaw installed from a wheel outside a protected user home.
ROSCLAW_ACCEPTANCE_PYTHON=/opt/rosclaw/bin/python \
  scripts/acceptance/daemon_cross_uid.sh
ROSCLAW_ACCEPTANCE_PYTHON=/opt/rosclaw/bin/python \
  scripts/acceptance/daemon_systemd.sh
```

`security-check` returns success only when:

- daemon and client PIDs differ;
- daemon and client UIDs differ;
- `ROSCLAW_DAEMON_UID` pins the kernel-authenticated daemon peer;
- the daemon owns the socket and its protected runtime directory;
- the Agent reaches the socket through the configured client group, while the
  socket and directory remain non-world-accessible and non-client-writable;
- the durable ledger is healthy and its database, anchor, and key are not
  readable or writable by the Agent UID;
- the Agent is not in `dialout`;
- the Agent cannot read/write detected serial devices.

`daemon_cross_uid.sh` uses two existing low-privilege Linux accounts, rejects a
source-tree import, verifies daemon-only RPC denial and forged REAL denial, and
restarts the daemon to prove durable ownership and Receipt recovery.
`daemon_systemd.sh` additionally launches a transient service with the reference
`DevicePolicy`, directory modes, empty capability set, and filesystem hardening.
Neither script creates users or accesses hardware. They do not prove a site's
SROS2 policy, vendor-credential isolation, exact device/CAN ACLs, network
firewalls, or physical E-stop; those remain deployment acceptance items.

The daemon protocol relies on Linux pathname socket permissions and peer
credentials described by
[`unix(7)`](https://man7.org/linux/man-pages/man7/unix.7.html).

## Operational Semantics

- `request_action` returning `ok=true` means rosclawd processed the request.
  Read the embedded receipt; it may correctly be `BLOCKED` or `FAILED`.
- Agent-facing MCP requests default to `SHADOW` when `execution_mode` is
  omitted. REAL must always be explicit and independently permitted.
- MCP audit logs recursively redact permit IDs and credential-like fields and
  are written under a `0700` directory to a `0600` file.
- Action IDs are immutable. Reusing an ID with changed content is rejected, and
  status, receipt, and cancellation are restricted to the submitting peer UID
  or the daemon service UID.
- A REAL permit is rejected if the Agent changes its arguments, body snapshot,
  explicit capability, execution mode, deadline, expected effect, or
  verification policy.
- `cancel_action` cancels only work that has not started. It never claims an
  active robot stopped.
- `emergency_stop` bypasses the action queue, but software ACK is not physical
  stop evidence and does not replace a certified E-stop. The daemon and ROS
  compatibility CLIs return a nonzero status unless physical stop observation
  is present.
- Stopping rosclawd latches E-stop, rejects new work, waits for active daemon
  work, stops Runtime, and removes only the socket inode it created.
- Permit use counts, immutable action IDs, transitions, and terminal receipts
  survive restart in the authenticated ledger. Runtime memory is bounded and
  may evict terminal records; older records remain queryable from the ledger.
- Startup currently verifies and replays the complete ledger. Operators must
  monitor ledger growth; automatic compaction and archival are not implemented.
- Running without a durable ledger remains supported only for direct component
  tests and embedded development. In that mode REAL history is retained in
  memory and fails closed with `ACTION_HISTORY_FULL`; it is not restart-safe.

## MCP Tools

The Agent surface adds four daemon-backed tools:

| Tool | Meaning |
|---|---|
| `get_runtime_status` | Daemon identity, queue, drivers, executors, permits, and latch |
| `request_action` | Submit SHADOW/REAL intent; REAL requires an exact daemon-issued permit |
| `get_action_status` | Read queue state and terminal receipt |
| `cancel_action` | Cancel queued work only |

`emergency_stop` is also daemon-backed. No MCP tool can register a driver,
register an executor, issue a permit, publish an arbitrary topic, or write raw
device bytes.
