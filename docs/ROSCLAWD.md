# rosclawd Control Plane

`rosclawd` is the experimental local daemon boundary for ROSClaw physical
actions. CLI, MCP, SDK, and Agent processes are northbound clients. The daemon
owns the action queue, daemon-issued permits, resource leases, driver/executor
registration, emergency-stop latch, and execution receipts.

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

The Phase 2 implementation provides:

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
- bounded in-memory action retention that evicts only terminal non-REAL records;
- refusal to replace files, symlinks, live sockets, or unresponsive sockets;
- MCP action and emergency tools that call the daemon instead of constructing a
  physical Runtime in the Agent process.

The base daemon intentionally loads no hardware pack or REAL executor. REAL
actions therefore fail closed unless a trusted daemon-side integration
registers both the executor and permit. Permit issuance and hardware-pack
loading are not exposed to Agent RPC in this phase. A future REAL pack must
validate its body policy, calibration, mapping, and action limits on the daemon
side before registering a permit; the base daemon does not yet provide that
pack-specific validation.

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
```

`security-check` returns success only when:

- daemon and client PIDs differ;
- daemon and client UIDs differ;
- the socket is not world-writable;
- the Agent is not in `dialout`;
- the Agent cannot read/write detected serial devices.

It does not prove SROS2 policy, vendor-credential isolation, CAN ACLs, network
firewalls, or a physical E-stop. Those remain deployment acceptance items.

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
- Queue, retained action, and permit state are in-memory in this prototype and
  do not survive a daemon restart. Terminal non-REAL records are evicted at the
  configured retention limit. Terminal REAL records are never evicted during a
  process lifetime because doing so could make an action ID executable twice;
  a full REAL history fails closed with `ACTION_HISTORY_FULL` until an operator
  archives evidence and restarts the daemon.
- A production REAL deployment therefore requires a durable, integrity-checked
  permit-consumption and action-ID ledger plus an operator-reviewed restart
  recovery procedure. The current in-memory authority must not be used as
  production replay protection across daemon restarts.

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
