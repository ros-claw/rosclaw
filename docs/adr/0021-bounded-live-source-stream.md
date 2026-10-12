# ADR 0021: bounded passive ECM stream supervision

Status: local implementation; actual SIM and integration gates pending.

G1/L1 remain stopped point-in-time evidence. The new SDK-only reader requires an
independently accepted complete segment-zero prefix, its separate SHA256, one
original observer PID/epoch, and bounded segment policy. It has no ROS imports,
Runtime, action client, publisher, admission or permit authority.

SourceWitness optionally emits bounded exclusive segments with SHA256 chaining,
strict cursor and independent random process epoch. Default single-file captures
retain their existing 600-sample/30-MB behavior. New streaming configuration is
explicit and closed: at most 64 segments, 600 samples/segment, 30 MB/segment and
120 MB total. No deletion, rotation overwrite or automatic rearming occurs.
OpenSSL Crypto is already present in the pinned SDK; no host install is needed.

Every live poll verifies original file identities, append-only consumed prefixes,
sealed segment metadata/hash links, exact sample sequence and bounded complete
lines. EOF partial lines are allowed only while the latest complete sample stays
fresh. No-new-sample, process/source drift, paused/reset World, inventory/ancestry
changes, gaps, corrupt lines and capacity exhaustion latch permanently. Freshness
uses monotonic wall time, at most 500 ms, independently of advancing SIM time.
Legitimate robot root motion and wheel joint motion preserve fixed local member
geometry and ancestry. Admission after segment-zero rollover is deliberately
unsupported in this first version: recapture is required, never history skipping.

`capacity_covers` only reports a conservative evidence budget under an externally
bound record-size/duration ceiling. Dispatch still requires fresh stopped G1/L1,
current ROS health, Body/run/daemon/session/lease policy and canonical MCP. The
reader does not evaluate current L1 or control a leased sink. L2 remains NOT_RUN.

The separate InventoryFault plugin is never loaded by default. It only removes
one actual owned lidar-kind component after a unique run/model/sensor trigger in
a disposable fixture. Other operator actors reset/pause or replace the owned
model. Observer-kill/stale target the in-process Gazebo observer; World loss/pause
is a disclosed collateral effect, not an isolated observer-causality experiment.
All seven planned live cases must retain original data and cleanup evidence.
