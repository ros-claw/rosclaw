# ADR 0020: terminal passive SIM source continuity

Status: groundwork; not wired to execution.

A successful point-in-time G1/L1 review cannot establish that the same World,
entity and collision/sensor inventory persist during later navigation. Add a
pure SDK helper bound to the operator's separately SHA-pinned original ECM sample,
run, World/model identities and an at-most-500ms monotonic freshness budget.
The operator must first independently accept source binding and fresh L1.
This helper does not validate that initial binding, grant a lease, authorize an
action or replace fresh ROS/navigation checks.

Repeated reads of an identical fresh sample are allowed. Each new sample must
strictly advance simulator time, iterations and observer counters. World/model
and member entity IDs, collision geometry/local poses, sensor types and static
World poses must remain unchanged. The navigating robot's root pose may change.
Malformed, incomplete, paused/reset, stale/future, replaced or changed source
permanently trips this guard; later healthy-looking data cannot silently rearm it.
A new guard requires a new independent source/L1 review.

The existing generic SDK witness remains source-only and no lease, velocity
forwarding, MCP, daemon or Runtime path is changed. Before enabling L2, integrate
this evidence with the existing leased sink, independent current G1/L1 admission,
measured pose/contact/speed/stop checks and terminal fault behavior. Validate real
entity replacement/reset/observer-loss faults in new Worlds and preserve the
canonical MCP request_action to independent rosclawd boundary. This ADR is not
physical navigation acceptance.
