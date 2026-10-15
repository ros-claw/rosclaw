# Preserve unavailable reads during an actual socket fault

The final-source 6f229d60 matrix independently passed its first twelve Worlds,
including all three normal seeds and canonical Action Lease expiry. In the
rosbridge-disconnect World the actual four original sockets were closed, the
independent actor verified stable stop, and old action/lease requests were
refused. A canonical get_action_status read timed out at the existing 0.5 second
client bound. The capture owner immediately entered cleanup, so the original
terminal receipt was not collected. That World remains FAILED.

The capture loop now preserves a temporary unavailable read as UNKNOWN and
continues read-only polling only after verifying the original socket fault's
policy/run/Body/daemon/PID binding, all four actual closed sockets, no new
connection and no physics-server signal. Only the specific daemon-unavailable
timeout can enter this branch. At most 32 original read failures are recorded.
Every retry remains inside the original goal observation deadline; that deadline
is checked before and after each status read. No RPC, action, lease, source or
proof timeout is extended. Other errors still terminate the original capture.

This change fabricates no action state, stop result or receipt. Acceptance still
requires the real canonical terminal receipt, independent original physics and
actual old-action/old-lease refusal. The 11 pure software contracts verify the
fault bindings and rejection of unrelated or unconfirmed faults. Actual new
socket-fault and full final-source/main Worlds are still required.

The first helper incorrectly matched RuntimeError although the actual MCP
adapter emits MCPError. Source f24aa693 again failed to drain its receipt and is
preserved. The branch now requires the real structured MCPError type, exact
DAEMON_UNAVAILABLE code, get_action_status operation and original timeout text,
in addition to all original fault bindings and the unchanged absolute deadline.
Tests use that actual exception class and reject other codes and operations.

A direct contract test passes the actual DaemonUnavailableError through the
real RuntimeClient error adapter and then through the fault-context reviewer;
the propagated code is DAEMON_UNAVAILABLE, not the adapter fallback label.
