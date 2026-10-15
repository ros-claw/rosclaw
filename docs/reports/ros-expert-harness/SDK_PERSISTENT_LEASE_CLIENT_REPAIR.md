# Isolated SDK persistent lease-service client

Source 0e127647 passed all five G4 before/after/rollback episodes in 15 fresh
Worlds, but its G3 seed303 yaw action stopped on a rosbridge renewal response
deadline. Its position action completed with independently verified stop. The
original failure remains failed; no physical or CI attempt is relabeled.

The exact installed upstream services.py SHA256 is
47fd035f751a90464a40a8dfebcc820d83ea5fdc33d88f6b46225d59dbd55ddd.
That implementation creates a new DDS service client on the ROS executor for
every call, waits for discovery, and destroys it after the response. Applying
that path to a 50 ms guarded heartbeat introduces repeated discovery and
executor scheduling. The separate cadence fix avoids an added 50 ms sleep but
does not remove this per-call client lifecycle.

The disposable governed SDK stack now runs the original ROS bridge server with
one narrowly scoped service-call wrapper. It retains a client only for the
manifest-declared current-World SetBool lease service. Every invocation checks
its actual live service type and readiness, sends a fresh actual ROS request,
waits for its own future, and returns the actual response, including refusal.
Responses are never cached. Other services use the unmodified upstream path.
The original client is created through upstream run_on_executor. SDK node
shutdown destroys the retained client along with the original ROS bridge node.

The wrapper refuses an unexpected upstream source hash, host execution,
non-localhost ROS domain policy, missing governed SDK role, changed run/service
or malformed SetBool arguments. It adds no ROS node, publisher, physical route,
lease deadline or execution authority. The SDK owner records client creation
and actual response timing in a bounded original audit. Existing canonical MCP,
rosclawd, immutable Body, proof, source cursor, source freshness and permanent
sink-latch checks are unchanged. Existing 150 ms guarded RPC bounds are retained.

Nine software tests cover single-client reuse with fresh responses, actual
refusals, upstream fallback, node scoping, malformed requests, changed types,
service loss and cancellation on response timeout. Actual new seed303/lease
Worlds and final-source/main matrices are still required before this repair is
accepted as physical evidence.
