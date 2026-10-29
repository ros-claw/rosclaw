# Daemon-owned robot-state observations

`get_robot_state` reads `observation.robot_state` through the authenticated
rosclawd Unix socket. The Agent-side MCP facade does not initialize a local
Runtime, open a collector, start ROS/DDS, or substitute fixture observations.
The RPC accepts no caller-selected collector, topic, Body, or configuration.

The daemon projects only its already-configured Sense instance. An absent Sense
returns `SENSE_NOT_CONFIGURED`; synthetic mock, unconfigured, empty replay,
unknown and stub sources remain unavailable. A source failure is
`SENSE_READ_FAILED`. Older daemons without the RPC remain unavailable; there is
no local Runtime fallback. Explicit MCP fixture mode remains synthetic and
requires no daemon.

A genuine recorded source stays `REPLAY` / `RECORDED`, preserving its original
state timestamp. A readable current source is `UNKNOWN` execution mode with
`OBSERVED` or `STALE` trust. Source labels, transport freshness and raw
self-reported binding flags do not establish an immutable Body, physical evidence
domain, action readiness, or REAL authority. Every response has
`usable_for_real_execution=false`, an observation owner and daemon generation.
This endpoint neither enrolls a sensor nor enables Sense when it is disabled.

Software acceptance includes the full authenticated UDS path from MCP through
DaemonClient/server/service, original source and time preservation, unavailable
and replay semantics, changed values without a local cache, malformed response
rejection, and forbidden Runtime construction during reads. The fixture is
same-UID software evidence; it is not ROS, physical, process-separation or
cross-UID acceptance. Existing physical-action admission is unchanged.

This change covers the robot-state observation route. Other legacy
Memory/Practice/Skill adapters have their own local data lifecycle; this contract
does not claim a complete migration of every adapter to daemon-owned storage.

An installation-only MCP quick probe may report a missing daemon or disabled
Sense as a visible readiness limitation, provided the get_robot_state envelope
explicitly says UNKNOWN / UNAVAILABLE and grants no REAL use. This does not
qualify a robot or a running daemon. The same errors in product-demo or other
operations, malformed provenance, and an older daemon lacking the method remain
probe failures. Full daemon and physical qualification remain separate gates.
