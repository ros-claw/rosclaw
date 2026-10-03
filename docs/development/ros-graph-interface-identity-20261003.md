# ROS graph interface identity

An actual native Kimi ROS 2 benchmark supplied topic and service entries with
`type`, which ROSClaw's public discovery summaries also emit. The old graph
reader discarded those declarations, and compilation returned success with
empty interface types. The original benchmark graph, manifest and results
remain unchanged.

The graph reader now accepts `type` as an explicit alias for `msg_type`,
`srv_type` and `action_type`. Conflicting declarations are rejected. The
compiler refuses missing, empty, whitespace-padded or non-string types,
including snapshots constructed directly in Python. The CLI returns an error
and does not write a success manifest for an untyped graph.

The 22 new cases reproduced the old behavior and passed after the repair.
The connector unit cohort passed 190 tests, with 10 integration tests excluded
and one existing dependency deprecation warning. These tests prove graph
identity and compilation behavior; they do not prove a live rosbridge transport
or physical execution. An offline/DDS fixture remains execution-ineligible.
