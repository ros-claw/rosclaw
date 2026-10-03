# Offline ROS graph endpoint binding

The 2026-10-03 parallel ROS2 observation wave found a concrete provenance gap:
an offline snapshot with `ros.endpoint=fixture://...` produced a compiled
manifest whose top-level endpoint falsely identified rosbridge on
`127.0.0.1:9090`. The compiler also replaced explicitly declared remote
rosbridge hosts and ports with that default.

The legacy capability provider already rejects non-dry-run inference and
requires the daemon action gateway, so this finding does not demonstrate
robot command execution. It does demonstrate unintended transport selection:
provider loading constructed its default rosbridge transport before loading a
static manifest, and later health/discovery could inspect that unrelated
endpoint. No socket, DDS domain or live ROS graph was accessed by this fix's
tests.

Graph compilation now preserves explicit ws/wss endpoints. Non-rosbridge or
absent endpoints become `transport=offline`, retain their source endpoint,
and declare `execution_eligible=false`. A saved DDS/fixture graph is discovery
evidence and has no implied rosbridge connection.

Static provider loading reads the manifest first. Offline metadata can be
loaded and inspected with no transport; health reports `ROS_METADATA_ONLY`.
Both new manifests and pre-fix manifests whose source endpoint is a fixture
remain blocked from transport creation, even if runtime settings contain a
different explicit rosbridge URL. Missing runtime endpoints never default to
localhost. A ws/wss manifest with an explicitly matching runtime endpoint
retains its transport path; silently retargeting it is rejected and requires
rediscovery. Dry runs remain mock-only. None of this grants real execution
authority or removes the existing daemon gateway requirement.

Seven meaningful regression cases failed against the original code, including
the two offline provider loads caught by a transport-constructor spy. Eleven
endpoint binding tests now cover offline provenance, legacy forged defaults,
absent endpoints, explicit remote host/port preservation, source/runtime
mismatch and a matching explicit static provider load. Broader connector
verification excludes the live integration suite.
