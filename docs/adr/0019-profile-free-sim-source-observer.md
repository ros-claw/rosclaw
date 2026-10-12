# ADR 0019: Profile-free SIM source observation inputs

Status: accepted for source-only implementation; no new L2 authority.

The existing SDK witness assumes a known profile, fixed topics and exactly two
wheel-contact streams. A new robot must not be admitted by adding another
profile or a model-name branch. Extend the same SDK witness with independently
pinned operator source inputs: World/entity/run/frame, ROS routes, contact
streams, drive message type, and source URDF/World SHA-256. Require exact closed
schema, immutable parsed fields, and unique typed route names.

Ground-contact exemptions must be exact scoped collision names derived from
static horizontal planes in the source SDF, never substring matching or an
arbitrary caller assertion that a wall is ground. At least one explicitly
selected live ground-contact stream witnesses the contact pipeline; there is
no fixed wheel count. Preserve complete original SDK contact messages in a
bounded stream for independent recomputation. A generic source observer uses
actual Gazebo contacts and does not claim the legacy room-specific geometric
collision proxy.

The generic mode is strictly source-only: no velocity publisher, no lease
activation, no cleaning service and no action client. A bootstrap can expose
read-only routes before Body compilation. Binding the compiled immutable Body
must match source URDF, run, frame, observation and lease routes. No Runtime,
second daemon or alternate Agent action interface is introduced. Legacy known
fixtures keep their existing path until separately migrated.

Live known-fixture verification, generic code/protocol freeze and explicit
held-out selection are required before claiming new-robot readiness. L2 must
add independently verified entity/source continuity and complete canonical
Nav2 stop/fault acceptance before the generic observer can enable its existing
leased sink. Neither synthetic tests nor a source-only observer permits L2.
