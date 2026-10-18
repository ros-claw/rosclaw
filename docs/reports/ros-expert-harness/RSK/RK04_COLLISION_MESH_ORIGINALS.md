# Original collision mesh inspection

The primitive-only discovery and source-binding path correctly leaves STL
collision robots unknown. `ros inspect-collision-assets` adds the next bounded,
offline observation: original STL bytes matched to exact package URIs and
SHA256, all-vertex extrema, URDF scale, collision origin, fixed transforms and
conservative movable-subtree bounds. No robot-specific name or dimensions enter
the parser. ASCII and binary STL are parsed without an external mesh loader.

The CLI takes an expanded original URDF, explicit base frame and a JSON list of
{uri, sha256, content_base64}. Inputs must be bounded regular nonsymlink files.
There is no URI fetching, path resolution, mesh repair, simplification, primitive
replacement or filesystem access in the geometry code. Truncated facets,
nonfinite coordinates, oversized counts, wrong hashes, traversal, duplicate or
missing original URIs and unsupported scales fail closed. Output is exclusive.

The optional mesh-originals argument produces an offline v2 collision envelope.
Without that argument the old v1 outputs, unsupported-mesh behavior and
historical contracts are unchanged. Runtime discovery, Body compilation and L1
still reject an unbound mesh; this change deliberately grants no new readiness
or action authority. A hash establishes consistency, not source authenticity.

Before admission can be supported, exact original resources must be tied to the
owned prepared source, actual loaded SDF/ECM geometry URI and scale, readonly
resource bytes, and continuous current-source checks. These prerequisites are
still pending. An offline complete envelope must never be presented as loaded
physics, actual collision response, verified Body or a passed navigation gate.

Post-freeze screening of dd150, a300, w200, r100 and do150 revealed this common
format prerequisite. They are now development evidence for mesh support;
subsequent reports must disclose that screening when discussing unseen robots.
No World was started for those source screening checks and no collision shape
was changed.

## Source and continuity wiring under acceptance

The optional prepared-source observation retains the original collision manifest
JSON bytes and SHA, explicitly separated from ROS interface observations. The
SDK probe opt-in can read only /assets/collision-asset-originals.json inside the
owned isolated SDK. Source binding compares the original document and each
digest-named staged STL against the independent launch/capture asset inventory,
and requires converted/World/loaded ECM mesh URIs and scales to agree. It checks
the current observation against the immutable Body source.

The SDK source-resource guard rereads the exact manifest and mesh bytes and
checks file and directory identities. Symlink components, resource changes or
replacement latch failure, even after a rollback. CurrentSimulationFacts applies
that guard before any health result; the existing 500ms collection and 250ms
proof gates remain unchanged. No new runtime, driver, executor or control path is
introduced. The default path without a prepared collision observation remains
unchanged.

These source predicates and synthetic failure injections are not sufficient to
claim actual mesh admission. Real pinned SDK conversion, loaded ECM, fresh L1,
canonical navigation, strict original review and exact source/main delivery are
still pending at the time of this development record.
