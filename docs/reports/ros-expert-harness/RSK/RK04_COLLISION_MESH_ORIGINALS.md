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

## Moving-link evidence

The first real DD150 After was blocked before any goal by a source frame check.
Original ECM audit found zero positional drift (about 1e-16m) but a legitimate
wheel-link rotation of about 0.00284rad. The old member check assumed every joint
remained at zero; that assumption does not describe a new supported mechanism.
The failed World and source review are retained.

The optional readonly SourceWitness joint observation records actual link poses,
loaded joint type/topology/child anchor/axis/bounds and any already available
unwrapped JointPosition component. It never creates an ECM component, publishes
a topic or requests actuation. Default observation bytes are unchanged unless
the explicit observe_joint_kinematics option is set.

Core checks every link against the original joint tree: fixed links cannot move,
revolute links can only rotate around the declared axis, and prismatic links can
only translate along it. Bounded rotary joints require actual unwrapped state;
continuous semantics must come from the original URDF. Member transforms are
then checked against source placement using observed, source-constrained links.
This is not a shape-symmetry exception or a larger tolerance. Current facts use
the same independently accepted initial source precision and recheck the latest
ECM sample before returning a guarded observation.

Gazebo 8 creates joint Pose from SDF semantic pose; SDF 14 resolves that pose
relative to the child link by default. Axis data are resolved into joint frame.
These API semantics were checked against the installed SDK headers and official
[gz-sim8 SdfEntityCreator](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/SdfEntityCreator.cc)
and [sdf14 Joint](https://github.com/gazebosim/sdformat/blob/sdf14/src/Joint.cc)
source. Actual pinned SDK observation is still required to accept the new path.

## Original-byte parsing cache and development limits

Only immutable offline STL facts are cached, keyed by the complete original bytes,
with at most eight inputs of at most 8MB each. The cached value is immutable JSON;
each public result is separately decoded, so caller mutation cannot poison later
facts. URI/SHA checks, fresh manifest/resource reads, file and directory identity,
loaded geometry, joint observations and the health/proof clocks are never cached.
Changed bytes must still pass the complete parser and independent source guards.

The first frozen-Core A300 driver-only World failed its bounded current-capture
condition without a Body or goals. Offline profiling at the original source time
found repeated STL parsing dominated candidate review (about 267--274ms per
review on that host); the byte cache reduced repeated historical processing to
about 42ms. These timings diagnose an implementation cost, not a passed live
freshness gate. The original failed freeze stays negative. A300 becomes known
development material after any Core optimization; no blind transfer is claimed
from a subsequent retry.

Known DD150 development also retained an After whose position action completed
but yaw failed: Nav2's localization-based success differed from independent
physical orientation. The compiler now preserves explicitly tighter software goal
checker settings. A stricter operator-selected yaw profile produced a subsequent
two-goal After with independent stop/contact/age review, but the physical yaw
margin was small. This does not calibrate localization or prove generic
robustness. Independent physical predicates remain unchanged.

## Separate static integrity from current observations

A300 diagnostic Worlds retained zero-command SDK admission failures. One measured
World ruled out container PID exhaustion (kernel peak451, limit512, max events0).
New bounded startup records then showed daemon observation completed at about
119ms, while first facts failed when its 100ms freshness lookahead expired.
These are distinct observations; earlier socket-close failures are retained and
are not retrospectively assigned a proven common cause.

The closed Body workspace's exact-byte static compiler replay now has at most
two cache entries of at most 10MB each. The no-follow reader still rereads every
file, directory inventory and identity on each check. Different bytes or pinned
hashes require a new replay; identical-byte symlinks still fail. Cached values
are immutable JSON and public dictionaries are fresh copies. Current inspection
time and historical snapshot age are recalculated on every call. No live health,
readiness, source binding, calibration state or daemon generation is cached;
the admitted static profile requires the original empty calibration document.

The SDK validates this static integrity during constructor preparation, before
opening the expiring health window. It polls and verifies actual source/joint
observations before evaluating current ROS health, and retains the minimum of
the unchanged source, Native and health deadlines. The same 500ms source, 250ms
proof and 100ms lookahead constraints remain. The existing strict safe YAML
parser and its alias/duplicate rejection remain unchanged. Offline timings are
development evidence, not a new physical pass.

## Preserve sink heartbeat during passive replay

Daemon-side diagnostics found available client slots and no thread-start errors.
The original physics witness then identified the preceding event: sink state
changed from known to `sink status stale ...` while `_try_admission` synchronously
constructed and authenticated CurrentFacts. The independent sink-status bound
remains 250ms. Worker socket errors after owner shutdown are secondary in these
measured Worlds; the earlier health-window expiry remains a separate failure.

Mesh collectors are now prepared in a readonly background job during the existing
unadmitted phase. The sink's status timer continues; no daemon client, lease,
monitor, proof or publisher is created by preparation. A constructor result must
match the current Body/run and original enrollment SHA. Its marker says NONE /
NOT_GRANTED. Actual admission still authenticates the original operator context
and calls current facts again; the monitor's independent checks remain required.
The primitive-only SDK path is preserved. The operator waits for and independently
checks passive preparation and fresh zero-command status before starting the
existing canonical journey. Preparation cannot stand in for L1 or L2.

## Avoid gaps between expiring independent observations

The passive-preparation World obtained actual SDK admission acknowledgment, then
failed its post-admission observation window before any sink command. Original
worker records showed 83--88ms collections with about 50ms validity left at
publication. This exposed a real gap between successive expiring results.

Only static encodings are now reused: eight exact bounded base64/digest keys,
two exact original manifest documents, and two escaped prepared-document JSON
literals. Public fact dictionaries/lists are separately reconstructed. Changed
payload, SHA, URI or document still fails its original check. Actual resource
bytes, identities, dynamic fields and deadlines are never cached.

Snapshot hashing streams exactly the previous canonical JSON bytes, including
the complete original document and every live field. It does not normalize or
omit any source value. Escape/Unicode/mutation tests and 16 historical sealed
models from eight actual Worlds match the original canonical hash calculation.
There is no snapshot schema or hash change. All cache key/value sizes and entry
counts are bounded; the legacy path without prepared collision bytes is retained.
Offline warm historical health cost fell to about 11--12ms on the measured host;
actual sustained validity remains a separate new-World requirement.


## Actual costmap freshness is still mandatory

After encoding optimization, A300 produced actual guarded velocity commands and
73 published proofs, then the global costmap's original message age plus Native
capture/processing delay and the unchanged 100ms lookahead exceeded 500ms. The
original FAILED receipt and zero-speed output remain preserved. Software rates
are not physical freshness evidence.

The compiler now preserves explicitly requested faster costmap publication up
to the observed driver-rate ceiling of this bounded software profile; default
output remains unchanged. Update frequency, geometry, driver limits and all
health/proof bounds remain unchanged. Readonly Native parameter collection now
also retains actual update/publication frequencies. New World acceptance must
check both readbacks and measured original stream ages.

Nav2's tagged [1.3.13 Costmap2DROS source](https://github.com/ros-navigation/navigation2/blob/1.3.13/nav2_costmap_2d/src/costmap_2d_ros.cpp) uses a wall-rate update loop and a
node-clock publication cycle. This explains why a configured SIM-time frequency
need not equal measured wall-time delivery; it is not independent authentication
of the installed binary or a substitute for actual stream measurement.

## Match the bounded task, software approach zone and wall-time budget

The faster-publication A300 World retained an original TIMED_OUT receipt at the
operator's 8s action deadline. Its data-age pipeline stayed active through actual
motion. The previous radius-scaled approach zone (~0.249m) covered the entire
~0.159m goal, causing slow approach over the whole short task in a slow SIM.

The reusable RPP profile now bounds goal-approach scaling distance by twice the
current task position tolerance, while retaining maximum, desired and minimum
approach speeds. Collision checks and full Body footprint are unchanged. The
next operator phase uses 14s per goal inside the original 16s Body duration bound,
leaving 2s reserve. Goal poses, all eight Body limits, source/proof/status ages and
100ms lookahead remain unchanged. This is known development configuration work,
not evidence of model autonomy or a passed unfamiliar-Body gate.


### Optional exact-byte lossless transport development

The twelfth A300 development World (task approach) still failed on original
Native lifecycle, costmap and parameter ages after guarded motion. It is retained
as a failure. Its original Native frame is approximately 3.68 MB; a local
lossless gzip level-1 check reduced it to 1.41 MB before base64 (about 27 ms
encode and 10 ms decode on this host). This is a transport hypothesis, not a
proven root cause or a physical acceptance. An optional bounded gzip/base64
envelope now preserves the exact original Native JSON bytes, SHA and capture
time. SDK captures retain both wire and decoded originals, including the
original receive clock taken before decompression. No observation age is
renewed. Default native publication remains raw; only owned mesh SDK stacks
select the optional transport. Production freshness, source, proof, status and
lookahead limits remain unchanged. Corrupt, oversized, truncated, concatenated,
noncanonical or nested envelopes fail closed. No A300 completion is claimed.
