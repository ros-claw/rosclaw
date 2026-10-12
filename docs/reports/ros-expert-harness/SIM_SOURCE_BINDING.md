# Independent SIM source binding

A compiled navigation Body can be internally consistent while Gazebo loads a
smaller or different physical model. `review_sim_source_binding` adds an
independent, read-only correspondence review; it never admits a Body or
requests an action. [ADR 0017](../../adr/0017-independent-sim-source-binding.md)
records the trust boundary and supported scope.

The operator pins both the launch manifest and a separately captured running
instance receipt, outside the checked workspace. The checker reuses the
12-file workspace inspector, original navigation review and
`SimulationNavigationBinding`. It compares exact URDF conversion lineage,
read-only source/resource inventory, container/image/PID, actual Gazebo ECM
World/entities/Static state, collision dimensions and effective transforms,
sensor mounts, native ROS domain/frames/routes, and independent physics
Body/run/time/pose observations. The evidence is returned separately; no file
is added to the compiled workspace.

`MATCHED` emits `SIM_SOURCE_MATCHED_NOT_AUTHORIZED`. Contradictory facts emit
`MISMATCH`; missing, stale or unsupported facts emit `UNKNOWN`. All results have
`execution_authority=NONE`, `real_authority=false`, and L1/L2 `NOT_RUN`.
The full current navigation review is retained. Its three localization
freshness/covariance checks belong to G2; source identity still requires a
unique typed localization publisher. A source match can coexist with blocked
live navigation. There is no arbitrary `allow_unknown` input.

## Actual implementation environment

The existing acceptance image is pinned to
`sha256:d16320799584a60035548fb298243bdd8b0d2cb5760675a9cb46b608ea211e56`,
ARM64 ROS 2 Jazzy, Gazebo Sim **8.15.0**. Installed headers expose Geometry,
Collision, Pose, ParentEntity, Sensor, Static and World, but no ModelSdf.
The C++17 observer uses const ECM PostUpdate, no publishers/services/ECM writes,
exclusive no-follow output and bounded 10 Hz capture. SourceFilePath is
informative, not a claimed original-SDF hash. Its actual `-Werror` build uses
system Protobuf MODULE discovery before Gazebo to avoid a conflicting ORTools
vendor configuration.

The operator stages a fresh known fixture with
`gazebo_source_witness/prepare.py`, retaining the actual
`gz sdf -p /assets/robot.urdf` output and all resolved visual-resource hashes.
Runtime mounts `/repo`, `/assets` and `/observer` read-only, with a read-only
rootfs, no network, dropped capabilities and a separate writable evidence
mount. `source_binding_fixture.py` runs inside that owned SDK container;
the independent host supervisor captures/pins Docker and asset state and calls
the pure checker. ROS SDK imports remain in `integrations/ros_probe`.
No new Runtime, executor, permission service or raw Agent tool is introduced.

The passive physics witness creates **no velocity publisher** and cannot enable
a lease. A separate operator initializer consumes the pinned known-fixture
pose and 5 mm / 0.01 rad uncertainty, calls existing SetInitialPose once after
actual lifecycle activation, and exits. This is disclosed SIM setup input,
not autonomous unknown-robot initialization. No covariance threshold, source
timestamp or observed value is changed. The duplicate-publisher fault creates
an endpoint and sends zero messages.

## Frozen source acceptance

On source `637ccafd2d3d17ba7fbe2d4920262563afdcd242`:

| Scope | Result |
| --- | --- |
| ROS/MCP/architecture | 943 passed, 2 skipped, 11 deselected |
| Practice | 183 passed, 9 skipped |
| mypy, including source binder | 126 source files passed |
| New source and publisher-race contracts | 68 passed |
| Actual Gazebo/ROS source matrix | 9 independent Worlds |
| Independent raw native/model/source recomputation | 9/9 exact |

| Actual isolated SIM case | Expected observed result |
| --- | --- |
| Known Body compiled from actual fresh native observations | MATCHED |
| Actual World collision width halved, exact URDF conversion unchanged | MISMATCH |
| Host replaces source asset during a read-only-mounted run | MISMATCH |
| Actual World reset, independently observed ROS clock rollback | UNKNOWN |
| Source witness exits | UNKNOWN |
| Source witness has a different ROS node name | UNKNOWN |
| Actual ROS domain 102, externally expected 101 | MISMATCH |
| Additional dynamic entity in the World | MISMATCH |
| Actual extra velocity publisher endpoint, zero messages | UNKNOWN |

The source binder's full result was independently replayed from original ECM
and independently rebuilt native models. Witness traces record zero received
control messages, zero published velocity messages and lease disabled. The
fixture dispatches zero navigation goals. All owned containers stop with PID 0.
No L1/L2 task or held-out robot acceptance is claimed.

Body/manifest/run/world pins, witness implementation, namespace/frame and
container replacement also have a separately labelled actual-capture REPLAY
matrix (12 negative cases plus an original control). These are not extra live
ROS faults. Original build/observation failures remain separate evidence.

## Problems found through actual execution

The old known Waffle URDF base collision is 0.266×0.266×0.094 m; the old physical
SDF used 0.265×0.265×0.089 m. Fresh source staging derives physics from the exact
URDF conversion. This does not retroactively promote or revoke historical
navigation receipts beyond their original scope.

CLI YAML plain-scalar parsing folded XML whitespace: 7753 source bytes became
6779 live parameter bytes. Typed YAML parameters preserve the exact description.
The SDK capture node uses simulator time, avoiding its own mixed-clock defect.
Repeated stationary zero-covariance AMCL initialization produced tiny negative
variances; those UNKNOWNs and the bounded failed attempts remain intact.
Explicit operator initialization supplies nondegenerate uncertainty without
changing the review gate. Native Probe now waits for actual publisher QoS
before subscribing, preventing subscriber-first discovery from losing a
transient-local map. Reset review pairs ROS/ECM monotonic observations rather
than trusting a successful service response or an old ECM file.

## Limits and delivery boundary

P0 supports expanded primitive robot collisions and static primitive/plane
World geometry, at a stationary configuration with explicitly bounded pose
precision. Moving articulation, mesh collisions, unresolved frames/includes,
additional dynamic robots and nonidentity World/map alignment fail closed.
ECM geometry is not an inspection of the physics engine's private solver or
an independent measurement of every inertial parameter. Same-UID operator
capture is an operational trust boundary, not adversarial authentication or
REAL certification.

G0 (#670) is integrated at actual main `85fe9af0`; its four original push gates,
merge-tree proof and actual pinned CLI checks passed. G1 is delivered in a
separate PR after that parent closure. PR/actual-main status is tracked in the
local execution journal and implementation report; the frozen source acceptance
above never claims acceptance for an unrelated commit. Full v1 remains open.
