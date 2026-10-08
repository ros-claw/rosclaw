# Passive Gazebo 8 physical-state producer (Native physical NOT_RUN)

This world plugin reads `const EntityComponentManager` in PostUpdate. It has no
actuator/service, ECM mutation, localization estimate, brush state or verifier.
A single packet binds the actual body and obstacle model poses, actual collision
primitives and model-relative offsets at one physics SIM timestamp. Run/Body/
attachment identities and the exact closed scene model set are configured by
the SIM fixture launcher, never by an action argument. Unknown/missing models,
geometry, mesh, articulated obstacles and invalid/unbounded envelopes produce
an incomplete fault packet, never a partial free-space observation.

Boxes, spheres and cylinders use a conservative model-centered 3D enclosing
disc for XY projection; tall shapes can therefore overblock. This is a safe
bound, not an exact surface or height-filtered brush-plane intersection.

Build in the unchanged acceptance image with no network or simulation:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S /source -B /build -DCMAKE_BUILD_TYPE=Release \
  -DProtobuf_DIR=Protobuf_DIR-NOTFOUND \
  -DCMAKE_IGNORE_PREFIX_PATH=/opt/ros/jazzy/opt/ortools_vendor
cmake --build /build -j1
cd /build
GZ_PARTITION=reh_n03_offline_ecm_contract ctest --output-on-failure
ldd -r /build/librosclaw_passive_physics.so
```

The explicit ignored prefix prevents OR-Tools' different Protobuf SDK from
being selected for Gazebo's installed message ABI. It changes build discovery
only, not image packages, generated SDK headers or running Nav2/Gazebo. The
fixed image uses Gazebo8.15.0/SDFormat14.9.0/system Protobuf3.21.12; runtime
`libgz-msgs10.so` links `libprotobuf.so.32`. Initial incompatible-prefix and
const-config API failures were retained and fixed, not suppressed.

Validation: C++17 `-Wall -Wextra -Werror` compile/link PASS. One CTest executes
actual SDK component fixtures, yielding9 JSON records (one valid, three repeated
read-only checks, five distinct failure cases); no Gazebo server/physics/actuator
or publisher invocation. The generated JSON was decoded independently with
Python; no partial fault packet contains a body/obstacle claim. Initial shared
library `ldd -r` had no unresolved symbols. Further parser/transport, actual
plugin loading and dynamic Native D2/D4/D5/D6 are pending.
No actual third Body was selected or inspected.

The launcher accepts `--physics-fixture CONFIG --physics-plugin LIBRARY` only
with `--brush-binding BINDING`. The versioned configuration pins the library
SHA256, run/Body/attachment/mission identities, original static-map denominator,
exact brush and explicitly approved known-fixture world/map identity. It defines
1–32 uniquely named bounded box obstacles initially parked outside the room.
Preparation adds the world plugin and a GZ_TO_ROS StringMsg bridge, then starts
the observer with `dynamic_physics=true`. Unknown scene models, changed masks,
unapproved transforms, source mismatches, bad bytes or repeated preparation
are rejected before simulator startup. Actual geometry still comes exclusively
from independent PostUpdate packets. These CLI inputs do not create a receipt,
authorize an action or prove the library loaded. 17 offline preparation tests
and the 509-test ROS suite passed; fresh physical execution remains NOT_RUN.

Daemon-configured dynamic repair additionally requires the actual physical Body
radius, distinct from cleaner width. Its time-paired route filter dilates
occupied cell squares conservatively, uses only the current legal-center
component and prohibits corner cuts or guessed entry into a distant component.
Only its filtered centers are offered to the existing greedy/pose-aware goal
selection; Nav2 still plans all connecting paths. A fixed admission SIM deadline
and the original wall deadline bound execution. Waiting uses a separate actuator
hold service: drive zero, brush OFF, commands inhibited, lease renewal cannot
enable the brush. Hold release alone grants no brush credit. Polls consume no
goal or retry count; withdrawal requires measured enabled revisits. Faults,
unpaired route input or computation exhaustion stop without partial proposals.
These contracts passed the 517-test ROS suite and scoped mypy; actual dynamic
Native episodes and their complete canonical receipts remain NOT_RUN.

The scripted runner must explicitly request `--dynamic-physics`. It compares
the newly compiled Body and actual measured static map with the prepared brush
and physics bindings and the observer's `physics_ready.json` source seal. Mission
id, closed binding, plugin bytes, geometry hash and initial packet digest must
match before creating daemon configuration. A prepared dynamic scene cannot
silently run with static accounting. Admission files do not replace rosclawd's
fresh observation wait, action guards or replay. 27 preparation/admission tests,
527 ROS tests, required mypy over121 files and Practice183 passed/9 skipped.

Temporal admission now validates a complete paired source packet before any
ON service. Repeated actions for the same source mission share the first SIM
and wall deadlines. A daemon-owned exclusive admission record refuses silently
restarting that mission's budget after process restart, including partial/crash
records. Navigation actions do not start the coverage mission's clocks.532 ROS
tests and scoped mypy passed, including wrong mission, missing source, expiry,
negative clock and process-restart refusal; physical acceptance is still NOT_RUN.

The packet parser also rejects oversized JSON integers before floating-point
conversion, bounds source counters to their SDK unsigned domain and SIM time to
the SDK nanosecond clock domain, and requires typed model/primitive identities.
Malformed names or dimensions cannot escape as unhandled overflow/type errors.
540 ROS tests,50 focused parser/observer tests and scoped mypy/CI Ruff passed;
the8 new cases are synthetic corruption tests, not physical fault episodes.

The source uses Gazebo8 `worldPose` and actual `Geometry`/`Collision`/`Pose`
components. It avoids concurrent `generate_world_sdf`, whose Gazebo8 implementation
explicitly notes an ECM thread-safety TODO:
https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/SimulationRunner.cc
