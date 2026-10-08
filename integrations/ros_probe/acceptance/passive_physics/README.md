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
plugin loading, scene/frame admission and dynamic Native D2/D4/D5/D6 are pending.
No actual third Body was selected or inspected.

The source uses Gazebo8 `worldPose` and actual `Geometry`/`Collision`/`Pose`
components. It avoids concurrent `generate_world_sdf`, whose Gazebo8 implementation
explicitly notes an ECM thread-safety TODO:
https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/SimulationRunner.cc
