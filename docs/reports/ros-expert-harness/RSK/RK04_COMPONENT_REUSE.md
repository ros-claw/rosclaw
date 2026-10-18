# RK-04 first increment: existing ROS2 component reuse

This increment creates a source-bound application dependency/reuse plan and an
offline bundle of operator-selected original assets. It is a prerequisite for
RK-04, not completion of unfamiliar-Body navigation or automatic stack synthesis.
It grants no installation, driver, Body, ROS write or execution authority.

## Original prerequisites

An operator explicitly selects the existing command topic, actual consumer node,
Twist/TwistStamped type and driver package. A fresh complete original graph/QoS
snapshot must show that selected typed subscriber and installed package. There
is no conventional command topic, robot-model branch or inferred drive interface.
Missing selected prerequisites produce `NEEDS_DRIVER`, meaning adaptation is
required; this is not an assertion that no driver exists anywhere. Incomplete or
ambiguous observations remain UNKNOWN and request read-only measurement.
Endpoint/package presence is not verified driver physics.

The target capability graph separates navigation dependencies from coverage,
cleaner enable/disable and independent coverage verification. Package inventory
alone is not version evidence. The component review compares SHA-pinned original
package.xml contents with explicit exact-version and ROS/distro requirements.
Unknown versions, missing dependencies and conflicts remain explicit. A matching
comparison proves neither ABI, launch/BT plugins nor successful execution; an
independent owner must authenticate the original installed files and image.

## Plan and bundle

`ros application-plan` consumes a sealed operational graph and system model,
original expanded URDF, navigation requirements, actual installed manifest XML
and a typed request. It preserves the graph's SEALED_REPLAY/OBSERVATION domain
and original review time. Historical files cannot renew current L1 or authority.
The existing generic navigation observation reviewer supplies Body/frame/stream/
geometry prerequisites without a model-specific profile.

If required stack observations are absent, the result is
`NEEDS_APPLICATION_BRINGUP`, containing dependencies and validation steps only.
This first increment does not claim to generate a stack from a bare driver.
Coverage always requires its separate cleaner/planner/measurement gate.

`ros application-staging` accepts only a recomputed
`CONDITIONAL_REUSE_STAGING` navigation plan. It assembles exactly five explicitly
selected original roles: Nav2 parameters, bringup descriptor, BT XML, installed
launch reference and existing driver reference. Each role carries original content
SHA, package/version, relative source path, target snapshot and URDF pins. Nav2
parameters must explicitly cover the selected lifecycle nodes; bringup must target
the selected namespace. Driver references must exactly reproduce the observed
selection. Relative paths, duplicate roles, version/source changes, XML entities,
YAML aliases, stale plans and extra rehashed plan claims are rejected.

Outputs are a JSON bundle with bounded inline file contents and references, not
an installed workspace. References are operator-declared reuse materials requiring
independent original review; hashes do not authenticate installed-file provenance.
The compiler neither executes launch files nor verifies their runtime use. The
existing World owner must materialize and review staging, prove which assets were
actually loaded, bind current Source/Body, obtain fresh full L1 and submit guarded
navigation exclusively through MCP → rosclawd. Independent stop, zero contacts,
source continuity, 500 ms source / 250 ms proof and Body limits remain unchanged.

```sh
PYTHONPATH=src:. .venv/bin/python -m rosclaw.entrypoint ros application-plan \
  --operational-graph graph.json --snapshot system.json --urdf robot.urdf \
  --requirements requirements.json --component-manifests installed-manifests.json \
  --request request.json --output application-plan.json --json
PYTHONPATH=src:. .venv/bin/python -m rosclaw.entrypoint ros application-staging \
  --operational-graph graph.json --snapshot system.json --urdf robot.urdf \
  --requirements requirements.json --component-manifests installed-manifests.json \
  --request request.json --plan application-plan.json --artifacts originals.json \
  --output application-bundle.json --json
```

Inputs are bounded regular files; symlinks are refused. Output creation is
exclusive. No discovery, installation or physical action occurs in these commands.
The adjacent generated schemas describe request, plan and reusable artifact roles.

## Remaining RK-04 gates

Freeze the generic Harness before selecting a truly independent non-Waffle/Burger/
J100 Body; record its existing ROS2 driver prerequisite and all operator adaptation.
Use original driver/interface/URDF and installed component evidence to build and
review actual source-bound staging, then repeat independent fresh World source,
L1, canonical navigation and stop/collision/continuity acceptance. An observed
working stack reused by this increment is not evidence of bare-driver bringup.
Coverage cleaning is a separate subsequent capability gate. No new Body acceptance,
full RK-04 completion, source/main delivery or physical success is implied here.
