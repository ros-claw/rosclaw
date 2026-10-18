# Driver-only navigation proposal

`ros application-bootstrap` compiles a proposed ROS2 SIM Nav2 application from
an existing, observed differential-drive interface. Unlike application-plan,
it does not require a running map, localization node, navigation action, or
Nav2 lifecycle stack. Those capabilities are explicitly planned and unadmitted.

Inputs are a sealed model, operational graph, expanded original URDF, explicit
owned driver bringup descriptor, reusable Nav2 software template as JSON,
SHA-bound installed package.xml originals, and an explicit request. Each current
base stream, dynamic required TF, description and driver/manager parameter
readback retains its original receive clock and must satisfy 500ms. The selected
controller class must actually be diff_drive_controller/DiffDriveController;
missing driver evidence produces NEEDS_DRIVER or a bounded read-only probe need.
There is no guessed cmd_vel endpoint, kinematics, package version or robot model.

The request supplies the exact source revision, URDF SHA, namespace, manager,
typed control endpoint, target map frame, proposed Body id, eight navigation
limits and exact component policies. Core validates those limits against the
original driver readback, derives geometry from the current URDF and binds
software frames/topics to current interfaces. Policies remain operator input,
not authority. Planner/controller software classes are explicit Nav2 profiles.
Coverage libraries are not part of the emitted navigation profile.

Output is a JSON proposal containing nav2.yaml, bringup.json, requirements.json
and navigation-policy.json with input and output hashes. Core neither writes the
files to a ROS workspace nor installs packages or starts a World. The map and
installed BT originals, explicit initial localization, source adapter, new World,
immutable Body and full fresh L1 remain independent validation requirements.
Proposal content is not a deployable or admitted application by itself.

The optional prepared SDK descriptor flag start_navigation_stack=false is valid
only for the explicit isolated driver_onboarding source-only role. It skips
Nav2 and AMCL initialization; an UNKNOWN navigation candidate is expected and
must retain a zero-command unadmitted sink. The default descriptor behavior is
unchanged. This role never makes an execution_config, grants a lease or sends a
navigation goal. A later After must use a new independently owned World and the
existing MCP→rosclawd path with unchanged source/proof freshness checks.

Current code tests are synthetic contracts, not new robot or physical
acceptance. Source/main execution, constructor/type checks, original package/BT
provenance, unchanged historical replays and normal PR/CI must be reported
separately before closing the gate. A robot inspected during compiler development
cannot be called an unseen test robot for that compiler.
