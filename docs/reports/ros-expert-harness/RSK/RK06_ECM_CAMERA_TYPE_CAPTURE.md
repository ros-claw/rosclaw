# Read-only ECM camera type capture

The source witness previously emitted `UNKNOWN` for camera sensors even when
Gazebo had created typed native components. Independent source binding correctly
refused an otherwise valid camera-equipped navigation source before any goal.

The SDK observer now reads `Camera`, `DepthCamera`, and `RgbdCamera` component
metadata and checks the native SDF type against the component. It emits the
exact `camera`, `depth_camera`, or `rgbd_camera` type. Missing, inconsistent or
multiple camera components remain `UNKNOWN`. Sensor names and declared input
labels do not determine the measured type. The existing lidar, IMU, navsat and
contact paths retain their prior behavior.

This is identity/type evidence only. It does not verify image delivery, depth
quality, calibration, perception, navigation, Body/L1 admission or hardware.
All original sensor identities, poses, source comparisons and unknown/type
mismatch rejection remain mandatory. No source sensor is removed or ignored.
The Core `src` tree, motion path, limits and freshness rules are unchanged.

Opt-in SDK tests (`ROSCLAW_SOURCE_WITNESS_TESTS=ON`, then `ctest`) exercise eight
synthetic stock ECM component cases with zero Worlds, ROS nodes or drivers:
three exact native types, three contradictory types, absence and ambiguity.
Python source-binding contracts separately require exact measured/declaration
agreement and preserve UNKNOWN/MISMATCH behavior. Native component tests are
not physical evidence; new owned source/main physics acceptance is separate.

The accompanying deployment change increases bounded APT transport retries from
three to eight after an exact-main deployment failed resolving the official ROS
repository. Signed repositories, fail-on-partial-index updates, package checks
and archived build provenance remain mandatory. The failed original workflow is
retained; validation uses a new source commit and its original workflows.
