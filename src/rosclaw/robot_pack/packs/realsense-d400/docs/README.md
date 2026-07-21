# RealSense D400 Read-only Robot Pack

This pack binds a D405 or D435i device identity to an e-URDF Body, the
`librealsense-mcp` adapter contract, the `camera.capture_rgbd` capability, and
an evidence-producing verification suite.

The adapter source is locked to commit
`fdea4c3cfd03e7acf1adb664a9ffca5733d44b59`. Install the Pack metadata only
with `rosclaw robot add realsense`, or explicitly install its native adapter
dependencies with `rosclaw robot add realsense --install-adapter`. A different
adapter commit is reported as `version_mismatch` and cannot pass read-only
verification.

The pack is perception-only. It exposes no actuator, firmware update, or
calibration-write capability. Agent requests for real frames must enter through
`rosclawd`; direct Agent access to `pyrealsense2` is outside the supported
boundary.

Installation does not claim that a camera or adapter is present. Contract
verification can establish H1. H3 requires a live device, real frame artifacts,
their hashes and receipt, plus independent physical observation evidence.
