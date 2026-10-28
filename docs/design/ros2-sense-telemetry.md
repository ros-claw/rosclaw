# Explicit ROS2 Sense telemetry

`ROS2Collector` reads explicitly configured BatteryState and Imu interfaces in an
operator-owned Sense/daemon process. It owns and closes its ROS context, executor
and subscription thread. Installing it does not give Native agents ROS transport
access. The default collector remains `unconfigured`.

Operators can set `RuntimeConfig.sense_collector="ros2"`,
`sense_ros2_topics`, and `sense_ros2_use_sim_time`. A standalone SenseRuntime
accepts the same values through `SenseConfig.extra`:

```python
{
    "ros2_topics": {
        "battery": {"topic": "/robot/battery_state", "max_age_s": 3.0},
        "imu": {"topic": "/robot/imu", "max_age_s": 1.5},
    },
    "ros2_use_sim_time": True,
}
```

Roles have fixed message types and explicit absolute topic names. Discovery does
not enroll arbitrary topics. These options do not change motion Driver limits,
TF checks, watchdog, permissions, or action admission.

Both original header age and monotonic receive age must satisfy the configured
limit. Only an advancing original header updates a sample. Queries retain the
original timestamps. Clock resets invalidate the previous epoch. Silence,
transport failure and stop invalidate usable values. An unclosed context/thread
prevents restart. Missing or nonfinite measurements remain unknown.

Battery percentage is converted from 0..1 to percent; discharge current becomes
positive `current_draw`. An absent battery stays unknown. IMU unavailable
covariance markers and invalid quaternions are not interpreted as zero rotation.
See the upstream [BatteryState](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/BatteryState.msg)
and [Imu](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/Imu.msg)
definitions.

`robot_id` is an attribution label. This collector does not verify a hardware
serial, publisher plugin/version or immutable Body binding. It reports
`body_identity_verified=false` and `action_authority=false`; other body dimensions
stay unknown. Concrete Body/Driver binding, installed-artifact and Native task
integration remain separate qualifications. The collector does not certify
hardware support, robot safety, coverage capability, or cleaning completion.

## Verification

Ordinary tests check source ages, epochs, unknown values, partial data and
ownership cleanup without opening ROS. The integration test uses actual typed
ROS messages in an isolated namespace and checks data receipt, silence, stale
replay, timestamps, SenseRuntime projection, shutdown and fresh data after
restart. Its values are fixtures, not robot measurements.

The `ROS2 Sense DDS` CI job runs a Jazzy fixture without network, GPU/device mounts
or capabilities, and records the image and installed package versions. The
package build depends on that check. Reproduce in an isolated Docker environment:

```bash
docker build -f tests/sense/integration/Dockerfile -t rosclaw/sense-ros2-fixture:local .
docker run --rm --init --network none --read-only --user 1000:1000 \
  --cap-drop ALL --security-opt no-new-privileges:true --cpus 1 --memory 512m \
  --pids-limit 128 --tmpfs /tmp:rw,size=64m rosclaw/sense-ros2-fixture:local
```
