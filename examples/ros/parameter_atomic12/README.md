# Parameter Atomic12 protocol example

ROSClaw Native authored this standalone ROS 2 parameter scenario, transcript oracle,
and pure regression suite. This archive preserves its exact three source files.
The companion `verification.json` records independent local validation.

Run the pure tests with the standard library only:

```sh
python -I -B examples/ros/parameter_atomic12/test_parameter_atomic_scenario.py -v
```

The 20 methods include three regressions for a real API mismatch: rclpy
`Parameter.Type` is a plain `Enum`, so its integer wire code comes from `.value`.
The old 17 methods and their assertions remain unchanged. Synthetic transcript
fixtures test the oracle; they are not evidence of service execution.

`PARAMETER_ATOMIC12_V1.md` preserves the author's original execution status at
writing time. The later independent local run passed 20 methods and 39 subtests.
Actual ROS 2 execution of the six cases and twelve RPCs remains unverified.
The CI workflow here runs only pure tests and imports no ROS SDK. It does not
launch nodes or send robot commands. The runtime entry point is explicitly opt-in
and needs a separately supervised, isolated ROS 2 environment.
