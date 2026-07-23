# LeRobot RH56 Reference Policy Plugin

This independently installable worker plugin registers
`rosclaw_rh56_reference` with LeRobot 0.6. It is isolated from the core
ROSClaw environment because LeRobot requires Python 3.12 and brings a separate
machine-learning dependency stack.

The `policies/rh56_reference_policy_v1/` directory is a deterministic,
non-trained policy fixture. It validates policy loading, preprocessing,
postprocessing, action-contract propagation, and guarded rollout behavior; it
does not establish task performance or hardware readiness.

Install and validate the supported runtime through the core CLI:

```bash
rosclaw setup lerobot --reference-policy rh56
rosclaw lerobot doctor --json
```

Do not import this package into the core ROSClaw process. The persistent
LeRobot worker loads it inside the isolated runtime.
