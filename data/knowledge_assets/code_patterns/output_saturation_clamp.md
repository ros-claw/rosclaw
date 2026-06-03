---
pattern_id: output_saturation_clamp
safety_label: Velocity_Divergence
applicable_symptoms: [output_saturation_clamp]
domain: Control_Locomotion
source: curated
---

# Commanded velocity diverges to ±∞ when the integrator has no clamp

**Domain**: `Control_Locomotion`
**Safety label**: `Velocity_Divergence`

## Fix

Wrap every commanded velocity through `torch.clamp(v_cmd, -v_max, v_max)` where `v_max` is read from the platform's robot_specifications YAML. Add an integral-leak term (`integ *= 0.99` per step) when in steady state.

## Anti-pattern

Adding only a soft-start ramp on the user-side command — once internal feedback diverges, the ramp can't stop the integrator alone.

## Cross-domain analogies (curated)

- **Memory_Reasoning** → Same as the sliding-window KV-cache: cap the magnitude of the running state, not just the input.
  - related fix: Bound the integrator state itself (`integ = clamp(integ, -I_MAX, I_MAX)`), mirroring the KV sliding window.

## Patch

```diff
--- output_saturation_clamp.before.py+++ output_saturation_clamp.after.py@@ -1 +1,2 @@-v_cmd = pid_step(error, dt)     # unbounded output
+v_cmd_raw = pid_step(error, dt)
+v_cmd = torch.clamp(v_cmd_raw, -V_MAX, V_MAX)

```
