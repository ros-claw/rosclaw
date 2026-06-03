---
pattern_id: anti_windup_pid
safety_label: Torque_Overflow
applicable_symptoms: [anti_windup_pid]
domain: Control_Locomotion
source: curated
---

# PID integral wind-up drives actuator into torque saturation

**Domain**: `Control_Locomotion`
**Safety label**: `Torque_Overflow`

## Fix

Apply conditional integration: stop accumulating the integral term whenever the actuator output is saturated AND the error direction would push further into saturation. Clamp `tau_cmd` with `torch.clamp(tau, -tau_max, tau_max)`.

## Anti-pattern

Cranking up Kp/Ki to fix a tracking error during saturation — this only deepens the wind-up and amplifies the eventual oscillation when the load reverses direction.

## Cross-domain analogies (curated)

- **Systems_Compute** → Same back-pressure principle as bounded-queue producer-consumer: stop the producer when downstream is full.
  - related fix: Treat actuator saturation as a back-pressure signal and pause the integrator the way you'd pause a queue writer.
- **Learning_Training** → Clamp gradient analogue — gradient clipping prevents one outlier from blowing up a step, just as anti-windup prevents one saturated cycle from poisoning the next.
  - related fix: Reuse the `clip_grad_norm_` mental model: an upper bound that fires only when the magnitude exceeds a known physical limit.

## Patch

```diff
--- anti_windup_pid.before.py+++ anti_windup_pid.after.py@@ -1,4 +1,7 @@ def pid_step(err, dt):
-    integ += err * dt           # unconditional integration
-    tau = Kp*err + Ki*integ + Kd*derr
-    return tau                  # no output limiter
+    tau_uncl = Kp*err + Ki*integ + Kd*derr
+    tau = torch.clamp(tau_uncl, -tau_max, tau_max)
+    saturated = tau != tau_uncl
+    if not (saturated and same_sign(err, tau_uncl)):
+        integ += err * dt        # conditional integration
+    return tau

```
