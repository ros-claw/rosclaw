---
pattern_id: pattern_isef
applicable_symptoms: [isef]
domain: Systems_Compute
---

# Industrial control systems (PLCs) are vulnerable to unauthorized start/stop commands via network protocols like Modbus and S7.

**Domain**: `Systems_Compute`

## Fix

Use the ISF framework to send crafted PLC control commands (e.g., Schneider_CPU_Command, Siemens_300_400_CPU_Control) for security testing and exploitation.

## Anti-pattern

_(no anti-pattern documented in source)_

## Cross-domain analogies

- **Perception_Vision** → Train control systems on simulated network traffic with injected anomalies and adversarial commands.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Planning_Decision** → Use hierarchical decomposition to validate commands via step-by-step subgoal checks.
  - related fix: Use chain-of-thought reasoning to decompose tasks into step-by-step subgoals before predicting actions.
- **Learning_Training** → Use causal intervention to block unauthorized commands' influence on control logic.
  - related fix: Use causal counterfactual reasoning to remove the influence of sensitive attributes on predictions by intervening on the causal graph.

## Patch

```diff
--- isef.before.py
+++ isef.after.py
@@ -1,2 +1,3 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Industrial control systems (PLCs) are vulnerable to unauthorized start/stop commands via network protocols like Modbus and S7.

+# Fix    : Use the ISF framework to send crafted PLC control commands (e.g., Schneider_CPU_Command, Siemens_300_400_CPU_Control) for security testing and exploitation.

```
