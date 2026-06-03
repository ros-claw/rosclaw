---
pattern_id: pattern_local_metric_trajectory_control
applicable_symptoms: [local_metric_trajectory_control]
domain: Control_Locomotion
---

# Robot navigation in unstructured environments fails due to reliance on global path pre-computation, leading to collisions when obstacles appear unexpectedly.

**Domain**: `Control_Locomotion`

## Fix

Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Anti-pattern

Global path pre-computation without dynamic obstacle avoidance.

## Cross-domain analogies

- **Perception_Vision** → Use atomic motion primitives as grounded units to reactively compose local paths.
  - related fix: Use actional atomic concepts (natural language phrases combining atomic action and object) as a compact grounded unit to bridge visual and linguistic features.
- **Planning_Decision** → Active request for local corrective feedback when global path uncertainty arises.
  - related fix: Enable the agent to actively request and interpret multimodal instructions (natural language and visual cues) from a human assistant when uncertain.
- **Learning_Training** → Use vision-only closed-loop replanning to replace precomputed global paths.
  - related fix: Use video-only input modality (no depth or map) combined with domain randomization to eliminate sensor fidelity and geometry transfer gaps

## Patch

```diff
--- local_metric_trajectory_control.before.py
+++ local_metric_trajectory_control.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Robot navigation in unstructured environments fails due to reliance on global path pre-computation, leading to collisions when obstacles appear unexpectedly.

+# Fix    : Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

+# Avoid  : Global path pre-computation without dynamic obstacle avoidance.

```
