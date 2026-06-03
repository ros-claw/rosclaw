---
pattern_id: pattern_mapless_path_planning
applicable_symptoms: [mapless_path_planning]
domain: Planning_Decision
---

# Robot cannot navigate to semantic goals in unknown environments without a pre-built metric map.

**Domain**: `Planning_Decision`

## Fix

Use a video world model to simulate future trajectories and select actions that reach the semantic goal, enabling zero-shot mapless navigation.

## Anti-pattern

Classical path planning that relies on pre-built metric maps fails in unknown or changing environments.

## Cross-domain analogies

- **Perception_Vision** → Project sensory data into a top-down semantic grid to enable goal-directed navigation without a pre-built map.
  - related fix: Project sensory data into a bird's-eye-view grid representation, distilled from visual foundation models, to enable structured spatial reasoning for planning.
- **Learning_Training** → Use video-only input with domain randomization to enable goal-directed navigation without requiring a pre-built metric map.
  - related fix: Use video-only input modality (no depth or map) combined with domain randomization to eliminate sensor fidelity and geometry transfer gaps
- **Control_Locomotion** → Use reinforcement learning to map sensor observations directly to navigation actions without a pre-built map.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- mapless_path_planning.before.py
+++ mapless_path_planning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Robot cannot navigate to semantic goals in unknown environments without a pre-built metric map.

+# Fix    : Use a video world model to simulate future trajectories and select actions that reach the semantic goal, enabling zero-shot mapless navigation.

+# Avoid  : Classical path planning that relies on pre-built metric maps fails in unknown or changing environments.

```
