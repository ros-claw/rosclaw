---
pattern_id: pattern_vision_and_language_navigation_in_the_continuous_environment_vln_ce
applicable_symptoms: [vision_and_language_navigation_in_the_continuous_environment_vln_ce]
domain: Planning_Decision
---

# VLN agents fail to generalize from discrete to continuous environments due to mismatched action spaces and lack of low-level control.

**Domain**: `Planning_Decision`

## Fix

Use continuous action spaces with low-level controllers (e.g., PID) and train with reinforcement learning or imitation learning on continuous trajectories.

## Anti-pattern

Discrete action spaces (e.g., turn left/right, move forward) used in simulated discrete environments.

## Cross-domain analogies

- **Perception_Vision** → Use joint cross-modal reasoning to unify discrete action planning with continuous low-level control.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Learning_Training** → Use synthetic continuous action data from simulators to augment training for low-level control generalization.
  - related fix: Use synthetic instruction generation via speaker model and large-scale unlabeled 3D scans, then train with imitation learning on the augmented dataset.
- **Control_Locomotion** → Use a standardized benchmark with continuous action spaces to train hierarchical low-level control.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- vision_and_language_navigation_in_the_continuous_environment_vln_ce.before.py
+++ vision_and_language_navigation_in_the_continuous_environment_vln_ce.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize from discrete to continuous environments due to mismatched action spaces and lack of low-level control.

+# Fix    : Use continuous action spaces with low-level controllers (e.g., PID) and train with reinforcement learning or imitation learning on continuous trajectories.

+# Avoid  : Discrete action spaces (e.g., turn left/right, move forward) used in simulated discrete environments.

```
