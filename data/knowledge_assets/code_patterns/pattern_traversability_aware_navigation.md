---
pattern_id: pattern_traversability_aware_navigation
applicable_symptoms: [traversability_aware_navigation]
domain: Planning_Decision
---

# Traditional navigation treats all drivable areas as equally traversable, leading to risky or inefficient paths on deformable, slippery, or uneven terrain.

**Domain**: `Planning_Decision`

## Fix

Integrate a learned continuous traversability metric (from RGB-D or lidar) directly into the path planning costmap to prefer routes that minimize risk, energy, or slippage.

## Anti-pattern

Geometry-only costmaps that ignore terrain properties like deformability or friction.

## Cross-domain analogies

- **Perception_Vision** → Active perception selects informative viewpoints to reduce terrain uncertainty for safer path planning.
  - related fix: Integrate active perception with semantic mapping: agent selects viewpoints to reduce ambiguity while building a task-driven semantic map from RGB-D or lidar data.
- **Learning_Training** → Use synthetic data augmentation to generate diverse terrain traversability examples for robust training.
  - related fix: Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.
- **Control_Locomotion** → Use lightweight terrain-aware MLP to predict traversability costs from proprioception at high frequency.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- traversability_aware_navigation.before.py
+++ traversability_aware_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Traditional navigation treats all drivable areas as equally traversable, leading to risky or inefficient paths on deformable, slippery, or uneven terrain.

+# Fix    : Integrate a learned continuous traversability metric (from RGB-D or lidar) directly into the path planning costmap to prefer routes that minimize risk, energy, or slippage.

+# Avoid  : Geometry-only costmaps that ignore terrain properties like deformability or friction.

```
