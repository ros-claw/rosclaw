---
pattern_id: pattern_unitree_b2
applicable_symptoms: [unitree_b2]
domain: Planning_Decision
---

# Quadruped robot navigation fails in dynamic environments over long distances due to limited planning horizon and lack of open-vocabulary goal specification.

**Domain**: `Planning_Decision`

## Fix

LOVON framework: hierarchical planning with long-range open-vocabulary object navigation, using a high-level semantic planner and low-level locomotion controller.

## Anti-pattern

Traditional navigation methods that rely on fixed goal definitions and short-horizon planning.

## Cross-domain analogies

- **Perception_Vision** → Use spherical ray constraints to regularize long-horizon navigation sampling for distortion-aware dynamic alignment.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use egocentric expert demonstration data to train open-vocabulary long-horizon planners via imitation learning.
  - related fix: Use VLN-Ego dataset: large-scale egocentric video + expert action pairs from Habitat simulator for imitation learning (behavioral cloning) of navigation policies directly from first-person observations.
- **Control_Locomotion** → Use closed-loop verification to retry alternative actions when a planned path is blocked.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- unitree_b2.before.py
+++ unitree_b2.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Quadruped robot navigation fails in dynamic environments over long distances due to limited planning horizon and lack of open-vocabulary goal specification.

+# Fix    : LOVON framework: hierarchical planning with long-range open-vocabulary object navigation, using a high-level semantic planner and low-level locomotion controller.

+# Avoid  : Traditional navigation methods that rely on fixed goal definitions and short-horizon planning.

```
