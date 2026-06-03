---
pattern_id: pattern_legged_robots
applicable_symptoms: [legged_robots]
domain: Planning_Decision
---

# Legged robots struggle to efficiently explore unknown environments and locate specific object categories without prior map knowledge, especially when balancing semantic information gain, localization cost, and safety.

**Domain**: `Planning_Decision`

## Fix

Use Decision-Driven Semantic Object Exploration (DD-SOE) algorithm, which provides a sequential decision-making framework that balances semantic information gain, localization cost, and safety to guide exploration behavior.

## Anti-pattern

Random or heuristic exploration without explicit semantic objectives or safety constraints.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic ray constraints to regularize exploration sampling offsets for distortion-aware semantic gain.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use concatenated circuitous trajectories to force exploration over direct goal-seeking.
  - related fix: Use R4R dataset (concatenated R2R paths) to create longer, circuitous trajectories that better differentiate instruction-following agents from goal-seeking ones.
- **Control_Locomotion** → Use standardized benchmark tasks to train hierarchical exploration policies balancing semantic gain, localization cost, and safety.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- legged_robots.before.py
+++ legged_robots.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Legged robots struggle to efficiently explore unknown environments and locate specific object categories without prior map knowledge, especially when balancing semantic information gain, localization cost, and safety.

+# Fix    : Use Decision-Driven Semantic Object Exploration (DD-SOE) algorithm, which provides a sequential decision-making framework that balances semantic information gain, localization cost, and safety to guide exploration behavior.

+# Avoid  : Random or heuristic exploration without explicit semantic objectives or safety constraints.

```
