---
pattern_id: pattern_humanoid_occupancy
applicable_symptoms: [humanoid_occupancy]
domain: Perception_Vision
---

# Humanoid robots suffer from kinematic self-occlusion and sensor blind spots due to body parts obstructing sensor views, leading to incomplete occupancy perception.

**Domain**: `Perception_Vision`

## Fix

Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.

## Anti-pattern

Single-modality occupancy mapping that ignores self-occlusion and sensor layout constraints.

## Cross-domain analogies

- **Planning_Decision** → Hierarchical decomposition of body coordination into unified occlusion-aware subgoals.
  - related fix: Hierarchical planner (VLM-driven) decomposes language instructions into subgoals, paired with a whole-body policy trained in simulation that coordinates locomotion and manipulation in a unified action space, enabling zero-shot sim-to-real transfer.
- **Learning_Training** → Use synthetic self-view generation from unobserved poses to augment training data for occlusion-aware perception.
  - related fix: Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.
- **Control_Locomotion** → Use standardized occlusion benchmarks to train perception models on self-occluded viewpoints.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- humanoid_occupancy.before.py
+++ humanoid_occupancy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Humanoid robots suffer from kinematic self-occlusion and sensor blind spots due to body parts obstructing sensor views, leading to incomplete occupancy perception.

+# Fix    : Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.

+# Avoid  : Single-modality occupancy mapping that ignores self-occlusion and sensor layout constraints.

```
