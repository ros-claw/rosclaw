---
pattern_id: pattern_global_topological_path_planning
applicable_symptoms: [global_topological_path_planning]
domain: Planning_Decision
---

# Long-horizon navigation fails due to computational cost and brittleness of global metric maps over large distances or unstructured environments.

**Domain**: `Planning_Decision`

## Fix

Use a topological graph of object-level sub-goals (e.g., 'doorway', 'kitchen counter') for global planning, with local metric control (e.g., DWA, MPC) for sub-goal execution.

## Anti-pattern

Traditional path planning relying on metric maps (occupancy grids, point clouds) for long-horizon navigation.

## Cross-domain analogies

- **Perception_Vision** → Use learned semantic representations to compress long-horizon navigation into hierarchical, cost-efficient abstractions.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Learning_Training** → Use counterfactual trajectory rollouts to prune irrelevant local subgoals, enabling sparse hierarchical planning.
  - related fix: Use counterfactual trajectory demonstrations: generate and analyze alternative paths that could have been taken, then contrast them with expert trajectories to focus learning on the most critical features for navigation cost inference.
- **Control_Locomotion** → Distill local reactive policies from global planners with closed-loop depth-based fine-tuning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- global_topological_path_planning.before.py
+++ global_topological_path_planning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon navigation fails due to computational cost and brittleness of global metric maps over large distances or unstructured environments.

+# Fix    : Use a topological graph of object-level sub-goals (e.g., 'doorway', 'kitchen counter') for global planning, with local metric control (e.g., DWA, MPC) for sub-goal execution.

+# Avoid  : Traditional path planning relying on metric maps (occupancy grids, point clouds) for long-horizon navigation.

```
