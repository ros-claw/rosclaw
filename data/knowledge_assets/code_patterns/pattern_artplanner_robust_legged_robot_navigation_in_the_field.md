---
pattern_id: pattern_artplanner_robust_legged_robot_navigation_in_the_field
applicable_symptoms: [artplanner_robust_legged_robot_navigation_in_the_field]
domain: Planning_Decision
---

# Legged robot navigation fails in unstructured outdoor terrain due to inaccurate elevation maps and high planning latency.

**Domain**: `Planning_Decision`

## Fix

ArtPlanner: an elevation mapping and path planning framework that uses a 2.5D grid with adaptive resolution and a fast A* variant to handle rough terrain in real time.

## Anti-pattern

Traditional elevation mapping with fixed resolution and standard A* planning suffers from high computational cost and poor terrain representation in field environments.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with proximal and distal sub-networks for real-time terrain mapping and planning.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Learning_Training** → End-to-end learning from raw sensor data to actions bypasses explicit terrain modeling and reduces planning latency.
  - related fix: Train a neural network end-to-end to map sensor observations directly to actions using reinforcement learning or imitation learning, without building an explicit world model.
- **Control_Locomotion** → Use reinforcement learning to map sensor data directly to navigation actions, bypassing explicit elevation mapping.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- artplanner_robust_legged_robot_navigation_in_the_field.before.py
+++ artplanner_robust_legged_robot_navigation_in_the_field.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Legged robot navigation fails in unstructured outdoor terrain due to inaccurate elevation maps and high planning latency.

+# Fix    : ArtPlanner: an elevation mapping and path planning framework that uses a 2.5D grid with adaptive resolution and a fast A* variant to handle rough terrain in real time.

+# Avoid  : Traditional elevation mapping with fixed resolution and standard A* planning suffers from high computational cost and poor terrain representation in field environments.

```
