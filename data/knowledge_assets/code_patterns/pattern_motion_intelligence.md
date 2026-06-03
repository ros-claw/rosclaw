---
pattern_id: pattern_motion_intelligence
applicable_symptoms: [motion_intelligence]
domain: Planning_Decision
---

# Navigation system computes paths that are physically infeasible for the robot to follow due to ignoring kinematics, dynamics, or environmental constraints.

**Domain**: `Planning_Decision`

## Fix

Combine learning-based motion planners with classical control to generate collision-free trajectories that respect kinematics, dynamics, and environmental constraints, and adapt plans in real time to dynamic obstacles.

## Anti-pattern

Using purely geometric motion planning without accounting for robot dynamics or real-time adaptation.

## Cross-domain analogies

- **Perception_Vision** → Use learned sampling points to selectively attend to feasible trajectory regions instead of full path space.
  - related fix: Use deformable cross-attention with learned sampling points to selectively attend to relevant image features instead of the entire grid.
- **Learning_Training** → Use closed-loop verification to filter planned paths against a learned feasibility model.
  - related fix: Self-Refining Data Flywheel: generate synthetic navigation trajectories via a teacher policy, filter with a learned verifier, and iteratively retrain the student policy on the augmented data.
- **Control_Locomotion** → Train a model-free policy with domain randomization to fuse visual and kinematic constraints for feasible path generation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- motion_intelligence.before.py
+++ motion_intelligence.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation system computes paths that are physically infeasible for the robot to follow due to ignoring kinematics, dynamics, or environmental constraints.

+# Fix    : Combine learning-based motion planners with classical control to generate collision-free trajectories that respect kinematics, dynamics, and environmental constraints, and adapt plans in real time to dynamic obstacles.

+# Avoid  : Using purely geometric motion planning without accounting for robot dynamics or real-time adaptation.

```
