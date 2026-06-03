---
pattern_id: pattern_extreme_parkour_robot
applicable_symptoms: [extreme_parkour_robot]
domain: Control_Locomotion
---

# Low-cost legged robot with imprecise actuation and a single low-frequency, jittery depth camera fails to perform agile parkour maneuvers reliably.

**Domain**: `Control_Locomotion`

## Fix

Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Anti-pattern

Traditional model-based controllers requiring precise state estimation and high-fidelity actuation.

## Cross-domain analogies

- **Perception_Vision** → Use pinhole projection to model actuation noise as extrinsic uncertainty, enabling closed-loop calibration.
  - related fix: Use pinhole camera projection model with intrinsic matrix K and extrinsic matrix [R|t] to map 3D world points to 2D image coordinates, enabling local-to-world transformations.
- **Planning_Decision** → Use discrete-continuous benchmark environments with dynamic obstacles and real-world validation to evaluate robust parkour policies under sensor noise.
  - related fix: HA-VLN benchmark with discrete-continuous environments, dynamic multi-human interactions, real-world validation, and an open leaderboard to evaluate human-aware navigation policies.
- **Learning_Training** → Use the robot's own execution traces to filter and retain only high-confidence parkour sequences for iterative policy refinement.
  - related fix: Self-Refining Data Flywheel (SRDF): after initial training, use the Navigator model itself to filter and score candidate trajectories, retaining only high-confidence or high-reward pairs for iterative fine-tuning.

## Patch

```diff
--- extreme_parkour_robot.before.py
+++ extreme_parkour_robot.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Low-cost legged robot with imprecise actuation and a single low-frequency, jittery depth camera fails to perform agile parkour maneuvers reliably.

+# Fix    : Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

+# Avoid  : Traditional model-based controllers requiring precise state estimation and high-fidelity actuation.

```
