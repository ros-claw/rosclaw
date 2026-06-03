---
pattern_id: pattern_slam
applicable_symptoms: [slam]
domain: Perception_Vision
---

# Video world models fail to capture accurate robot motion and spatial understanding when trained only on raw visual data.

**Domain**: `Perception_Vision`

## Fix

Use SLAM-derived trajectories as ground-truth motion tendency references to supervise or condition video world model predictions.

## Anti-pattern

Training video world models solely on visual reconstruction fidelity without explicit motion priors.

## Cross-domain analogies

- **Planning_Decision** → Cache visual-spatial trajectories from exploration for retrieval, bypassing full video world model inference.
  - related fix: Cache task-location trajectories from an exploration phase and retrieve them for reuse, bypassing full planning pipeline when a known task at a known location is encountered.
- **Learning_Training** → Use a motion model to generate synthetic visual trajectories from unlabeled robot actions, then augment training data.
  - related fix: Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.
- **Control_Locomotion** → Train a policy that directly maps noisy depth images to motor commands using domain randomization and sim-to-real transfer.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- slam.before.py
+++ slam.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Video world models fail to capture accurate robot motion and spatial understanding when trained only on raw visual data.

+# Fix    : Use SLAM-derived trajectories as ground-truth motion tendency references to supervise or condition video world model predictions.

+# Avoid  : Training video world models solely on visual reconstruction fidelity without explicit motion priors.

```
