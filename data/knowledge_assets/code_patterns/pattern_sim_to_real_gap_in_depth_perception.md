---
pattern_id: pattern_sim_to_real_gap_in_depth_perception
applicable_symptoms: [sim_to_real_gap_in_depth_perception]
domain: Perception_Vision
---

# Depth policies trained on clean simulated depth maps fail on real sensors due to missing noise, occlusions, and lighting artifacts.

**Domain**: `Perception_Vision`

## Fix

Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.

## Anti-pattern

Using idealized depth maps from simulation without augmentation.

## Cross-domain analogies

- **Planning_Decision** → Use a staged pipeline to decompose depth estimation into noise, occlusion, and lighting sub-tasks.
  - related fix: Use the TOFRA five-stage pipeline (Transition, Observation, Fusion, Reward-policy construction, Action) to structure navigation systems, integrating sensing, social, and motion intelligence across stages.
- **Learning_Training** → Train a depth-to-noise model to generate synthetic noisy depth maps from clean data for augmentation.
  - related fix: Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.
- **Control_Locomotion** → Train lightweight policy on simulated depth with injected noise and occlusions.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- sim_to_real_gap_in_depth_perception.before.py
+++ sim_to_real_gap_in_depth_perception.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Depth policies trained on clean simulated depth maps fail on real sensors due to missing noise, occlusions, and lighting artifacts.

+# Fix    : Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.

+# Avoid  : Using idealized depth maps from simulation without augmentation.

```
