---
pattern_id: pattern_visual_locomotion_rl_policy
applicable_symptoms: [visual_locomotion_rl_policy]
domain: Control_Locomotion
---

# Legged robots fail to traverse complex terrain or avoid small obstacles when relying solely on proprioception or high-level commands without visual adaptation.

**Domain**: `Control_Locomotion`

## Fix

Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Anti-pattern

Using only proprioceptive feedback or open-loop control without visual input leads to poor performance on uneven terrain and cluttered environments.

## Cross-domain analogies

- **Perception_Vision** → Pre-train on grounded visual-terrain annotations for entity-level obstacle alignment.
  - related fix: Pre-train on GEL-R2R, a dataset with grounded entity-level annotations, to enable cross-modal alignment at the entity level.
- **Planning_Decision** → Augment locomotion control with visual semantic and passibility features to filter foothold candidates.
  - related fix: Augment waypoint predictor with semantic and passibility features from the environment (e.g., obstacle labels, terrain traversability) to filter or score candidate waypoints.
- **Learning_Training** → Use human demonstration data to train visual adaptation policies for terrain-aware locomotion.
  - related fix: Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.

## Patch

```diff
--- visual_locomotion_rl_policy.before.py
+++ visual_locomotion_rl_policy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Legged robots fail to traverse complex terrain or avoid small obstacles when relying solely on proprioception or high-level commands without visual adaptation.

+# Fix    : Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

+# Avoid  : Using only proprioceptive feedback or open-loop control without visual input leads to poor performance on uneven terrain and cluttered environments.

```
