---
pattern_id: pattern_navid
applicable_symptoms: [navid]
domain: Planning_Decision
---

# Sim-to-real gap in VLN due to reliance on maps, odometry, or depth sensors that are noisy or unavailable in real environments

**Domain**: `Planning_Decision`

## Fix

Use a video-based VLM that directly outputs actions from a monocular RGB video stream, eliminating maps, odometry, and depth inputs

## Anti-pattern

Map-based or depth-based navigation methods that require explicit geometric representations

## Cross-domain analogies

- **Perception_Vision** → Fuse multiple noisy or missing modalities via a sensor layout strategy to maximize coverage and handle gaps.
  - related fix: Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.
- **Learning_Training** → Use human demonstration data to train a policy that directly maps observations to actions, bypassing noisy maps.
  - related fix: Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.
- **Control_Locomotion** → Train a model-free RL policy with domain randomization fusing vision and commands to replace noisy maps with learned visual navigation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- navid.before.py
+++ navid.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap in VLN due to reliance on maps, odometry, or depth sensors that are noisy or unavailable in real environments

+# Fix    : Use a video-based VLM that directly outputs actions from a monocular RGB video stream, eliminating maps, odometry, and depth inputs

+# Avoid  : Map-based or depth-based navigation methods that require explicit geometric representations

```
