---
pattern_id: pattern_rl_navigation_policy
applicable_symptoms: [rl_navigation_policy]
domain: Planning_Decision
---

# Classical navigation stacks fail in dynamic, crowded environments due to reliance on explicit maps and global planners.

**Domain**: `Planning_Decision`

## Fix

Train an end-to-end RL policy that maps raw sensor data (lidar, depth) directly to motor commands using reward signals for goal success, collision avoidance, and energy efficiency.

## Anti-pattern

Using pre-built metric maps with global planners that cannot adapt to moving obstacles or changing layouts.

## Cross-domain analogies

- **Perception_Vision** → Train a neural network on simulated crowd trajectories to predict local navigation actions.
  - related fix: Use deep learning models (e.g., CNNs) trained on simulated galaxy merger images to automatically classify merger stages.
- **Learning_Training** → Unified shared representation model co-trained on diverse crowd scenarios for adaptive real-time planning.
  - related fix: Unified multi-task model co-trained on all VLNVerse benchmark tasks (goal-oriented navigation, language-guided exploration, instruction following) using shared transformer-based cross-modal attention and common visual-linguistic embedding space.
- **Control_Locomotion** → Train a model-free RL policy with domain randomization fusing raw sensor inputs for closed-loop dynamic obstacle avoidance.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- rl_navigation_policy.before.py
+++ rl_navigation_policy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Classical navigation stacks fail in dynamic, crowded environments due to reliance on explicit maps and global planners.

+# Fix    : Train an end-to-end RL policy that maps raw sensor data (lidar, depth) directly to motor commands using reward signals for goal success, collision avoidance, and energy efficiency.

+# Avoid  : Using pre-built metric maps with global planners that cannot adapt to moving obstacles or changing layouts.

```
