---
pattern_id: pattern_navila_legged_robot_vision_language_action_model_for_navigation
applicable_symptoms: [navila_legged_robot_vision_language_action_model_for_navigation]
domain: Planning_Decision
---

# Legged robot navigation with language instructions fails due to high-level planning not translating to low-level motor commands, causing execution errors.

**Domain**: `Planning_Decision`

## Fix

Two-stage architecture: VLM outputs mid-level action tokens (e.g., 'move forward 0.5m') which are then executed by a separate RL-based locomotion policy that maps visual observations to motor commands.

## Anti-pattern

End-to-end VLM directly outputting low-level motor commands, which fails due to lack of grounding in robot dynamics and control.

## Cross-domain analogies

- **Perception_Vision** → Dual-view prompt: fuse high-level plan and low-level state into a single action token.
  - related fix: Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.
- **Learning_Training** → Pretrain motor primitives on demonstration data before hierarchical planning.
  - related fix: Use supervised fine-tuning (SFT) on expert demonstration trajectories to bootstrap a behavioral prior before reinforcement learning
- **Control_Locomotion** → Closed-loop verification reconciles high-level language plans with real-time motor commands.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- navila_legged_robot_vision_language_action_model_for_navigation.before.py
+++ navila_legged_robot_vision_language_action_model_for_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Legged robot navigation with language instructions fails due to high-level planning not translating to low-level motor commands, causing execution errors.

+# Fix    : Two-stage architecture: VLM outputs mid-level action tokens (e.g., 'move forward 0.5m') which are then executed by a separate RL-based locomotion policy that maps visual observations to motor commands.

+# Avoid  : End-to-end VLM directly outputting low-level motor commands, which fails due to lack of grounding in robot dynamics and control.

```
