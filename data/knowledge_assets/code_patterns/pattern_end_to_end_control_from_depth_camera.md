---
pattern_id: pattern_end_to_end_control_from_depth_camera
applicable_symptoms: [end_to_end_control_from_depth_camera]
domain: Control_Locomotion
---

# Classical modular perception-actuation-control pipelines are brittle in novel environments due to hand-crafted subsystems requiring independent calibration and tuning.

**Domain**: `Control_Locomotion`

## Fix

Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Anti-pattern

Classical modular pipeline with separate perception, state estimation, and control modules.

## Cross-domain analogies

- **Perception_Vision** → Use SLAM-derived trajectories as ground-truth references to supervise end-to-end locomotion policy learning.
  - related fix: Use SLAM-derived trajectories as ground-truth motion tendency references to supervise or condition video world model predictions.
- **Planning_Decision** → Unify perception, actuation, and control into a differentiable end-to-end model for adaptive fine-tuning.
  - related fix: Unify perception, planning, and control into a single differentiable computation graph with a learned model that can be fine-tuned via backpropagation.
- **Learning_Training** → Use two-stage training: supervised imitation then reinforcement fine-tuning for end-to-end locomotion policy.
  - related fix: Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward

## Patch

```diff
--- end_to_end_control_from_depth_camera.before.py
+++ end_to_end_control_from_depth_camera.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Classical modular perception-actuation-control pipelines are brittle in novel environments due to hand-crafted subsystems requiring independent calibration and tuning.

+# Fix    : Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

+# Avoid  : Classical modular pipeline with separate perception, state estimation, and control modules.

```
