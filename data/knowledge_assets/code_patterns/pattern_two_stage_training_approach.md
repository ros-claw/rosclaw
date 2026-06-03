---
pattern_id: pattern_two_stage_training_approach
applicable_symptoms: [two_stage_training_approach]
domain: Learning_Training
---

# VLN agent trained purely with supervised learning fails to generalize to unseen environments and ambiguous instructions

**Domain**: `Learning_Training`

## Fix

Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward

## Anti-pattern

Pure supervised imitation learning without RL fine-tuning

## Cross-domain analogies

- **Perception_Vision** → Regularize latent path offsets with geometric scene constraints to align vision and language without explicit mapping.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Planning_Decision** → Use closed-loop verification of candidate actions to score alignment during training.
  - related fix: Visibility-based Viewpoint Decision module that scores candidate poses based on visibility and semantic alignment to resolve the last mile problem.
- **Control_Locomotion** → Use EB-Manipulation-style standardized benchmarks to train VLN agents on precise spatial reasoning and closed-loop verification.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- two_stage_training_approach.before.py
+++ two_stage_training_approach.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent trained purely with supervised learning fails to generalize to unseen environments and ambiguous instructions

+# Fix    : Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward

+# Avoid  : Pure supervised imitation learning without RL fine-tuning

```
