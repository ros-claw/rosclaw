---
pattern_id: pattern_zero_shot_instruction_following
applicable_symptoms: [zero_shot_instruction_following]
domain: Planning_Decision
---

# Instruction following agents fail to generalize to novel environments without environment-specific fine-tuning.

**Domain**: `Planning_Decision`

## Fix

Decompose instruction understanding and path planning into factor graph inference using pre-trained LLMs and visual foundation models, without task-specific fine-tuning.

## Anti-pattern

Fully supervised instruction following requiring training on each target environment.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic ray constraints to regularize action offsets for geometry-aware instruction alignment without environment-specific tuning.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use sensor-calibrated synthetic training to simulate novel environments with realistic noise.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.
- **Control_Locomotion** → Train a safety critic to detect novel context and trigger a fallback policy.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- zero_shot_instruction_following.before.py
+++ zero_shot_instruction_following.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Instruction following agents fail to generalize to novel environments without environment-specific fine-tuning.

+# Fix    : Decompose instruction understanding and path planning into factor graph inference using pre-trained LLMs and visual foundation models, without task-specific fine-tuning.

+# Avoid  : Fully supervised instruction following requiring training on each target environment.

```
