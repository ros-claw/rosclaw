---
pattern_id: pattern_pre_training_and_fine_tuning_paradigm_for_vln
applicable_symptoms: [pre_training_and_fine_tuning_paradigm_for_vln]
domain: Learning_Training
---

# VLN agents fail to generalize to unseen environments when trained from scratch on limited task-specific data

**Domain**: `Learning_Training`

## Fix

Pre-train on large-scale image-text-action triplets via self-supervised learning, then fine-tune on downstream VLN tasks

## Anti-pattern

Training VLN models from scratch on task-specific datasets without pre-training

## Cross-domain analogies

- **Perception_Vision** → Augment training data with simulated distributional shifts to match target deployment conditions.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Planning_Decision** → End-to-end differentiable training with fine-tuning via backpropagation on unified representations.
  - related fix: Unify perception, planning, and control into a single differentiable computation graph with a learned model that can be fine-tuned via backpropagation.
- **Control_Locomotion** → Train a separate safety critic to override the primary policy when confidence in novel scenes is low.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- pre_training_and_fine_tuning_paradigm_for_vln.before.py
+++ pre_training_and_fine_tuning_paradigm_for_vln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize to unseen environments when trained from scratch on limited task-specific data

+# Fix    : Pre-train on large-scale image-text-action triplets via self-supervised learning, then fine-tune on downstream VLN tasks

+# Avoid  : Training VLN models from scratch on task-specific datasets without pre-training

```
