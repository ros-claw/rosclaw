---
pattern_id: pattern_two_stage_training_pipeline_for_reverie
applicable_symptoms: [two_stage_training_pipeline_for_reverie]
domain: Planning_Decision
---

# VLN agent fails to localize remote target objects from high-level instructions and navigate efficiently in real indoor environments.

**Domain**: `Planning_Decision`

## Fix

Two-stage training pipeline: first pretrain with cross-modal alignment sub-tasks (scene grounding and object grounding) without action supervision, then train a memory-augmented attentive action decoder to generate action sequences.

## Anti-pattern

End-to-end training without explicit cross-modal alignment or memory augmentation.

## Cross-domain analogies

- **Perception_Vision** → Use a transformer-based 3D decoder to predict a continuous semantic occupancy map from visual observations for goal-directed navigation.
  - related fix: Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.
- **Learning_Training** → Train an end-to-end policy mapping visual observations and instructions directly to actions, bypassing explicit map building.
  - related fix: Train a neural network end-to-end to map sensor observations directly to actions using reinforcement learning or imitation learning, without building an explicit world model.
- **Control_Locomotion** → End-to-end neural policy mapping instructions and visual input directly to navigation actions.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- two_stage_training_pipeline_for_reverie.before.py
+++ two_stage_training_pipeline_for_reverie.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to localize remote target objects from high-level instructions and navigate efficiently in real indoor environments.

+# Fix    : Two-stage training pipeline: first pretrain with cross-modal alignment sub-tasks (scene grounding and object grounding) without action supervision, then train a memory-augmented attentive action decoder to generate action sequences.

+# Avoid  : End-to-end training without explicit cross-modal alignment or memory augmentation.

```
