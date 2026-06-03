---
pattern_id: pattern_scene_grounding_sub_task
applicable_symptoms: [scene_grounding_sub_task]
domain: Planning_Decision
---

# Agent fails to stop at the correct location during navigation, stopping early or missing the goal due to poor alignment between visual scenes and language instructions.

**Domain**: `Planning_Decision`

## Fix

Scene Grounding sub-task: pretrain a cross-modal alignment model using pairs of visual environments and language instructions to learn when and where to stop, with a binary output indicating whether the current viewpoint is the goal.

## Anti-pattern

Training directly on the full REVERIE task without explicit scene grounding pretraining.

## Cross-domain analogies

- **Perception_Vision** → Use a transformer-based 3D decoder to predict continuous goal occupancy from vision and language.
  - related fix: Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.
- **Learning_Training** → Use adaptive gradient alignment to stabilize visual-language goal localization without batch-dependent tuning.
  - related fix: Use adaptive gradient clipping (AGC) and Scaled Weight Standardization to train deep networks without batch normalization.
- **Control_Locomotion** → Apply closed-loop verification: after each stop attempt, check alignment and retry until goal match is confirmed.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- scene_grounding_sub_task.before.py
+++ scene_grounding_sub_task.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Agent fails to stop at the correct location during navigation, stopping early or missing the goal due to poor alignment between visual scenes and language instructions.

+# Fix    : Scene Grounding sub-task: pretrain a cross-modal alignment model using pairs of visual environments and language instructions to learn when and where to stop, with a binary output indicating whether the current viewpoint is the goal.

+# Avoid  : Training directly on the full REVERIE task without explicit scene grounding pretraining.

```
