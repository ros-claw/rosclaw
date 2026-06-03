---
pattern_id: pattern_zero_shot_object_goal_visual_navigation
applicable_symptoms: [zero_shot_object_goal_visual_navigation]
domain: Planning_Decision
---

# Traditional object goal navigation fails to generalize to novel object classes not seen during training, requiring retraining or fine-tuning for each new target category.

**Domain**: `Planning_Decision`

## Fix

Use semantic embeddings (e.g., from language models or knowledge graphs) to compute similarity between target description and observed scene features, enabling zero-shot navigation to unseen object classes.

## Anti-pattern

Training class-specific visual features for every possible target object category.

## Cross-domain analogies

- **Perception_Vision** → Use learned semantic representations to enable zero-shot generalization to novel object categories.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Learning_Training** → Train on multiple object-goal datasets jointly to learn shared object representations for zero-shot generalization.
  - related fix: Jointly train the VLN model on multiple annotated datasets (RxR and R2R) using multitask learning to learn shared visual and linguistic representations, improving robustness and generalization.
- **Control_Locomotion** → Use reinforcement learning to map visual observations directly to navigation actions for novel objects.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- zero_shot_object_goal_visual_navigation.before.py
+++ zero_shot_object_goal_visual_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Traditional object goal navigation fails to generalize to novel object classes not seen during training, requiring retraining or fine-tuning for each new target category.

+# Fix    : Use semantic embeddings (e.g., from language models or knowledge graphs) to compute similarity between target description and observed scene features, enabling zero-shot navigation to unseen object classes.

+# Avoid  : Training class-specific visual features for every possible target object category.

```
