---
pattern_id: pattern_gc_vln_instruction_as_graph_constraints_for_training_free_vision_and_language_na
applicable_symptoms: [gc_vln_instruction_as_graph_constraints_for_training_free_vision_and_language_na]
domain: Planning_Decision
---

# VLN agent ignores landmark cues in long instructions

**Domain**: `Planning_Decision`

## Fix

Represent instructions as graph constraints (landmark nodes + spatial edges) and prune action space via constraint satisfaction at each step

## Anti-pattern

Training-based VLN methods that require expensive data collection and fail to generalize to unseen instructions

## Cross-domain analogies

- **Perception_Vision** → Apply Laplacian variance filtering to preprocess instruction embeddings for salient landmark weighting.
  - related fix: Apply Laplacian Variance Filtering to stabilize camera feed before detection.
- **Learning_Training** → Use back-translation to generate landmark-diverse instruction variants from paths.
  - related fix: Use back-translation: generate new instructions from paths and new paths from instructions using a pre-trained model, combined with environmental dropout to create diverse training triplets.
- **Control_Locomotion** → Use closed-loop verification to retry alternative landmark-triggered actions when visual cues are ignored.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- gc_vln_instruction_as_graph_constraints_for_training_free_vision_and_language_na.before.py
+++ gc_vln_instruction_as_graph_constraints_for_training_free_vision_and_language_na.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions

+# Fix    : Represent instructions as graph constraints (landmark nodes + spatial edges) and prune action space via constraint satisfaction at each step

+# Avoid  : Training-based VLN methods that require expensive data collection and fail to generalize to unseen instructions

```
