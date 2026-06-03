---
pattern_id: pattern_self_supervised_learning
applicable_symptoms: [self_supervised_learning]
domain: Learning_Training
---

# VLN models require costly human annotations and fail to generalize to unseen environments without task-specific fine-tuning.

**Domain**: `Learning_Training`

## Fix

Pre-train on large-scale image-text-action triplets using self-supervised pretext tasks to learn generic representations that transfer to new navigation tasks.

## Anti-pattern

Supervised learning with manually annotated labels for each navigation task.

## Cross-domain analogies

- **Perception_Vision** → Fuse self-supervised losses from multiple auxiliary tasks with attention to prioritize informative training signals.
  - related fix: Implement a sensing intelligence pipeline that fuses multiple sensor modalities (e.g., vision, depth, IMU) into a coherent representation, with attention mechanisms to prioritize salient input.
- **Planning_Decision** → Use a structured 3D scene graph with LLM reasoning to enable zero-shot VLN without human annotations.
  - related fix: Use a multi-modal 3D scene graph that encodes object categories, spatial relations, and hierarchical structure, combined with a large language model for zero-shot goal reasoning and path planning.
- **Control_Locomotion** → Use diffusion policies to model diverse navigation trajectories from unlabeled video, enabling zero-shot generalization.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- self_supervised_learning.before.py
+++ self_supervised_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN models require costly human annotations and fail to generalize to unseen environments without task-specific fine-tuning.

+# Fix    : Pre-train on large-scale image-text-action triplets using self-supervised pretext tasks to learn generic representations that transfer to new navigation tasks.

+# Avoid  : Supervised learning with manually annotated labels for each navigation task.

```
