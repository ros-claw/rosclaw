---
pattern_id: pattern_model_free_learning
applicable_symptoms: [model_free_learning]
domain: Learning_Training
---

# Model-based control requires explicit dynamics or terrain models, which are hard to obtain and generalize poorly to novel environments.

**Domain**: `Learning_Training`

## Fix

Train a neural network end-to-end to map sensor observations directly to actions using reinforcement learning or imitation learning, without building an explicit world model.

## Anti-pattern

Using privileged terrain maps or model-based control that requires precomputed dynamics models.

## Cross-domain analogies

- **Perception_Vision** → Incremental object-centric mapping inspires learning dynamics as sparse, updatable local models from online interaction.
  - related fix: Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.
- **Planning_Decision** → Embed physical constraints into network computations to bypass explicit dynamics modeling.
  - related fix: Plug-and-play proximal alternating-minimization network (PAN) that embeds physical constraints into network computations, enabling efficient alternating minimization for collision-free trajectory generation.
- **Control_Locomotion** → Use standardized benchmark tasks to learn implicit dynamics models from interaction data.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- model_free_learning.before.py
+++ model_free_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Model-based control requires explicit dynamics or terrain models, which are hard to obtain and generalize poorly to novel environments.

+# Fix    : Train a neural network end-to-end to map sensor observations directly to actions using reinforcement learning or imitation learning, without building an explicit world model.

+# Avoid  : Using privileged terrain maps or model-based control that requires precomputed dynamics models.

```
