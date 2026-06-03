---
pattern_id: pattern_envdrop
applicable_symptoms: [envdrop]
domain: Learning_Training
---

# VLN agent overfits to specific visual patterns and fails to generalize from seen to unseen environments, especially when visual observations are partially missing.

**Domain**: `Learning_Training`

## Fix

Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.

## Anti-pattern

Standard training without visual dropout leads to poor generalization and large performance gap between seen and unseen environments.

## Cross-domain analogies

- **Perception_Vision** → Use hierarchical open-vocabulary graph structure to decouple visual patterns from navigation policy.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Planning_Decision** → Decouple visual pattern encoding into separate modules with a unified scoring function to select robust features.
  - related fix: Explicitly decouple observation, reasoning, and correction into separate modules; formulate long-term memory construction as an optimization problem using a unified scoring function to select key frames from historical candidates.
- **Control_Locomotion** → Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- envdrop.before.py
+++ envdrop.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent overfits to specific visual patterns and fails to generalize from seen to unseen environments, especially when visual observations are partially missing.

+# Fix    : Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.

+# Avoid  : Standard training without visual dropout leads to poor generalization and large performance gap between seen and unseen environments.

```
