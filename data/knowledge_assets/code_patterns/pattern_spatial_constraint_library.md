---
pattern_id: pattern_spatial_constraint_library
applicable_symptoms: [spatial_constraint_library]
domain: Planning_Decision
---

# VLN agent fails to ground natural language spatial relations into structured constraints for navigation

**Domain**: `Planning_Decision`

## Fix

Use a Spatial Constraint Library with predefined spatial relationship types, retrieved via DAG queries within the GC-VLN framework

## Anti-pattern

Using ad-hoc or hand-coded spatial relation parsing without a structured constraint library

## Cross-domain analogies

- **Perception_Vision** → Apply Laplacian variance filtering to preprocess language embeddings to suppress noisy spatial relation signals before grounding.
  - related fix: Apply Laplacian Variance Filtering to stabilize camera feed before detection.
- **Learning_Training** → Train a latent language-to-spatial constraint model enabling internal simulation of instruction compliance.
  - related fix: Train a neural world model that predicts future latent states and rewards from current observations and actions, enabling model-based planning and mental simulation without direct environment interaction.
- **Control_Locomotion** → Use lightweight learned grounding to directly map language to navigation constraints at inference time.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- spatial_constraint_library.before.py
+++ spatial_constraint_library.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to ground natural language spatial relations into structured constraints for navigation

+# Fix    : Use a Spatial Constraint Library with predefined spatial relationship types, retrieved via DAG queries within the GC-VLN framework

+# Avoid  : Using ad-hoc or hand-coded spatial relation parsing without a structured constraint library

```
