---
pattern_id: pattern_navigator
applicable_symptoms: [navigator]
domain: Learning_Training
---

# Language-guided navigation models trained on noisy instruction-trajectory pairs suffer from performance degradation due to low-quality data.

**Domain**: `Learning_Training`

## Fix

Self-Refining Data Flywheel (SRDF): after initial training, use the Navigator model itself to filter and score candidate trajectories, retaining only high-confidence or high-reward pairs for iterative fine-tuning.

## Anti-pattern

Training on a large, possibly noisy dataset without automatic data filtering.

## Cross-domain analogies

- **Perception_Vision** → Use structured 3D cell aggregation to discretize noisy trajectories into clean, unified spatial-action cells for robust training.
  - related fix: Voxelize the physical world into structured 3D cells and aggregate multi-view 2D features into that unified 3D space via 2D-3D spatial sampling, then jointly predict 3D occupancy, room layout, and bounding boxes through multi-task learning.
- **Planning_Decision** → Use two-stage training: coarse policy from clean data, then fine-tune with noisy data via closed-loop verification.
  - related fix: Modular architecture with two-stage process: coarse path generation from language, then low-level controller for smooth trajectory following
- **Control_Locomotion** → Use diffusion to model diverse trajectory distributions from noisy instruction data.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- navigator.before.py
+++ navigator.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Language-guided navigation models trained on noisy instruction-trajectory pairs suffer from performance degradation due to low-quality data.

+# Fix    : Self-Refining Data Flywheel (SRDF): after initial training, use the Navigator model itself to filter and score candidate trajectories, retaining only high-confidence or high-reward pairs for iterative fine-tuning.

+# Avoid  : Training on a large, possibly noisy dataset without automatic data filtering.

```
