---
pattern_id: pattern_multi_floor_abstraction
applicable_symptoms: [multi_floor_abstraction]
domain: Planning_Decision
---

# Mobile robots fail to plan paths across multiple floors because staircases are not distinguished from ramps or elevators, and cross-floor connectivity is not modeled.

**Domain**: `Planning_Decision`

## Fix

Multi-Floor Abstraction: hierarchical environment representation with stair-aware obstacle mapping and cross-floor topology modeling, linking floor-level occupancy grids via edge nodes for staircases.

## Anti-pattern

Treating all vertical connectors uniformly without explicit stair geometry modeling.

## Cross-domain analogies

- **Perception_Vision** → Use a pre-trained multimodal embedding to unify floor-transition semantics for zero-shot cross-floor planning.
  - related fix: Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.
- **Learning_Training** → Model cross-floor connectivity with occlusion-aware ray casting to distinguish staircases from ramps.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.
- **Control_Locomotion** → Use a standardized benchmark to train agents on precise cross-floor connectivity perception and action selection.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- multi_floor_abstraction.before.py
+++ multi_floor_abstraction.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Mobile robots fail to plan paths across multiple floors because staircases are not distinguished from ramps or elevators, and cross-floor connectivity is not modeled.

+# Fix    : Multi-Floor Abstraction: hierarchical environment representation with stair-aware obstacle mapping and cross-floor topology modeling, linking floor-level occupancy grids via edge nodes for staircases.

+# Avoid  : Treating all vertical connectors uniformly without explicit stair geometry modeling.

```
