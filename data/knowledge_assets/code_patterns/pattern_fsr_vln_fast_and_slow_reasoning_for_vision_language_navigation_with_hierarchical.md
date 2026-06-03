---
pattern_id: pattern_fsr_vln_fast_and_slow_reasoning_for_vision_language_navigation_with_hierarchical
applicable_symptoms: [fsr_vln_fast_and_slow_reasoning_for_vision_language_navigation_with_hierarchical]
domain: Planning_Decision
---

# VLN agents fail to generalize to long-horizon tasks due to coarse scene representations and lack of hierarchical reasoning.

**Domain**: `Planning_Decision`

## Fix

Hierarchical Multi-Modal Scene Graph (HMSG) combining geometric, semantic, and topological maps for coarse-to-fine localization, plus Fast and Slow Reasoning (FSR) using VLM for goal selection.

## Anti-pattern

Flat scene graphs or single-level representations that cannot handle room-to-object granularity.

## Cross-domain analogies

- **Perception_Vision** → Predict 3D occupancy and semantics hierarchically for long-horizon VLN planning.
  - related fix: Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.
- **Learning_Training** → Hybrid supervised waypoint prediction with hierarchical RL for long-horizon planning.
  - related fix: Hybrid algorithm combining supervised learning for position prediction (waypoint predictor) with reinforcement learning for continuous control, trained jointly in simulation and real environments without requiring autonomous physical flight during training.
- **Control_Locomotion** → Use hierarchical decomposition with standardized sub-tasks for precise spatial reasoning.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- fsr_vln_fast_and_slow_reasoning_for_vision_language_navigation_with_hierarchical.before.py
+++ fsr_vln_fast_and_slow_reasoning_for_vision_language_navigation_with_hierarchical.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize to long-horizon tasks due to coarse scene representations and lack of hierarchical reasoning.

+# Fix    : Hierarchical Multi-Modal Scene Graph (HMSG) combining geometric, semantic, and topological maps for coarse-to-fine localization, plus Fast and Slow Reasoning (FSR) using VLM for goal selection.

+# Avoid  : Flat scene graphs or single-level representations that cannot handle room-to-object granularity.

```
