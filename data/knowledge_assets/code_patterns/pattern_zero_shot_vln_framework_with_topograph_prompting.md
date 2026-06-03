---
pattern_id: pattern_zero_shot_vln_framework_with_topograph_prompting
applicable_symptoms: [zero_shot_vln_framework_with_topograph_prompting]
domain: Planning_Decision
---

# Zero-shot VLN agents fail in continuous environments without task-specific training, achieving low success rates on R2R-CE and RxR-CE benchmarks.

**Domain**: `Planning_Decision`

## Fix

Combine an abstract obstacle map-based waypoint predictor with a multimodal LLM prompted by a topological graph and visitation history to select waypoints and generate low-level actions.

## Anti-pattern

Training-based VLN methods that require environment-specific fine-tuning and cannot generalize to unseen environments.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic visual imagination to augment training for zero-shot continuous navigation.
  - related fix: Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.
- **Learning_Training** → Use counterfactual trajectory contrast to identify critical navigation cues for zero-shot generalization.
  - related fix: Use counterfactual trajectory demonstrations: generate and analyze alternative paths that could have been taken, then contrast them with expert trajectories to focus learning on the most critical features for navigation cost inference.
- **Control_Locomotion** → Use closed-loop verification with standardized continuous-action benchmarks to train spatial reasoning.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- zero_shot_vln_framework_with_topograph_prompting.before.py
+++ zero_shot_vln_framework_with_topograph_prompting.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Zero-shot VLN agents fail in continuous environments without task-specific training, achieving low success rates on R2R-CE and RxR-CE benchmarks.

+# Fix    : Combine an abstract obstacle map-based waypoint predictor with a multimodal LLM prompted by a topological graph and visitation history to select waypoints and generate low-level actions.

+# Avoid  : Training-based VLN methods that require environment-specific fine-tuning and cannot generalize to unseen environments.

```
