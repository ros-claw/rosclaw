---
pattern_id: pattern_language_inferred_factor_graph_for_instruction_following_lifgif
applicable_symptoms: [language_inferred_factor_graph_for_instruction_following_lifgif]
domain: Planning_Decision
---

# Zero-shot instruction following fails in novel environments without pre-training or pre-mapping

**Domain**: `Planning_Decision`

## Fix

Construct a factor graph jointly representing spatial landmarks and instruction semantics, then perform inference to infer the correct path

## Anti-pattern

Object Goal Navigation and Vision Language Navigation baselines that require pre-training or pre-mapped environments

## Cross-domain analogies

- **Perception_Vision** → Use a transformer-based 3D decoder to predict occupancy and semantics from multi-camera images for zero-shot planning.
  - related fix: Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.
- **Learning_Training** → Use video-only input with domain randomization to eliminate reliance on pre-maps for zero-shot instruction following.
  - related fix: Use video-only input modality (no depth or map) combined with domain randomization to eliminate sensor fidelity and geometry transfer gaps
- **Control_Locomotion** → Use diffusion policies to sample diverse candidate plans and discretize high-level actions for robust zero-shot instruction following.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- language_inferred_factor_graph_for_instruction_following_lifgif.before.py
+++ language_inferred_factor_graph_for_instruction_following_lifgif.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Zero-shot instruction following fails in novel environments without pre-training or pre-mapping

+# Fix    : Construct a factor graph jointly representing spatial landmarks and instruction semantics, then perform inference to infer the correct path

+# Avoid  : Object Goal Navigation and Vision Language Navigation baselines that require pre-training or pre-mapped environments

```
