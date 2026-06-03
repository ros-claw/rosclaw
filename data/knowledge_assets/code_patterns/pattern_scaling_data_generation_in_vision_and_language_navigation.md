---
pattern_id: pattern_scaling_data_generation_in_vision_and_language_navigation
applicable_symptoms: [scaling_data_generation_in_vision_and_language_navigation]
domain: Learning_Training
---

# VLN agents overfit to limited training environments and fail to generalize to unseen scenes due to insufficient data diversity.

**Domain**: `Learning_Training`

## Fix

ScaleVLN: large-scale synthetic data generation by combining 3D scene graphs with LLM-generated instructions and augmenting with panoramic views and object-level grounding.

## Anti-pattern

Training only on human-annotated datasets like R2R without synthetic augmentation.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with proximal and distal sub-networks to separate near-field and far-field feature processing.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Planning_Decision** → Use reinforcement learning with rule-based policies to generate diverse synthetic trajectories for data augmentation.
  - related fix: Fine-tune a vision-language model via reinforcement learning with rule-based policies and a long-horizon planner that generates actions using value-based rewards.
- **Control_Locomotion** → Use diffusion policies to model diverse navigation trajectories from limited training environments.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- scaling_data_generation_in_vision_and_language_navigation.before.py
+++ scaling_data_generation_in_vision_and_language_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents overfit to limited training environments and fail to generalize to unseen scenes due to insufficient data diversity.

+# Fix    : ScaleVLN: large-scale synthetic data generation by combining 3D scene graphs with LLM-generated instructions and augmenting with panoramic views and object-level grounding.

+# Avoid  : Training only on human-annotated datasets like R2R without synthetic augmentation.

```
