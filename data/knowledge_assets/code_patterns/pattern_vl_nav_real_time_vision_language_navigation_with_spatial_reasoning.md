---
pattern_id: pattern_vl_nav_real_time_vision_language_navigation_with_spatial_reasoning
applicable_symptoms: [vl_nav_real_time_vision_language_navigation_with_spatial_reasoning]
domain: Planning_Decision
---

# VLN agents fail to balance following human instructions with exploring unknown areas, leading to low success rates in real-time navigation.

**Domain**: `Planning_Decision`

## Fix

Use curiosity-driven weighting to combine CVL scores (spatial distribution from visual-language features) with exploration bonus for goal selection, then employ a traditional planner for obstacle avoidance.

## Anti-pattern

Pure instruction-following without exploration weighting

## Cross-domain analogies

- **Perception_Vision** → Use a fixed-size instruction memory bottleneck to compress and prioritize navigation goals.
  - related fix: Use a cross-attention bottleneck: project arbitrary input to a fixed-size latent array via cross-attention, then process with iterative self-attention in latent space.
- **Learning_Training** → Use self-supervised pseudo-labeling to generate exploration rewards from instruction following.
  - related fix: Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.
- **Control_Locomotion** → Multi-expert distillation with DAgger enables instruction-following and exploration via iterative closed-loop imitation.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- vl_nav_real_time_vision_language_navigation_with_spatial_reasoning.before.py
+++ vl_nav_real_time_vision_language_navigation_with_spatial_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to balance following human instructions with exploring unknown areas, leading to low success rates in real-time navigation.

+# Fix    : Use curiosity-driven weighting to combine CVL scores (spatial distribution from visual-language features) with exploration bonus for goal selection, then employ a traditional planner for obstacle avoidance.

+# Avoid  : Pure instruction-following without exploration weighting

```
