---
pattern_id: pattern_td_stp_target_driven_structured_transformer_planner
applicable_symptoms: [td_stp_target_driven_structured_transformer_planner]
domain: Planning_Decision
---

# VLN agents struggle with long-horizon navigation and spatial awareness in complex indoor environments, leading to suboptimal success rates on R2R and REVERIE benchmarks.

**Domain**: `Planning_Decision`

## Fix

Use a hierarchical transformer that explicitly estimates long-term navigation targets and incorporates room layout into structured attention for global planning.

## Anti-pattern

Prior methods lacked explicit target prediction and room layout awareness, resulting in lower success rates.

## Cross-domain analogies

- **Perception_Vision** → Use cross-view semantic alignment to enforce spatial consistency across long-horizon planning steps.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Learning_Training** → Use causal intervention to remove spurious correlations from visual features for robust long-horizon planning.
  - related fix: Use causal counterfactual reasoning to remove the influence of sensitive attributes on predictions by intervening on the causal graph.
- **Control_Locomotion** → Train an end-to-end policy from noisy visual inputs to actions with domain randomization for robust long-horizon navigation.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- td_stp_target_driven_structured_transformer_planner.before.py
+++ td_stp_target_driven_structured_transformer_planner.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents struggle with long-horizon navigation and spatial awareness in complex indoor environments, leading to suboptimal success rates on R2R and REVERIE benchmarks.

+# Fix    : Use a hierarchical transformer that explicitly estimates long-term navigation targets and incorporates room layout into structured attention for global planning.

+# Avoid  : Prior methods lacked explicit target prediction and room layout awareness, resulting in lower success rates.

```
