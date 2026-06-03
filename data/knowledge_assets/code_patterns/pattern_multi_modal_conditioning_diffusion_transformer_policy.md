---
pattern_id: pattern_multi_modal_conditioning_diffusion_transformer_policy
applicable_symptoms: [multi_modal_conditioning_diffusion_transformer_policy]
domain: Planning_Decision
---

# VLN agent's low-level actions are jerky or inaccurate when conditioned only on high-level goals, causing poor real-time trajectory following.

**Domain**: `Planning_Decision`

## Fix

Use a Diffusion Transformer policy with multi-modal conditioning (pixel goals + latent features) as System 1 in a dual-system architecture to generate smooth, continuous trajectories in real time.

## Anti-pattern

Using a single monolithic policy that directly maps high-level goals to low-level actions without explicit pixel goals or latent feature conditioning.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with closed-loop verification from VLM and LiDAR layers.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Learning_Training** → Use synthetic low-level action sequences to augment training data for smoother trajectory prediction.
  - related fix: Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.
- **Control_Locomotion** → Train a low-level safety shield policy that overrides jerky actions when trajectory deviation exceeds a threshold.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- multi_modal_conditioning_diffusion_transformer_policy.before.py
+++ multi_modal_conditioning_diffusion_transformer_policy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent's low-level actions are jerky or inaccurate when conditioned only on high-level goals, causing poor real-time trajectory following.

+# Fix    : Use a Diffusion Transformer policy with multi-modal conditioning (pixel goals + latent features) as System 1 in a dual-system architecture to generate smooth, continuous trajectories in real time.

+# Avoid  : Using a single monolithic policy that directly maps high-level goals to low-level actions without explicit pixel goals or latent feature conditioning.

```
