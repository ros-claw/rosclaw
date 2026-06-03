---
pattern_id: pattern_volume_state_estimation
applicable_symptoms: [volume_state_estimation]
domain: Planning_Decision
---

# VLN agent lacks closed-loop state estimation from online volumetric data, causing poor next-step prediction without full map reconstruction.

**Domain**: `Planning_Decision`

## Fix

Volume State Estimation: encode current Volumetric Environment Representation into a latent state, feed into episodic memory for next-step prediction.

## Anti-pattern

Full map reconstruction before navigation step prediction.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic scene parsing to extract local occupancy from 360° views for closed-loop state estimation.
  - related fix: Use panoramic scene parsing on equirectangular or cubemap representations to extract floorplans, wall boundaries, and free-space regions from a single 360° RGB image.
- **Learning_Training** → Use pretrained vision-language models to generate synthetic volumetric state estimates for closed-loop next-step prediction.
  - related fix: Use Marky, a multilingual instruction generator that produces visually grounded instruction-trajectory pairs at scale (4.2M pairs) by leveraging pretrained vision-language models and spatial alignment.
- **Control_Locomotion** → Closed-loop local volumetric fusion with real-time traversability feedback replaces full map reconstruction for next-step prediction.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- volume_state_estimation.before.py
+++ volume_state_estimation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent lacks closed-loop state estimation from online volumetric data, causing poor next-step prediction without full map reconstruction.

+# Fix    : Volume State Estimation: encode current Volumetric Environment Representation into a latent state, feed into episodic memory for next-step prediction.

+# Avoid  : Full map reconstruction before navigation step prediction.

```
