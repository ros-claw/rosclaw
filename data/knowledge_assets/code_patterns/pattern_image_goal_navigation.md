---
pattern_id: pattern_image_goal_navigation
applicable_symptoms: [image_goal_navigation]
domain: Planning_Decision
---

# Agent cannot navigate to a location specified only by a goal image without metric coordinates or semantic labels.

**Domain**: `Planning_Decision`

## Fix

Use a learned visual representation (e.g., Siamese network or contrastive embedding) to match current observation to goal image, combined with an exploration policy (e.g., frontier-based or RL) that moves to reduce embedding distance.

## Anti-pattern

Classical visual navigation relying on metric maps or GPS coordinates.

## Cross-domain analogies

- **Perception_Vision** → Use 3D-GS to generate novel-view goal images from sparse real captures, enabling visual navigation without metric coordinates.
  - related fix: Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.
- **Learning_Training** → Provide diverse goal image datasets with standardized evaluation to improve navigation generalization.
  - related fix: Provide large-scale, diverse offline datasets (e.g., RL Unplugged) with standardized evaluation protocols for fair comparison.
- **Control_Locomotion** → Use camera images as direct goal-conditioned input to a learned navigation policy, bypassing explicit mapping.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- image_goal_navigation.before.py
+++ image_goal_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Agent cannot navigate to a location specified only by a goal image without metric coordinates or semantic labels.

+# Fix    : Use a learned visual representation (e.g., Siamese network or contrastive embedding) to match current observation to goal image, combined with an exploration policy (e.g., frontier-based or RL) that moves to reduce embedding distance.

+# Avoid  : Classical visual navigation relying on metric maps or GPS coordinates.

```
