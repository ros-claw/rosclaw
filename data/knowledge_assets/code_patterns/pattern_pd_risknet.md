---
pattern_id: pattern_pd_risknet
applicable_symptoms: [pd_risknet]
domain: Perception_Vision
---

# Locomotion policies fail to anticipate dynamic obstacles in far-field regions, leading to collisions or inefficient paths.

**Domain**: `Perception_Vision`

## Fix

Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.

## Anti-pattern

Single-scale spatial processing without temporal dynamics or far-field awareness.

## Cross-domain analogies

- **Planning_Decision** → Use curiosity-driven weighting to prioritize far-field dynamic obstacle prediction in locomotion planning.
  - related fix: Use curiosity-driven weighting to combine CVL scores (spatial distribution from visual-language features) with exploration bonus for goal selection, then employ a traditional planner for obstacle avoidance.
- **Learning_Training** → Apply multi-scale dropout to perception features to prevent over-reliance on near-field cues.
  - related fix: Apply dropout operations at multiple feature scales (activations, channels, spatial regions, entire feature maps) to regularize training.
- **Control_Locomotion** → Use a standardized benchmark with far-field dynamic obstacle tasks to train perception for anticipatory spatial reasoning.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- pd_risknet.before.py
+++ pd_risknet.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Locomotion policies fail to anticipate dynamic obstacles in far-field regions, leading to collisions or inefficient paths.

+# Fix    : Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.

+# Avoid  : Single-scale spatial processing without temporal dynamics or far-field awareness.

```
