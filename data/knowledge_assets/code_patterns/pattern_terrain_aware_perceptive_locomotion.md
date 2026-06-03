---
pattern_id: pattern_terrain_aware_perceptive_locomotion
applicable_symptoms: [terrain_aware_perceptive_locomotion]
domain: Control_Locomotion
---

# Legged robots fail to maintain stability and traverse uneven or complex terrains (e.g., gravel, stairs, slopes) without prior knowledge of the environment.

**Domain**: `Control_Locomotion`

## Fix

Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Anti-pattern

Using only proprioceptive feedback or flat-ground assumptions for locomotion control on uneven terrain.

## Cross-domain analogies

- **Perception_Vision** → Use terrain-response variance filtering to discard or downweight unstable footholds.
  - related fix: Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.
- **Planning_Decision** → Hierarchical decomposition with structured attention for global terrain layout estimation.
  - related fix: Use a hierarchical transformer that explicitly estimates long-term navigation targets and incorporates room layout into structured attention for global planning.
- **Learning_Training** → Randomly mask sensor inputs during training to force reliance on proprioceptive priors.
  - related fix: Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.

## Patch

```diff
--- terrain_aware_perceptive_locomotion.before.py
+++ terrain_aware_perceptive_locomotion.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Legged robots fail to maintain stability and traverse uneven or complex terrains (e.g., gravel, stairs, slopes) without prior knowledge of the environment.

+# Fix    : Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

+# Avoid  : Using only proprioceptive feedback or flat-ground assumptions for locomotion control on uneven terrain.

```
