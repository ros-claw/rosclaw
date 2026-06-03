---
pattern_id: pattern_reverie_dataset
applicable_symptoms: [reverie_dataset]
domain: Planning_Decision
---

# VLN agents struggle to follow high-level language instructions and identify target objects in photorealistic environments.

**Domain**: `Planning_Decision`

## Fix

Actional Atomic-Concept Learning (AACL): learn atomic-level action representations from language instructions to improve navigation and grounding.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic ray constraints to regularize cross-view attention offsets for distortion-aware instruction alignment.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use a convolutional stem for local perception then global attention for long-range language-vision alignment.
  - related fix: Use a convolutional stem followed by Transformer blocks with global attention to model dependencies across 100kb+ distances
- **Control_Locomotion** → Use a safety-critic to override VLN actions when instruction-objective risk is high.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- reverie_dataset.before.py
+++ reverie_dataset.after.py
@@ -1,2 +1,3 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents struggle to follow high-level language instructions and identify target objects in photorealistic environments.

+# Fix    : Actional Atomic-Concept Learning (AACL): learn atomic-level action representations from language instructions to improve navigation and grounding.

```
