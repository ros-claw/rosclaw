---
pattern_id: pattern_back_translation_for_navigation
applicable_symptoms: [back_translation_for_navigation]
domain: Learning_Training
---

# VLN agent overfits to limited human-annotated instruction-path pairs, failing to generalize to unseen environments.

**Domain**: `Learning_Training`

## Fix

Use back-translation: generate new instructions from paths and new paths from instructions using a pre-trained model, combined with environmental dropout to create diverse training triplets.

## Anti-pattern

Training only on human-annotated data without augmentation.

## Cross-domain analogies

- **Perception_Vision** → Use a transformer-based 3D decoder to predict latent navigation goals from multi-view observations, reducing reliance on paired annotations.
  - related fix: Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.
- **Planning_Decision** → Use topological graph priors to structure exploration and reduce reliance on sparse human annotations.
  - related fix: Combine an abstract obstacle map-based waypoint predictor with a multimodal LLM prompted by a topological graph and visitation history to select waypoints and generate low-level actions.
- **Control_Locomotion** → Use large-scale synthetic data generation with randomization to augment limited human annotations.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- back_translation_for_navigation.before.py
+++ back_translation_for_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent overfits to limited human-annotated instruction-path pairs, failing to generalize to unseen environments.

+# Fix    : Use back-translation: generate new instructions from paths and new paths from instructions using a pre-trained model, combined with environmental dropout to create diverse training triplets.

+# Avoid  : Training only on human-annotated data without augmentation.

```
