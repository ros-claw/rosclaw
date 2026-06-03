---
pattern_id: pattern_action_prediction_with_image_generation
applicable_symptoms: [action_prediction_with_image_generation]
domain: Planning_Decision
---

# VLN agents struggle to align language instructions with visual observations, leading to poor next-step prediction and navigation failures.

**Domain**: `Planning_Decision`

## Fix

Use APIG: a proxy pre-training task that generates the pixel-level next view conditioned on the full instruction and navigation history, forcing fine-grained language-vision alignment.

## Anti-pattern

Standard VLN pre-training without explicit visual generation fails to capture detailed spatial-semantic correspondences.

## Cross-domain analogies

- **Perception_Vision** → Cross-view augmentation enforces cross-modal alignment between language and visual observations during training.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Learning_Training** → Use full-kinematics alignment to ground language instructions in high-fidelity visual dynamics.
  - related fix: Use full-kinematics agents with a robust physics engine to enable realistic motion dynamics and high-fidelity simulation, reducing sim-to-real gap.
- **Control_Locomotion** → Use diffusion policies to discretize language-aligned action spaces for robust next-step prediction.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- action_prediction_with_image_generation.before.py
+++ action_prediction_with_image_generation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents struggle to align language instructions with visual observations, leading to poor next-step prediction and navigation failures.

+# Fix    : Use APIG: a proxy pre-training task that generates the pixel-level next view conditioned on the full instruction and navigation history, forcing fine-grained language-vision alignment.

+# Avoid  : Standard VLN pre-training without explicit visual generation fails to capture detailed spatial-semantic correspondences.

```
