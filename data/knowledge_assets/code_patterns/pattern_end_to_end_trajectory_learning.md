---
pattern_id: pattern_end_to_end_trajectory_learning
applicable_symptoms: [end_to_end_trajectory_learning]
domain: Planning_Decision
---

# Navigation policies fail to generalize across diverse scenes and instruction formats when using hand-crafted features or heuristic planners.

**Domain**: `Planning_Decision`

## Fix

End-to-end trajectory learning with Vision-Language-Exploration pre-training over a million diverse RGB-D trajectories, directly mapping raw sensor observations to continuous commands.

## Anti-pattern

Modular approaches that separate perception, planning, and control (classical robotics pipeline).

## Cross-domain analogies

- **Perception_Vision** → Use a pre-trained vision-language model to learn a shared embedding space for zero-shot transfer across scenes and instructions.
  - related fix: Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.
- **Learning_Training** → Use modular expert policies with dynamic selection for scene-specific navigation.
  - related fix: Train three specialized RL experts (reaching, squeezing, avoiding) and combine them via dynamic weighting based on individual capability performance.
- **Control_Locomotion** → Train an end-to-end policy that maps raw scene observations and instructions directly to actions.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- end_to_end_trajectory_learning.before.py
+++ end_to_end_trajectory_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation policies fail to generalize across diverse scenes and instruction formats when using hand-crafted features or heuristic planners.

+# Fix    : End-to-end trajectory learning with Vision-Language-Exploration pre-training over a million diverse RGB-D trajectories, directly mapping raw sensor observations to continuous commands.

+# Avoid  : Modular approaches that separate perception, planning, and control (classical robotics pipeline).

```
