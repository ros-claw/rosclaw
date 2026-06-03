---
pattern_id: pattern_a_new_path_scaling_vision_and_language_navigation_with_synthetic_instructions_an
applicable_symptoms: [a_new_path_scaling_vision_and_language_navigation_with_synthetic_instructions_an]
domain: Learning_Training
---

# VLN agents trained on limited human-annotated instructions fail to generalize to new environments and long instructions.

**Domain**: `Learning_Training`

## Fix

Use synthetic instruction generation via speaker model and large-scale unlabeled 3D scans, then train with imitation learning on the augmented dataset.

## Anti-pattern

Training only on human-annotated RxR dataset without synthetic augmentation.

## Cross-domain analogies

- **Perception_Vision** → Multi-task learning on shared representations to jointly predict diverse instruction features.
  - related fix: Multi-task learning jointly predicting 3D occupancy, room layout, and object bounding boxes from a shared volumetric representation
- **Planning_Decision** → Decompose long instructions into entity-level subgoals for localized training supervision.
  - related fix: Fine-grained entity-level alignment: map each entity phrase (e.g., 'the red chair') to a specific visual landmark independently, rather than aligning the whole instruction globally.
- **Control_Locomotion** → Closed-loop verification using local perceptual grounding to reconcile high-level instructions with real-time visual feedback.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- a_new_path_scaling_vision_and_language_navigation_with_synthetic_instructions_an.before.py
+++ a_new_path_scaling_vision_and_language_navigation_with_synthetic_instructions_an.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents trained on limited human-annotated instructions fail to generalize to new environments and long instructions.

+# Fix    : Use synthetic instruction generation via speaker model and large-scale unlabeled 3D scans, then train with imitation learning on the augmented dataset.

+# Avoid  : Training only on human-annotated RxR dataset without synthetic augmentation.

```
