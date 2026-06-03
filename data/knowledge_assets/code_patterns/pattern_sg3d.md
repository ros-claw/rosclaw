---
pattern_id: pattern_sg3d
applicable_symptoms: [sg3d]
domain: Planning_Decision
---

# Embodied navigation agents struggle to generalize across diverse 3D scenes and fail to answer semantic queries during navigation.

**Domain**: `Planning_Decision`

## Fix

Use the SG3D benchmark to evaluate navigation and question-answering performance, and adopt the MTU3D multi-task unified architecture that integrates mapping, planning, and obstacle avoidance into a single model.

## Anti-pattern

Previous state-of-the-art methods that treat navigation and question-answering as separate tasks.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic scene parsing for holistic spatial-semantic grounding to answer queries during navigation.
  - related fix: Use panoramic scene parsing on equirectangular or cubemap representations to extract floorplans, wall boundaries, and free-space regions from a single 360° RGB image.
- **Learning_Training** → Use adaptive normalization and gradient scaling to stabilize scene-agnostic feature learning for robust semantic navigation.
  - related fix: Use adaptive gradient clipping (AGC) and Scaled Weight Standardization to train deep networks without batch normalization.
- **Control_Locomotion** → Use closed-loop verification to retry alternative semantic queries when a scene interpretation fails.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- sg3d.before.py
+++ sg3d.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied navigation agents struggle to generalize across diverse 3D scenes and fail to answer semantic queries during navigation.

+# Fix    : Use the SG3D benchmark to evaluate navigation and question-answering performance, and adopt the MTU3D multi-task unified architecture that integrates mapping, planning, and obstacle avoidance into a single model.

+# Avoid  : Previous state-of-the-art methods that treat navigation and question-answering as separate tasks.

```
