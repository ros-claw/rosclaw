---
pattern_id: pattern_visual_jittering
applicable_symptoms: [visual_jittering]
domain: Perception_Vision
---

# High-frequency camera motion on legged robots causes successive image frames to shift unpredictably, degrading object detection, feature tracking, and visual odometry, especially over long-range navigation.

**Domain**: `Perception_Vision`

## Fix

Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.

## Anti-pattern

Using raw camera streams without filtering on legged platforms.

## Cross-domain analogies

- **Planning_Decision** → Use hierarchical decomposition to stabilize vision by grounding each frame to local features.
  - related fix: Use a cross-modal translator module that maps language instructions into a sequence of sub-goals, each grounded in visual landmarks, and a hierarchical policy that executes sub-goals sequentially.
- **Learning_Training** → Penalize large frame-to-frame feature shifts to stabilize visual perception across motion.
  - related fix: Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.
- **Control_Locomotion** → Use terrain-aware motion prediction to compensate for camera shake via adaptive frame stabilization.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- visual_jittering.before.py
+++ visual_jittering.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: High-frequency camera motion on legged robots causes successive image frames to shift unpredictably, degrading object detection, feature tracking, and visual odometry, especially over long-range navigation.

+# Fix    : Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.

+# Avoid  : Using raw camera streams without filtering on legged platforms.

```
