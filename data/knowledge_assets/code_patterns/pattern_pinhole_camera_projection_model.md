---
pattern_id: pattern_pinhole_camera_projection_model
applicable_symptoms: [pinhole_camera_projection_model]
domain: Perception_Vision
---

# Visual observations from different camera poses cannot be fused into a consistent world coordinate frame for semantic mapping.

**Domain**: `Perception_Vision`

## Fix

Use pinhole camera projection model with intrinsic matrix K and extrinsic matrix [R|t] to map 3D world points to 2D image coordinates, enabling local-to-world transformations.

## Anti-pattern

Using a non-rectilinear or multi-viewpoint camera model that distorts straight lines and complicates geometric consistency.

## Cross-domain analogies

- **Planning_Decision** → Use hierarchical neural radiance fields to predict and align multi-pose observations into a consistent coordinate frame.
  - related fix: Use a Lookahead Exploration Strategy that constructs a navigable future path tree via Hierarchical Neural Radiance Representation Model (HNR) to evaluate candidate locations in parallel based on multi-level semantic features.
- **Learning_Training** → Train a privileged pose-aware encoder, then distill into a pose-agnostic student via a consistency loss.
  - related fix: Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.
- **Control_Locomotion** → Use depth-based pose alignment to dynamically adapt coordinate frame fusion.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- pinhole_camera_projection_model.before.py
+++ pinhole_camera_projection_model.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Visual observations from different camera poses cannot be fused into a consistent world coordinate frame for semantic mapping.

+# Fix    : Use pinhole camera projection model with intrinsic matrix K and extrinsic matrix [R|t] to map 3D world points to 2D image coordinates, enabling local-to-world transformations.

+# Avoid  : Using a non-rectilinear or multi-viewpoint camera model that distorts straight lines and complicates geometric consistency.

```
