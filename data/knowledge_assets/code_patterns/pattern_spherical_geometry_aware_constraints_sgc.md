---
pattern_id: pattern_spherical_geometry_aware_constraints_sgc
applicable_symptoms: [spherical_geometry_aware_constraints_sgc]
domain: Perception_Vision
---

# Panoramic camera distortion causes feature matching drift and poor geometric alignment in humanoid egocentric perception.

**Domain**: `Perception_Vision`

## Fix

Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.

## Anti-pattern

Standard undistortion-based feature matching fails under severe radial distortion of wide-FOV panoramic imagery.

## Cross-domain analogies

- **Planning_Decision** → Active multimodal querying corrects distortion by cross-referencing visual features with human-verified cues.
  - related fix: Enable the agent to actively request and interpret multimodal instructions (natural language and visual cues) from a human assistant when uncertain.
- **Learning_Training** → Freeze high-level panoramic features; train local alignment separately.
  - related fix: Train System 1 (VLM) and System 2 (local navigation policy) separately: freeze or fine-tune the VLM on high-level tasks, and train the navigation policy via RL or IL on environment-specific interactions.
- **Control_Locomotion** → Train a separate distortion-aware correction policy that overrides nominal features when alignment risk exceeds a threshold.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- spherical_geometry_aware_constraints_sgc.before.py
+++ spherical_geometry_aware_constraints_sgc.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Panoramic camera distortion causes feature matching drift and poor geometric alignment in humanoid egocentric perception.

+# Fix    : Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.

+# Avoid  : Standard undistortion-based feature matching fails under severe radial distortion of wide-FOV panoramic imagery.

```
