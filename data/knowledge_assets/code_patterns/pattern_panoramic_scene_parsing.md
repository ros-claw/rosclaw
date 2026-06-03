---
pattern_id: pattern_panoramic_scene_parsing
applicable_symptoms: [panoramic_scene_parsing]
domain: Perception_Vision
---

# Standard narrow-FOV scene parsers fail to capture holistic spatial layout from 360° images, limiting embodied agents' spatial awareness for navigation.

**Domain**: `Perception_Vision`

## Fix

Use panoramic scene parsing on equirectangular or cubemap representations to extract floorplans, wall boundaries, and free-space regions from a single 360° RGB image.

## Anti-pattern

Using traditional narrow-field-of-view parsers that cannot reconstruct full 360° scene structure.

## Cross-domain analogies

- **Planning_Decision** → Use semantic landmarks as sparse 360° cues for holistic spatial layout parsing.
  - related fix: Use semantic signage (e.g., 'Oncology Wing') as navigation cues for instruction-based wayfinding.
- **Learning_Training** → Use the parser itself to score and select high-quality 360° views for iterative self-training.
  - related fix: Self-Refining Data Flywheel (SRDF): after initial training, use the Navigator model itself to filter and score candidate trajectories, retaining only high-confidence or high-reward pairs for iterative fine-tuning.
- **Control_Locomotion** → Use panoramic depth maps to dynamically adapt attention and parsing across full 360° views.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- panoramic_scene_parsing.before.py
+++ panoramic_scene_parsing.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Standard narrow-FOV scene parsers fail to capture holistic spatial layout from 360° images, limiting embodied agents' spatial awareness for navigation.

+# Fix    : Use panoramic scene parsing on equirectangular or cubemap representations to extract floorplans, wall boundaries, and free-space regions from a single 360° RGB image.

+# Avoid  : Using traditional narrow-field-of-view parsers that cannot reconstruct full 360° scene structure.

```
