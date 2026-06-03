---
pattern_id: pattern_realistic_depth_images_synthetic_method
applicable_symptoms: [realistic_depth_images_synthetic_method]
domain: Learning_Training
---

# Sim-to-real gap for depth-based perception causes policy collapse on real hardware due to unrealistic self-occlusion and sensor noise in synthetic depth images.

**Domain**: `Learning_Training`

## Fix

Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.

## Anti-pattern

Naive rendering without modeling self-occlusion or sensor noise.

## Cross-domain analogies

- **Perception_Vision** → Simulate multi-scale depth realism by blending coarse synthetic layouts with fine real sensor noise patterns.
  - related fix: Use a coarse-to-fine pyramid (e.g., U-Net or FPN) that downsamples to capture coarse layout and upsamples to recover fine details, then fuse or sequentially feed multi-scale features.
- **Planning_Decision** → Use hierarchical decomposition to ground depth perception in real-world sensor signatures.
  - related fix: Use a cross-modal translator module that maps language instructions into a sequence of sub-goals, each grounded in visual landmarks, and a hierarchical policy that executes sub-goals sequentially.
- **Control_Locomotion** → Use closed-loop verification to detect and reject unrealistic depth artifacts by comparing against alternative sensor readings.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- realistic_depth_images_synthetic_method.before.py
+++ realistic_depth_images_synthetic_method.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap for depth-based perception causes policy collapse on real hardware due to unrealistic self-occlusion and sensor noise in synthetic depth images.

+# Fix    : Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.

+# Avoid  : Naive rendering without modeling self-occlusion or sensor noise.

```
