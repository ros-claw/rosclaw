---
pattern_id: pattern_birds_eye_view_bev_perceptual_representations
applicable_symptoms: [birds_eye_view_bev_perceptual_representations]
domain: Perception_Vision
---

# Planning in autonomous driving fails to capture spatial layout and navigation costs due to lack of structured top-down representation.

**Domain**: `Perception_Vision`

## Fix

Project sensory data into a bird's-eye-view grid representation, distilled from visual foundation models, to enable structured spatial reasoning for planning.

## Anti-pattern

Using raw perspective camera images directly for planning without explicit spatial aggregation.

## Cross-domain analogies

- **Planning_Decision** → Use dual separate encoders for spatial layout and navigation cost, updated via sliding window.
  - related fix: Dual implicit memory with separate 2D semantic encoder (Qwen2.5-VL) and 3D spatial encoder (VGGT), updated via sliding window for dynamic incremental history.
- **Learning_Training** → Use closed-loop data aggregation to iteratively refine a learned top-down cost map from on-policy driving experience.
  - related fix: Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).
- **Control_Locomotion** → Use a lightweight learned top-down cost map from simulation, executed at low resolution for real-time spatial planning.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- birds_eye_view_bev_perceptual_representations.before.py
+++ birds_eye_view_bev_perceptual_representations.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Planning in autonomous driving fails to capture spatial layout and navigation costs due to lack of structured top-down representation.

+# Fix    : Project sensory data into a bird's-eye-view grid representation, distilled from visual foundation models, to enable structured spatial reasoning for planning.

+# Avoid  : Using raw perspective camera images directly for planning without explicit spatial aggregation.

```
