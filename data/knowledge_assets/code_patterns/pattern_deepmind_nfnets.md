---
pattern_id: pattern_deepmind_nfnets
applicable_symptoms: [deepmind_nfnets]
domain: Learning_Training
---

# Batch normalization introduces training instability and batch-size dependence, limiting model performance and scalability.

**Domain**: `Learning_Training`

## Fix

Use adaptive gradient clipping (AGC) and Scaled Weight Standardization to train deep networks without batch normalization.

## Anti-pattern

Standard batch normalization with small batch sizes or distributed training.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic normalization across all batch dimensions to remove batch-size dependence.
  - related fix: Use panoramic scene parsing on equirectangular or cubemap representations to extract floorplans, wall boundaries, and free-space regions from a single 360° RGB image.
- **Planning_Decision** → Replace batch normalization with implicit normalization via auxiliary geometric consistency tasks.
  - related fix: Fine-tune a long-horizon visual-geometry backbone with auxiliary tasks (metric scale grounding, scene geometry reconstruction, implicit geometry bootstrapping) to output metric-scale predictions and condition the policy on implicit geometry, enabling fully end-to-end navigation without a separate localization module.
- **Control_Locomotion** → Distill batch normalization into per-sample adaptive scaling with closed-loop fine-tuning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- deepmind_nfnets.before.py
+++ deepmind_nfnets.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Batch normalization introduces training instability and batch-size dependence, limiting model performance and scalability.

+# Fix    : Use adaptive gradient clipping (AGC) and Scaled Weight Standardization to train deep networks without batch normalization.

+# Avoid  : Standard batch normalization with small batch sizes or distributed training.

```
