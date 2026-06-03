---
pattern_id: pattern_opengraph
applicable_symptoms: [opengraph]
domain: Perception_Vision
---

# Existing 3D mapping methods for outdoor environments fail to recognize arbitrary object categories not seen during training and lack hierarchical structure for navigation.

**Domain**: `Perception_Vision`

## Fix

Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.

## Anti-pattern

Closed-vocabulary 3D segmentation methods that cannot generalize to unseen object classes.

## Cross-domain analogies

- **Planning_Decision** → Pre-training on diverse visual data with masked object modeling and hierarchical prediction objectives.
  - related fix: Pre-training on large-scale vision-and-language navigation data with masked language modeling and action prediction objectives
- **Learning_Training** → Use synthetic data generation and hierarchical imitation learning to map unseen objects.
  - related fix: Use synthetic instruction generation via speaker model and large-scale unlabeled 3D scans, then train with imitation learning on the augmented dataset.
- **Control_Locomotion** → Multi-expert distillation with DAgger could enable incremental learning of novel object categories via iterative expert-guided data augmentation.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- opengraph.before.py
+++ opengraph.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Existing 3D mapping methods for outdoor environments fail to recognize arbitrary object categories not seen during training and lack hierarchical structure for navigation.

+# Fix    : Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.

+# Avoid  : Closed-vocabulary 3D segmentation methods that cannot generalize to unseen object classes.

```
