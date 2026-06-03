---
pattern_id: pattern_deepmind_cs_gan
applicable_symptoms: [deepmind_cs_gan]
domain: Perception_Vision
---

# MRI reconstruction from undersampled measurements suffers from slow iterative optimization and poor image quality.

**Domain**: `Perception_Vision`

## Fix

Use a GAN with a compressed sensing loss to directly reconstruct images from undersampled k-space data in a single feedforward pass.

## Anti-pattern

Traditional compressed sensing methods rely on iterative optimization with handcrafted sparsity priors.

## Cross-domain analogies

- **Planning_Decision** → Use reactive inference to predict missing k-space data, enabling fast non-iterative reconstruction.
  - related fix: Use reactive planning with human intent inference and collision avoidance that generalizes beyond scripted motion, as modeled in HAPS 2.0 dataset and HA-VLN 2.0 benchmark.
- **Learning_Training** → Jointly train on multiple undersampling patterns to learn shared reconstruction priors for faster, higher-quality MRI.
  - related fix: Jointly train the VLN model on multiple annotated datasets (RxR and R2R) using multitask learning to learn shared visual and linguistic representations, improving robustness and generalization.
- **Control_Locomotion** → Pre-train a library of efficient reconstruction primitives via deep learning, decoupling acquisition from iterative optimization.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- deepmind_cs_gan.before.py
+++ deepmind_cs_gan.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: MRI reconstruction from undersampled measurements suffers from slow iterative optimization and poor image quality.

+# Fix    : Use a GAN with a compressed sensing loss to directly reconstruct images from undersampled k-space data in a single feedforward pass.

+# Avoid  : Traditional compressed sensing methods rely on iterative optimization with handcrafted sparsity priors.

```
