---
pattern_id: pattern_data_domain_gap
applicable_symptoms: [data_domain_gap]
domain: Perception_Vision
---

# VLA models for aerial navigation fail to generalize from synthetic training data to real-world deployment due to distributional mismatch in lighting, textures, weather, and sensor noise.

**Domain**: `Perception_Vision`

## Fix

Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.

## Anti-pattern

Training on purely synthetic data without photorealism or covering edge cases.

## Cross-domain analogies

- **Planning_Decision** → Use hierarchical decomposition to ground synthetic aerial features into real-world visual sub-goals.
  - related fix: Use a cross-modal translator module that maps language instructions into a sequence of sub-goals, each grounded in visual landmarks, and a hierarchical policy that executes sub-goals sequentially.
- **Learning_Training** → Use synthetic domain randomization with real-world priors for imitation learning.
  - related fix: Use synthetic instruction generation via speaker model and large-scale unlabeled 3D scans, then train with imitation learning on the augmented dataset.
- **Control_Locomotion** → Train domain-randomized RL policy fusing real sensor noise with synthetic data for closed-loop visual adaptation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- data_domain_gap.before.py
+++ data_domain_gap.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLA models for aerial navigation fail to generalize from synthetic training data to real-world deployment due to distributional mismatch in lighting, textures, weather, and sensor noise.

+# Fix    : Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.

+# Avoid  : Training on purely synthetic data without photorealism or covering edge cases.

```
