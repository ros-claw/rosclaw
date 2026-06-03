---
pattern_id: pattern_dataset_aggregation_dagger
applicable_symptoms: [dataset_aggregation_dagger]
domain: Learning_Training
---

# Behavioral cloning suffers from covariate shift: the policy encounters states during deployment that differ from the expert demonstration distribution, leading to compounding errors.

**Domain**: `Learning_Training`

## Fix

Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).

## Anti-pattern

Behavioral cloning with a fixed dataset of expert demonstrations.

## Cross-domain analogies

- **Perception_Vision** → Use 3D-GS-style data augmentation to generate on-policy states from sparse expert demonstrations.
  - related fix: Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.
- **Planning_Decision** → Joint training on expert and self-generated trajectories reduces distribution mismatch.
  - related fix: Unified architecture with shared route and language encoders feeding two decoders (action prediction and instruction generation), trained jointly on both objectives via pretrain-then-fine-tune.
- **Control_Locomotion** → Use terrain-aware adaptation to condition the policy on state distribution cues, reducing covariate shift.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- dataset_aggregation_dagger.before.py
+++ dataset_aggregation_dagger.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Behavioral cloning suffers from covariate shift: the policy encounters states during deployment that differ from the expert demonstration distribution, leading to compounding errors.

+# Fix    : Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).

+# Avoid  : Behavioral cloning with a fixed dataset of expert demonstrations.

```
