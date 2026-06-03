---
pattern_id: pattern_jetson_orin_agx
applicable_symptoms: [jetson_orin_agx]
domain: Systems_Compute
---

# Open-vocabulary feature extraction and semantic mapping on mobile robots require cloud connectivity due to high computational demands.

**Domain**: `Systems_Compute`

## Fix

Deploy Jetson Orin AGX as onboard computer to run open-vocabulary feature extraction and semantic mapping at real-time rates without cloud dependency.

## Anti-pattern

Using cloud-dependent computing for real-time semantic mapping on mobile robots.

## Cross-domain analogies

- **Perception_Vision** → Lightweight plug-and-play augmentation enforces local consistency to reduce cloud dependency.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Planning_Decision** → Predict future semantic features locally to reduce cloud dependency for mapping.
  - related fix: Train a visual imagination module that predicts future visual observations conditioned on language instructions and current visual input, then integrate imagined features into the navigation policy via cross-modal attention.
- **Learning_Training** → Use domain randomization to train a lightweight on-device feature extractor robust to real-world variability.
  - related fix: Domain randomization, system identification, or sim-to-real transfer techniques

## Patch

```diff
--- jetson_orin_agx.before.py
+++ jetson_orin_agx.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Open-vocabulary feature extraction and semantic mapping on mobile robots require cloud connectivity due to high computational demands.

+# Fix    : Deploy Jetson Orin AGX as onboard computer to run open-vocabulary feature extraction and semantic mapping at real-time rates without cloud dependency.

+# Avoid  : Using cloud-dependent computing for real-time semantic mapping on mobile robots.

```
