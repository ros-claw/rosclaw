---
pattern_id: pattern_noise_aware_modeling
applicable_symptoms: [noise_aware_modeling]
domain: Learning_Training
---

# Sim-to-real gap causes depth-based policies to fail on real hardware due to unrealistic synthetic depth images.

**Domain**: `Learning_Training`

## Fix

Augment synthetic depth images with noise patterns (Gaussian blur, quantization artifacts, dropout) during training.

## Anti-pattern

Training on clean synthetic depth images without noise injection.

## Cross-domain analogies

- **Perception_Vision** → Replace depth input with language descriptions of geometry from a vision-language model.
  - related fix: Replace visual features with language-based representations (e.g., captions from a vision-language model) for navigation policy input.
- **Planning_Decision** → Use closed-loop verification to detect depth discrepancies and trigger adaptive retraining.
  - related fix: Adaptive replanning via Advisor module (assess alternatives) and Arborist module (restructure plan) using LLM reasoning.
- **Control_Locomotion** → Train a lightweight policy on simulated depth with noise augmentation to enable real-time inference on real hardware.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- noise_aware_modeling.before.py
+++ noise_aware_modeling.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap causes depth-based policies to fail on real hardware due to unrealistic synthetic depth images.

+# Fix    : Augment synthetic depth images with noise patterns (Gaussian blur, quantization artifacts, dropout) during training.

+# Avoid  : Training on clean synthetic depth images without noise injection.

```
