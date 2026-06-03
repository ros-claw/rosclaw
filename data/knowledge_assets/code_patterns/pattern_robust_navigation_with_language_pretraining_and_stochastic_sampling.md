---
pattern_id: pattern_robust_navigation_with_language_pretraining_and_stochastic_sampling
applicable_symptoms: [robust_navigation_with_language_pretraining_and_stochastic_sampling]
domain: Planning_Decision
---

# VLN agent ignores landmark cues in long instructions

**Domain**: `Planning_Decision`

## Fix

Use language pretraining (BERT) to encode instructions and stochastic sampling during decoding to improve robustness to instruction variations

## Anti-pattern

Deterministic greedy decoding from LSTM-based seq2seq models

## Cross-domain analogies

- **Perception_Vision** → Apply Laplacian variance filtering to prioritize high-salience landmark cues over noisy distractors.
  - related fix: Apply Laplacian Variance Filtering to stabilize camera feed before detection.
- **Learning_Training** → Use back-translation to generate diverse landmark-augmented instructions from unlabeled trajectories.
  - related fix: Use back translation: generate new paths and instructions from unlabeled trajectory data via a learned translator, combined with environmental dropout for visual perturbations.
- **Control_Locomotion** → Map visual features to actions to incorporate landmark cues into navigation decisions.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- robust_navigation_with_language_pretraining_and_stochastic_sampling.before.py
+++ robust_navigation_with_language_pretraining_and_stochastic_sampling.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions

+# Fix    : Use language pretraining (BERT) to encode instructions and stochastic sampling during decoding to improve robustness to instruction variations

+# Avoid  : Deterministic greedy decoding from LSTM-based seq2seq models

```
