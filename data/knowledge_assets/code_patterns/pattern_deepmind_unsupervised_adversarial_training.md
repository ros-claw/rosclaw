---
pattern_id: pattern_deepmind_unsupervised_adversarial_training
applicable_symptoms: [deepmind_unsupervised_adversarial_training]
domain: Learning_Training
---

# Adversarial training requires large labeled datasets, which are expensive or unavailable.

**Domain**: `Learning_Training`

## Fix

Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.

## Anti-pattern

Standard adversarial training with supervised labels.

## Cross-domain analogies

- **Perception_Vision** → Use variance-based filtering to discard low-quality training samples that degrade adversarial robustness.
  - related fix: Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.
- **Planning_Decision** → Use synthetic data from diverse agent constraints to generate labeled adversarial examples.
  - related fix: CapNav benchmark: evaluate VLMs on 5 agent types with distinct constraints across 45 scenes, 473 tasks, and 2,365 QA pairs to test capability-aware navigation.
- **Control_Locomotion** → Use a separate generative model to override the primary classifier when adversarial risk exceeds a threshold.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- deepmind_unsupervised_adversarial_training.before.py
+++ deepmind_unsupervised_adversarial_training.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Adversarial training requires large labeled datasets, which are expensive or unavailable.

+# Fix    : Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.

+# Avoid  : Standard adversarial training with supervised labels.

```
