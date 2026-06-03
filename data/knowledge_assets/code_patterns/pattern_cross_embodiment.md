---
pattern_id: pattern_cross_embodiment
applicable_symptoms: [cross_embodiment]
domain: Learning_Training
---

# Single-embodiment policies require complete retraining when hardware changes, limiting scalable deployment.

**Domain**: `Learning_Training`

## Fix

Train a single policy on shared representations that abstract away physical differences across robot morphologies.

## Anti-pattern

Training separate policies for each robot morphology.

## Cross-domain analogies

- **Perception_Vision** → Active multi-policy fusion or predictive adaptation to handle hardware variations.
  - related fix: Active mitigation strategies such as multi-sensor fusion or predictive reacquisition to handle blind zones
- **Planning_Decision** → Pre-train a cross-embodiment policy using masked hardware parameters and action prediction objectives.
  - related fix: Pre-training on large-scale vision-and-language navigation data with masked language modeling and action prediction objectives
- **Control_Locomotion** → Use hierarchical decomposition to separate embodiment-specific low-level control from shared high-level planning.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- cross_embodiment.before.py
+++ cross_embodiment.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Single-embodiment policies require complete retraining when hardware changes, limiting scalable deployment.

+# Fix    : Train a single policy on shared representations that abstract away physical differences across robot morphologies.

+# Avoid  : Training separate policies for each robot morphology.

```
