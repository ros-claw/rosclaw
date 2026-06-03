---
pattern_id: pattern_deepmind_functional_regularisation_for_continual_learning
applicable_symptoms: [deepmind_functional_regularisation_for_continual_learning]
domain: Learning_Training
---

# Catastrophic forgetting in continual learning: model performance on previous tasks degrades sharply when learning new tasks.

**Domain**: `Learning_Training`

## Fix

Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.

## Anti-pattern

Standard fine-tuning without regularisation leads to catastrophic forgetting.

## Cross-domain analogies

- **Perception_Vision** → Use grounded entity-level replay to anchor prior task knowledge during new task learning.
  - related fix: Pre-train on GEL-R2R, a dataset with grounded entity-level annotations, to enable cross-modal alignment at the entity level.
- **Planning_Decision** → Represent instructions as graph constraints to structure task memory and prune parameter updates via constraint satisfaction.
  - related fix: Represent instructions as graph constraints (landmark nodes + spatial edges) and prune action space via constraint satisfaction at each step
- **Control_Locomotion** → Train a single policy with domain randomization to handle task distribution shifts, preventing catastrophic forgetting.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- deepmind_functional_regularisation_for_continual_learning.before.py
+++ deepmind_functional_regularisation_for_continual_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Catastrophic forgetting in continual learning: model performance on previous tasks degrades sharply when learning new tasks.

+# Fix    : Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.

+# Avoid  : Standard fine-tuning without regularisation leads to catastrophic forgetting.

```
