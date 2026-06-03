---
pattern_id: pattern_deepmind_counterfactual_fairness
applicable_symptoms: [deepmind_counterfactual_fairness]
domain: Learning_Training
---

# ML models exhibit bias due to sensitive attributes in training data, leading to unfair predictions.

**Domain**: `Learning_Training`

## Fix

Use causal counterfactual reasoning to remove the influence of sensitive attributes on predictions by intervening on the causal graph.

## Anti-pattern

Demographic parity or equalized odds without causal modeling can still encode proxy discrimination.

## Cross-domain analogies

- **Perception_Vision** → Use cross-view semantic alignment to enforce invariant feature representations across sensitive attribute subgroups during training.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Planning_Decision** → Use hierarchical scene graph construction to structure training data, mitigating bias by decomposing sensitive attributes.
  - related fix: Use hierarchical scene graph construction from a semantic object map to provide structured, open-vocabulary environment context to the LLM, enabling multi-step plan generation and real-time re-planning.
- **Control_Locomotion** → Train a single policy that maps noisy inputs to outputs via domain randomization to mitigate bias from sensitive attributes.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- deepmind_counterfactual_fairness.before.py
+++ deepmind_counterfactual_fairness.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: ML models exhibit bias due to sensitive attributes in training data, leading to unfair predictions.

+# Fix    : Use causal counterfactual reasoning to remove the influence of sensitive attributes on predictions by intervening on the causal graph.

+# Avoid  : Demographic parity or equalized odds without causal modeling can still encode proxy discrimination.

```
