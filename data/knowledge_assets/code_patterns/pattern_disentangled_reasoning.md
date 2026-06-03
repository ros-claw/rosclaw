---
pattern_id: pattern_disentangled_reasoning
applicable_symptoms: [disentangled_reasoning]
domain: Planning_Decision
---

# Embodied navigation agents using direct perception-to-action mapping lack interpretability and robustness, making failures hard to diagnose.

**Domain**: `Planning_Decision`

## Fix

Decompose navigation decision into three stages: imagination (generate goal representation), observation selection (align goal with sensory input), and action determination (compute motor commands).

## Anti-pattern

Direct action prediction collapses perception and action into a single black-box model.

## Cross-domain analogies

- **Perception_Vision** → Implicit latent state estimation via learned geometric priors.
  - related fix: Fine-tune a long-horizon visual-geometry backbone to predict metric-scale depth and pose directly from images, enabling implicit state estimation and dense geometry reconstruction without external sensors.
- **Learning_Training** → Use full-kinematics simulation to generate interpretable intermediate state representations for closed-loop verification.
  - related fix: Use full-kinematics agents with a robust physics engine to enable realistic motion dynamics and high-fidelity simulation, reducing sim-to-real gap.
- **Control_Locomotion** → Distill multi-expert policies with closed-loop verification to improve interpretability and robustness.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- disentangled_reasoning.before.py
+++ disentangled_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied navigation agents using direct perception-to-action mapping lack interpretability and robustness, making failures hard to diagnose.

+# Fix    : Decompose navigation decision into three stages: imagination (generate goal representation), observation selection (align goal with sensory input), and action determination (compute motor commands).

+# Avoid  : Direct action prediction collapses perception and action into a single black-box model.

```
