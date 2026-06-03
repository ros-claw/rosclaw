---
pattern_id: pattern_counterfactual_inverse_reinforcement_learning_irl
applicable_symptoms: [counterfactual_inverse_reinforcement_learning_irl]
domain: Learning_Training
---

# Standard IRL methods fail to distinguish essential environmental cues from irrelevant ones, leading to ambiguous cost functions and slow convergence.

**Domain**: `Learning_Training`

## Fix

Use counterfactual trajectory demonstrations: generate and analyze alternative paths that could have been taken, then contrast them with expert trajectories to focus learning on the most critical features for navigation cost inference.

## Anti-pattern

Standard IRL that treats all observed features equally without counterfactual reasoning.

## Cross-domain analogies

- **Perception_Vision** → Use implicit geometric priors to learn latent cost features that filter irrelevant cues.
  - related fix: Fine-tune a long-horizon visual-geometry backbone to predict metric-scale depth and pose directly from images, enabling implicit state estimation and dense geometry reconstruction without external sensors.
- **Planning_Decision** → Use structured evaluation benchmarks to isolate essential cues via standardized task decomposition.
  - related fix: Use the ODYSSEY benchmark for structured evaluation, which provides standardized tasks that test the interplay of mobility and dexterity under natural language instructions, enabling systematic assessment of long-horizon planning and coordination.
- **Control_Locomotion** → Closed-loop attention weighting that refines cost features using real-time relevance feedback.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- counterfactual_inverse_reinforcement_learning_irl.before.py
+++ counterfactual_inverse_reinforcement_learning_irl.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Standard IRL methods fail to distinguish essential environmental cues from irrelevant ones, leading to ambiguous cost functions and slow convergence.

+# Fix    : Use counterfactual trajectory demonstrations: generate and analyze alternative paths that could have been taken, then contrast them with expert trajectories to focus learning on the most critical features for navigation cost inference.

+# Avoid  : Standard IRL that treats all observed features equally without counterfactual reasoning.

```
