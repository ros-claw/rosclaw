---
pattern_id: pattern_supervised_reinforcement_asynchronous_learning
applicable_symptoms: [supervised_reinforcement_asynchronous_learning]
domain: Learning_Training
---

# Quadcopter cannot learn to follow natural language instructions in continuous control without risky and costly autonomous flight in the physical environment during training.

**Domain**: `Learning_Training`

## Fix

Hybrid algorithm combining supervised learning for position prediction (waypoint predictor) with reinforcement learning for continuous control, trained jointly in simulation and real environments without requiring autonomous physical flight during training.

## Anti-pattern

End-to-end reinforcement learning requiring autonomous flight in the physical environment during training.

## Cross-domain analogies

- **Perception_Vision** → Use language embeddings as policy input to enable sim-to-real transfer without physical flight.
  - related fix: Replace visual features with language-based representations (e.g., captions from a vision-language model) for navigation policy input.
- **Planning_Decision** → Use online semantic mapping and LLM parsing to simulate instruction-grounded training without physical flight.
  - related fix: Online visual-language mapping that builds and updates a semantic map from visual observations, combined with an LLM-based instruction parser and DD-PPO local controller.
- **Control_Locomotion** → Use sim-to-real transfer with domain randomization to train on language-conditioned control.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- supervised_reinforcement_asynchronous_learning.before.py
+++ supervised_reinforcement_asynchronous_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Quadcopter cannot learn to follow natural language instructions in continuous control without risky and costly autonomous flight in the physical environment during training.

+# Fix    : Hybrid algorithm combining supervised learning for position prediction (waypoint predictor) with reinforcement learning for continuous control, trained jointly in simulation and real environments without requiring autonomous physical flight during training.

+# Avoid  : End-to-end reinforcement learning requiring autonomous flight in the physical environment during training.

```
