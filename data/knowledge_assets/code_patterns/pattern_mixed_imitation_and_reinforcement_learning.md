---
pattern_id: pattern_mixed_imitation_and_reinforcement_learning
applicable_symptoms: [mixed_imitation_and_reinforcement_learning]
domain: Learning_Training
---

# Pure imitation learning lacks exploration and fails on unseen states; pure RL is sample-inefficient and suffers from cold-start exploration.

**Domain**: `Learning_Training`

## Fix

Mixed Imitation and Reinforcement Learning (MIRL): bootstrap policy via off-policy imitation learning, then refine with on-policy RL, gradually shifting weight from imitation to RL.

## Anti-pattern

Behavioral cloning alone (no exploration) or pure deep RL alone (sample-inefficient).

## Cross-domain analogies

- **Perception_Vision** → Augment imitation learning with simulated exploration noise and failure states to bridge training and deployment distribution.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Planning_Decision** → Closed-loop hierarchical decomposition with confidence scoring enables structured exploration and feedback for sample-efficient RL.
  - related fix: Closed-loop hierarchical chain-of-thought: decompose navigation into multi-turn QA with confidence scoring for each step, fine-tune InternVL2 (2B) with LoRA on simulation data.
- **Control_Locomotion** → Multi-expert distillation with DAgger enables guided exploration and efficient RL fine-tuning for cold-start and unseen states.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- mixed_imitation_and_reinforcement_learning.before.py
+++ mixed_imitation_and_reinforcement_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Pure imitation learning lacks exploration and fails on unseen states; pure RL is sample-inefficient and suffers from cold-start exploration.

+# Fix    : Mixed Imitation and Reinforcement Learning (MIRL): bootstrap policy via off-policy imitation learning, then refine with on-policy RL, gradually shifting weight from imitation to RL.

+# Avoid  : Behavioral cloning alone (no exploration) or pure deep RL alone (sample-inefficient).

```
