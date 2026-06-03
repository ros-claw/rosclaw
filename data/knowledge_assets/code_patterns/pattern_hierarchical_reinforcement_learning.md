---
pattern_id: pattern_hierarchical_reinforcement_learning
applicable_symptoms: [hierarchical_reinforcement_learning]
domain: Planning_Decision
---

# Long-horizon navigation and locomotion tasks are difficult to learn with flat RL due to high-dimensional action spaces and sparse rewards.

**Domain**: `Planning_Decision`

## Fix

Hierarchical RL with a high-level navigation planner issuing subgoals to a low-level locomotion controller, both trained via model-free RL.

## Anti-pattern

Flat RL policies that directly map observations to joint torques fail on long-horizon tasks with varied terrain.

## Cross-domain analogies

- **Perception_Vision** → Use a generative adversarial hierarchy to directly output subgoal sequences in a single forward pass.
  - related fix: Use a GAN with a compressed sensing loss to directly reconstruct images from undersampled k-space data in a single feedforward pass.
- **Learning_Training** → Decompose long-horizon tasks into shorter subtasks solved by parallel workers with synchronized policy updates.
  - related fix: Distribute PPO training across multiple workers with synchronized gradient updates (DD-PPO).
- **Control_Locomotion** → Hierarchical decomposition via pre-trained motor primitives reduces action dimensionality and reward sparsity for long-horizon tasks.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- hierarchical_reinforcement_learning.before.py
+++ hierarchical_reinforcement_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon navigation and locomotion tasks are difficult to learn with flat RL due to high-dimensional action spaces and sparse rewards.

+# Fix    : Hierarchical RL with a high-level navigation planner issuing subgoals to a low-level locomotion controller, both trained via model-free RL.

+# Avoid  : Flat RL policies that directly map observations to joint torques fail on long-horizon tasks with varied terrain.

```
