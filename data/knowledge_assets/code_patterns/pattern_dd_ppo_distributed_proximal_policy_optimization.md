---
pattern_id: pattern_dd_ppo_distributed_proximal_policy_optimization
applicable_symptoms: [dd_ppo_distributed_proximal_policy_optimization]
domain: Learning_Training
---

# Training PPO on a single machine is slow and sample-inefficient for embodied RL tasks.

**Domain**: `Learning_Training`

## Fix

Distribute PPO training across multiple workers with synchronized gradient updates (DD-PPO).

## Anti-pattern

Single-worker PPO training with sequential rollouts.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic rollout generation to augment real experience, improving sample efficiency in PPO training.
  - related fix: Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings
- **Planning_Decision** → Use token compression to reduce sequence length in PPO's trajectory buffer, cutting memory and computation per update.
  - related fix: FantasyVLN: a method that addresses token inflation while preserving visual chain-of-thought reasoning.
- **Control_Locomotion** → Use lightweight policy distillation to reduce model size and enable faster, sample-efficient PPO training.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- dd_ppo_distributed_proximal_policy_optimization.before.py
+++ dd_ppo_distributed_proximal_policy_optimization.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Training PPO on a single machine is slow and sample-inefficient for embodied RL tasks.

+# Fix    : Distribute PPO training across multiple workers with synchronized gradient updates (DD-PPO).

+# Avoid  : Single-worker PPO training with sequential rollouts.

```
