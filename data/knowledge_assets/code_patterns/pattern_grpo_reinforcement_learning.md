---
pattern_id: pattern_grpo_reinforcement_learning
applicable_symptoms: [grpo_reinforcement_learning]
domain: Learning_Training
---

# Long-horizon manipulation and navigation policies suffer from reasoning inconsistency and control instability across extended action sequences.

**Domain**: `Learning_Training`

## Fix

Use GRPO (Group Relative Policy Optimization) as a second-stage RL fine-tuning after supervised chain-of-thought alignment, optimizing policy relative to a group of sampled trajectories via group-relative advantage estimation.

## Anti-pattern

Using only supervised fine-tuning or chain-of-thought distillation without RL fine-tuning leads to poor execution robustness in long-horizon tasks.

## Cross-domain analogies

- **Perception_Vision** → Cross-modal alignment pretraining can stabilize long-horizon policies by enforcing consistency between sequential action embeddings and goal tokens.
  - related fix: Cross-modal alignment pretraining using contrastive or attention-based losses to align visual object features with language tokens.
- **Planning_Decision** → Use cross-modal attention to fuse state and goal embeddings at each step for consistent reasoning.
  - related fix: Use a cross-modal attention mechanism to fuse visual features and language embeddings at each step, enabling the agent to align instruction phrases with visual landmarks.
- **Control_Locomotion** → Use reinforcement learning to map observations directly to actions, bypassing long-horizon reasoning instability.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- grpo_reinforcement_learning.before.py
+++ grpo_reinforcement_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon manipulation and navigation policies suffer from reasoning inconsistency and control instability across extended action sequences.

+# Fix    : Use GRPO (Group Relative Policy Optimization) as a second-stage RL fine-tuning after supervised chain-of-thought alignment, optimizing policy relative to a group of sampled trajectories via group-relative advantage estimation.

+# Avoid  : Using only supervised fine-tuning or chain-of-thought distillation without RL fine-tuning leads to poor execution robustness in long-horizon tasks.

```
