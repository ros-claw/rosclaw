---
pattern_id: pattern_dynamic_chain_of_navigation
applicable_symptoms: [dynamic_chain_of_navigation]
domain: Planning_Decision
---

# Navigation agents fail to generalize across diverse instruction types (spatial, temporal, object-referential) without task-specific training.

**Domain**: `Planning_Decision`

## Fix

Dynamic Chain-of-Navigation (DCoN): convert linguistic instructions into a unified chain of navigation subtasks using pre-trained VLMs, enabling zero-shot planning across instruction types.

## Anti-pattern

Training separate models for each instruction type or using fixed task-specific planners.

## Cross-domain analogies

- **Perception_Vision** → Use a fixed-size latent instruction bottleneck via cross-attention to handle diverse query types.
  - related fix: Use a cross-attention bottleneck: project arbitrary input to a fixed-size latent array via cross-attention, then process with iterative self-attention in latent space.
- **Learning_Training** → Train a latent instruction model to simulate diverse task contexts for zero-shot generalization.
  - related fix: Train a neural world model that predicts future latent states and rewards from current observations and actions, enabling model-based planning and mental simulation without direct environment interaction.
- **Control_Locomotion** → Train a single policy with domain-randomized instruction embeddings to map diverse queries directly to actions.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- dynamic_chain_of_navigation.before.py
+++ dynamic_chain_of_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents fail to generalize across diverse instruction types (spatial, temporal, object-referential) without task-specific training.

+# Fix    : Dynamic Chain-of-Navigation (DCoN): convert linguistic instructions into a unified chain of navigation subtasks using pre-trained VLMs, enabling zero-shot planning across instruction types.

+# Avoid  : Training separate models for each instruction type or using fixed task-specific planners.

```
