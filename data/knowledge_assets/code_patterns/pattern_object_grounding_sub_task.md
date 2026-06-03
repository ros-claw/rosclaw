---
pattern_id: pattern_object_grounding_sub_task
applicable_symptoms: [object_grounding_sub_task]
domain: Perception_Vision
---

# Agent fails to attend to relevant objects in a scene when given natural language instructions, leading to poor object navigation and instruction following.

**Domain**: `Perception_Vision`

## Fix

Cross-modal alignment pretraining using contrastive or attention-based losses to align visual object features with language tokens.

## Anti-pattern

Training without explicit object grounding, resulting in the agent ignoring language references to objects.

## Cross-domain analogies

- **Planning_Decision** → Remove the perfect-attention assumption by training with noisy, egocentric visual inputs and uncertain grounding.
  - related fix: Train and evaluate agents in continuous environments with raw egocentric observations, uncertain localization, and fine-grained motor control, removing the graph assumption.
- **Learning_Training** → Use a learned language-conditioned world model to simulate attention shifts and verify object relevance before acting.
  - related fix: Train a neural world model that predicts future latent states and rewards from current observations and actions, enabling model-based planning and mental simulation without direct environment interaction.
- **Control_Locomotion** → Multi-expert distillation with DAgger and RL fine-tuning using depth images.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- object_grounding_sub_task.before.py
+++ object_grounding_sub_task.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Agent fails to attend to relevant objects in a scene when given natural language instructions, leading to poor object navigation and instruction following.

+# Fix    : Cross-modal alignment pretraining using contrastive or attention-based losses to align visual object features with language tokens.

+# Avoid  : Training without explicit object grounding, resulting in the agent ignoring language references to objects.

```
