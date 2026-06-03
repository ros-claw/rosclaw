---
pattern_id: pattern_navmorph_a_self_evolving_world_model_for_vision_and_language_navigation_in_conti
applicable_symptoms: [navmorph_a_self_evolving_world_model_for_vision_and_language_navigation_in_conti]
domain: Planning_Decision
---

# VLN agents fail to adapt to dynamic obstacles and changing environments in continuous settings, leading to navigation failures.

**Domain**: `Planning_Decision`

## Fix

Self-evolving world model that continuously updates its internal representation of the environment using online interaction data, enabling adaptive navigation.

## Anti-pattern

Static world models that do not update during deployment, causing poor generalization to unseen obstacles.

## Cross-domain analogies

- **Perception_Vision** → Integrate learned semantic representations with real-time closed-loop verification to adapt navigation decisions to dynamic obstacles.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Learning_Training** → Two-stage training: supervised imitation then reinforcement fine-tuning for dynamic adaptation.
  - related fix: Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward
- **Control_Locomotion** → Use diffusion policies to model multi-modal paths and discretize continuous action spaces for adaptive obstacle avoidance.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- navmorph_a_self_evolving_world_model_for_vision_and_language_navigation_in_conti.before.py
+++ navmorph_a_self_evolving_world_model_for_vision_and_language_navigation_in_conti.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to adapt to dynamic obstacles and changing environments in continuous settings, leading to navigation failures.

+# Fix    : Self-evolving world model that continuously updates its internal representation of the environment using online interaction data, enabling adaptive navigation.

+# Avoid  : Static world models that do not update during deployment, causing poor generalization to unseen obstacles.

```
