---
pattern_id: pattern_environment_action_planning
applicable_symptoms: [environment_action_planning]
domain: Planning_Decision
---

# Navigation policies fail to bind perception to decision-making, leading to reactive but myopic behavior that ignores long-term goals.

**Domain**: `Planning_Decision`

## Fix

Learn a mapping from environmental features (occupancy grids, depth images, semantic labels) to motion primitives or subgoals, forming an environment-action planning relation within a dual-relation reasoning framework.

## Anti-pattern

Using only state-transition reasoning that models environment evolution without considering agent actions.

## Cross-domain analogies

- **Perception_Vision** → Active perception with semantic mapping inspires a closed-loop verification mechanism that actively queries perception to reduce decision myopia.
  - related fix: Integrate active perception with semantic mapping: agent selects viewpoints to reduce ambiguity while building a task-driven semantic map from RGB-D or lidar data.
- **Learning_Training** → Use synthetic goal-conditioned scene graphs to augment training with long-horizon planning examples.
  - related fix: ScaleVLN: large-scale synthetic data generation by combining 3D scene graphs with LLM-generated instructions and augmenting with panoramic views and object-level grounding.
- **Control_Locomotion** → Train a model-free RL policy with domain randomization that fuses visual inputs with goal embeddings for closed-loop long-horizon planning.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- environment_action_planning.before.py
+++ environment_action_planning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation policies fail to bind perception to decision-making, leading to reactive but myopic behavior that ignores long-term goals.

+# Fix    : Learn a mapping from environmental features (occupancy grids, depth images, semantic labels) to motion primitives or subgoals, forming an environment-action planning relation within a dual-relation reasoning framework.

+# Avoid  : Using only state-transition reasoning that models environment evolution without considering agent actions.

```
