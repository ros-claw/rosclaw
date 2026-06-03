---
pattern_id: pattern_habitat_web_learning_embodied_object_search_strategies_from_human_demonstrations
applicable_symptoms: [habitat_web_learning_embodied_object_search_strategies_from_human_demonstrations]
domain: Learning_Training
---

# Embodied object-search agents trained in simulation fail to generalize to real-world environments due to lack of diverse, human-like exploration strategies.

**Domain**: `Learning_Training`

## Fix

Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.

## Anti-pattern

Reinforcement learning from scratch in simulation without human priors leads to poor real-world transfer.

## Cross-domain analogies

- **Perception_Vision** → Train exploration policies on simulated trajectories with injected human-like biases and noise.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Planning_Decision** → Use LLM-driven reasoning to dynamically compose diverse exploration strategies from a skill tree.
  - related fix: AINav: adaptive interactive navigation using LLM-driven reasoning, a primitive skill tree, and RL-trained interaction skills (push, slide, climb) to proactively manipulate obstacles and replan on the fly.
- **Control_Locomotion** → Pre-train a library of diverse exploration behaviors via RL, decoupling strategy acquisition from task planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- habitat_web_learning_embodied_object_search_strategies_from_human_demonstrations.before.py
+++ habitat_web_learning_embodied_object_search_strategies_from_human_demonstrations.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied object-search agents trained in simulation fail to generalize to real-world environments due to lack of diverse, human-like exploration strategies.

+# Fix    : Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.

+# Avoid  : Reinforcement learning from scratch in simulation without human priors leads to poor real-world transfer.

```
