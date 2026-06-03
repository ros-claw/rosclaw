---
pattern_id: pattern_topological_map_navigation
applicable_symptoms: [topological_map_navigation]
domain: Planning_Decision
---

# LLM-based navigation agents struggle with long-horizon tasks because they lack a structured memory of visited locations and object associations, leading to inefficient replanning and failure to leverage past experience.

**Domain**: `Planning_Decision`

## Fix

Use a topological map that stores viewpoints, objects, and spatial relationships as a graph, serving as the global action space for an LLM planner to select next navigation actions via node selection instead of continuous coordinates.

## Anti-pattern

Using metric maps or continuous coordinate regression for action selection in LLM-based navigation.

## Cross-domain analogies

- **Perception_Vision** → Use joint cross-modal memory to fuse spatial and semantic history for structured long-horizon replanning.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Learning_Training** → End-to-end training with a unified memory representation replaces modular handcrafted submodules.
  - related fix: Train a single neural network end-to-end from raw sensor inputs to control outputs using a reward signal (e.g., reinforcement learning), allowing the network to discover internal representations that directly optimize the desired behavior.
- **Control_Locomotion** → Multi-expert distillation provides a structured memory of diverse strategies for long-horizon replanning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- topological_map_navigation.before.py
+++ topological_map_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: LLM-based navigation agents struggle with long-horizon tasks because they lack a structured memory of visited locations and object associations, leading to inefficient replanning and failure to leverage past experience.

+# Fix    : Use a topological map that stores viewpoints, objects, and spatial relationships as a graph, serving as the global action space for an LLM planner to select next navigation actions via node selection instead of continuous coordinates.

+# Avoid  : Using metric maps or continuous coordinate regression for action selection in LLM-based navigation.

```
