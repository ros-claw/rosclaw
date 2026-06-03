---
pattern_id: pattern_skill_library
applicable_symptoms: [skill_library]
domain: Control_Locomotion
---

# High-level planners lack robust, reusable low-level motor primitives for mobile manipulation, leading to brittle task execution under environmental perturbations.

**Domain**: `Control_Locomotion`

## Fix

Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Anti-pattern

Hand-coded or monolithic control policies that are not reusable across tasks and fail under perturbations.

## Cross-domain analogies

- **Perception_Vision** → Active perception via task-driven viewpoint selection inspires adaptive motor primitive selection to reduce execution brittleness.
  - related fix: Integrate active perception with semantic mapping: agent selects viewpoints to reduce ambiguity while building a task-driven semantic map from RGB-D or lidar data.
- **Planning_Decision** → Use a large pretrained model zero-shot to directly generate low-level motor primitives from high-level task commands.
  - related fix: Use a large language model (LLM) in a zero-shot manner to directly predict navigation actions from natural language instructions without any task-specific fine-tuning.
- **Learning_Training** → Use supervised pre-training on diverse motion primitives to bootstrap a robust low-level skill library before high-level planning.
  - related fix: Use supervised fine-tuning (SFT) on expert demonstration trajectories to bootstrap a behavioral prior before reinforcement learning

## Patch

```diff
--- skill_library.before.py
+++ skill_library.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: High-level planners lack robust, reusable low-level motor primitives for mobile manipulation, leading to brittle task execution under environmental perturbations.

+# Fix    : Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

+# Avoid  : Hand-coded or monolithic control policies that are not reusable across tasks and fail under perturbations.

```
