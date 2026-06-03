---
pattern_id: pattern_large_language_models
applicable_symptoms: [large_language_models]
domain: Planning_Decision
---

# LLM-based task planning fails to adapt when initial plan fails or new information arrives, leading to rigid execution in dynamic environments.

**Domain**: `Planning_Decision`

## Fix

Adaptive replanning via Advisor module (assess alternatives) and Arborist module (restructure plan) using LLM reasoning.

## Anti-pattern

Static task planning without replanning capability.

## Cross-domain analogies

- **Perception_Vision** → Active perception with semantic mapping inspires closed-loop plan revision via selective information gathering.
  - related fix: Integrate active perception with semantic mapping: agent selects viewpoints to reduce ambiguity while building a task-driven semantic map from RGB-D or lidar data.
- **Learning_Training** → Train a single planner on abstracted task representations that generalize across execution contexts.
  - related fix: Train a single policy on shared representations that abstract away physical differences across robot morphologies.
- **Control_Locomotion** → Use closed-loop verification with standardized failure recovery tasks to retrain planner adaptability.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- large_language_models.before.py
+++ large_language_models.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: LLM-based task planning fails to adapt when initial plan fails or new information arrives, leading to rigid execution in dynamic environments.

+# Fix    : Adaptive replanning via Advisor module (assess alternatives) and Arborist module (restructure plan) using LLM reasoning.

+# Avoid  : Static task planning without replanning capability.

```
