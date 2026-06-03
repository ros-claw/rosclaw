---
pattern_id: pattern_uni_navid
applicable_symptoms: [uni_navid]
domain: Planning_Decision
---

# Existing navigation models are task-specific and fail to generalize across diverse navigation sub-tasks (instruction following, object searching, QA, people tracking) without fine-tuning.

**Domain**: `Planning_Decision`

## Fix

Unified video-based Vision-Language-Action (VLA) model trained on 3.6M multi-task navigation samples with harmonized input/output configurations.

## Anti-pattern

Task-specific models that require separate fine-tuning for each navigation sub-task.

## Cross-domain analogies

- **Perception_Vision** → Use task-agnostic landmark saliency as a universal perceptual anchor for diverse navigation objectives.
  - related fix: Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).
- **Learning_Training** → Hierarchical decomposition with reusable skill modules enables zero-shot generalization across navigation sub-tasks.
  - related fix: Option Keyboard: a hierarchical RL framework where a high-level policy selects and composes pre-trained low-level skills (options) via a learned combination mechanism, enabling zero-shot generalization to new tasks.
- **Control_Locomotion** → Use diffusion policies to model diverse navigation tasks as multi-modal action distributions over sub-task primitives.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- uni_navid.before.py
+++ uni_navid.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Existing navigation models are task-specific and fail to generalize across diverse navigation sub-tasks (instruction following, object searching, QA, people tracking) without fine-tuning.

+# Fix    : Unified video-based Vision-Language-Action (VLA) model trained on 3.6M multi-task navigation samples with harmonized input/output configurations.

+# Avoid  : Task-specific models that require separate fine-tuning for each navigation sub-task.

```
