---
pattern_id: pattern_vision_language_model_vlm
applicable_symptoms: [vision_language_model_vlm]
domain: Planning_Decision
---

# VLMs exhibit poor generalization in embodied navigation across different robot morphologies, environments, or action spaces without additional adaptation.

**Domain**: `Planning_Decision`

## Fix

Fine-tune a pre-trained VLM on simulated embodied experience to act as a navigation policy, conditioning on visual history and goals (as in FiLM-Nav).

## Anti-pattern

Directly applying zero-shot VLMs to embodied navigation tasks without fine-tuning.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical open-vocabulary graph decomposition with incremental projection enables cross-morphology action space generalization.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Learning_Training** → Use GRPO to fine-tune VLMs on diverse morphology-relative trajectory groups for zero-shot adaptation.
  - related fix: Use GRPO (Group Relative Policy Optimization) as a second-stage RL fine-tuning after supervised chain-of-thought alignment, optimizing policy relative to a group of sampled trajectories via group-relative advantage estimation.
- **Control_Locomotion** → Use closed-loop trial-and-error to adapt actions when morphology or environment mismatches occur.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- vision_language_model_vlm.before.py
+++ vision_language_model_vlm.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLMs exhibit poor generalization in embodied navigation across different robot morphologies, environments, or action spaces without additional adaptation.

+# Fix    : Fine-tune a pre-trained VLM on simulated embodied experience to act as a navigation policy, conditioning on visual history and goals (as in FiLM-Nav).

+# Avoid  : Directly applying zero-shot VLMs to embodied navigation tasks without fine-tuning.

```
