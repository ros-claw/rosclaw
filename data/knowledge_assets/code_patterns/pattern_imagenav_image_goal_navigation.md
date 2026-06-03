---
pattern_id: pattern_imagenav_image_goal_navigation
applicable_symptoms: [imagenav_image_goal_navigation]
domain: Planning_Decision
---

# Agent fails to navigate to a location specified by a target image when starting far away and environment is partially observable.

**Domain**: `Planning_Decision`

## Fix

Train a navigation policy using a mixture of goal modalities including ImageNav, ObjectNav, and LangNav to improve generalization across diverse goal representations.

## Anti-pattern

Using only point-goal or language-goal navigation without visual goal matching.

## Cross-domain analogies

- **Perception_Vision** → End-to-end implicit metric prediction from raw observations to bypass explicit mapping.
  - related fix: Fine-tune a long-horizon visual-geometry backbone to predict metric-scale depth and pose directly from images, enabling implicit state estimation and dense geometry reconstruction without external sensors.
- **Learning_Training** → Use group-relative advantage estimation over sampled trajectory rollouts to refine navigation policies under partial observability.
  - related fix: Use GRPO (Group Relative Policy Optimization) as a second-stage RL fine-tuning after supervised chain-of-thought alignment, optimizing policy relative to a group of sampled trajectories via group-relative advantage estimation.
- **Control_Locomotion** → Distill multiple navigation policies via DAgger and fine-tune with RL using visual observations.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- imagenav_image_goal_navigation.before.py
+++ imagenav_image_goal_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Agent fails to navigate to a location specified by a target image when starting far away and environment is partially observable.

+# Fix    : Train a navigation policy using a mixture of goal modalities including ImageNav, ObjectNav, and LangNav to improve generalization across diverse goal representations.

+# Avoid  : Using only point-goal or language-goal navigation without visual goal matching.

```
