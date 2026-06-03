---
pattern_id: pattern_open_vocabulary_object_goal_navigation
applicable_symptoms: [open_vocabulary_object_goal_navigation]
domain: Planning_Decision
---

# Open-vocabulary object-goal navigation agents fail to generalize to novel objects unseen during training, leading to poor performance in unseen environments.

**Domain**: `Planning_Decision`

## Fix

Combine visual grounding (e.g., CLIP-based detectors) with semantic mapping and language-conditioned hierarchical exploration policies, as in LOVON.

## Anti-pattern

Using a fixed set of object categories and closed-vocabulary detectors.

## Cross-domain analogies

- **Perception_Vision** → Multi-modal fusion of vision, language, and proprioception to maximize semantic coverage and handle novel object gaps.
  - related fix: Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.
- **Learning_Training** → Progressive distillation of object representations from broad to narrow categories to improve generalization to novel objects.
  - related fix: Two-stage progressive knowledge distillation: first distill from large teacher to medium student, then from medium to small student, achieving 1/7 model size with same accuracy.
- **Control_Locomotion** → Use reinforcement learning to map visual observations directly to navigation actions for novel objects.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- open_vocabulary_object_goal_navigation.before.py
+++ open_vocabulary_object_goal_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Open-vocabulary object-goal navigation agents fail to generalize to novel objects unseen during training, leading to poor performance in unseen environments.

+# Fix    : Combine visual grounding (e.g., CLIP-based detectors) with semantic mapping and language-conditioned hierarchical exploration policies, as in LOVON.

+# Avoid  : Using a fixed set of object categories and closed-vocabulary detectors.

```
