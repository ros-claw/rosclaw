---
pattern_id: pattern_hierarchical_planning_high_level_low_level
applicable_symptoms: [hierarchical_planning_high_level_low_level]
domain: Planning_Decision
---

# Standard integrated planning and control leads to computational bottlenecks and fragility in cluttered environments, especially for long-horizon tasks.

**Domain**: `Planning_Decision`

## Fix

Decompose navigation into a high-level planner (abstract waypoint selection, long-horizon reasoning) and a low-level controller (local obstacle-avoiding motion).

## Anti-pattern

Tightly coupled planning and control that recomputes low-level trajectories from scratch on path blockage.

## Cross-domain analogies

- **Perception_Vision** → Use deformable attention to selectively sample critical state-action pairs instead of full replanning.
  - related fix: Use deformable cross-attention with learned sampling points to selectively attend to relevant image features instead of the entire grid.
- **Learning_Training** → Hierarchical decomposition of planning into coarse-to-fine stages reduces computational load and improves robustness.
  - related fix: Two-stage progressive knowledge distillation: first distill from large teacher to medium student, then from medium to small student, achieving 1/7 model size with same accuracy.
- **Control_Locomotion** → Use diffusion policies to discretize long-horizon plans into tractable action chunks.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- hierarchical_planning_high_level_low_level.before.py
+++ hierarchical_planning_high_level_low_level.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Standard integrated planning and control leads to computational bottlenecks and fragility in cluttered environments, especially for long-horizon tasks.

+# Fix    : Decompose navigation into a high-level planner (abstract waypoint selection, long-horizon reasoning) and a low-level controller (local obstacle-avoiding motion).

+# Avoid  : Tightly coupled planning and control that recomputes low-level trajectories from scratch on path blockage.

```
