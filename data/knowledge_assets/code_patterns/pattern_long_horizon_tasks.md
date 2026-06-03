---
pattern_id: pattern_long_horizon_tasks
applicable_symptoms: [long_horizon_tasks]
domain: Planning_Decision
---

# Agent loses coherence or fails to complete complex missions requiring many sequential actions and sub-goals.

**Domain**: `Planning_Decision`

## Fix

Use hierarchical task planning to decompose long-horizon missions into sub-tasks executed by lower-level policies.

## Anti-pattern

Flat planning over full action sequence without hierarchical decomposition.

## Cross-domain analogies

- **Perception_Vision** → Use frame-quality gating to discard low-confidence planning steps that degrade long-horizon coherence.
  - related fix: Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.
- **Learning_Training** → Use adaptive gradient clipping to stabilize long-horizon planning by scaling action steps.
  - related fix: Use adaptive gradient clipping (AGC) and Scaled Weight Standardization to train deep networks without batch normalization.
- **Control_Locomotion** → Pre-train a library of reusable subgoal policies via RL, decoupling subgoal acquisition from mission planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- long_horizon_tasks.before.py
+++ long_horizon_tasks.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Agent loses coherence or fails to complete complex missions requiring many sequential actions and sub-goals.

+# Fix    : Use hierarchical task planning to decompose long-horizon missions into sub-tasks executed by lower-level policies.

+# Avoid  : Flat planning over full action sequence without hierarchical decomposition.

```
