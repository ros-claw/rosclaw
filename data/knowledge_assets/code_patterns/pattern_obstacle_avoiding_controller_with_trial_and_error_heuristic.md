---
pattern_id: pattern_obstacle_avoiding_controller_with_trial_and_error_heuristic
applicable_symptoms: [obstacle_avoiding_controller_with_trial_and_error_heuristic]
domain: Control_Locomotion
---

# Navigation agents get permanently stuck on obstacles during long-horizon plan execution in continuous environments.

**Domain**: `Control_Locomotion`

## Fix

Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Anti-pattern

Relying solely on high-level plans without reactive obstacle handling.

## Cross-domain analogies

- **Perception_Vision** → Closed-loop verification enforces obstacle avoidance consistency during long-horizon execution.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Planning_Decision** → Use state-adaptive expert gating to switch between recovery and path-tracking controllers when stuck.
  - related fix: State-Adaptive Mixture of Experts (SAME): adaptively selects expert modules based on current state and instruction, enabling shared navigation knowledge with task-specific exploitation.
- **Learning_Training** → Use randomized obstacle properties during training to prevent policy overfitting to specific stuck scenarios.
  - related fix: Use domain randomization: vary simulation parameters (friction, mass, lighting, delay) randomly during training to improve policy robustness to real-world conditions.

## Patch

```diff
--- obstacle_avoiding_controller_with_trial_and_error_heuristic.before.py
+++ obstacle_avoiding_controller_with_trial_and_error_heuristic.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents get permanently stuck on obstacles during long-horizon plan execution in continuous environments.

+# Fix    : Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

+# Avoid  : Relying solely on high-level plans without reactive obstacle handling.

```
