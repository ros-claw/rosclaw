---
pattern_id: pattern_help_anna_task
applicable_symptoms: [help_anna_task]
domain: Planning_Decision
---

# VLN agent fails to follow interactive task instructions involving object manipulation and human interaction

**Domain**: `Planning_Decision`

## Fix

Pre-training on large-scale vision-and-language navigation data with masked language modeling and action prediction objectives

## Anti-pattern

Training from scratch on the Help, Anna! task without pre-training

## Cross-domain analogies

- **Perception_Vision** → Use multi-sensor fusion to combine visual and haptic feedback for reacquiring lost interaction context.
  - related fix: Active mitigation strategies such as multi-sensor fusion or predictive reacquisition to handle blind zones
- **Learning_Training** → Pre-train on diverse human-object interaction trajectories via self-supervised learning, then fine-tune on interactive task instructions.
  - related fix: Pre-train on large-scale image-text-action triplets via self-supervised learning, then fine-tune on downstream VLN tasks
- **Control_Locomotion** → Use lightweight hierarchical decomposition: high-level planner at low frequency, low-level reactive policy at high frequency.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- help_anna_task.before.py
+++ help_anna_task.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to follow interactive task instructions involving object manipulation and human interaction

+# Fix    : Pre-training on large-scale vision-and-language navigation data with masked language modeling and action prediction objectives

+# Avoid  : Training from scratch on the Help, Anna! task without pre-training

```
