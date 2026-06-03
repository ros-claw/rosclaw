---
pattern_id: pattern_odyssey
applicable_symptoms: [odyssey]
domain: Planning_Decision
---

# Long-horizon mobile manipulation tasks (e.g., door opening, object retrieval) fail due to lack of coordination between locomotion and manipulation, and poor sim-to-real transfer.

**Domain**: `Planning_Decision`

## Fix

Hierarchical planner (VLM-driven) decomposes language instructions into subgoals, paired with a whole-body policy trained in simulation that coordinates locomotion and manipulation in a unified action space, enabling zero-shot sim-to-real transfer.

## Anti-pattern

Separate locomotion and manipulation controllers that require manual coordination and fine-tuning per task.

## Cross-domain analogies

- **Perception_Vision** → Train a hierarchical policy on simulated multi-task data to coordinate locomotion and manipulation.
  - related fix: Use deep learning models (e.g., CNNs) trained on simulated galaxy merger images to automatically classify merger stages.
- **Learning_Training** → Pretrain a joint locomotion-manipulation policy on diverse offline data, then fine-tune on task-specific sim-to-real.
  - related fix: Pretrain a vision-language model on large-scale web data to learn cross-modal alignment, then fine-tune on navigation tasks
- **Control_Locomotion** → Train a model-free RL policy with domain randomization fusing vision and actions for coordinated locomotion-manipulation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- odyssey.before.py
+++ odyssey.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon mobile manipulation tasks (e.g., door opening, object retrieval) fail due to lack of coordination between locomotion and manipulation, and poor sim-to-real transfer.

+# Fix    : Hierarchical planner (VLM-driven) decomposes language instructions into subgoals, paired with a whole-body policy trained in simulation that coordinates locomotion and manipulation in a unified action space, enabling zero-shot sim-to-real transfer.

+# Avoid  : Separate locomotion and manipulation controllers that require manual coordination and fine-tuning per task.

```
