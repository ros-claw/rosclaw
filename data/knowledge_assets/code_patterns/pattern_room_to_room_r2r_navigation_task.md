---
pattern_id: pattern_room_to_room_r2r_navigation_task
applicable_symptoms: [room_to_room_r2r_navigation_task]
domain: Planning_Decision
---

# VLN agent fails to generalize to unseen buildings when following natural language instructions, leading to low success rate and inefficient paths.

**Domain**: `Planning_Decision`

## Fix

Use the Room-to-Room (R2R) benchmark with standardized dataset, simulator, and evaluation metrics (SR, SPL) to train and evaluate vision-and-language navigation models.

## Anti-pattern

Training on seen environments without generalization to novel layouts.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic visual imaginations of unseen building layouts to pretrain the planner with an auxiliary alignment loss.
  - related fix: Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.
- **Learning_Training** → Use a local encoder followed by global attention to model long-range spatial dependencies across unseen buildings.
  - related fix: Use a convolutional stem followed by Transformer blocks with global attention to model dependencies across 100kb+ distances
- **Control_Locomotion** → Distill high-level VLN policies into lightweight recurrent models for real-time closed-loop waypoint control.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- room_to_room_r2r_navigation_task.before.py
+++ room_to_room_r2r_navigation_task.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to generalize to unseen buildings when following natural language instructions, leading to low success rate and inefficient paths.

+# Fix    : Use the Room-to-Room (R2R) benchmark with standardized dataset, simulator, and evaluation metrics (SR, SPL) to train and evaluate vision-and-language navigation models.

+# Avoid  : Training on seen environments without generalization to novel layouts.

```
