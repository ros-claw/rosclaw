---
pattern_id: pattern_room_to_room_r2r_task
applicable_symptoms: [room_to_room_r2r_task]
domain: Planning_Decision
---

# VLN agents fail to generalize to unseen environments and instructions, as measured by the R2R benchmark's private test set.

**Domain**: `Planning_Decision`

## Fix

Train on the R2R dataset (Matterport3D) with step-by-step instruction following and evaluate on unseen test splits to measure generalization.

## Anti-pattern

Training only on seen environments without a held-out test set leads to overfitting and poor generalization.

## Cross-domain analogies

- **Perception_Vision** → Fine-tune a visual-language backbone to predict navigation actions and spatial reasoning directly from egocentric images and instructions.
  - related fix: Fine-tune a long-horizon visual-geometry backbone to predict metric-scale depth and pose directly from images, enabling implicit state estimation and dense geometry reconstruction without external sensors.
- **Learning_Training** → Hierarchical decomposition with learned skill composition for zero-shot generalization.
  - related fix: Option Keyboard: a hierarchical RL framework where a high-level policy selects and composes pre-trained low-level skills (options) via a learned combination mechanism, enabling zero-shot generalization to new tasks.
- **Control_Locomotion** → Train a model-free RL policy with domain randomization fusing visual and language inputs for real-time adaptation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- room_to_room_r2r_task.before.py
+++ room_to_room_r2r_task.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize to unseen environments and instructions, as measured by the R2R benchmark's private test set.

+# Fix    : Train on the R2R dataset (Matterport3D) with step-by-step instruction following and evaluate on unseen test splits to measure generalization.

+# Avoid  : Training only on seen environments without a held-out test set leads to overfitting and poor generalization.

```
