---
pattern_id: pattern_supervised_fine_tuning_sft
applicable_symptoms: [supervised_fine_tuning_sft]
domain: Learning_Training
---

# VLN agent initialized randomly fails to generalize to novel environments after RL fine-tuning

**Domain**: `Learning_Training`

## Fix

Use supervised fine-tuning (SFT) on expert demonstration trajectories to bootstrap a behavioral prior before reinforcement learning

## Anti-pattern

Training RL from scratch without SFT initialization

## Cross-domain analogies

- **Perception_Vision** → Use cross-view consistency regularization during RL fine-tuning to align agent representations across environments.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Planning_Decision** → Use differentiable world-frame mapping as an explicit spatial prior to initialize RL policy.
  - related fix: Build an explicit semantic map in the world reference frame using differentiable pinhole camera projection, then feed it into a control policy to generate continuous velocity commands.
- **Control_Locomotion** → Closed-loop verification using local perceptual feedback to correct high-level navigation policies during training.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- supervised_fine_tuning_sft.before.py
+++ supervised_fine_tuning_sft.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent initialized randomly fails to generalize to novel environments after RL fine-tuning

+# Fix    : Use supervised fine-tuning (SFT) on expert demonstration trajectories to bootstrap a behavioral prior before reinforcement learning

+# Avoid  : Training RL from scratch without SFT initialization

```
