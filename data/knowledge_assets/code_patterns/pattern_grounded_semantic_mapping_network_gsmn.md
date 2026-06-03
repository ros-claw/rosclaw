---
pattern_id: pattern_grounded_semantic_mapping_network_gsmn
applicable_symptoms: [grounded_semantic_mapping_network_gsmn]
domain: Planning_Decision
---

# Visual navigation agents fail to follow high-level language instructions in novel environments due to lack of explicit spatial reasoning.

**Domain**: `Planning_Decision`

## Fix

Build an explicit semantic map in the world reference frame using differentiable pinhole camera projection, then feed it into a control policy to generate continuous velocity commands.

## Anti-pattern

End-to-end methods that map raw pixels directly to actions without explicit spatial representations.

## Cross-domain analogies

- **Perception_Vision** → Cross-view semantic alignment enforces spatial-language consistency for instruction following.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Learning_Training** → Use supervised behavioral cloning from expert demonstrations to pre-train spatial reasoning before RL fine-tuning.
  - related fix: Use supervised fine-tuning (SFT) on expert demonstration trajectories to bootstrap a behavioral prior before reinforcement learning
- **Control_Locomotion** → Train an end-to-end policy mapping language and visual inputs directly to actions with domain randomization for spatial reasoning.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- grounded_semantic_mapping_network_gsmn.before.py
+++ grounded_semantic_mapping_network_gsmn.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Visual navigation agents fail to follow high-level language instructions in novel environments due to lack of explicit spatial reasoning.

+# Fix    : Build an explicit semantic map in the world reference frame using differentiable pinhole camera projection, then feed it into a control policy to generate continuous velocity commands.

+# Avoid  : End-to-end methods that map raw pixels directly to actions without explicit spatial representations.

```
