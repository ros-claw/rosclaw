---
pattern_id: pattern_hop_history_and_order_aware_pre_training_for_vision_and_language_navigation
applicable_symptoms: [hop_history_and_order_aware_pre_training_for_vision_and_language_navigation]
domain: Planning_Decision
---

# VLN agent ignores landmark cues in long instructions

**Domain**: `Planning_Decision`

## Fix

History-and-Order Pre-training (HOP) that encodes temporal order and visual history via masked language modeling and trajectory-order prediction

## Anti-pattern

Standard VLN models that treat instructions as unordered bag-of-words or ignore temporal dynamics

## Cross-domain analogies

- **Perception_Vision** → Apply spherical-aware constraints to regularize attention offsets, aligning landmark cues with instruction geometry.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use closed-loop verification to filter and retain only high-reward landmark-following trajectories for fine-tuning.
  - related fix: Self-Refining Data Flywheel (SRDF): after initial training, use the Navigator model itself to filter and score candidate trajectories, retaining only high-confidence or high-reward pairs for iterative fine-tuning.
- **Control_Locomotion** → Fuse visual observations with language commands via domain-randomized RL to enable closed-loop landmark attention.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- hop_history_and_order_aware_pre_training_for_vision_and_language_navigation.before.py
+++ hop_history_and_order_aware_pre_training_for_vision_and_language_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions

+# Fix    : History-and-Order Pre-training (HOP) that encodes temporal order and visual history via masked language modeling and trajectory-order prediction

+# Avoid  : Standard VLN models that treat instructions as unordered bag-of-words or ignore temporal dynamics

```
