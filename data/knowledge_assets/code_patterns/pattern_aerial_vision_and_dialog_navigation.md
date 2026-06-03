---
pattern_id: pattern_aerial_vision_and_dialog_navigation
applicable_symptoms: [aerial_vision_and_dialog_navigation]
domain: Planning_Decision
---

# VLN agent fails to follow natural language instructions in aerial navigation due to lack of dialog history and spatial reasoning in 3D environments.

**Domain**: `Planning_Decision`

## Fix

Aerial Vision-and-Dialog Navigation (AVDN) dataset and model that uses a transformer-based architecture to fuse visual observations, dialog history, and spatial memory for step-by-step navigation.

## Anti-pattern

Standard VLN models that ignore dialog context or treat navigation as a single-shot instruction following.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic visual dialog to train an auxiliary spatial reasoning loss.
  - related fix: Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.
- **Learning_Training** → Use synthetic data generation to create diverse aerial instruction-trajectory pairs with dialog history.
  - related fix: Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.
- **Control_Locomotion** → Fuse egocentric vision and language commands into a PPO policy with domain randomization for real-time spatial reasoning.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- aerial_vision_and_dialog_navigation.before.py
+++ aerial_vision_and_dialog_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to follow natural language instructions in aerial navigation due to lack of dialog history and spatial reasoning in 3D environments.

+# Fix    : Aerial Vision-and-Dialog Navigation (AVDN) dataset and model that uses a transformer-based architecture to fuse visual observations, dialog history, and spatial memory for step-by-step navigation.

+# Avoid  : Standard VLN models that ignore dialog context or treat navigation as a single-shot instruction following.

```
