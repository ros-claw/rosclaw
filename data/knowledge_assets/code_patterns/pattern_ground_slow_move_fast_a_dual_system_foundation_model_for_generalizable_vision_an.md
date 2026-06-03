---
pattern_id: pattern_ground_slow_move_fast_a_dual_system_foundation_model_for_generalizable_vision_an
applicable_symptoms: [ground_slow_move_fast_a_dual_system_foundation_model_for_generalizable_vision_an]
domain: Planning_Decision
---

# VLN agents struggle with real-time obstacle avoidance and smooth trajectory generation under high-frequency control loops when using large VLMs end-to-end.

**Domain**: `Planning_Decision`

## Fix

Dual-system architecture: a slow (2 Hz) VLM-based global planner predicts pixel-level goals, and a fast (30 Hz) lightweight Diffusion Transformer generates smooth, collision-free trajectories conditioned on the goal and current RGB.

## Anti-pattern

End-to-end large VLMs running at low frequency for both planning and control, leading to jerky motion and poor obstacle avoidance.

## Cross-domain analogies

- **Perception_Vision** → Pre-train on grounded trajectory-action pairs to enable fine-grained cross-modal alignment for control.
  - related fix: Pre-train on GEL-R2R, a dataset with grounded entity-level annotations, to enable cross-modal alignment at the entity level.
- **Learning_Training** → Use adversarial training to refine trajectory outputs against a real-time collision discriminator.
  - related fix: Adversarial Variational Autoencoder (aVAE) combining VAE with adversarial training: use a discriminator to distinguish real data from VAE-generated samples, and train the VAE encoder/decoder to fool the discriminator.
- **Control_Locomotion** → Distill large VLMs into lightweight experts with DAgger and RL fine-tuning for real-time closed-loop control.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- ground_slow_move_fast_a_dual_system_foundation_model_for_generalizable_vision_an.before.py
+++ ground_slow_move_fast_a_dual_system_foundation_model_for_generalizable_vision_an.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents struggle with real-time obstacle avoidance and smooth trajectory generation under high-frequency control loops when using large VLMs end-to-end.

+# Fix    : Dual-system architecture: a slow (2 Hz) VLM-based global planner predicts pixel-level goals, and a fast (30 Hz) lightweight Diffusion Transformer generates smooth, collision-free trajectories conditioned on the goal and current RGB.

+# Avoid  : End-to-end large VLMs running at low frequency for both planning and control, leading to jerky motion and poor obstacle avoidance.

```
