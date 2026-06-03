---
pattern_id: pattern_visual_prompt_vp_technique
applicable_symptoms: [visual_prompt_vp_technique]
domain: Perception_Vision
---

# Perception hallucinations and poor spatial understanding cause navigation agents to fail in zero-shot settings.

**Domain**: `Perception_Vision`

## Fix

Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.

## Anti-pattern

Single-view input without structured prompting leads to hallucinated or ambiguous perceptual cues.

## Cross-domain analogies

- **Planning_Decision** → Use multi-modal diffusion conditioning to fuse pixel and latent features for robust spatial grounding.
  - related fix: Use a Diffusion Transformer policy with multi-modal conditioning (pixel goals + latent features) as System 1 in a dual-system architecture to generate smooth, continuous trajectories in real time.
- **Learning_Training** → Use closed-loop verification to filter synthetic visual data, iteratively retraining the perception model.
  - related fix: Self-Refining Data Flywheel: generate synthetic navigation trajectories via a teacher policy, filter with a learned verifier, and iteratively retrain the student policy on the augmented data.
- **Control_Locomotion** → Train model-free RL with domain randomization fusing vision and commands to reduce hallucinations.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- visual_prompt_vp_technique.before.py
+++ visual_prompt_vp_technique.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Perception hallucinations and poor spatial understanding cause navigation agents to fail in zero-shot settings.

+# Fix    : Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.

+# Avoid  : Single-view input without structured prompting leads to hallucinated or ambiguous perceptual cues.

```
