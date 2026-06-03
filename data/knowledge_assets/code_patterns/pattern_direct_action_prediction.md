---
pattern_id: pattern_direct_action_prediction
applicable_symptoms: [direct_action_prediction]
domain: Planning_Decision
---

# Direct action prediction lacks robustness and interpretability in complex navigation tasks, leading to poor performance compared to reasoning-based methods.

**Domain**: `Planning_Decision`

## Fix

Use chain-of-thought reasoning to decompose tasks into step-by-step subgoals before predicting actions.

## Anti-pattern

Direct end-to-end mapping from perception to action without intermediate reasoning.

## Cross-domain analogies

- **Perception_Vision** → Use a structured projection model to map abstract action spaces into interpretable world-relative coordinates.
  - related fix: Use pinhole camera projection model with intrinsic matrix K and extrinsic matrix [R|t] to map 3D world points to 2D image coordinates, enabling local-to-world transformations.
- **Learning_Training** → Apply random feature masking to action inputs to force reliance on higher-level reasoning.
  - related fix: Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.
- **Control_Locomotion** → Distill reasoning trajectories via DAgger and fine-tune with RL for robust, interpretable navigation.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- direct_action_prediction.before.py
+++ direct_action_prediction.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Direct action prediction lacks robustness and interpretability in complex navigation tasks, leading to poor performance compared to reasoning-based methods.

+# Fix    : Use chain-of-thought reasoning to decompose tasks into step-by-step subgoals before predicting actions.

+# Avoid  : Direct end-to-end mapping from perception to action without intermediate reasoning.

```
