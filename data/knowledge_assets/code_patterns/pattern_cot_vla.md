---
pattern_id: pattern_cot_vla
applicable_symptoms: [cot_vla]
domain: Planning_Decision
---

# Token inflation from generating imagined visual observations makes real-time navigation impractical due to high computational cost and latency.

**Domain**: `Planning_Decision`

## Fix

FantasyVLN: a method that addresses token inflation while preserving visual chain-of-thought reasoning.

## Anti-pattern

CoT-VLA's naive generation of full visual observations as intermediate tokens.

## Cross-domain analogies

- **Perception_Vision** → Use a compact latent projection to map imagined tokens into a fixed coordinate space.
  - related fix: Use pinhole camera projection model with intrinsic matrix K and extrinsic matrix [R|t] to map 3D world points to 2D image coordinates, enabling local-to-world transformations.
- **Learning_Training** → Use off-policy cheap priors to bootstrap, then shift to expensive on-policy refinement only when needed.
  - related fix: Mixed Imitation and Reinforcement Learning (MIRL): bootstrap policy via off-policy imitation learning, then refine with on-policy RL, gradually shifting weight from imitation to RL.
- **Control_Locomotion** → Use standardized low-level action benchmarks to prune unnecessary visual tokens.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- cot_vla.before.py
+++ cot_vla.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Token inflation from generating imagined visual observations makes real-time navigation impractical due to high computational cost and latency.

+# Fix    : FantasyVLN: a method that addresses token inflation while preserving visual chain-of-thought reasoning.

+# Avoid  : CoT-VLA's naive generation of full visual observations as intermediate tokens.

```
