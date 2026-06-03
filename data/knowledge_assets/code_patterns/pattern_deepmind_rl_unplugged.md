---
pattern_id: pattern_deepmind_rl_unplugged
applicable_symptoms: [deepmind_rl_unplugged]
domain: Learning_Training
---

# Offline RL algorithms overfit to limited data and fail to generalize to unseen states.

**Domain**: `Learning_Training`

## Fix

Provide large-scale, diverse offline datasets (e.g., RL Unplugged) with standardized evaluation protocols for fair comparison.

## Anti-pattern

Training offline RL on small, homogeneous datasets leads to poor policy performance.

## Cross-domain analogies

- **Perception_Vision** → Fuse diverse offline datasets into a unified latent state representation to improve generalization.
  - related fix: Propose a multimodal occupancy perception system that fuses vision, depth, and other sensor data into a unified occupancy representation for humanoid robots.
- **Planning_Decision** → Multi-expert decomposition with online closed-loop correction prevents offline overfitting by exploring unseen states.
  - related fix: Train three RL experts (reaching, squeezing, avoiding) and fine-tune a VLA model (SigLIP+Qwen2-7B) with multi-expert learning, then deploy with online teacher-student training using 4 fisheye cameras on Unitree GO2.
- **Control_Locomotion** → Use closed-loop rollouts with learned dynamics to correct offline value estimates online.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- deepmind_rl_unplugged.before.py
+++ deepmind_rl_unplugged.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Offline RL algorithms overfit to limited data and fail to generalize to unseen states.

+# Fix    : Provide large-scale, diverse offline datasets (e.g., RL Unplugged) with standardized evaluation protocols for fair comparison.

+# Avoid  : Training offline RL on small, homogeneous datasets leads to poor policy performance.

```
