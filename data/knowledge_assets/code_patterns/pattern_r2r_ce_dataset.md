---
pattern_id: pattern_r2r_ce_dataset
applicable_symptoms: [r2r_ce_dataset]
domain: Planning_Decision
---

# VLN agents trained on discrete action spaces fail to generalize to continuous environments with free movement, leading to poor navigation performance.

**Domain**: `Planning_Decision`

## Fix

Adapt existing discrete VLN datasets (e.g., RxR) to continuous action spaces by converting human-annotated paths into continuous trajectories in photorealistic simulators (Matterport3D), enabling training and evaluation of agents in realistic continuous environments.

## Anti-pattern

Using discrete action spaces (e.g., R2R dataset) for VLN in continuous environments, which limits agent motion realism and causes sim-to-real gap.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with proximal and distal sub-networks for continuous action spaces.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Learning_Training** → Apply random masking to action space dimensions during training to force reliance on language instructions.
  - related fix: Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.
- **Control_Locomotion** → Use multi-expert distillation with DAgger to train on continuous trajectories from discrete action priors.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- r2r_ce_dataset.before.py
+++ r2r_ce_dataset.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents trained on discrete action spaces fail to generalize to continuous environments with free movement, leading to poor navigation performance.

+# Fix    : Adapt existing discrete VLN datasets (e.g., RxR) to continuous action spaces by converting human-annotated paths into continuous trajectories in photorealistic simulators (Matterport3D), enabling training and evaluation of agents in realistic continuous environments.

+# Avoid  : Using discrete action spaces (e.g., R2R dataset) for VLN in continuous environments, which limits agent motion realism and causes sim-to-real gap.

```
