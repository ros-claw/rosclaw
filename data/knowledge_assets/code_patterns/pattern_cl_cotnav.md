---
pattern_id: pattern_cl_cotnav
applicable_symptoms: [cl_cotnav]
domain: Planning_Decision
---

# VLM-driven object-goal navigation suffers from hallucinations and unreliable decisions in unseen environments, leading to low success rate and SPL.

**Domain**: `Planning_Decision`

## Fix

Hierarchical Chain-of-Thought (H-CoT) prompting with closed-loop confidence feedback: decompose navigation into scene-level, region-level, and action-level subgoals, and re-evaluate low-confidence actions via visual re-evaluation cycle.

## Anti-pattern

Prior methods like EmbCLIP and LM-Nav lack confidence-based feedback and hierarchical decomposition, resulting in lower success rate and SPL.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic ray constraints as spherical priors to regularize VLM action sampling, reducing hallucinated offsets.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Hybrid supervised waypoint prediction with closed-loop RL verification to ground VLM decisions.
  - related fix: Hybrid algorithm combining supervised learning for position prediction (waypoint predictor) with reinforcement learning for continuous control, trained jointly in simulation and real environments without requiring autonomous physical flight during training.
- **Control_Locomotion** → Train an end-to-end policy with domain randomization to map noisy VLM outputs directly to actions.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- cl_cotnav.before.py
+++ cl_cotnav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLM-driven object-goal navigation suffers from hallucinations and unreliable decisions in unseen environments, leading to low success rate and SPL.

+# Fix    : Hierarchical Chain-of-Thought (H-CoT) prompting with closed-loop confidence feedback: decompose navigation into scene-level, region-level, and action-level subgoals, and re-evaluate low-confidence actions via visual re-evaluation cycle.

+# Avoid  : Prior methods like EmbCLIP and LM-Nav lack confidence-based feedback and hierarchical decomposition, resulting in lower success rate and SPL.

```
