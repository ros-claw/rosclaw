---
pattern_id: pattern_system_2
applicable_symptoms: [system_2]
domain: Planning_Decision
---

# VLN agent lacks high-level reasoning for long-horizon navigation, relying on reactive low-level control that fails to produce coherent mid-term goals.

**Domain**: `Planning_Decision`

## Fix

Use a VLM-based global planner (System 2) to predict mid-term waypoint goals via image-grounded reasoning, then pass them to a low-level controller (System 1) for execution.

## Anti-pattern

End-to-end reactive policies without explicit hierarchical decomposition.

## Cross-domain analogies

- **Perception_Vision** → Cross-modal alignment pretraining can structure latent goal representations for hierarchical planning.
  - related fix: Cross-modal alignment pretraining using contrastive or attention-based losses to align visual object features with language tokens.
- **Learning_Training** → Use a speaker model to generate synthetic high-level subgoal sequences from unannotated trajectories for hierarchical planning.
  - related fix: Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.
- **Control_Locomotion** → Incorporate environmental priors into the policy to dynamically generate subgoals for long-horizon navigation.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- system_2.before.py
+++ system_2.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent lacks high-level reasoning for long-horizon navigation, relying on reactive low-level control that fails to produce coherent mid-term goals.

+# Fix    : Use a VLM-based global planner (System 2) to predict mid-term waypoint goals via image-grounded reasoning, then pass them to a low-level controller (System 1) for execution.

+# Avoid  : End-to-end reactive policies without explicit hierarchical decomposition.

```
