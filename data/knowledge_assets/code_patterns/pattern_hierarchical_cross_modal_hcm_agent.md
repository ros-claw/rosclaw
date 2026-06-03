---
pattern_id: pattern_hierarchical_cross_modal_hcm_agent
applicable_symptoms: [hierarchical_cross_modal_hcm_agent]
domain: Planning_Decision
---

# VLN agents struggle with long trajectories and continuous 3D environments due to monolithic policies that fail to decompose navigation into manageable subgoals.

**Domain**: `Planning_Decision`

## Fix

Hierarchical high- and low-level policies: high-level selects subgoals from visual and linguistic inputs, low-level executes continuous motor commands to reach subgoals; modularized training decouples reasoning and imitation.

## Anti-pattern

End-to-end monolithic policies that jointly learn reasoning and motor control, leading to poor sample efficiency and generalization.

## Cross-domain analogies

- **Perception_Vision** → Use imagined subgoals from text to decompose long trajectories via auxiliary loss.
  - related fix: Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings
- **Learning_Training** → Use subgoal randomization during training to decompose long trajectories into robust, variable subgoals.
  - related fix: Use domain randomization: vary simulation parameters (friction, mass, lighting, delay) randomly during training to improve policy robustness to real-world conditions.
- **Control_Locomotion** → Use visual subgoal decomposition to break long trajectories into manageable local navigation tasks.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- hierarchical_cross_modal_hcm_agent.before.py
+++ hierarchical_cross_modal_hcm_agent.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents struggle with long trajectories and continuous 3D environments due to monolithic policies that fail to decompose navigation into manageable subgoals.

+# Fix    : Hierarchical high- and low-level policies: high-level selects subgoals from visual and linguistic inputs, low-level executes continuous motor commands to reach subgoals; modularized training decouples reasoning and imitation.

+# Avoid  : End-to-end monolithic policies that jointly learn reasoning and motor control, leading to poor sample efficiency and generalization.

```
