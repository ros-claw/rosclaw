---
pattern_id: pattern_cross_modal_planner
applicable_symptoms: [cross_modal_planner]
domain: Planning_Decision
---

# Navigation agents fail to ground linguistic commands in spatial representations, leading to incoherent long-horizon plans.

**Domain**: `Planning_Decision`

## Fix

Cross-modal transformer with attention aligning topological map features and language instruction embeddings to output subgoal sequences.

## Anti-pattern

Using separate unimodal planners for language and map without cross-modal fusion.

## Cross-domain analogies

- **Perception_Vision** → Use text-to-imagination grounding loss to align language with spatial embeddings.
  - related fix: Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings
- **Learning_Training** → Use human demonstration data to train agents to map language commands to spatial plans.
  - related fix: Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.
- **Control_Locomotion** → Incorporate linguistic context into planning policy to dynamically adapt subgoal selection and spatial reasoning.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- cross_modal_planner.before.py
+++ cross_modal_planner.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents fail to ground linguistic commands in spatial representations, leading to incoherent long-horizon plans.

+# Fix    : Cross-modal transformer with attention aligning topological map features and language instruction embeddings to output subgoal sequences.

+# Avoid  : Using separate unimodal planners for language and map without cross-modal fusion.

```
