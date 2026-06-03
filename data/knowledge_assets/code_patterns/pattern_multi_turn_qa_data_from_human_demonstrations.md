---
pattern_id: pattern_multi_turn_qa_data_from_human_demonstrations
applicable_symptoms: [multi_turn_qa_data_from_human_demonstrations]
domain: Memory_Reasoning
---

# VLN agent fails to decompose high-level navigation goals into intermediate reasoning steps, leading to poor compositional decision-making.

**Domain**: `Memory_Reasoning`

## Fix

Convert human demonstration trajectories into multi-turn QA pairs to fine-tune VLM for chain-of-thought reasoning.

## Anti-pattern

Training VLM directly on raw action sequences without explicit reasoning decomposition.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with proximal sub-goals and distal context fusion enables compositional stepwise reasoning.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Planning_Decision** → Hierarchical decomposition via multi-expert training for goal sub-tasks.
  - related fix: Train three RL experts (reaching, squeezing, avoiding) and fine-tune a VLA model (SigLIP+Qwen2-7B) with multi-expert learning, then deploy with online teacher-student training using 4 fisheye cameras on Unitree GO2.
- **Learning_Training** → Use synthetic hierarchical decomposition to generate intermediate subgoal trajectories from structured environment graphs.
  - related fix: Use Marky to programmatically generate 4.2 million synthetic instruction–trajectory pairs from structured environment representations and action sequences.

## Patch

```diff
--- multi_turn_qa_data_from_human_demonstrations.before.py
+++ multi_turn_qa_data_from_human_demonstrations.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to decompose high-level navigation goals into intermediate reasoning steps, leading to poor compositional decision-making.

+# Fix    : Convert human demonstration trajectories into multi-turn QA pairs to fine-tune VLM for chain-of-thought reasoning.

+# Avoid  : Training VLM directly on raw action sequences without explicit reasoning decomposition.

```
