---
pattern_id: pattern_transformer_based_cross_modal_planner
applicable_symptoms: [transformer_based_cross_modal_planner]
domain: Planning_Decision
---

# VLN agent fails to align visual topological map nodes with language instructions, leading to incoherent subgoal sequences.

**Domain**: `Planning_Decision`

## Fix

Use a Transformer backbone with cross-modal attention to fuse visual topological map nodes and language instructions, generating a coherent sequence of subgoals.

## Anti-pattern

Using separate unimodal encoders without cross-modal attention for planning.

## Cross-domain analogies

- **Perception_Vision** → Use joint cross-modal reasoning to align visual nodes with language subgoals.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Learning_Training** → Use synthetic instruction-map alignment pairs to train subgoal grounding.
  - related fix: Train a transformer agent on 4.2 million synthetic instruction-trajectory pairs generated at scale, reducing reliance on human demonstrations.
- **Control_Locomotion** → Train an end-to-end policy mapping language and visual inputs directly to subgoal sequences via RL with domain randomization.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- transformer_based_cross_modal_planner.before.py
+++ transformer_based_cross_modal_planner.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to align visual topological map nodes with language instructions, leading to incoherent subgoal sequences.

+# Fix    : Use a Transformer backbone with cross-modal attention to fuse visual topological map nodes and language instructions, generating a coherent sequence of subgoals.

+# Avoid  : Using separate unimodal encoders without cross-modal attention for planning.

```
