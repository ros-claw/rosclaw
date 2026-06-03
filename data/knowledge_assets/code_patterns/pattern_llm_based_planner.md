---
pattern_id: pattern_llm_based_planner
applicable_symptoms: [llm_based_planner]
domain: Planning_Decision
---

# LLM-based planner fails to ground natural language commands in physical environments, leading to plans that ignore object semantics or fail to adapt to scene changes.

**Domain**: `Planning_Decision`

## Fix

Use hierarchical scene graph construction from a semantic object map to provide structured, open-vocabulary environment context to the LLM, enabling multi-step plan generation and real-time re-planning.

## Anti-pattern

Using raw LLM without structured scene representation, resulting in plans that hallucinate objects or ignore spatial constraints.

## Cross-domain analogies

- **Perception_Vision** → Cross-modal alignment pretraining can ground language tokens in physical object features for adaptive planning.
  - related fix: Cross-modal alignment pretraining using contrastive or attention-based losses to align visual object features with language tokens.
- **Learning_Training** → Two-stage training: supervised grounding then reinforcement fine-tuning for adaptive plan execution.
  - related fix: Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward
- **Control_Locomotion** → Use lightweight closed-loop verification at high frequency to ground LLM plans in real-time scene feedback.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- llm_based_planner.before.py
+++ llm_based_planner.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: LLM-based planner fails to ground natural language commands in physical environments, leading to plans that ignore object semantics or fail to adapt to scene changes.

+# Fix    : Use hierarchical scene graph construction from a semantic object map to provide structured, open-vocabulary environment context to the LLM, enabling multi-step plan generation and real-time re-planning.

+# Avoid  : Using raw LLM without structured scene representation, resulting in plans that hallucinate objects or ignore spatial constraints.

```
