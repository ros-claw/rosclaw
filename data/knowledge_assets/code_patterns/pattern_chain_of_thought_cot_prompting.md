---
pattern_id: pattern_chain_of_thought_cot_prompting
applicable_symptoms: [chain_of_thought_cot_prompting]
domain: Planning_Decision
---

# Embodied agents fail to decompose long-horizon navigation tasks into manageable subgoals, leading to poor perception and decision-making in complex environments.

**Domain**: `Planning_Decision`

## Fix

Use Hierarchical Chain-of-Thought (H-CoT) prompting to decompose high-level goals into stepwise subgoals (e.g., region → furniture → object), enabling compositional reasoning for navigation and manipulation.

## Anti-pattern

Flat reasoning without intermediate steps, which fails to integrate sensor data and semantic cues effectively.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic subgoal generation from task descriptions with auxiliary alignment loss for hierarchical decomposition.
  - related fix: Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.
- **Learning_Training** → Regularize subgoal predictions to remain consistent with prior task structure via output constraints.
  - related fix: Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.
- **Control_Locomotion** → Use blocked-action heuristic to trigger dynamic subgoal decomposition.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- chain_of_thought_cot_prompting.before.py
+++ chain_of_thought_cot_prompting.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied agents fail to decompose long-horizon navigation tasks into manageable subgoals, leading to poor perception and decision-making in complex environments.

+# Fix    : Use Hierarchical Chain-of-Thought (H-CoT) prompting to decompose high-level goals into stepwise subgoals (e.g., region → furniture → object), enabling compositional reasoning for navigation and manipulation.

+# Avoid  : Flat reasoning without intermediate steps, which fails to integrate sensor data and semantic cues effectively.

```
