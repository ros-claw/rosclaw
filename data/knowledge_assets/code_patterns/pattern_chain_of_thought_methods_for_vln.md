---
pattern_id: pattern_chain_of_thought_methods_for_vln
applicable_symptoms: [chain_of_thought_methods_for_vln]
domain: Memory_Reasoning
---

# VLN agent ignores landmark cues in long instructions

**Domain**: `Memory_Reasoning`

## Fix

Use chain-of-thought prompting to decompose long instructions into step-by-step reasoning before action

## Anti-pattern

End-to-end VLN without explicit reasoning steps

## Cross-domain analogies

- **Perception_Vision** → Active perception guides viewpoint selection to reduce landmark ambiguity in long instructions.
  - related fix: Integrate active perception with semantic mapping: agent selects viewpoints to reduce ambiguity while building a task-driven semantic map from RGB-D or lidar data.
- **Planning_Decision** → Use spatial-temporal chain-of-thought to decompose instructions into landmark-grounded reasoning steps.
  - related fix: Use open-source LLMs with spatial-temporal chain-of-thought reasoning that decomposes navigation into instruction comprehension, progress estimation, and decision-making, enhanced by fine-grained object and spatial knowledge.
- **Learning_Training** → Use diverse, standardized instruction-trajectory datasets to prevent overfitting to specific landmark patterns.
  - related fix: Provide large-scale, diverse offline datasets (e.g., RL Unplugged) with standardized evaluation protocols for fair comparison.

## Patch

```diff
--- chain_of_thought_methods_for_vln.before.py
+++ chain_of_thought_methods_for_vln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions

+# Fix    : Use chain-of-thought prompting to decompose long instructions into step-by-step reasoning before action

+# Avoid  : End-to-end VLN without explicit reasoning steps

```
