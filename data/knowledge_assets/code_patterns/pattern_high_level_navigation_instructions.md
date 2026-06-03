---
pattern_id: pattern_high_level_navigation_instructions
applicable_symptoms: [high_level_navigation_instructions]
domain: Planning_Decision
---

# Natural language navigation instructions are too abstract for robots to map directly to low-level control, causing execution failures.

**Domain**: `Planning_Decision`

## Fix

Use a Grounded Semantic Mapping Network (GSMN) to decompose high-level instructions into spatially grounded action sequences for visual navigation.

## Anti-pattern

Directly mapping language to low-level commands without semantic grounding.

## Cross-domain analogies

- **Perception_Vision** → Map natural language instructions to learned hierarchical control representations for closed-loop execution.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Learning_Training** → Use hierarchical decomposition: train a high-level policy for instruction interpretation, then a low-level policy for control via reinforcement.
  - related fix: Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward
- **Control_Locomotion** → Use reinforcement learning to map abstract language tokens directly to low-level motor commands.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- high_level_navigation_instructions.before.py
+++ high_level_navigation_instructions.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Natural language navigation instructions are too abstract for robots to map directly to low-level control, causing execution failures.

+# Fix    : Use a Grounded Semantic Mapping Network (GSMN) to decompose high-level instructions into spatially grounded action sequences for visual navigation.

+# Avoid  : Directly mapping language to low-level commands without semantic grounding.

```
