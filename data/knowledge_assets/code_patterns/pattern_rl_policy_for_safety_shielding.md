---
pattern_id: pattern_rl_policy_for_safety_shielding
applicable_symptoms: [rl_policy_for_safety_shielding]
domain: Control_Locomotion
---

# Primary navigation policy proposes actions that lead to unsafe states (e.g., tipping over, collision, excessive joint stress) during legged robot navigation.

**Domain**: `Control_Locomotion`

## Fix

Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Anti-pattern

Relying solely on a single task-optimized policy without safety constraints.

## Cross-domain analogies

- **Perception_Vision** → Jointly process proprioception and terrain data for cross-modal safety reasoning.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Planning_Decision** → Use hierarchical decomposition with a high-level safety planner issuing constraints to a low-level locomotion controller.
  - related fix: Hierarchical RL with a high-level navigation planner issuing subgoals to a low-level locomotion controller, both trained via model-free RL.
- **Learning_Training** → Pretrain on safe static poses then fine-tune on dynamic locomotion data.
  - related fix: Two-stage curriculum: pretrain on large-scale web-scraped image-text pairs (Conceptual Captions) then fine-tune on embodied path-instruction data.

## Patch

```diff
--- rl_policy_for_safety_shielding.before.py
+++ rl_policy_for_safety_shielding.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Primary navigation policy proposes actions that lead to unsafe states (e.g., tipping over, collision, excessive joint stress) during legged robot navigation.

+# Fix    : Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

+# Avoid  : Relying solely on a single task-optimized policy without safety constraints.

```
