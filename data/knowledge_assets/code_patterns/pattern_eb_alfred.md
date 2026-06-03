---
pattern_id: pattern_eb_alfred
applicable_symptoms: [eb_alfred]
domain: Planning_Decision
---

# Long-horizon task decomposition from natural language instructions fails to produce executable subgoal sequences in embodied environments.

**Domain**: `Planning_Decision`

## Fix

Use ALFRED environment's predefined action space and scene configurations to decompose natural language goals into ordered high-level action steps (e.g., 'pick up the book', 'move to desk', 'place on desk').

## Anti-pattern

Directly mapping language to low-level motor commands without intermediate subgoal planning.

## Cross-domain analogies

- **Perception_Vision** → Dual-view decomposition: fuse complementary task perspectives into a single planning prompt.
  - related fix: Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.
- **Learning_Training** → Use closed-loop verification to iteratively refine subgoal sequences via backward reconstruction of instructions.
  - related fix: Train a Transformer-based Speaker jointly with a Listener in a Double Back-Translation loop, where the Speaker generates instructions from paths and the Listener reconstructs paths from instructions, enforcing instruction-path consistency through iterative refinement.
- **Control_Locomotion** → Use a safety-critic to verify and override subgoal proposals exceeding execution risk.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- eb_alfred.before.py
+++ eb_alfred.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon task decomposition from natural language instructions fails to produce executable subgoal sequences in embodied environments.

+# Fix    : Use ALFRED environment's predefined action space and scene configurations to decompose natural language goals into ordered high-level action steps (e.g., 'pick up the book', 'move to desk', 'place on desk').

+# Avoid  : Directly mapping language to low-level motor commands without intermediate subgoal planning.

```
