---
pattern_id: pattern_low_level_planner
applicable_symptoms: [low_level_planner]
domain: Planning_Decision
---

# Long-horizon navigation agents fail to recover from execution errors (e.g., stuck or off-course) during trajectory following.

**Domain**: `Planning_Decision`

## Fix

Exploration-Verification strategy: alternate between advancing along the predicted trajectory and checking for successful completion; if error detected, trigger corrective motion.

## Anti-pattern

Open-loop execution of high-level sub-instructions without per-step verification.

## Cross-domain analogies

- **Perception_Vision** → Replace visual features with closed-loop language verification to detect and correct execution errors.
  - related fix: Replace visual features with language-based representations (e.g., captions from a vision-language model) for navigation policy input.
- **Learning_Training** → Apply dropout-like random masking of subgoals to prevent over-reliance on any single recovery path.
  - related fix: Apply dropout operations at multiple feature scales (activations, channels, spatial regions, entire feature maps) to regularize training.
- **Control_Locomotion** → Closed-loop recovery by systematic alternative action search when deviation is detected.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- low_level_planner.before.py
+++ low_level_planner.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon navigation agents fail to recover from execution errors (e.g., stuck or off-course) during trajectory following.

+# Fix    : Exploration-Verification strategy: alternate between advancing along the predicted trajectory and checking for successful completion; if error detected, trigger corrective motion.

+# Avoid  : Open-loop execution of high-level sub-instructions without per-step verification.

```
