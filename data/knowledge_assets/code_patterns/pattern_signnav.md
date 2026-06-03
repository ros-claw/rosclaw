---
pattern_id: pattern_signnav
applicable_symptoms: [signnav]
domain: Planning_Decision
---

# Navigation agents fail in novel large-scale indoor environments because they cannot interpret human-readable signs (text, symbols, arrows) to dynamically plan paths.

**Domain**: `Planning_Decision`

## Fix

SignNav task: agent interprets semantic hints from signage and updates plan in real-time as new signs are encountered.

## Anti-pattern

Traditional navigation relying solely on geometric maps or pre-defined paths.

## Cross-domain analogies

- **Perception_Vision** → Use SLAM-derived semantic maps as ground-truth references to supervise sign interpretation for path planning.
  - related fix: Use SLAM-derived trajectories as ground-truth motion tendency references to supervise or condition video world model predictions.
- **Learning_Training** → Use back-translation to generate synthetic sign-guided paths from layout maps, then train with visual dropout for robust sign interpretation.
  - related fix: Use back-translation: generate new instructions from paths and new paths from instructions using a pre-trained model, combined with environmental dropout to create diverse training triplets.
- **Control_Locomotion** → Use standardized sign-interpretation benchmarks to train agents on precise visual-linguistic reasoning for dynamic path planning.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- signnav.before.py
+++ signnav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents fail in novel large-scale indoor environments because they cannot interpret human-readable signs (text, symbols, arrows) to dynamically plan paths.

+# Fix    : SignNav task: agent interprets semantic hints from signage and updates plan in real-time as new signs are encountered.

+# Avoid  : Traditional navigation relying solely on geometric maps or pre-defined paths.

```
