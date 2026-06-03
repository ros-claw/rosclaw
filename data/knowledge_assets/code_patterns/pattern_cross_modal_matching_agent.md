---
pattern_id: pattern_cross_modal_matching_agent
applicable_symptoms: [cross_modal_matching_agent]
domain: Planning_Decision
---

# Discrete VLN models fail in continuous environments due to discrete-to-continuous action gap, causing navigation failures.

**Domain**: `Planning_Decision`

## Fix

Use a waypoint predictor to convert continuous states into discrete high-level action targets (node-to-node jumps), enabling discrete VLN models to navigate continuous environments.

## Anti-pattern

Directly applying discrete VLN models to continuous environments without waypoint abstraction.

## Cross-domain analogies

- **Perception_Vision** → Use language-based continuous representations to bridge the discrete-to-continuous action gap.
  - related fix: Replace visual features with language-based representations (e.g., captions from a vision-language model) for navigation policy input.
- **Learning_Training** → Apply domain randomization to action spaces during training to bridge discrete-continuous gap.
  - related fix: Domain randomization, system identification, or sim-to-real transfer techniques
- **Control_Locomotion** → Use standardized continuous-action benchmarks to train discrete-to-continuous bridging.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- cross_modal_matching_agent.before.py
+++ cross_modal_matching_agent.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Discrete VLN models fail in continuous environments due to discrete-to-continuous action gap, causing navigation failures.

+# Fix    : Use a waypoint predictor to convert continuous states into discrete high-level action targets (node-to-node jumps), enabling discrete VLN models to navigate continuous environments.

+# Avoid  : Directly applying discrete VLN models to continuous environments without waypoint abstraction.

```
