---
pattern_id: pattern_eb_manipulation
applicable_symptoms: [eb_manipulation]
domain: Control_Locomotion
---

# Embodied agents fail to execute precise low-level manipulation tasks (e.g., pick-and-place, peg insertion) due to insufficient spatial reasoning and perception accuracy.

**Domain**: `Control_Locomotion`

## Fix

Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Anti-pattern

Relying solely on high-level task planning without fine-grained control and perception leads to poor manipulation performance.

## Cross-domain analogies

- **Perception_Vision** → Pre-train on grounded entity-level annotations to improve spatial reasoning for precise manipulation.
  - related fix: Pre-train on GEL-R2R, a dataset with grounded entity-level annotations, to enable cross-modal alignment at the entity level.
- **Planning_Decision** → Use a topological graph of keyframes and affordances as a discrete action space for precise manipulation planning.
  - related fix: Use a topological map that stores viewpoints, objects, and spatial relationships as a graph, serving as the global action space for an LLM planner to select next navigation actions via node selection instead of continuous coordinates.
- **Learning_Training** → Use domain randomization to vary object poses and textures during training for robust spatial reasoning.
  - related fix: Domain randomization, system identification, or sim-to-real transfer techniques

## Patch

```diff
--- eb_manipulation.before.py
+++ eb_manipulation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied agents fail to execute precise low-level manipulation tasks (e.g., pick-and-place, peg insertion) due to insufficient spatial reasoning and perception accuracy.

+# Fix    : Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

+# Avoid  : Relying solely on high-level task planning without fine-grained control and perception leads to poor manipulation performance.

```
