---
pattern_id: pattern_ainav
applicable_symptoms: [ainav]
domain: Planning_Decision
---

# Robot cannot reach navigation goals in cluttered environments where no direct viable path exists because it only avoids obstacles instead of interacting with them.

**Domain**: `Planning_Decision`

## Fix

AINav: adaptive interactive navigation using LLM-driven reasoning, a primitive skill tree, and RL-trained interaction skills (push, slide, climb) to proactively manipulate obstacles and replan on the fly.

## Anti-pattern

Static path planning that avoids obstacles without interaction, failing when no clear path exists.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical open-vocabulary graph segmentation enables interaction with unknown objects by reclassifying obstacles as manipulable entities.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Learning_Training** → Use closed-loop data aggregation to iteratively learn corrective interactions with obstacles.
  - related fix: Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).
- **Control_Locomotion** → Train an end-to-end policy via RL to directly map sensor inputs to obstacle-interaction actions.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- ainav.before.py
+++ ainav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Robot cannot reach navigation goals in cluttered environments where no direct viable path exists because it only avoids obstacles instead of interacting with them.

+# Fix    : AINav: adaptive interactive navigation using LLM-driven reasoning, a primitive skill tree, and RL-trained interaction skills (push, slide, climb) to proactively manipulate obstacles and replan on the fly.

+# Avoid  : Static path planning that avoids obstacles without interaction, failing when no clear path exists.

```
