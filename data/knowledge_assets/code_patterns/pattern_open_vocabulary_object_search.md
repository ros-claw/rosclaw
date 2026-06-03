---
pattern_id: pattern_open_vocabulary_object_search
applicable_symptoms: [open_vocabulary_object_search]
domain: Planning_Decision
---

# Robot cannot find objects described by natural language in unstructured outdoor environments without prior maps or pre-enumerated object categories.

**Domain**: `Planning_Decision`

## Fix

Combine foundation models for zero-shot language understanding and visual grounding with geometric exploration to plan and execute efficient search trajectories.

## Anti-pattern

Pre-training on specific object classes and relying on pre-built maps.

## Cross-domain analogies

- **Perception_Vision** → Use open-vocabulary hierarchical 3D graphs to map natural language queries to incremental LiDAR and vision features.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Learning_Training** → Use full-kinematics agents with robust physics to ground language in continuous, dynamic outdoor environments.
  - related fix: Use full-kinematics agents with a robust physics engine to enable realistic motion dynamics and high-fidelity simulation, reducing sim-to-real gap.
- **Control_Locomotion** → Train an end-to-end policy mapping language and sensor data to actions via large-scale RL with domain randomization.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- open_vocabulary_object_search.before.py
+++ open_vocabulary_object_search.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Robot cannot find objects described by natural language in unstructured outdoor environments without prior maps or pre-enumerated object categories.

+# Fix    : Combine foundation models for zero-shot language understanding and visual grounding with geometric exploration to plan and execute efficient search trajectories.

+# Avoid  : Pre-training on specific object classes and relying on pre-built maps.

```
