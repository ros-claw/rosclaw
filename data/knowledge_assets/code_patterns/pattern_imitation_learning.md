---
pattern_id: pattern_imitation_learning
applicable_symptoms: [imitation_learning]
domain: Learning_Training
---

# Imitation learning from human demonstrations is expensive and limited in scale due to the need for human teleoperation.

**Domain**: `Learning_Training`

## Fix

Train a transformer agent on 4.2 million synthetic instruction-trajectory pairs generated at scale, reducing reliance on human demonstrations.

## Anti-pattern

Training RL agents from scratch without a strong behavioral prior.

## Cross-domain analogies

- **Perception_Vision** → Incremental object-centric mapping suggests using cheap, scalable data sources to build task representations frame-by-frame.
  - related fix: Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.
- **Planning_Decision** → Combine learning-based imitation with classical control to generate scalable, constraint-aware policies from limited human demonstrations.
  - related fix: Combine learning-based motion planners with classical control to generate collision-free trajectories that respect kinematics, dynamics, and environmental constraints, and adapt plans in real time to dynamic obstacles.
- **Control_Locomotion** → Use EB-Manipulation-style standardized benchmarks to auto-generate synthetic demonstrations for imitation learning.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- imitation_learning.before.py
+++ imitation_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Imitation learning from human demonstrations is expensive and limited in scale due to the need for human teleoperation.

+# Fix    : Train a transformer agent on 4.2 million synthetic instruction-trajectory pairs generated at scale, reducing reliance on human demonstrations.

+# Avoid  : Training RL agents from scratch without a strong behavioral prior.

```
