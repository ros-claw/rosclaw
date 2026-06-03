---
pattern_id: pattern_scene_graphs
applicable_symptoms: [scene_graphs]
domain: Memory_Reasoning
---

# Embodied agents lack a compact symbolic representation of environment structure, making high-level reasoning and planning inefficient or brittle.

**Domain**: `Memory_Reasoning`

## Fix

Represent the environment as a scene graph: nodes for objects/regions, edges for spatial/semantic relations, constructed incrementally using a VLM during exploration.

## Anti-pattern

Relying on raw pixel maps or exhaustive 3D reconstruction for spatial reasoning.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with open-vocabulary symbolic abstraction from sensor data.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Planning_Decision** → Incremental 3D Gaussian map updates suggest sliding-window symbolic abstraction for compact, online environment representation.
  - related fix: Incremental 3D Gaussian Splatting for online scene reconstruction and localization, using a sliding window of keyframes to update a 3D Gaussian map and match the goal image via rendering-based pose estimation.
- **Learning_Training** → Use self-supervised pseudo-label generation to create compact symbolic environment representations for efficient planning.
  - related fix: Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.

## Patch

```diff
--- scene_graphs.before.py
+++ scene_graphs.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied agents lack a compact symbolic representation of environment structure, making high-level reasoning and planning inefficient or brittle.

+# Fix    : Represent the environment as a scene graph: nodes for objects/regions, edges for spatial/semantic relations, constructed incrementally using a VLM during exploration.

+# Avoid  : Relying on raw pixel maps or exhaustive 3D reconstruction for spatial reasoning.

```
